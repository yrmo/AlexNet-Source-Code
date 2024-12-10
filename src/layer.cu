/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <helper_cuda.h>
#include <iostream>

#include <layer_kernels.cuh>
#include <layer.cuh>
#include <data.cuh>
#include <util.cuh>
#include <cudaconv2.cuh>
#include <matrix.h>
#include <set>

using namespace std;

/* 
 * =======================
 * Layer
 * =======================
 */

Layer::Layer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool trans) : 
             _convNetGPU(convNetGPU),  _trans(trans) {
    _name = pyDictGetString(paramsDict, "name");
    _type = pyDictGetString(paramsDict, "type");
    
    _numGradProducersNext = 0;
    _foundGradConsumers = false;
    _gradConsumer = pyDictGetInt(paramsDict, "gradConsumer");
    _actsTarget = pyDictGetInt(paramsDict, "actsTarget");
    _actsGradTarget = pyDictGetInt(paramsDict, "actsGradTarget");
    _conserveMem = pyDictGetInt(paramsDict, "conserveMem");
    _numOutputs = pyDictGetInt(paramsDict, "outputs");
    _deviceID = _convNetGPU == NULL ? -1 : _convNetGPU->getDeviceID(); // DataLayer doesnt have a device ID

    _bwdTerminal = false;
    _rcvdBInputMsgs = 0;

    PyObject* quantF = PyDict_GetItemString(paramsDict, "quantF");
    PyObject* quantB = PyDict_GetItemString(paramsDict, "quantB");
    _fwdQuantizer = &Quantizer::make(quantF);
    _bwdQuantizer = &Quantizer::make(quantB);
}

void Layer::shuffle(intv& v) {
    for (int i = 0; i < v.size(); ++i) {
        int r1 = rand() % v.size();
        int r2 = rand() % v.size();
        int tmp = v[r1];
        v[r1] = v[r2];
        v[r2] = tmp;
    }
}

void Layer::fpropNext(PASS_TYPE passType) {
    set<int> devices; // The set of devices onto which I have copied my output
    // If I must copy my outputs to a GPU on another PCH, make sure
    // I copy them to host memory.
    for (intv::iterator it = _nextDeviceIDs.begin(); it != _nextDeviceIDs.end(); ++it) {
        int d = *it;
        if (!NVMatrix::canAccessDevice(_deviceID, d)) {
            _fwdQuantizer->quantize(*_outputs[_deviceID], _hostMemFwd);
            break;
        }
    }
    // Sync so that we don't send out messages about data that hasn't been formed yet
    if (_nextDeviceIDs.size() > 1 || (_nextDeviceIDs.size() == 1 && _nextDeviceIDs[0] != _deviceID)) {
        cudaDeviceSynchronize();
    }
    for (int i = 0; i < _next.size(); i++) {
        int d = _next[i]->getDeviceID();
        if (d != _deviceID && devices.count(d) == 0) {
            // Copy my output to next layer's GPU
            if (NVMatrix::canAccessDevice(_deviceID, d)) {
                // Clone the matrix because the next layers in this thread may transpose it and stuff
                _next[i]->getConvNetGPU().enqueueMessage(new CopyMessage(&_outputs[_deviceID]->getClone(), _outputs[d]));
            } else { // This will go through host
                _next[i]->getConvNetGPU().enqueueMessage(new DequantizeMessage(_fwdQuantizer, _outputs[d]));
            }

            devices.insert(d);
        }
        // Inform the next layer that my output is ready
        _next[i]->getConvNetGPU().enqueueMessage(new FpropMessage(_name, _next[i]->getName(), passType));
    }
}

void Layer::truncBwdActs() {
    // Only truncate actsGrad if I own it
    if (_conserveMem && _actsGradTarget < 0) {
        for (map<int,NVMatrix*>::iterator it = _actsGrad.begin(); it != _actsGrad.end(); ++it) {
            getActsGrad(it->first).truncate();
        }
    }
    if (_conserveMem) {
        getActs().truncate();
        for (map<int,NVMatrix*>::iterator it = _outputs.begin(); it != _outputs.end(); ++it) {
            NVMatrix::setDeviceID(it->first);
            getActs(it->first).truncate();
        }
        NVMatrix::setDeviceID(_deviceID);
    }
}

void Layer::fprop(PASS_TYPE passType) {
    _rcvdFInputs++;
    if (_rcvdFInputs == _prev.size()) {
        assert(_deviceID == NVMatrix::getDeviceID());
        NVMatrixV v;
        for (int i = 0; i < _prev.size(); i++) {
            v.push_back(&_prev[i]->getActs(_deviceID));
        }
        fprop(v, passType);
    }
}

void Layer::fprop(NVMatrix& v, PASS_TYPE passType) {
    NVMatrixV vl;
    vl.push_back(&v);
    fprop(vl, passType);
}

void Layer::fprop(NVMatrixV& v, PASS_TYPE passType) {
    assert(v.size() == _prev.size());
    _inputs.clear();
    _inputs.insert(_inputs.begin(), v.begin(), v.end());
    _outputs[_deviceID] = _actsTarget < 0 ? _outputs[_deviceID] : _inputs[_actsTarget];
    _rcvdFInputs = _prev.size();
    for (NVMatrixV::iterator it = v.begin(); it != v.end(); ++it) {
        (*it)->transpose(_trans);
    }
    getActs().transpose(_trans);
    
    // First do fprop on the input whose acts matrix I'm sharing, if any
    if (_actsTarget >= 0) {
        fpropActs(_actsTarget, 0, passType);
    }
    // Then add the rest of the inputs to that
    for (int i = 0; i < _prev.size(); i++) {
        if (i != _actsTarget) {
            fpropActs(i, _actsTarget >= 0 || i > 0, passType);
        }
    }
    fpropNext(passType);
}

void Layer::bprop(PASS_TYPE passType) {
    //printf("layer %s got bprop, total rcvd %d/%d\n", _name.c_str(), getRcvdBInputs(_deviceID) + _rcvdBInputMsgs, _numGradProducersNext);
    if (getRcvdBInputs(_deviceID) + _rcvdBInputMsgs == _numGradProducersNext) {
        // Sum the contributions to my activity gradients from each GPU
        // and store the result in this matrix called v.
#ifdef USE_PD
        NVMatrix* v = &getActsGrad();
        v->transpose(_trans);
        bool doCopy = getRcvdBInputs(_deviceID) == 0;
        seti interested;
        interested.insert(_nextDeviceIDs.begin(), _nextDeviceIDs.end());
        interested.erase(_deviceID);
        while (interested.size() > 0) {
            int d = getPipeDispenser().getPipe(interested);
            //cout << _name << " got pipe " << d << endl;
            interested.erase(d);
            NVMatrix& mat = getActsGrad(d); // This lives on the other device
            mat.transpose(_trans);
            if (mat.getNumElements() != 0 && getRcvdBInputs(d) > 0) {
                if (!NVMatrix::canAccessDevice(_deviceID, d)) {
                    // Copy the gradients produced by device d from his GPU to CPU memory
                    // (since a direct GPU-GPU copy is impossible in this case)

                    NVMatrix::setDeviceID(d);
                    _bwdQuantizer->quantize(mat, _hostMemBwd);
                    // I have verified that synchronization *is* necessary here.
                    // Even without any explicit mention of streams, kernel calls
                    // from the same host thread on different devices execute simultaneously.
                    cudaDeviceSynchronize();
                    getPipeDispenser().freePipe(d);
                    NVMatrix::setDeviceID(_deviceID);
//                    getPipeDispenser().getPipe(_deviceID);
                    _bwdQuantizer->dequantize(*v, !doCopy, 1);
//                    cudaDeviceSynchronize();
//                    getPipeDispenser().freePipe(_deviceID);
                } else { 
//                    getPipeDispenser().getPipe(_deviceID);
                    v->add(mat, !doCopy, 1);
                    cudaDeviceSynchronize();
//                    getPipeDispenser().freePipe(_deviceID);
                    getPipeDispenser().freePipe(d);
                }
                doCopy = false;
            } else {
                getPipeDispenser().freePipe(d);
            }
        }
#else
        NVMatrix* v = NULL;
        bool skipMine = false;
        for (intv::iterator it = _nextDeviceIDs.begin(); it != _nextDeviceIDs.end(); ++it) {
            int d = *it;
            if (d == _deviceID && skipMine) {
                continue;
            }
            NVMatrix& mat = getActsGrad(d); // This lives on the other device
            mat.transpose(_trans);
            if (mat.getNumElements() != 0 && getRcvdBInputs(d) > 0) {
                bool doCopy = v == NULL && getRcvdBInputs(_deviceID) == 0;
                if (v == NULL) {
                    v = &getActsGrad();
                    // We have handled _actsGrad[_deviceID] so we don't
                    // have to consider it in the remainder of the loop.
                    skipMine = true;
                    v->transpose(_trans);
                }
                if (d != _deviceID) {
                    if (!NVMatrix::canAccessDevice(_deviceID, d)) {
                        // Copy the gradients produced by device d from his GPU to CPU memory
                        // (since a direct GPU-GPU copy is impossible in this case)

                        NVMatrix::setDeviceID(d);
                        _bwdQuantizer->quantize(mat, _hostMemBwd);
                        // I have verified that synchronization *is* necessary here.
                        // Even without any explicit mention of streams, kernel calls
                        // from the same host thread on different devices execute simultaneously.
                        cudaDeviceSynchronize();
                        NVMatrix::setDeviceID(_deviceID);

                        _bwdQuantizer->dequantize(*v, !doCopy, 1);
                    } else {
                        v->add(mat, !doCopy, 1);
                    }
                }
            }

        }
#endif
        // Increment so we never hit this code again
        incRcvdBInputs(_deviceID);
        // Cost layers won't have any actual actGrads, so just pass some
        // empty matrix rather than passing NULL (which would cause a segfault)
        bprop(v == NULL ? getActsGrad() : *v, passType);
        
        if (_bwdTerminal) {
            // I am a terminal node, so let the parent know that I'm done.
            cudaDeviceSynchronize();
            _convNetGPU->getConvNet().getMessageQueue().enqueue(new Message(BPROP_TERMINAL));
        }
    }
}

void Layer::bprop(NVMatrix& v, PASS_TYPE passType) {
    v.transpose(_trans);
    assert(_deviceID == NVMatrix::getDeviceID());
    for (int i = 0; i < _prev.size(); i++) {
        _inputs[i]->transpose(_trans);
        _prev[i]->getActsGrad().transpose(_trans);
    }
    getActs().transpose(_trans);
    bpropCommon(v, passType);
    
    if (isGradProducer()) {
        // First propagate activity gradient to all layers whose activity
        // gradient matrix I'm definitely not sharing.
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer() && isGradProducer(_prev[i]->getName()) && _actsGradTarget != i) {
                bpropActs(v, i, _prev[i]->getRcvdBInputs(_deviceID) > 0 ? 1 : 0, passType);
                _prev[i]->incRcvdBInputs(_deviceID);
            }
        }
        // Then propagate activity gradient to the layer whose activity gradient
        // matrix I'm sharing, if any.
        if (_actsGradTarget >= 0 && _prev[_actsGradTarget]->isGradConsumer() && isGradProducer(_prev[_actsGradTarget]->getName())) {
            bpropActs(v, _actsGradTarget, _prev[_actsGradTarget]->getRcvdBInputs(_deviceID) > 0 ? 1 : 0, passType);
            _prev[_actsGradTarget]->incRcvdBInputs(_deviceID);
        }
    }
    truncBwdActs();
    
    // This is necessary because the kernel calls that compute my backward acts
    // execute asynchronously. Therefore I don't want to tell other threads that I've
    // comptued bprop activities for them when in fact I've only called a function which
    // will eventually compute them.
    bool synced = false;

    if (isGradProducer()) {
        // First notify other threads that I have output for them.
        // This is a separate loop from the one below because I don't
        // want to do any more computation before telling the other threads
        // that they can proceed.
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer() && isGradProducer(_prev[i]->getName()) && _prev[i]->getDeviceID() != _deviceID) {
                if (!synced) {
                    cudaDeviceSynchronize();
                    synced = true;
                }
                _prev[i]->getConvNetGPU().enqueueMessage(new BpropMessage(_name, _prev[i]->getName(), passType));
            }
        }
        
        for (int i = 0; i < _prev.size(); i++) {
            if (_prev[i]->isGradConsumer() && isGradProducer(_prev[i]->getName()) && _prev[i]->getDeviceID() == _deviceID) {
                _prev[i]->bprop(passType);
            }
        }
    } 
}

void Layer::reset() {
    _rcvdFInputs = 0;
    _rcvdBInputMsgs = 0;
    for (map<int,int>::iterator it = _rcvdBInputs.begin(); it != _rcvdBInputs.end(); ++it) {
        it->second = 0;
    }
}

int Layer::getNumCases(NVMatrix& v) {
    int numCases = _convNetGPU->getConvNet().getWeightUpdateFreq() == 1 ? (_trans ? v.getNumRows() : v.getNumCols()) 
                 : _convNetGPU->getConvNet().getWeightUpdateFreq() * _convNetGPU->getConvNet().getMinibatchSize();
    return numCases;
}

int Layer::incRcvdBInputMsgs() {
    return ++_rcvdBInputMsgs;
}

string& Layer::getName() {
    return _name;
}

string& Layer::getType() {
    return _type;
}

int Layer::getRcvdFInputs() {
    return _rcvdFInputs;
}
int Layer::getRcvdBInputs(int deviceID) {
    return _rcvdBInputs[deviceID];
}

// TODO: make sure all this stuff is thread-safe
// it seems like it shouldn't be a problem for multiple threads to 
// simultaneously increment different elements of a map,
// as long as no one also tried to insert/delete stuff.
int Layer::incRcvdBInputs(int deviceID) {
    return ++_rcvdBInputs[deviceID];
}

void Layer::addNext(Layer* l) {
    _next.push_back(l);
    // Insert into a random position in _nextDeviceIDs
    // so that the backward message order is randomized
    // and good in expectation.
    if (count(_nextDeviceIDs.begin(), _nextDeviceIDs.end(), l->getDeviceID()) == 0) {
        int pos = rand() % (_nextDeviceIDs.size() + 1);
        _nextDeviceIDs.insert(_nextDeviceIDs.begin() + pos, l->getDeviceID());
//        _nextDeviceIDs.push_back(l->getDeviceID());
    }
}

void Layer::addPrev(Layer* l) {
    _prev.push_back(l);
}

bool Layer::hasGradProducerNext(string& layerName) {
    bool b = _next.size() == 0;
    for (int i = 0; i < _next.size(); i++) {
        b |= _next[i]->hasGradProducerNext(_name);
    }
    return b && isGradProducer(layerName);
}

void Layer::postInit() {
//    _outputs = _actsTarget < 0 ? new NVMatrix() : &_prev[_actsTarget]->getActs();
    _outputs[_deviceID] = _actsTarget < 0 ? new NVMatrix() : NULL;
    _actsGrad[_deviceID] = _actsGradTarget < 0 ? new NVMatrix() : &_prev[_actsGradTarget]->getActsGrad(_deviceID);
    for (int i = 0; i < _next.size(); ++i) {
        _numGradProducersNext += _next[i]->hasGradProducerNext(_name);
        int d = _next[i]->getDeviceID();
        if (_actsGrad.count(d) == 0) {
            _actsGrad[d] = new NVMatrix();
            _rcvdBInputs[d] = 0;
            _outputs[d] = new NVMatrix();
        }
    }
}

// Does this layer, or some layer below it, need the gradient
// for parameter updates?
// Only weight layers should be grad consumers themselves.
bool Layer::isGradConsumer() {
    if (!_foundGradConsumers) {
        for (int i = 0; i < _prev.size(); i++) {
            _gradConsumer |= _prev[i]->isGradConsumer();
        }
        _foundGradConsumers = true;
    }
    return _gradConsumer;
}

// Does this layer produce gradient for layers below?
bool Layer::isGradProducer() {
    return true;
}

bool Layer::isGradProducer(string& layerName) {
    return isGradProducer();
}

vector<Layer*>& Layer::getPrev() {
    return _prev;
}

vector<Layer*>& Layer::getNext() {
    return _next;
}

NVMatrix& Layer::getActs() {
    return getActs(getDeviceID());
}

NVMatrix& Layer::getActs(int deviceID) {
    assert(_outputs.count(deviceID) > 0);
    return *_outputs[deviceID];
}

NVMatrix& Layer::getActsGrad(int deviceID) {
    assert(_actsGrad.count(deviceID) > 0);
    return *_actsGrad[deviceID];
}

NVMatrix& Layer::getActsGrad() {
    return getActsGrad(NVMatrix::getDeviceID());
}

int Layer::getDeviceID() {
    return _deviceID;
}

ConvNetGPU& Layer::getConvNetGPU() {
    assert(_convNetGPU != NULL);
    return *_convNetGPU;
}

ConvNet& Layer::getConvNet() {
    return getConvNetGPU().getConvNet();
}

PipeDispenser& Layer::getPipeDispenser() {
    return getConvNet().getPipeDispenser();
}

void Layer::setBwdTerminal(bool t) {
    _bwdTerminal = t;
}

/* 
 * =======================
 * NeuronLayer
 * =======================
 */
NeuronLayer::NeuronLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) 
    : Layer(convNetGPU, paramsDict, true) {
    PyObject* neuronDict = PyDict_GetItemString(paramsDict, "neuron");
    _neuronType = pyDictGetString(neuronDict, "type");
    _neuron = &Neuron::makeNeuron(neuronDict);
}

void NeuronLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // Special optimization for cross-entropy objective with logistic units.
    // Better to just compute the input gradient in one go to avoid division by small numbers.
    bool doCrossEntGrad = _neuronType == "logistic" && _next.size() == 1 && _next[0]->getType() == "cost.crossent2" && _next[0]->getDeviceID() == _deviceID;
    if (doCrossEntGrad) {
        NVMatrix& labels = _next[0]->getPrev()[0]->getActs(_deviceID);
        float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();
        labels.transpose(_trans);
        if (scaleTargets == 0) {
            getActs().add(labels, -gradCoeff, gradCoeff, _prev[0]->getActsGrad());
        } else {
            getActs().applyTernary(AddGradientBinaryOperator<NVMatrixBinaryOps::WeightedAdd>(NVMatrixBinaryOps::WeightedAdd(-gradCoeff, gradCoeff)), labels, _prev[0]->getActsGrad(), _prev[0]->getActsGrad());
        }
    } else {
        _neuron->computeInputGrad(v, _prev[0]->getActsGrad(), scaleTargets > 0);
    }
}

void NeuronLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _neuron->activate(*_inputs[0], getActs());
}

string& NeuronLayer::getNeuronType() {
    return _neuronType;
}

/* 
 * =======================
 * WeightLayer
 * =======================
 * 
 * The  useGrad parameter here merely expresses a preference by the subclass. It may
 * be overridden by the superclass (WeightLayer) and in that case the subclass must follow its wishes.
 * So when computing gradient updates, the subclass must always first check weights.isUseGrad().
 * 
 * Note: biases always useGrad.
 */
WeightLayer::WeightLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool trans, bool useGrad, bool initWeights) : 
    Layer(convNetGPU, paramsDict, trans) {
    if (initWeights) {
        MatrixV& hWeights = *pyDictGetMatrixV(paramsDict, "weights");
        MatrixV& hWeightsInc = *pyDictGetMatrixV(paramsDict, "weightsInc");
        Matrix& hBiases = *pyDictGetMatrix(paramsDict, "biases");
        Matrix& hBiasesInc = *pyDictGetMatrix(paramsDict, "biasesInc");
        PyObject* pySchedW = PyDict_GetItemString(paramsDict, "schedW");
        PyObject* pySchedB = PyDict_GetItemString(paramsDict, "schedB");
        floatv& momW = *pyDictGetFloatV(paramsDict, "momW");
        float momB = pyDictGetFloat(paramsDict, "momB");
        floatv& epsW = *pyDictGetFloatV(paramsDict, "epsW");
        float epsB = pyDictGetFloat(paramsDict, "epsB");
        floatv& wc = *pyDictGetFloatV(paramsDict, "wc");
        floatv& wball = *pyDictGetFloatV(paramsDict, "wballNormed");
        float superEps = pyDictGetFloat(paramsDict, "superEps");
        float superMom = pyDictGetFloat(paramsDict, "superMom");
        useGrad |= superEps > 0; // if using super weight updates, must use gradient matrix

        // Source layers for shared weights
        stringv& weightSourceLayers = *pyDictGetStringV(paramsDict, "weightSourceLayers");
        // Weight matrix indices (inside the above source layers) for shared weights
        intv& weightSourceMatrixIndices = *pyDictGetIntV(paramsDict, "weightSourceMatrixIndices");

        for (int i = 0; i < weightSourceLayers.size(); i++) {
            string& srcLayerName = weightSourceLayers[i];
            int matrixIdx = weightSourceMatrixIndices[i];
            LearningRateSchedule& lrs = LearningRateSchedule::make(pySchedW, epsW[i]);
            if (srcLayerName == _name) { // Current layer
                _weights.addWeights(*new Weights(_weights[matrixIdx], lrs));
            } else if (srcLayerName != "") {
                WeightLayer& srcLayer = *static_cast<WeightLayer*>(&convNetGPU->getLayer(srcLayerName));
                Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
                _weights.addWeights(*new Weights(*srcWeights, lrs));
            } else {
                _weights.addWeights(*new Weights(*hWeights[i], *hWeightsInc[i], lrs, wc[i], wball[i], momW[i], superEps, useGrad));
            }
        }
        _biases = new Weights(hBiases, hBiasesInc,  LearningRateSchedule::make(pySchedB, epsB), 0, 0, momB, superEps, true);

        delete &weightSourceLayers;
        delete &weightSourceMatrixIndices;
        delete &hWeights;
        delete &hWeightsInc;
        delete &momW;
        delete &epsW;
        delete &wc;
        delete &wball;
    }
    _wStep = 0.02;
    _bStep = 0.05;
    _gradComputed = false;
}

void WeightLayer::bpropCommon(NVMatrix& v, PASS_TYPE passType) {
    if (_biases->getLearningRateSchedule().getBaseRate() > 0) {
        bpropBiases(v, passType);
        _biases->incNumUpdates();
    }
    for (int i = 0; i < _weights.getSize(); i++) {
        if (_weights[i].getLearningRateSchedule().getBaseRate() > 0) {
            bpropWeights(v, i, passType);
            // Increment its number of updates
            _weights[i].incNumUpdates();
//            printf("layer %s[%d] computing weight grad\n", _name.c_str(), i);
        }
    }
    _gradComputed = true;
}

bool WeightLayer::updateWeights() {
    if (_gradComputed && _convNetGPU->getConvNet().getNumBwdMiniPasses() == 0) {
        _weights.update(_convNetGPU->getConvNet().getTrainingProgress());
        _biases->update(_convNetGPU->getConvNet().getTrainingProgress());
        constrainWeights();
        _gradComputed = false;
        return true;
    }
    return false;
}

void WeightLayer::copyToCPU() {
    _weights.copyToCPU();
    _biases->copyToCPU();
}

void WeightLayer::copyToGPU() {
    _weights.copyToGPU();
    _biases->copyToGPU();
}

void WeightLayer::checkGradients() {
    for (int i = 0; i < _weights.getSize(); i++) {
        _convNetGPU->getConvNet().checkGradient(_name + " weights[" + tostr(i) + "]", _wStep, _weights[i]);
    }
    _convNetGPU->getConvNet().checkGradient(_name + " biases", _bStep, *_biases);
}

Weights& WeightLayer::getWeights(int idx) {
    return _weights[idx];
}

/* 
 * =======================
 * FCLayer
 * =======================
 */
FCLayer::FCLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool useGrad, bool initWeights) 
    : WeightLayer(convNetGPU, paramsDict, true, useGrad, initWeights) {
    _wStep = 0.01;
    _bStep = 0.01;
}

void FCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    getActs().addProduct(*_inputs[inpIdx], *_weights[inpIdx], scaleTargets, 1);
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void FCLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& weights_T = _weights[inpIdx].getW().getTranspose();
    _prev[inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);
    delete &weights_T;
}

void FCLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    float scaleBGrad = passType == PASS_GC ? 1.0f : 1.0f / getNumCases(v);
    float scaleInc = _biases->getNumUpdates() > 0;
    _biases->getGrad().addSum(v, 0, scaleInc, scaleBGrad);
}

void FCLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = getNumCases(v);
    NVMatrix& prevActs_T = _inputs[inpIdx]->getTranspose();
    float progress = _convNetGPU->getConvNet().getTrainingProgress();
    if (_weights[inpIdx].isUseGrad()) {
        float scaleGrad = passType == PASS_GC ? 1 : 1.0f / numCases;
        float scaleInc = (_weights[inpIdx].getNumUpdates() > 0);
        _weights[inpIdx].getGrad().addProduct(prevActs_T, v, scaleInc, scaleGrad);
    } else {
        float scaleGrad = passType == PASS_GC ? 1 : _weights[inpIdx].getEps(progress) / numCases;
        float scaleInc =  (passType == PASS_GC ? _weights[inpIdx].getNumUpdates() > 0 
                                               : (_weights[inpIdx].getNumUpdates() == 0 ? _weights[inpIdx].getMom() : 1.0f));
        _weights[inpIdx].getInc().addProduct(prevActs_T, v, scaleInc, scaleGrad);
    }
    
    delete &prevActs_T;
}

void FCLayer::constrainWeights() {
    for (int i = 0; i < _weights.getSize(); i++) {
        if (_weights[i].getWBall() > 0 && _weights[i].isOwner() && _weights[i].getLearningRateSchedule().getBaseRate() > 0) {
            NVMatrix norm, sqw; // Unfortunate extra weight matrix...
            _weights[i].getW().apply(NVMatrixOps::Square(), sqw);
            sqw.sum(0, norm);
            norm.apply(WeightConstraintOperator(_weights[i].getWBall()));
            _weights[i].getW().eltwiseMultByVector(norm);
        }
    }
}

/* 
 * =======================
 * TreeFCLayer
 * =======================
 */
TreeFCLayer::TreeFCLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) 
    : FCLayer(convNetGPU, paramsDict, true, false) {
    int rootLabel = pyDictGetInt(paramsDict, "rootLabel");

    SoftmaxTree* tree = new SoftmaxTree(rootLabel);
    
    PyObject* pyTree = PyDict_GetItemString(paramsDict, "tree");
    makeTree(pyTree, tree->getRoot());
    
    MatrixV& hWeights = *pyDictGetMatrixV(paramsDict, "weights");
    MatrixV& hWeightsInc = *pyDictGetMatrixV(paramsDict, "weightsInc");
    Matrix& hBiases = *pyDictGetMatrix(paramsDict, "biases");
    Matrix& hBiasesInc = *pyDictGetMatrix(paramsDict, "biasesInc");
    
    floatv& momW = *pyDictGetFloatV(paramsDict, "momW");
    float momB = pyDictGetFloat(paramsDict, "momB");
    floatv& epsW = *pyDictGetFloatV(paramsDict, "epsW");
    float epsB = pyDictGetFloat(paramsDict, "epsB");
    floatv& wc = *pyDictGetFloatV(paramsDict, "wc");
    
    // This class does not support learning rate schedules for now.
    _treeWeights = new TreeWeights(*tree, *hWeights[0], *hWeightsInc[0], *new LearningRateSchedule(epsW[0]), wc[0], momW[0]);
    _biases = new Weights(hBiases, hBiasesInc, *new LearningRateSchedule(epsB), 0, 0, momB, false, true);
    _weights.addWeights(*_treeWeights);
    
    _wStep = 0.001;
    
    delete &hWeights;
    delete &hWeightsInc;
    delete &momW;
    delete &epsW;
    delete &wc;
}

void TreeFCLayer::makeTree(PyObject* pyTree, SoftmaxNode& rootNode) {
    PyObject* pyChildren = PyList_GetItem(pyTree, rootNode.getLabel());
    int numChildren = PyList_GET_SIZE(pyChildren);
    for (int c = 0; c < numChildren; ++c) {
        int childLabel = PyLong_AsLong(PyList_GetItem(pyChildren, c));
        SoftmaxNode& childNode = rootNode.addChild(childLabel);
        makeTree(pyTree, childNode);
    }
}

void TreeFCLayer::fpropActs(int inpIdx, float scaleTargets, uint passType) {
    if (passType == PASS_GC) {
        _treeWeights->makeWeights();
    }
    FCLayer::fpropActs(inpIdx, scaleTargets, passType);
}

void TreeFCLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, uint passType) {
    FCLayer::bpropActs(v, inpIdx, scaleTargets, passType);
    if (passType == PASS_GC) {
        _treeWeights->distributeGradients();
    }
}

void TreeFCLayer::constrainWeights() {
}

void TreeFCLayer::checkGradients() {
    DummyWeights dum = DummyWeights(_treeWeights->getCPUW(), _treeWeights->getCPUWInc(),
                                    _treeWeights->getAllW(), _treeWeights->getAllInc(),
                                    _treeWeights->getAllGrad());
    _convNetGPU->getConvNet().checkGradient(_name + " weights", _wStep, dum);
    _convNetGPU->getConvNet().checkGradient(_name + " biases", _bStep, *_biases);
}

/* 
 * =======================
 * LocalLayer
 * =======================
 */
LocalLayer::LocalLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool useGrad) 
    : WeightLayer(convNetGPU, paramsDict, false, useGrad, true) {
    _padding = pyDictGetIntV(paramsDict, "padding");
    _stride = pyDictGetIntV(paramsDict, "stride");
    _filterSize = pyDictGetIntV(paramsDict, "filterSize");
    _channels = pyDictGetIntV(paramsDict, "channels");
    _imgSize = pyDictGetIntV(paramsDict, "imgSize");
    _numFilters = pyDictGetInt(paramsDict, "filters");
    _groups = pyDictGetIntV(paramsDict, "groups");
    _filterChannels = pyDictGetIntV(paramsDict, "filterChannels");
    _randSparse = pyDictGetIntV(paramsDict, "randSparse");
    _overSample = pyDictGetIntV(paramsDict, "overSample");
    _filterPixels = pyDictGetIntV(paramsDict, "filterPixels");
    _imgPixels = pyDictGetIntV(paramsDict, "imgPixels");
    
    _modulesX = pyDictGetInt(paramsDict, "modulesX");
    _modules = pyDictGetInt(paramsDict, "modules");

    // It's a vector on the heap to be consistent with all the others...
    _filterConns = new vector<FilterConns>();
    PyObject* pyFilterConns = PyDict_GetItemString(paramsDict, "filterConns");
    for (int i = 0; i < _randSparse->size(); i++) {
        FilterConns fc;
        if (_randSparse->at(i)) {
            fc.hFilterConns = getIntA(PyList_GET_ITEM(pyFilterConns, i));
        }
        _filterConns->push_back(fc);
    }
}

void LocalLayer::copyToGPU() {
    WeightLayer::copyToGPU();
    for  (int i = 0; i < _prev.size(); i++) {
        if (_randSparse->at(i)) { // Copy to GPU vector that describes sparse random connectivity
            cudaMalloc(&_filterConns->at(i).dFilterConns, sizeof(int) * _groups->at(i) * _filterChannels->at(i));
            cudaMemcpy(_filterConns->at(i).dFilterConns, _filterConns->at(i).hFilterConns,
                       sizeof(int) * _groups->at(i) * _filterChannels->at(i), cudaMemcpyHostToDevice);
            getLastCudaError("cudaMemcpy: failed");
        }
    }
}

/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) 
    : LocalLayer(convNetGPU, paramsDict, true) {
    _partialSum = pyDictGetInt(paramsDict, "partialSum");
    _sharedBiases = pyDictGetInt(paramsDict, "sharedBiases");
    _weightContrastNormMin = pyDictGetFloatV(paramsDict, "wcNormMin");
    _weightContrastNormMax = pyDictGetFloatV(paramsDict, "wcNormMax");
}

void ConvLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        convFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                             _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        convFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }

    if (scaleTargets == 0) {
        if (_sharedBiases) {
            getActs().reshape(_numFilters, getActs().getNumElements() / _numFilters);
            getActs().addVector(_biases->getW());
            getActs().reshape(_numFilters * _modules, getActs().getNumElements() / (_numFilters * _modules));
        } else {
            getActs().addVector(_biases->getW());
        }
    }
}

void ConvLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = getNumCases(v);
    float scaleBGrad = passType == PASS_GC ? 1.0f : 1.0f / numCases;
    float scaleInc = _biases->getNumUpdates() > 0;
    if (_sharedBiases) {
        v.reshape(_numFilters, v.getNumElements() / _numFilters);
        _biases->getGrad().addSum(v, 1, scaleInc, scaleBGrad);
        v.reshape(_numFilters * _modules, v.getNumElements() / (_numFilters * _modules));
    } else {
        _biases->getGrad().addSum(v, 1, scaleInc, scaleBGrad);
    }
}

void ConvLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = getNumCases(v);

    NVMatrix& tgt = _partialSum > 0 ? _weightGradTmp : _weights[inpIdx].getGrad();
    float scaleWGrad = passType == PASS_GC ? 1.0f : 1.0f / numCases;
    float scaleTargets = _weights[inpIdx].getNumUpdates() > 0 && _partialSum == 0; // ? 1 : 0;
    
    if (_randSparse->at(inpIdx)) {
        convWeightActsSparse(*_inputs[inpIdx], v, tgt, _filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx), _modulesX, _modulesX,
                             _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                             _filterChannels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
    } else {
        convWeightActs(*_inputs[inpIdx], v, tgt, _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx),
                       _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
    }
    if (_partialSum > 0) {
        scaleTargets = _weights[inpIdx].getNumUpdates() > 0;
        //cout << _name << " scale inc: " << scaleTargets << " scale grad " << scaleWGrad << endl;
        _weightGradTmp.reshape(_modules / _partialSum, _filterChannels->at(inpIdx) * _filterPixels->at(inpIdx) * _numFilters);
        _weights[inpIdx].getGrad().addSum(_weightGradTmp, 0, scaleTargets, 1);
        _weights[inpIdx].getGrad().reshape(_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx), _numFilters);
    }
}

void ConvLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        NVMatrix& tgt = _overSample->at(inpIdx) > 1 ? _actGradTmp : _prev[inpIdx]->getActsGrad();

        convImgActsSparse(v, *_weights[inpIdx], tgt, _filterConns->at(inpIdx).dFilterConns,
                          _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx),
                          _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
        if (_overSample->at(inpIdx) > 1) {
            _actGradTmp.reshape(_overSample->at(inpIdx), _actGradTmp.getNumElements() / _overSample->at(inpIdx));
            _actGradTmp.sum(0, _prev[inpIdx]->getActsGrad());
            _prev[inpIdx]->getActsGrad().reshape(_prev[inpIdx]->getActsGrad().getNumElements() / v.getNumCols(), v.getNumCols());
        }
    } else {
        convImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
}

void ConvLayer::truncBwdActs() {
    LocalLayer::truncBwdActs();
    if (_conserveMem) {
        _weightGradTmp.truncate();
        _actGradTmp.truncate();
    }
}

void ConvLayer::constrainWeights() {
    for (int i = 0; i < _weights.getSize(); i++) {
        if (_weightContrastNormMax->at(i) > 0 && _weights[i].isOwner() && _weights[i].getLearningRateSchedule().getBaseRate() > 0) {
            float fz = _weights[i].getW().getNumRows();
            NVMatrix tmp;
            _weights[i].getW().sum(0, tmp); 
            _weights[i].getW().addVector(tmp, -1.0f / fz, _weights[i].getGrad());
            // Now _weights[i].getGrad() contains zero-mean filters
            _weights[i].getGrad().apply(NVMatrixOps::Square());
            _weights[i].getGrad().sum(0, tmp);
//            tmp.apply(NVMatrixOps::Sqrt());
//            tmp.scale(1.0f / fz);
            
//            tmp.scale(1.0f / (fz * _weightContrastNorm->at(i)));
            tmp.apply(WeightContrastNormOperator(_weightContrastNormMin->at(i), _weightContrastNormMax->at(i), 1.0f / fz));
            // Now tmp has the stdev
            _weights[i].getW().eltwiseMultByVector(tmp);
        }
        // It's pretty silly to do both these things but whatever
        if (_weights[i].getWBall() > 0 && _weights[i].isOwner() && _weights[i].getLearningRateSchedule().getBaseRate() > 0) {
            NVMatrix norm;
            _weights[i].getW().apply(NVMatrixOps::Square(), _weights[i].getGrad());
            _weights[i].getGrad().sum(0, norm);

            norm.apply(WeightConstraintOperator(_weights[i].getWBall()));
            _weights[i].getW().eltwiseMultByVector(norm);
        }
    }
}

/* 
 * =======================
 * LocalUnsharedLayer
 * =======================
 */
LocalUnsharedLayer::LocalUnsharedLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) 
    : LocalLayer(convNetGPU, paramsDict, false) {
}

void LocalUnsharedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                              _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(), _imgSize->at(inpIdx), _modulesX, _modulesX, _padding->at(inpIdx),
                        _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);

    }  
    if (scaleTargets == 0) {
        getActs().addVector(_biases->getW());
    }
}

void LocalUnsharedLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
    int numCases = getNumCases(v);
    float scaleBGrad = passType == PASS_GC ? 1.0f : 1.0f / numCases;
    float scaleInc = _biases->getNumUpdates() > 0;
    _biases->getGrad().addSum(v, 1, scaleInc, scaleBGrad);
}

void LocalUnsharedLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
    int numCases = getNumCases(v);
    float progress = _convNetGPU->getConvNet().getTrainingProgress();
    NVMatrix& tgt = _weights[inpIdx].isUseGrad() ? _weights[inpIdx].getGrad() : _weights[inpIdx].getInc();
    float scaleInc, scaleWGrad;
    if (_weights[inpIdx].isUseGrad()) {
        scaleInc = _weights[inpIdx].getNumUpdates() > 0;
        scaleWGrad = passType == PASS_GC ? 1.0f : 1.0f / numCases; // eps / numCases
    } else {
        scaleInc =  (passType == PASS_GC ? _weights[inpIdx].getNumUpdates() > 0 
                                         : (_weights[inpIdx].getNumUpdates() == 0 ? _weights[inpIdx].getMom() : 1.0f));
        scaleWGrad = passType == PASS_GC ? 1.0f : _weights[inpIdx].getEps(progress) / numCases; // eps / numCases
    }
    
    if (_randSparse->at(inpIdx)) {
        localWeightActsSparse(*_inputs[inpIdx], v, tgt, _filterConns->at(inpIdx).dFilterConns,
                              _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx), _padding->at(inpIdx), _stride->at(inpIdx),
                              _channels->at(inpIdx), _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleInc, scaleWGrad);
    } else {
        localWeightActs(*_inputs[inpIdx], v, tgt, _imgSize->at(inpIdx), _modulesX, _modulesX, _filterSize->at(inpIdx),
                        _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleInc, scaleWGrad);
    }
}

void LocalUnsharedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_randSparse->at(inpIdx)) {
        localImgActsSparse(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(), _filterConns->at(inpIdx).dFilterConns,
                           _imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx), _stride->at(inpIdx), _channels->at(inpIdx),
                           _filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    } else {
        localImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(),_imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
                    _padding->at(inpIdx),  _stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
    }
}

void LocalUnsharedLayer::constrainWeights() {
    for (int i = 0; i < _weights.getSize(); i++) {
        if (_weights[i].getWBall() > 0  && _weights[i].isOwner() && _weights[i].getLearningRateSchedule().getBaseRate() > 0) {
            normalizeLocalWeights(*_weights[i], _modules, _weights[i].getWBall());
        }
    }
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */
SoftmaxLayer::SoftmaxLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) 
    : Layer(convNetGPU, paramsDict, true), _doLogregGrad(true) {
}

void SoftmaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& input = *_inputs[0];
    NVMatrix& max = input.max(1);
    input.addVector(max, -1, getActs());
    getActs().apply(NVMatrixOps::Exp());
    NVMatrix& sum = getActs().sum(1);
    getActs().eltwiseDivideByVector(sum);
    
    delete &max;
    delete &sum;
}

void SoftmaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 0);
    bool doLogregGrad = _doLogregGrad && (_next.size() == 1 && _next[0]->getType() == "cost.logreg" && _next[0]->getDeviceID() == _deviceID);
    if (doLogregGrad) {
        NVMatrix& labels = _next[0]->getPrev()[0]->getActs(_deviceID);
        float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();
        computeLogregSoftmaxGrad(labels, getActs(), _prev[0]->getActsGrad(), scaleTargets == 1, gradCoeff);
    } else {
        computeSoftmaxGrad(getActs(), v, _prev[0]->getActsGrad(), scaleTargets == 1);
    }
}

void SoftmaxLayer::setDoLogregGrad(bool b) {
    _doLogregGrad = b;
}

/* 
 * =======================
 * ConcatenationLayer
 * =======================
 */
ConcatenationLayer::ConcatenationLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict)
    : Layer(convNetGPU, paramsDict, false) {
    _copyOffsets = pyDictGetIntV(paramsDict, "copyOffsets");
    _copyOffsets->push_back(_numOutputs);
}

void ConcatenationLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    getActs().resize(_numOutputs, _inputs[inpIdx]->getNumCols());
    _inputs[inpIdx]->copy(getActs(), 0, -1, 0, -1, _copyOffsets->at(inpIdx), 0);
}

void ConcatenationLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& copySrc = v.sliceRows(_copyOffsets->at(inpIdx), _copyOffsets->at(inpIdx + 1)); // view
    _prev[inpIdx]->getActsGrad().add(copySrc, scaleTargets, 1);
    delete &copySrc;
}

/* 
 * =======================
 * EltwiseSumLayer
 * =======================
 */
EltwiseSumLayer::EltwiseSumLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
    _coeffs = pyDictGetFloatV(paramsDict, "coeffs");
}

void EltwiseSumLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (scaleTargets == 0) {
        _inputs[inpIdx]->scale(_coeffs->at(inpIdx), getActs());
    } else {
        getActs().add(*_inputs[inpIdx], _coeffs->at(inpIdx));
    }
}

void EltwiseSumLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (scaleTargets == 0 ) {
        v.scale(_coeffs->at(inpIdx), _prev[inpIdx]->getActsGrad());
    } else {
        assert(&_prev[inpIdx]->getActsGrad() != &v);
        _prev[inpIdx]->getActsGrad().add(v, scaleTargets, _coeffs->at(inpIdx));
    }
}

/* 
 * =======================
 * EltwiseMaxLayer
 * =======================
 */
EltwiseMaxLayer::EltwiseMaxLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
}

void EltwiseMaxLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (inpIdx == 1) { // First input, do nothing
        _inputs[inpIdx]->applyBinary(NVMatrixAggs::Max(), *_inputs[0], getActs());
    } else if (inpIdx > 1) {
        getActs().applyBinary(NVMatrixAggs::Max(), *_inputs[inpIdx]);
    }
}

void EltwiseMaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    computeEltwiseMaxGrad(v, *_inputs[inpIdx], getActs(), _prev[inpIdx]->getActsGrad(), scaleTargets != 0);
}

/* 
 * =======================
 * HiddenSexLayer
 * =======================
 */
HiddenSexLayer::HiddenSexLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
    _enable = pyDictGetInt(paramsDict, "enable");
    _keep = pyDictGetFloat(paramsDict, "keep");
}

void HiddenSexLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_enable && passType == PASS_TRAIN) {
        _sexMask.resize(*_inputs[inpIdx]);
        _sexMask.randomizeUniform();
        _sexMask.smallerThanScalar(_keep);
        _inputs[inpIdx]->eltwiseMult(_sexMask, getActs());
    } else {
        _inputs[inpIdx]->scale(_keep, getActs());
    }
}

void HiddenSexLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_enable && passType == PASS_TRAIN) {
        if (scaleTargets != 0) {
            v.applyTernary(AddGradientBinaryOperator<NVMatrixBinaryOps::Multiply>(NVMatrixBinaryOps::Multiply()),
                           _sexMask, _prev[inpIdx]->getActsGrad(), _prev[inpIdx]->getActsGrad());
        } else {
            v.eltwiseMult(_sexMask, _prev[inpIdx]->getActsGrad());
        }
    } else {
        if (scaleTargets != 0) {
             v.applyBinary(AddGradientOperator<NVMatrixOps::MultByScalar>(NVMatrixOps::MultByScalar(_keep)),
                           _prev[inpIdx]->getActsGrad(), _prev[inpIdx]->getActsGrad());
        } else {
            v.scale(_keep, _prev[inpIdx]->getActsGrad());
        }
    }
}

void HiddenSexLayer::truncBwdActs() {
    Layer::truncBwdActs();
    if (_conserveMem) {
        _sexMask.truncate();
    }
}

/* 
 * =======================
 * DataLayer
 * =======================
 */
DataLayer::DataLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
    _dataIdx = pyDictGetInt(paramsDict, "dataIdx");
    _useBuffer = false;
    _bufferMinibatchIdx = -1;
    _bufferData = NULL;
}

void DataLayer::fprop(PASS_TYPE passType) {
    throw string("No dava given!");
}

void DataLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
}

void DataLayer::fprop(NVMatrixV& data, PASS_TYPE passType) {
    throw string("Data layer requires CPU data!");
}

void DataLayer::startFprop(CPUData& data, PASS_TYPE passType) {
    copyData(data, false);
    fpropNext(passType);
}

void DataLayer::startFpropFromBuffer(PASS_TYPE passType) {
    _useBuffer = !_useBuffer;
    fpropNext(passType);
}

void DataLayer::fpropNext(PASS_TYPE passType) {
    for (int i = 0; i < _next.size(); i++) {
        // Inform this guy that my output is ready for him
        _next[i]->getConvNetGPU().enqueueMessage(new FpropMessage(_name, _next[i]->getName(), passType));
    }
}

void DataLayer::setBuffer(CPUData& data, int minibatchIdx) {
    _bufferData = &data;
    copyData(data, true);
    
    _bufferMinibatchIdx = minibatchIdx;
}

void DataLayer::copyData(CPUData& data, bool other) {
    Matrix& dataMatrix = data.getData(_dataIdx);
    int oldDeviceID = NVMatrix::getDeviceID();
    //StopWatchInterface *timer = NULL;
    //sdkCreateTimer(&timer);
    //sdkStartTimer(&timer);
    if (dataMatrix.isTrans()) {
        assert(dataMatrix.isView());
        _hostMemFwd.copyFromHost(dataMatrix, true);
    }
    for (intv::iterator it = _nextDeviceIDs.begin(); it != _nextDeviceIDs.end(); ++it) {
        int deviceID = *it;
        // Copy my output to this guy's GPU
        NVMatrix::setDeviceID(deviceID);
        if (dataMatrix.isTrans()) {
            _hostMemFwd.flipTrans(getActs(deviceID, other));
        } else {
            getActs(deviceID, other).copyFromHost(data.getData(_dataIdx), true);
        }
    }
    for (intv::iterator it = _nextDeviceIDs.begin(); it != _nextDeviceIDs.end(); ++it) {
        NVMatrix::setDeviceID(*it);
        cudaDeviceSynchronize();
    }
    NVMatrix::setDeviceID(oldDeviceID);
    //sdkStopTimer(&timer);
    //printf("data copy took %f\n", sdkGetTimerValue(&timer));
}

CPUData* DataLayer::getBufferData() {
    return _bufferData;
}

int DataLayer::getBufferMinibatchIdx() {
    return _bufferMinibatchIdx;
}

NVMatrix& DataLayer::getActs(int deviceID) {
    return getActs(deviceID, false);
}

NVMatrix& DataLayer::getActs(int deviceID, bool other) {
    return *(_useBuffer != other ? _outputs2[deviceID] : _outputs[deviceID]);
}

void DataLayer::postInit() {
    Layer::postInit();
    for (int i = 0; i < _next.size(); ++i) {
        int d = _next[i]->getDeviceID();
        if (_outputs2.count(d) == 0) {
            _outputs2[d] = new NVMatrix();
        }
    }
}

bool DataLayer::isGradProducer() {
    return false;
}

/* 
 * =====================
 * PoolLayer
 * =====================
 */
PoolLayer::PoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool trans) 
    : Layer(convNetGPU, paramsDict, trans) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _sizeX = pyDictGetInt(paramsDict, "sizeX");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _pool = pyDictGetString(paramsDict, "pool");
}

PoolLayer& PoolLayer::makePoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) {
    string _pool = pyDictGetString(paramsDict, "pool");
    if (_pool == "max") {
        return *new MaxPoolLayer(convNetGPU, paramsDict, false);
    } else if(_pool == "maxabs") {
        return *new MaxPoolLayer(convNetGPU, paramsDict, true);
    } else if(_pool == "avg") {
        return *new AvgPoolLayer(convNetGPU, paramsDict);
    } else if(_pool == "rand") {
        return *new RandomPoolLayer(convNetGPU, paramsDict);
    }
    throw string("Unknown pooling layer type ") + _pool;
}

/* 
 * =====================
 * AvgPoolLayer
 * =====================
 */
AvgPoolLayer::AvgPoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : PoolLayer(convNetGPU, paramsDict, false) {
}

void AvgPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, AvgPooler());
}

void AvgPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalAvgUndo(v, _prev[0]->getActsGrad(), _sizeX, _start, _stride, _outputsX, _imgSize, scaleTargets, 1);
}

/* 
 * =====================
 * MaxPoolLayer
 * =====================
 */
MaxPoolLayer::MaxPoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool abs) : PoolLayer(convNetGPU, paramsDict, false), _abs(abs) {
}

void MaxPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_abs) {
//        _inputs[0]->print(10,10);printf(" \n");
        convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, MaxAbsPooler());
    } else {
        convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, MaxPooler());
    }

}

void MaxPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalMaxUndo(*_inputs[0], v, getActs(), _prev[inpIdx]->getActsGrad(), _sizeX, _start, _stride, _outputsX, scaleTargets, 1);
}

/* 
 * =====================
 * RandomPoolLayer
 * =====================
 */
RandomPoolLayer::RandomPoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : PoolLayer(convNetGPU, paramsDict, false){
    _doMax = pyDictGetInt(paramsDict, "doMax");
    printf("domax: %d\n", _doMax);
}

void RandomPoolLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (_doMax) {
        convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX, MaxPooler());
    } else {
        convLocalRandomPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride, _outputsX);
    }

}

void RandomPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convLocalMaxUndo(*_inputs[0], v, getActs(), _prev[inpIdx]->getActsGrad(), _sizeX, _start, _stride, _outputsX, scaleTargets, 1);
}

/*
 * =====================
 * RandomScaleLayer
 * =====================
 */
RandomScaleLayer::RandomScaleLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _maxScale = pyDictGetFloat(paramsDict, "maxScale");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _tgtSize = pyDictGetInt(paramsDict, "tgtSize");
    // The smallest size the image could be after rescaling
    _minScaledSize = _imgSize / _maxScale; 
    
    // The number of discrete scales we're considering
    int numScales = _imgSize - _minScaledSize + 1;
    
    // The total number of squares of size _tgtSize that we can extract
    // from all these scales
    double numCrops = numScales * (numScales + 1) * (2 * numScales + 1) / 6;
    
    // For each scale, record the fraction of the squares that it has.
    // This will be the probability of sampling this scale.
    _scaleProbs.push_back(1.0 / numCrops);
    for (int s = 1; s < numScales; ++s) {
        _scaleProbs.push_back(_scaleProbs[s-1] + (s + 1) * (s + 1) / numCrops);
//        cout << _scaleProbs.back() << endl;
    }
}
 
void RandomScaleLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (passType == PASS_TRAIN) {
        // _maxScale is in the range [1, 2) 
        float r = randf;
        int rescaledSize = _tgtSize;
        float scaleFactor = _maxScale;
        // Find which scale we have sampled
        for (int s = 0; s < _scaleProbs.size(); ++s) {
            if (r <= _scaleProbs[s]) {
                rescaledSize += s;
                float scaleFactorEnd = _imgSize / float(rescaledSize);
                float scaleFactorStart = max(1.0, _imgSize / (1.0 + rescaledSize));
//                printf("scaleFactorStart: %f, scaleFactorEnd: %f\n", scaleFactorStart, scaleFactorEnd);
                scaleFactor = scaleFactorStart + randf * (scaleFactorEnd - scaleFactorStart);
                break;
            }
        }
//        printf("Rescaled size: %d (r = %f), scale factor: %f\n", rescaledSize, r, scaleFactor);
        assert(rescaledSize >= _tgtSize);
        int maxStart = rescaledSize - _tgtSize;
        int startY = rand() % (1 + maxStart), startX = rand() % (1 + maxStart);
//        int startY = 0, startX = 0;
//        printf("starty: %d, startx: %d\n", startY, startX);
        if (rescaledSize  == _imgSize) {
//            printf("not resizing\n");
            convCrop(*_inputs[0], getActs(), rescaledSize, _tgtSize, startY, startX);
        } else {
            convResizeBilinear(*_inputs[0], _rescaledActs, _imgSize, rescaledSize, scaleFactor);
//            _rescaledActs.print(10,10);exit(0);
            convCrop(_rescaledActs, getActs(), rescaledSize, _tgtSize, startY, startX);
        }
        _rescaledActs.truncate(); // this'll have a different size each time so may as well truncate it.
    } else if (passType & PASS_MULTIVIEW_TEST) { // for now... 
        // definitely redo this later so that multiview cropping is handled in c
        _inputs[0]->copy(getActs());
    } else if (passType & PASS_TEST) { // Test on center patch
        int cropStart = (_imgSize - _tgtSize) / 2;
        convCrop(*_inputs[0], getActs(), _imgSize, _tgtSize, cropStart, cropStart);
//        convResizeBilinear(*_inputs[0], getActs(), _imgSize, _tgtSize, _maxScale);
    }
}

void RandomScaleLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * NailbedLayer
 * =====================
 */
NailbedLayer::NailbedLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _start = pyDictGetInt(paramsDict, "start");
    _stride = pyDictGetInt(paramsDict, "stride");
    _outputsX = pyDictGetInt(paramsDict, "outputsX");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void NailbedLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNails(*_inputs[0], getActs(), _channels, _imgSize, _start, _stride, 0, 1);
}

void NailbedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convBedOfNailsUndo(v, _prev[0]->getActsGrad(), _channels, _imgSize, _start, _stride, scaleTargets, 1);
}

/* 
 * =====================
 * GaussianBlurLayer
 * =====================
 */
GaussianBlurLayer::GaussianBlurLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _hFilter = pyDictGetMatrix(paramsDict, "filter");
}

void GaussianBlurLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convGaussianBlur(*_inputs[0], _filter, getActs(), true, _channels, 0, 1);
    convGaussianBlur(getActs(), _filter, getActs(), false, _channels, 0, 1);
}

void GaussianBlurLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& tgt1 = _prev[0]->getRcvdBInputs(_deviceID) > 0 ? _actGradsTmp : _prev[0]->getActsGrad();
    convGaussianBlur(v, _filter, tgt1, true, _channels, 0, 1);
    convGaussianBlur(tgt1, _filter, _prev[0]->getActsGrad(), false, _channels, scaleTargets, 1);
}

void GaussianBlurLayer::copyToGPU() {
    _filter.copyFromHost(*_hFilter, true);
}

 /* 
 * =====================
 * HorizontalReflectionLayer
 * =====================
 */
HorizontalReflectionLayer::HorizontalReflectionLayer(ConvNetGPU* convNet, PyObject* paramsDict) : Layer(convNet, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    assert(_channels >= 1 && _channels <= 3);
}

void HorizontalReflectionLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convReflectHorizontal(*_inputs[0], getActs(), _imgSize);
}

void HorizontalReflectionLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convReflectHorizontal(v, _prev[0]->getActsGrad(), _imgSize);
}

/* 
 * =====================
 * ResizeLayer
 * =====================
 */
ResizeLayer::ResizeLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
    _tgtSize = pyDictGetInt(paramsDict, "tgtSize");
    _scale = pyDictGetFloat(paramsDict, "scale");
}

void ResizeLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResizeBilinear(*_inputs[0], getActs(), _imgSize, _tgtSize, _scale);
}

// Can't do this
void ResizeLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToYUVLayer
 * =====================
 */
RGBToYUVLayer::RGBToYUVLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
}

void RGBToYUVLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToYUV(*_inputs[0], getActs());
}

// Can't do this
void RGBToYUVLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * RGBToLABLayer
 * =====================
 */
RGBToLABLayer::RGBToLABLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
    _center = pyDictGetInt(paramsDict, "center");
}

void RGBToLABLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convRGBToLAB(*_inputs[0], getActs(), _center);
}

// Can't do this
void RGBToLABLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(false);
}

/* 
 * =====================
 * ResponseNormLayer
 * =====================
 */
ResponseNormLayer::ResponseNormLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : Layer(convNetGPU, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _size = pyDictGetInt(paramsDict, "size");

    _scale = pyDictGetFloat(paramsDict, "scale");
    _pow = pyDictGetFloat(paramsDict, "pow");
}

void ResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNorm(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow);
}

void ResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormUndo(v, _denoms, *_inputs[0], getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ResponseNormLayer::truncBwdActs() {
    Layer::truncBwdActs();
    if (_conserveMem) {
        _denoms.truncate();
    }
}

CrossMapResponseNormLayer::CrossMapResponseNormLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : ResponseNormLayer(convNetGPU, paramsDict) {
    _blocked = pyDictGetInt(paramsDict, "blocked");
    _minDiv = pyDictGetFloat(paramsDict, "minDiv");
}

void CrossMapResponseNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMap(*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow, _minDiv, _blocked);
}

void CrossMapResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convResponseNormCrossMapUndo(v, _denoms, *_inputs[0], getActs(), _prev[0]->getActsGrad(), _channels, _size, _scale, _pow, _blocked, scaleTargets, 1);
}

/* 
 * =====================
 * ContrastNormLayer
 * =====================
 */
ContrastNormLayer::ContrastNormLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : ResponseNormLayer(convNetGPU, paramsDict) {
    _imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void ContrastNormLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    NVMatrix& images = *_inputs[0];
    convLocalPool(images, _meanDiffs, _channels, _size, -_size/2, 1, _imgSize, AvgPooler());
    //_meanDiffs.print(10,10);exit(0);
    _meanDiffs.add(images, -1, 1);
    convContrastNorm(images, _meanDiffs, _denoms, getActs(), _channels, _size, _scale, _pow);
    //images.print(5,5);
}

void ContrastNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convContrastNormUndo(v, _denoms, _meanDiffs, getActs(), _prev[inpIdx]->getActsGrad(), _channels, _size, _scale, _pow, scaleTargets, 1);
}

void ContrastNormLayer::truncBwdActs() {
    ResponseNormLayer::truncBwdActs();
    if (_conserveMem) {
        _meanDiffs.truncate();
    }
}

/* 
 * =====================
 * CostLayer
 * =====================
 */
CostLayer::CostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool trans) 
    : Layer(convNetGPU, paramsDict, trans) {
    _coeff = pyDictGetFloat(paramsDict, "coeff");
}

float CostLayer::getCoeff() {
    return _coeff;
}

void CostLayer::bprop(NVMatrix& v, PASS_TYPE passType) {
    if (_coeff != 0) {
        Layer::bprop(v, passType);
    }
}

void CostLayer::fprop(PASS_TYPE passType) {
    Layer::fprop(passType);
    if (_rcvdFInputs == _prev.size()) {
        cudaDeviceSynchronize();
        _convNetGPU->getConvNet().getMessageQueue().enqueue(new Message(FPROP_TERMINAL));
    }
}

bool CostLayer::isGradProducer() {
    return _coeff != 0;
}

doublev& CostLayer::getCost() {
    doublev& v = *new doublev();
    v.insert(v.begin(), _costv.begin(), _costv.end());
    return v;
}

CostLayer& CostLayer::makeCostLayer(ConvNetGPU* convNetGPU, string& type, PyObject* paramsDict) {
    if (type == "cost.crossent") {
        return *new CrossEntCostLayer(convNetGPU, paramsDict);
    } else if (type == "cost.crossent2") {
        return *new CrossEnt2CostLayer(convNetGPU, paramsDict);
    } else if (type == "cost.logreg") {
        return *new LogregCostLayer(convNetGPU, paramsDict);
    } else if (type == "cost.sum2") {
        return *new SumOfSquaresCostLayer(convNetGPU, paramsDict);
    } else if (type == "cost.gsum2") {
        return *new GatedSumOfSquaresCostLayer(convNetGPU, paramsDict);
    } else if (type == "cost.tica") {
        return *new TICACostLayer(convNetGPU, paramsDict);
    } else if (type == "cost.msm") {
        return *new MultiSoftmaxCostLayer(convNetGPU, paramsDict);
    } else if (type == "cost.rflickr") {
        return *new RobustFlickrCost(convNetGPU, paramsDict);
    }
    throw string("Unknown cost layer type ") + type;
}

/* 
 * =====================
 * CrossEntCostLayer
 * =====================
 */
CrossEntCostLayer::CrossEntCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : CostLayer(convNetGPU, paramsDict, false) {
}

void CrossEntCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getLeadingDim();
        NVMatrix& trueLabelLogProbs = getActs(), correctProbs;
        computeCrossEntCost(labels, probs, trueLabelLogProbs, correctProbs);
        _costv.clear();
        _costv.push_back(-trueLabelLogProbs.sum());
        _costv.push_back(numCases - correctProbs.sum());
    }
}

void CrossEntCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    NVMatrix& labels = *_inputs[0];
    NVMatrix& probs = *_inputs[1];
    NVMatrix& target = _prev[1]->getActsGrad();
    // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
    // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
    bool doWork = _prev[1]->getNext().size() > 1 || _prev[1]->getType() != "softmax";
    if (doWork) {
        computeCrossEntGrad(labels, probs, target, scaleTargets == 1, _coeff);
    }
}

/* 
 * =====================
 * CrossEnt2CostLayer
 * =====================
 */
CrossEnt2CostLayer::CrossEnt2CostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : CostLayer(convNetGPU, paramsDict, false) {
}

void CrossEnt2CostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getLeadingDim();
        labels.applyBinary(CrossEntOperator(), probs, getActs());
        _costv.clear();
        _costv.push_back(-getActs().sum());// / labels.getFollowingDim());
//        printf("-getActs().sum(): %f\n", -getActs().sum());
    }
}

void CrossEnt2CostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    NVMatrix& labels = *_inputs[0];
    NVMatrix& probs = *_inputs[1];
    NVMatrix& target = _prev[1]->getActsGrad();
    // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
    // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
//    printf("_prev[1]->getType():%s\n", _prev[1]->getType().c_str());
    bool doWork =   _prev[1]->getNext().size() > 1 
                    || _prev[1]->getType() != "neuron" 
                    || static_cast<NeuronLayer*>(_prev[1])->getNeuronType() != "logistic" 
                    ||  _prev[1]->getDeviceID() != _deviceID;
    if (doWork) {
        printf("Computing cross-ent gradient the stupid way\n");
        if (scaleTargets == 0) {
            labels.applyBinary(CrossEntGradientOperator(_coeff), probs, target);
        } else {
            labels.applyTernary(AddGradientBinaryOperator<CrossEntGradientOperator>(CrossEntGradientOperator(_coeff)), probs, target, target);
        }
    }
}

/* 
 * =====================
 * RobustFlickrCost
 * =====================
 */
RobustFlickrCost::RobustFlickrCost(ConvNetGPU* convNetGPU, PyObject* paramsDict) : CostLayer(convNetGPU, paramsDict, false) {
}

void RobustFlickrCost::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        int numCases = labels.getLeadingDim();
        labels.applyBinary(RobustFlickrCostOperator(), probs, getActs());
        _costv.clear();
        _costv.push_back(getActs().sum());// / labels.getFollowingDim());
//        printf("-getActs().sum(): %f\n", -getActs().sum());
    }
}

void RobustFlickrCost::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    assert(inpIdx == 1);
    NVMatrix& labels = *_inputs[0];
    NVMatrix& probs = *_inputs[1];
    NVMatrix& target = _prev[1]->getActsGrad();
    if (scaleTargets == 0) {
        labels.applyBinary(RobustFlickrCostGradientOperator(_coeff), probs, target);
    } else {
        labels.applyTernary(AddGradientBinaryOperator<RobustFlickrCostGradientOperator>(RobustFlickrCostGradientOperator(_coeff)), probs, target, target);
    }
}

/* 
 * =====================
 * LogregCostLayer
 * =====================
 */
LogregCostLayer::LogregCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : CostLayer(convNetGPU, paramsDict, false) {
    _topk = pyDictGetInt(paramsDict, "topk");
}

void LogregCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix* probs = _inputs[1];
        bool doCompute = !IS_MULTIVIEW_TEST(passType);
        if (!doCompute) {
            if (IS_MULTIVIEW_TEST_START(passType)) {
                probs->copy(_probsAccum);
                _numAccumed = 1;
            } else {
                _probsAccum.add(*probs);
                _numAccumed += 1;
            }
            if (IS_MULTIVIEW_TEST_END(passType)) {
                probs = &_probsAccum;
                probs->scale(1.0 / _numAccumed);
                doCompute = true;
            }
        }
        if (doCompute) {
            int numCases = labels.getNumElements();
            NVMatrix& trueLabelLogProbs = getActs();
            if (_topk == 1) {
                computeLogregCost(labels, *probs, trueLabelLogProbs, _correctProbs);
            } else {
                computeMultiSoftmaxCost(labels, *probs, *probs, trueLabelLogProbs, _correctProbs, _topkProbs, _topk, false);
            }
            _costv.clear();
            double top1 = _correctProbs.sum();
            _costv.push_back(-trueLabelLogProbs.sum());
            _costv.push_back(numCases - top1);
            _costv.push_back(numCases - (_topk == 1 ? top1 : _topkProbs.sum()));
        }
    }
}

NVMatrix& LogregCostLayer::getProbsAccum() {
    return _probsAccum;
}

void LogregCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
//    assert(inpIdx == 1);
    if (inpIdx == 1) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& probs = *_inputs[1];
        NVMatrix& target = _prev[1]->getActsGrad();
        // Numerical stability optimization: if the layer below me is a softmax layer, let it handle
        // the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
        bool doWork = _prev[1]->getNext().size() > 1 || _prev[1]->getType() != "softmax" || _prev[1]->getDeviceID() != _deviceID;
        if (doWork) {
            computeLogregGrad(labels, probs, target, scaleTargets == 1, _coeff);
        }
    }
}

/* 
 * =====================
 * MultiSoftmaxCostLayer
 * =====================
 */
MultiSoftmaxCostLayer::MultiSoftmaxCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : CostLayer(convNetGPU, paramsDict, false) {
    _setSize = pyDictGetInt(paramsDict, "setSize");
    _numOut = pyDictGetInt(paramsDict, "numOut");
    _threads = pyDictGetInt(paramsDict, "threads");
    
    for (int i = 0; i < _threads; i++) {
        B.push_back(new Matrix(_numOut + 1, _setSize + 1));
        B[i]->apply(Matrix::ONE);
        B[i]->scale(-INF);
    }
}

void MultiSoftmaxCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // This layer uses its two inputs together
    if (inpIdx == 0) {
        NVMatrix& labels = *_inputs[0];
        NVMatrix& energies = *_inputs[1];
        labels.copyToHost(_cpuLabels, true);
        Matrix& cpuLabels_T = _cpuLabels.transpose();

        NVMatrix energies_T;
        energies.transpose(energies_T);
        NVMatrix& max = energies_T.max(1);
        energies_T.addVector(max, -1);
        energies_T.copyToHost(_energies_T_CPU, true);

        MultiSoftmaxCPU_T_parallel(_energies_T_CPU, B, _cpuProbs, cpuLabels_T, _setSize, true);
        _probsT.copyFromHost(_cpuProbs, true);
        _probsT.transpose(getActs());
        
        computeCost(true);
        
        delete &max;
        delete &cpuLabels_T;
    }
}

void MultiSoftmaxCostLayer::computeCost(bool useEnergies) {
    NVMatrix& labels = *_inputs[0];
    NVMatrix& energies = *_inputs[1];
    int numCases = labels.getNumElements();
    NVMatrix trueLabelLogProbs, correctProbs, top5Probs;
    computeMultiSoftmaxCost(labels, getActs(), energies, trueLabelLogProbs, correctProbs, top5Probs, _setSize, useEnergies);
    _costv.clear();
    _costv.push_back(-trueLabelLogProbs.sum());
    _costv.push_back(numCases - correctProbs.sum());
    _costv.push_back(numCases - top5Probs.sum());
}

void MultiSoftmaxCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
//    assert(inpIdx == 1);
    if (inpIdx == 1) {
        NVMatrix& labels = *_inputs[0];
        
        labels.copyToHost(_cpuLabels, true);
        Matrix& cpuLabels_T = _cpuLabels.transpose();

        MultiSoftmaxCPU_T_parallel(_energies_T_CPU, B, _cpuProbs, cpuLabels_T, _setSize, false);

        // _cpuProbs now contains gradient
        _probsT.copyFromHost(_cpuProbs);
        _probsT.scale(_coeff);
        if (scaleTargets == 1) {
            _prev[1]->getActsGrad().add(_probsT);
        } else {
            _probsT.transpose(_prev[1]->getActsGrad());
            
        }
        delete &cpuLabels_T;
    }
}

/* 
 * =====================
 * SumOfSquaresCostLayer
 * =====================
 */
SumOfSquaresCostLayer::SumOfSquaresCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : CostLayer(convNetGPU, paramsDict, false) {
}

void SumOfSquaresCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _inputs[0]->apply(NVMatrixOps::Square(), getActs());
    _costv.clear();
    _costv.push_back(getActs().sum());
}

void SumOfSquaresCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    _prev[inpIdx]->getActsGrad().add(*_inputs[0], scaleTargets, -2 * _coeff);
}

/* 
 * =====================
 * GatedSumOfSquaresCostLayer
 * =====================
 */
GatedSumOfSquaresCostLayer::GatedSumOfSquaresCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : CostLayer(convNetGPU, paramsDict, false) {
}

void GatedSumOfSquaresCostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (inpIdx == 0) {
        _inputs[1]->apply(NVMatrixOps::Square(), _ungated);
        _ungated.eltwiseMultByVector(*_inputs[0], getActs());
        _costv.clear();
        _costv.push_back(getActs().sum());
    }
}

void GatedSumOfSquaresCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    if (inpIdx == 0) { // derivative w.r.t. gates
        _prev[inpIdx]->getActsGrad().addSum(_ungated, 0, scaleTargets, -_coeff);
    } else {
        _inputs[inpIdx]->eltwiseMultByVector(*_inputs[0], _ungated);
        _prev[inpIdx]->getActsGrad().add(_ungated, scaleTargets, -2 * _coeff);
    }   
}

/* 
 * =====================
 * TICACostLayer
 * =====================
 */
TICACostLayer::TICACostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict) : CostLayer(convNetGPU, paramsDict, false) {
    _channels = pyDictGetInt(paramsDict, "channels");
    _sizeX = pyDictGetInt(paramsDict, "sizeX");
}

// This one doesn't report any error measure.
void TICACostLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
    // TODO: make it report something when doing a grad check so it doesn't fail.
    // Otherwise it's pretty useless and consumes extra memory to report error numbers.
    convTICA(*_inputs[0], getActs(), _channels, _sizeX, scaleTargets, 1);
    _costv.clear();
    _costv.push_back(getActs().sum()); // TODO: this is wrong, because it contains reciprocals
}

void TICACostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
    convTICAGrad(*_inputs[0], getActs(), _prev[inpIdx]->getActsGrad(), _channels, _sizeX, scaleTargets, _coeff);
}
