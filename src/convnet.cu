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

#include <vector>
#include <iostream> 
#include <string>
#include <set>
#include <map>

#include <nvmatrix.cuh>
#include <nvmatrix_operators.cuh>
#include <matrix.h>
#include <convnet.cuh>
#include <util.cuh>

using namespace std;

/* 
 * =======================
 * ConvNet
 * =======================
 */
ConvNet::ConvNet(PyObject* layerParams, intv& deviceIDs, vector<intv*>& deviceCPUs, int minibatchSize, int weightUpdateFreq) : Thread(false) {
    _weightUpdateFreq = weightUpdateFreq;
    _numBwdMiniPasses = 0;
    _deviceIDs = &deviceIDs;
    _deviceCPUs = &deviceCPUs;
    _data = NULL;
    _trainingProgress = 0;
    _sync = new ThreadSynchronizer(deviceIDs.size() + 1);
    seti pipeSet;
    pipeSet.insert(deviceIDs.begin(), deviceIDs.end());
    _pd = new PipeDispenserNonBlocking(pipeSet);
    PyObject* layerList = PyDict_Values(layerParams);
    
    // Data layers live on the manager thread (in CPU memory)
    for (int i = 0; i < PyList_GET_SIZE(layerList); i++) {
        PyObject* paramsDict = PyList_GET_ITEM(layerList, i);
        string layerType = pyDictGetString(paramsDict, "type");
        if (layerType == "data") {
            DataLayer* d = new DataLayer(NULL, paramsDict);
            _dataLayers.push_back(d);
            _layerMap[d->getName()] = d;
        }
    }
    
    // Initialize GPU worker threads
    for (int d = 0; d < deviceIDs.size(); ++d) {
        ConvNetGPU* cng = new ConvNetGPU(layerList, deviceIDs[d], *deviceCPUs[d], this);
        
        _convNetThreads.push_back(cng);
        for (map<string, Layer*>::iterator it = cng->getLayerMap().begin(); it != cng->getLayerMap().end(); ++it) {
            _layerMap[it->first] = it->second;
        }
    }
    // Connect forward/backward links in graph
    for (map<string, Layer*>::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
        PyObject* paramsDict = PyDict_GetItemString(layerParams, it->first.c_str());
        PyObject* inputList = PyDict_GetItemString(paramsDict, "inputs");
        if (inputList != NULL) {
            for (int i = 0; i < PyList_GET_SIZE(inputList); i++) {
                string inputName = PyString_AsString(PyList_GetItem(inputList, i));
                it->second->addPrev(_layerMap[inputName]);
                _layerMap[inputName]->addNext(it->second);
            }
        }
    }
    
    _numFwdTerminal = 0;
    // Execute post-initialization stuff
    for (map<string, Layer*>::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
        it->second->postInit();
        _numFwdTerminal += it->second->getNext().size() == 0; // Number of terminal nodes going forward
    }
    // Find and count the terminal nodes in the backward pass
    set<string> visited, terminal;
    for (int t = 0; t < _convNetThreads.size(); t++) {
        vector<CostLayer*>& cl = _convNetThreads[t]->getCostLayers();
        for (int c = 0; c < cl.size(); c++) {
            findBwdTerminal(*cl[c], visited, terminal);
        }
    }
    _numBwdTerminal = terminal.size();
//    printf("num fwd terminals: %d, back terminals:\n", _numFwdTerminal);
//    for (set<string>::iterator it = terminal.begin(); it != terminal.end(); ++it) {
//        printf("%s\n", (*it).c_str());
//    }
    _dp = new DataProvider(minibatchSize);

    Py_DECREF(layerList);
    assert(_weightUpdateFreq > 0);
}

void ConvNet::findBwdTerminal(Layer& l, set<string>& visited, set<string> &terminal) {
    if (visited.count(l.getName()) == 0) { 
        visited.insert(l.getName());
        if (l.isGradConsumer()) {
            bool hasPrevConsumer = false;
            for (int i = 0; i < l.getPrev().size(); i++) {
                hasPrevConsumer |= l.getPrev()[i]->isGradConsumer();
            }
            if (!hasPrevConsumer || !l.isGradProducer()) {
                terminal.insert(l.getName());
                l.setBwdTerminal(true);
            } else if (l.isGradProducer()) {
                for (int i = 0; i < l.getPrev().size(); i++) {
                    if (l.getPrev()[i]->isGradConsumer()) {
                        findBwdTerminal(*l.getPrev()[i], visited, terminal);
                    }
                }
            }
        }
    }
}

void* ConvNet::run() {
    // The manager thread defaults to using the GPU of the first worker.
    // Put more logic here if this is inappropriate.
    NVMatrix::setDeviceID(_convNetThreads[0]->getDeviceID());
    for (int t = 0; t < _convNetThreads.size(); t++) {
        _convNetThreads[t]->start();
    }
    copyToGPU();
    while (true) {
        Worker* worker = _workerQueue.dequeue();
        worker->run();
        delete worker;
    }
    return NULL;
}

Queue<Worker*>& ConvNet::getWorkerQueue() {
    return _workerQueue;
}

Queue<WorkResult*>& ConvNet::getResultQueue() {
    return _resultQueue;
}

DataProvider& ConvNet::getDataProvider() {
    return *_dp;
}

Layer& ConvNet::operator[](string& name) {
    return *_layerMap[name];
}

Layer& ConvNet::getLayer(string& name) {
    return *_layerMap[name];
}

void ConvNet::sendMessage(MESSAGES msg, bool sync) {
    for (int i = 0; i < _convNetThreads.size(); i++) {
        _convNetThreads[i]->enqueueMessage(new Message(msg));
        if (sync) {
            _convNetThreads[i]->enqueueMessage(new Message(SYNC));
        }
    }
    
    if  (sync) {
        _sync->sync();
    }
}

void ConvNet::copyToCPU() {
    sendMessage(COPY_TO_CPU, true);
}

void ConvNet::copyToGPU() {
    sendMessage(COPY_TO_GPU, false);
}

void ConvNet::updateWeights() {
    sendMessage(UPDATE_WEIGHTS, true);
}

void ConvNet::reset() {
    sendMessage(RESET, false);
}

void ConvNet::fprop(PASS_TYPE passType) {
    assert(_data != NULL);
    reset();
    for (int i = 0; i < _dataLayers.size(); i++) {
        _dataLayers[i]->startFprop(*_data, passType);
    }
    waitForTerminals(_numFwdTerminal, FPROP_TERMINAL);
}

void ConvNet::fprop(CPUData& data, PASS_TYPE passType) {
    if (&data != _data) {
        delete _data;
    }
    _data = &data;
    fprop(passType);
}

void ConvNet::fprop(int miniIdx, PASS_TYPE passType) {
    delete _data;
    reset();
    if (miniIdx == 0 || miniIdx != _dataLayers[0]->getBufferMinibatchIdx()) {
        _data = &_dp->getMinibatch(miniIdx);
        for (int i = 0; i < _dataLayers.size(); i++) {
            _dataLayers[i]->startFprop(*_data, passType);
        }
    } else {
        _data = _dataLayers[0]->getBufferData();
        for (int i = 0; i < _dataLayers.size(); i++) {
            _dataLayers[i]->startFpropFromBuffer(passType);
        }
    }
    CPUData* nextData = miniIdx + 1 == _dp->getNumMinibatches() ? NULL : &_dp->getMinibatch(miniIdx + 1);
    if (nextData != NULL) {
        for (int i = 0; i < _dataLayers.size(); i++) {
            _dataLayers[i]->setBuffer(*nextData, miniIdx + 1);
        }
    }
    waitForTerminals(_numFwdTerminal, FPROP_TERMINAL);
}

void ConvNet::bprop(PASS_TYPE passType) {
    // Weights are updated when this is zero
    _numBwdMiniPasses = (_numBwdMiniPasses + 1) % _weightUpdateFreq;
    for (int i = 0; i < _convNetThreads.size(); i++) {
        _convNetThreads[i]->enqueueMessage(new BpropStartMessage(passType));;
    }
    waitForTerminals(_numBwdTerminal, BPROP_TERMINAL);
    reset();
}

void ConvNet::waitForTerminals(int numMsgs, MESSAGES msg) {
    int terminalsDone = 0;
    while(terminalsDone++ < numMsgs) {
        Message* m = _msgQueue.dequeue();
        assert(m->getMessageType() == msg);
        delete m;
    }
}

// Same as getCost() but adds results to given cost and returns it
Cost& ConvNet::getCost(Cost& cost) {
    Cost &tmp = getCost();
    cost += tmp;
    delete &tmp;
    return cost;
}

Cost& ConvNet::getCost() {
    Cost &tmp = *new Cost(_data->getNumCases());
    for (int i = 0; i < _convNetThreads.size(); i++) {
        Cost& tmp2 = _convNetThreads[i]->getCost(_data->getNumCases());
        tmp |= tmp2;
        delete &tmp2;
    }
    return tmp;
}

double ConvNet::getCostValue() {
    Cost& cost = getCost();
    double val = cost.getValue();
    delete &cost;
    return val;
}

Queue<Message*>& ConvNet::getMessageQueue() {
    return _msgQueue;
}

int ConvNet::getDeviceID(int gpuIdx) {
    if (gpuIdx < 0) {
        return -1;
    }
    return _deviceIDs->at(gpuIdx);
}

intv& ConvNet::getDeviceIDs() {
    return *_deviceIDs;
}

ThreadSynchronizer& ConvNet::getSync() {
    return *_sync;
}

PipeDispenser& ConvNet::getPipeDispenser() {
    return *_pd;
}

void ConvNet::syncWithChildren() {
    sendMessage(SYNC, false);
    _sync->sync();
}

int ConvNet::getWeightUpdateFreq() {
    return _weightUpdateFreq;
}

int ConvNet::getNumBwdMiniPasses() {
    return _numBwdMiniPasses;
}

int ConvNet::getMinibatchSize() {
    return _dp->getMinibatchSize();
}

void ConvNet::setTrainingProgress(double progress) {
	_trainingProgress = progress;
}

double ConvNet::getTrainingProgress() const {
	return _trainingProgress;
}

/*
 * Gradient checking stuff
 */
void ConvNet::checkGradients() {
    _numFailures = 0;
    _numTests = 0;
    fprop(0, PASS_GC);
    _baseErr = getCostValue();
    bprop(PASS_GC);
    
    for (map<string, Layer*>::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
        if (it->second->getDeviceID() >= 0) {
            NVMatrix::setDeviceID(it->second->getDeviceID());
            it->second->checkGradients();
        }
    }
    NVMatrix::setDeviceID(_convNetThreads[0]->getDeviceID());
    
    cout << "------------------------" << endl;
    if (_numFailures > 0) {
        cout << _numFailures << "/" << _numTests << " TESTS FAILED" << endl;
    } else {
        cout << "ALL " << _numTests << " TESTS PASSED" << endl;
    }
}

/*
 * name: weight matrix name
 * eps: finite difference step
 */
bool ConvNet::checkGradient(const string& name, float eps, Weights& weights) {
    Matrix numGrad(weights.getNumRows(), weights.getNumCols());
    Matrix diff(numGrad);
    numGrad.apply(Matrix::ZERO);
    Matrix weightsCPU;

    weights.getW().copyToHost(weightsCPU, true);

    for(int i = 0; i < weights.getNumRows(); i++) {
        for (int j = 0; j < weights.getNumCols(); j++) {
            float v = weightsCPU(i,j);
            weightsCPU(i,j) += eps;
            weights.getW().copyFromHost(weightsCPU);
            weightsCPU(i,j) = v;
            fprop(PASS_GC);
            double err = getCostValue();
            numGrad(i,j) = (err - _baseErr) / (_data->getNumCases() * eps);
            if (isnan(numGrad(i,j)) || isinf(numGrad(i,j))) {
                cout << "Numerical computation produced nan or inf when checking '" << name << "': " << numGrad(i,j) << endl;
                cout << "Consider reducing the sizes of the weights or finite difference steps." << endl;
                cout << "Exiting." << endl;
                exit(1);
            }
            weights.getW().copyFromHost(weightsCPU);
        }
    }
    Matrix gradCPU;
    weights.getGrad().copyToHost(gradCPU, true);
    gradCPU.scale(-1.0 / _data->getNumCases());
    float analNorm = gradCPU.norm();
    float numNorm = numGrad.norm();
    numGrad.subtract(gradCPU, diff);
    float relErr = diff.norm() / analNorm;
    bool fail = relErr >= GC_REL_ERR_THRESH;
    if (fail || !GC_SUPPRESS_PASSES) {
        cout << "========================" << endl;
        printf("(%s) %s GRADIENT CHECK\n", fail ? "****FAIL****" : "PASS", name.c_str());
        cout << "========================" << endl;
        cout << "Analytic:" << endl;
        gradCPU.print(0, 6, 0, 4);
        cout << "Numeric:" << endl;
        numGrad.print(0, 6, 0, 4);
        printf("Analytic norm: %e\n", analNorm);
        printf("Numeric norm:  %e\n", numNorm);
        printf("Relative error: %e\n", relErr);
    }
    _numTests++;
    _numFailures += fail;
    return fail;
}

/* 
 * =======================
 * ConvNetGPU
 * =======================
 */
ConvNetGPU::ConvNetGPU(PyObject* layerList, int deviceID, intv& deviceCPUs, ConvNet* convNet)
    : Thread(false, deviceCPUs), _deviceID(deviceID), _convNet(convNet) {
    try {
        int numLayers = PyList_GET_SIZE(layerList);

        for (int i = 0; i < numLayers; i++) {
            PyObject* paramsDict = PyList_GET_ITEM(layerList, i);
            int layerDeviceID = convNet->getDeviceID(pyDictGetInt(paramsDict, "gpu"));
            if (layerDeviceID == _deviceID) {
                initLayer(paramsDict);
            }
        }
    } catch (string& s) {
        cout << "Error creating ConvNet: " << s << endl;
        exit(1);
    }
}

void ConvNetGPU::initLayer(PyObject* paramsDict) {
    string type = pyDictGetString(paramsDict, "type");
    string name = pyDictGetString(paramsDict, "name");
    if (type == "fc") {
        _layerMap[name] = new FCLayer(this, paramsDict, false, true);
    } else if (type == "treefc") {
        _layerMap[name] = new TreeFCLayer(this, paramsDict);
    } else if (type == "conv") {
        _layerMap[name] = new ConvLayer(this, paramsDict);
    } else if (type == "local") {
        _layerMap[name] = new LocalUnsharedLayer(this, paramsDict);
    } else if (type == "pool") {
        _layerMap[name] = &PoolLayer::makePoolLayer(this, paramsDict);
    } else if (type == "rnorm") {
        _layerMap[name] = new ResponseNormLayer(this, paramsDict);
    } else if (type == "cmrnorm") {
        _layerMap[name] = new CrossMapResponseNormLayer(this, paramsDict);
    } else if (type == "cnorm") {
        _layerMap[name] = new ContrastNormLayer(this, paramsDict);
    } else if (type == "softmax") {
        _layerMap[name] = new SoftmaxLayer(this, paramsDict);
    } else if (type == "eltsum") {
        _layerMap[name] = new EltwiseSumLayer(this, paramsDict);
    } else if (type == "eltmax") {
        _layerMap[name] = new EltwiseMaxLayer(this, paramsDict);
    } else if (type == "neuron") {
        _layerMap[name] = new NeuronLayer(this, paramsDict);
    } else if (type == "nailbed") {
        _layerMap[name] = new NailbedLayer(this, paramsDict);
    } else if (type == "blur") {
        _layerMap[name] = new GaussianBlurLayer(this, paramsDict);
    } else if (type == "href") {
        _layerMap[name] = new HorizontalReflectionLayer(this, paramsDict);
    } else if (type == "resize") {
        _layerMap[name] = new ResizeLayer(this, paramsDict);
    } else if (type == "rgb2yuv") {
        _layerMap[name] = new RGBToYUVLayer(this, paramsDict);
    } else if (type == "rgb2lab") {
        _layerMap[name] = new RGBToLABLayer(this, paramsDict);
    } else if (type == "rscale") {
        _layerMap[name] = new RandomScaleLayer(this, paramsDict);
    } else if (type == "concat") {
        _layerMap[name] = new ConcatenationLayer(this, paramsDict);
    } else if (type == "hs") {
        _layerMap[name] = new HiddenSexLayer(this, paramsDict);
    } else if (strncmp(type.c_str(), "cost.", 5) == 0) {
        CostLayer *c = &CostLayer::makeCostLayer(this, type, paramsDict);
        _layerMap[name] = c;
        _costs.push_back(c);
    } else {
        throw string("Unknown layer type ") + type;
    }
}

/*
 * This executes in a new CPU thread so it's OK to initialize CUDA stuff here. 
 */
void ConvNetGPU::initCuda() { 
    NVMatrix::setDeviceID(_deviceID);
    checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    for (int i = 0; i < _convNet->getDeviceIDs().size(); i++) {
        int d = _convNet->getDeviceID(i);
        if (d != _deviceID) {
            if (NVMatrix::canAccessDevice(_deviceID, d)) {
                printf("Enabling peer access %d --> %d\n", NVMatrix::getDeviceID(), d);
                checkCudaErrors(cudaDeviceEnablePeerAccess(d, 0));
            } else {
                printf("No peer access %d --> %d\n", _deviceID, d);
            }
        }
    }
    NVMatrix::initCublas();
    NVMatrix::initRandom();
    srand(time(0));
}

void* ConvNetGPU::run() {
    initCuda();

    while (true) {
        Message* m = _msgQueue.dequeue();
        if (m->getMessageType() == FPROP_READY) {
            FpropMessage* msg = static_cast<FpropMessage*>(m);
            _layerMap[msg->getToLayer()]->fprop(msg->getPassType());
        } else if (m->getMessageType() == BPROP_READY) {
            BpropMessage* msg = static_cast<BpropMessage*>(m);
            _layerMap[msg->getToLayer()]->incRcvdBInputMsgs();
            _layerMap[msg->getToLayer()]->bprop(msg->getPassType());
        } else if (m->getMessageType() == BPROP_START) {
            BpropStartMessage* msg = static_cast<BpropStartMessage*>(m);
            for (int i = 0; i < _costs.size(); i++) {
                dynamic_cast<Layer*>(_costs[i])->bprop(msg->getPassType());
            }
        } else if (m->getMessageType() == SYNC) {
            _convNet->getSync().sync();
        } else if (m->getMessageType() == COPY_TO_CPU) {
            for (map<string,Layer*>::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
                it->second->copyToCPU();
            }
        } else if (m->getMessageType() == COPY_TO_GPU) {
            for (map<string,Layer*>::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
                it->second->copyToGPU();
            }
        } else if (m->getMessageType() == RESET) {
            for (map<string,Layer*>::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
                it->second->reset();
            }
        } else if (m->getMessageType() == UPDATE_WEIGHTS) {
            for (map<string,Layer*>::iterator it = _layerMap.begin(); it != _layerMap.end(); ++it) {
                it->second->updateWeights();
            }
        } else if (m->getMessageType() == RUNME) {
            RunMeMessage* msg = static_cast<RunMeMessage*>(m);
            msg->run();
        }
        delete m;
    }
    return NULL;
}

Cost& ConvNetGPU::getCost(int numCases) {
    return *new Cost(numCases, _costs);
}

Layer& ConvNetGPU::operator[](string& name) {
    return *_layerMap[name];
}

Layer& ConvNetGPU::getLayer(string& name) {
    return *_layerMap[name];
}

int ConvNetGPU::getDeviceID() {
    return _deviceID;
}

Queue<Message*>& ConvNetGPU::getMessageQueue() {
    return _msgQueue;
}

void ConvNetGPU::enqueueMessage(Message* msg) {
    getMessageQueue().enqueue(msg);
}

vector<CostLayer*>& ConvNetGPU::getCostLayers() {
    return _costs;
}

map<string, Layer*>& ConvNetGPU::getLayerMap() {
    return _layerMap;
}

ConvNet& ConvNetGPU::getConvNet() {
    return *_convNet;
}
