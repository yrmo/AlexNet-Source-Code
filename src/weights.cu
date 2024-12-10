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

#include <algorithm>
#include <weights.cuh>
#include <softmaxtree.cuh>
#include <lr.cuh>
#include "worker.cuh"

NVMatrix& Weights::operator*() const {
    return getW();
}

Weights::Weights(Weights& srcWeights, LearningRateSchedule& lrs)
    : _srcWeights(&srcWeights), _lrs(&lrs), _wc(0), _wball(0), _onGPU(false), _numUpdates(0),
        _weights(NULL), _weightsInc(NULL), _weightsGrad(NULL), _cleanup(false) {
    _hWeights = &srcWeights.getCPUW();
    _hWeightsInc = &srcWeights.getCPUWInc();
    _mom = srcWeights.getMom();
    _useGrad = srcWeights.isUseGrad();   
    _superEps = srcWeights.getSuperEps();
}

Weights::Weights(Matrix& hWeights, Matrix& hWeightsInc, LearningRateSchedule& lrs, float wc,
                 float wball, float mom, float superEps, bool useGrad, bool cleanup)
    : _srcWeights(NULL), _hWeights(&hWeights), _hWeightsInc(&hWeightsInc), _numUpdates(0),
        _lrs(&lrs), _wc(wc), _wball(wball), _mom(mom), _useGrad(useGrad), _superEps(superEps),
        _onGPU(false), _weights(NULL),_weightsInc(NULL), _weightsGrad(NULL), _cleanup(cleanup) {
    assert(_superEps <= 0 || _useGrad); // superWeights ==> useGrad
}

Weights::~Weights() {
	delete _lrs;
    if (_cleanup) {
        delete _hWeights;
        delete _hWeightsInc;
        if (_srcWeights == NULL) {
            delete _weights;
            delete _weightsInc;
            delete _weightsGrad;
        }
    }
}

NVMatrix& Weights::getW() const {
    assert(_onGPU);
    return *_weights;
}

NVMatrix& Weights::getInc() const {
    assert(_onGPU);
    return *_weightsInc;
}

NVMatrix& Weights::getGrad() const {
    assert(_onGPU);
    return _useGrad ? *_weightsGrad : *_weightsInc;
}

Matrix& Weights::getCPUW() const {
    return *_hWeights;
}

Matrix& Weights::getCPUWInc() const {
    return *_hWeightsInc;
}

int Weights::getNumRows() const {
    return _hWeights->getNumRows();
}

int Weights::getNumCols() const {
    return _hWeights->getNumCols();
}

void Weights::copyToCPU() {
    if (_srcWeights == NULL) {
        assert(_onGPU);
        _weights->copyToHost(*_hWeights);
        _weightsInc->copyToHost(*_hWeightsInc);
    }
}

// This function is assumed to be called in the order in which the layers
// were defined
void Weights::copyToGPU() {
    assert(!_onGPU);
    if (_srcWeights == NULL) {
        _weights = _weights == NULL ? new NVMatrix() : _weights;
        _weightsInc = _weightsInc == NULL ? new NVMatrix() : _weightsInc;
        _weights->copyFromHost(*_hWeights, true);
        _weightsInc->copyFromHost(*_hWeightsInc, true);
        _weightsGrad = _useGrad ? (_weightsGrad == NULL ? new NVMatrix(*_weights) : _weightsGrad) : NULL;
        _weightsGradAvg = _superEps > 0 ? new NVMatrix() : NULL;
        _weightsGrad2Avg = _superEps > 0 ? new NVMatrix() : NULL;
    } else {
        _weights = _srcWeights->_weights;
        _weightsInc = _srcWeights->_weightsInc;
        _weightsGrad = _srcWeights->_weightsGrad;
    }
    _onGPU = true;
}

#define SUPERMOM_THREADS    256
#define SUPERMOM_BLOCKS_MAX 4096

/*
 * V = eps * g / (G2 - G^2 + superEps)^.5 + mom * V
 */
__global__ void superMomUpdate(float* V, float* g, float* G, float* G2,
                               const float eps, const float mom, const float superEps, const int numElements) {
    const int tidx = blockIdx.x * SUPERMOM_THREADS + threadIdx.x;
    
    for (int t = tidx; t < numElements; t += gridDim.x * SUPERMOM_THREADS) {
        V[t] = /*mom*/0.9 * V[t] + eps * __fdividef(g[t], sqrtf(G2[t] - G[t] + superEps));
    }
}

// When _useGrad is false, weightsInc is assumed to contain the 
// entire, properly scaled weight increment.
// OTHERWISE, scale your gradient by 1 / numCases only.
// The scaling by epsW will be done in this routine.
void Weights::update(float progress) {
    // Only true owner of weights updates
    if (_srcWeights == NULL && _lrs->getBaseRate() > 0) {
        assert(_onGPU);
        if (_superEps <= 0) {
            if (_useGrad) {
                _weightsInc->add(*_weightsGrad, _mom, _lrs->getRate(progress));
            }
        } else {
            if (!_weightsGradAvg->isSameDims(*_weightsGrad)) {
                _weightsGradAvg->resize(*_weightsGrad);
                _weightsGrad2Avg->resize(*_weightsGrad);
                _weightsGradAvg->apply(NVMatrixOps::Zero());
                _weightsGrad2Avg->apply(NVMatrixOps::Zero());
            }
            _weightsGradAvg->add(*_weightsGrad, _mom, 1 - _mom);
            _weightsGrad2Avg->applyBinary(Grad2AvgOperator(_mom), *_weightsGrad);
            // Geoff version

            // Make sure all matrices are contiguous
            assert(_weightsGrad->isContiguous());
            assert(_weightsGradAvg->isContiguous());
            assert(_weightsGrad2Avg->isContiguous());
            assert(_weightsInc->isContiguous());
            // Make sure they all have the same transposedness
            assert(_weightsGrad->isTrans() == _weightsGradAvg->isTrans());
            assert(_weightsGradAvg->isTrans() == _weightsGrad2Avg->isTrans());
            assert(_weightsGrad2Avg->isTrans() == _weightsInc->isTrans());
            // Make sure they all have the same sizes
            assert(_weightsGrad->isSameDims(*_weightsGradAvg));
            assert(_weightsGradAvg->isSameDims(*_weightsGrad2Avg));
            assert(_weightsGrad2Avg->isSameDims(*_weightsInc));
            
            int numElements = _weights->getNumElements();
            dim3 blocks(std::min(DIVUP(numElements, SUPERMOM_THREADS), SUPERMOM_BLOCKS_MAX));
            dim3 threads(SUPERMOM_THREADS);
            //float super =  _superEps + 1000000*_weightsGrad2Avg->sum() / numElements;
            //printf("super: %f\n", super);
            superMomUpdate<<<blocks, threads>>>(_weightsInc->getDevData(), _weightsGrad->getDevData(),
                                                _weightsGradAvg->getDevData(), _weightsGrad2Avg->getDevData(),
                                                _lrs->getRate(progress), _mom, _superEps, numElements);
            getLastCudaError("superMomUpdate: Kernel execution failed");
            //_weightsInc->print(4,4);
            //_weightsGrad2Avg->print(5,5);exit(0);
            // Ilya version
        }
        if (_wc > 0) {
            _weightsInc->add(*_weights, -_wc * _lrs->getRate(progress));
        }
        _weights->add(*_weightsInc);
        _numUpdates = 0;
    }
}

int Weights::incNumUpdates() {
    if (_srcWeights != NULL) {
        return _srcWeights->incNumUpdates();
    }
    return _numUpdates++;
}

// Returns the number of times a gradient has been computed for this
// weight matrix during the current pass (interval between two calls of update())
// through the net. This number will only be greater than 1 if this weight matrix
// is *shared* by multiple layers in the net.
int Weights::getNumUpdates() const {
    if (_srcWeights != NULL) {
        return _srcWeights->getNumUpdates();
    }
    return _numUpdates;
}

float Weights::getEps(float progress) const {
    return _lrs->getRate(progress);
}

float Weights::getMom() const {
    return _mom;
}

float Weights::getWC() const {
    return _wc;
}

float Weights::getWBall() const {
    return _wball;
}

bool Weights::isUseGrad() const { // is good grammar
    return _useGrad;
}

bool Weights::isOwner() const {
    return _srcWeights == NULL;
}

float Weights::getSuperEps() const {
    return _superEps;
}

LearningRateSchedule& Weights::getLearningRateSchedule() const {
	return *_lrs;
}

/* 
 * ===============
 * TreeWeights
 * ===============
 */
TreeWeights::TreeWeights(SoftmaxTree& tree, Matrix& hWeights, Matrix& hWeightsInc, LearningRateSchedule& lrs, float wcBase, float mom)
 : _tree(&tree), Weights(hWeights, hWeightsInc, lrs, wcBase, 0, mom, 0, true) {
    assert(hWeights.isTrans());
    assert(hWeightsInc.isTrans());
}

NVMatrix& TreeWeights::getW() const {
    return *_leafWeights;
}

NVMatrix& TreeWeights::getInc() const {
    return *_leafInc;
}

NVMatrix& TreeWeights::getGrad() const {
    return *_leafGrad;
}

NVMatrix& TreeWeights::getAllW() const {
    return *_weights;
}

NVMatrix& TreeWeights::getAllInc() const {
    return *_weightsInc;
}

NVMatrix& TreeWeights::getAllGrad() const {
    return *_weightsGrad;
}

void TreeWeights::copyToGPU() {
    assert(!_onGPU);
    Weights::copyToGPU();
    _tree->finalize();
    _effWeights.resize(*_weights);
    _leafWeights = &_effWeights.sliceCols(0, _tree->getNumLeaves());
    _leafGrad = &_weightsGrad->sliceCols(0, _tree->getNumLeaves());
    _leafInc = &_weightsInc->sliceCols(0, _tree->getNumLeaves());
    assert(_leafWeights->isView());
    makeWeights();
}

int TreeWeights::getNumRows() const {
    return _tree->getNumNodes();
}

void TreeWeights::update(float progress) {
     // Only true owner of weights updates
    if (_lrs->getBaseRate() > 0) {
        assert(_onGPU);
        distributeGradients();
        _tree->updateWeights(*_weights, *_weightsInc, *_weightsGrad, _lrs->getRate(progress), _mom, _wc);
        makeWeights();
        _numUpdates = 0;
    }
}

void TreeWeights::makeWeights() {
    _tree->makeWeights(*_weights, _effWeights);
}

void TreeWeights::distributeGradients() {
    _tree->distributeGradients(*_weightsGrad);
}

/* 
 * ===============
 * DummyWeights
 * ===============
 */
DummyWeights::DummyWeights(Matrix& hWeights, Matrix& hWeightsInc,
                           NVMatrix& weights, NVMatrix& incs, NVMatrix& grads)
 : Weights(hWeights, hWeightsInc, *new LearningRateSchedule(0), 0, 0, 0, 0, true, false) {
    _onGPU = true;
    _weights = &weights;
    _weightsInc = &incs;
    _weightsGrad = &grads;
}

/* 
 * ===============
 * WeightList
 * ===============
 */
Weights& WeightList::operator[](const int idx) const {
    return *_weightList[idx];
}

WeightList::~WeightList() {
    for (int i = 0; i < _weightList.size(); i++) {
        delete _weightList[i];
    }
}

WeightList::WeightList() {
}


void WeightList::addWeights(Weights& w) {
    _weightList.push_back(&w);
}


void WeightList::update(float progress) {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->update(progress);
    }
}

void WeightList::copyToCPU() {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->copyToCPU();
    }
}

void WeightList::copyToGPU() {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->copyToGPU();
    }
}

int WeightList::getSize() const {
    return _weightList.size();
}
