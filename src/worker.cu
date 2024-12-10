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
#include <util.cuh>
#include <worker.cuh>

using namespace std;

/* 
 * ====================
 * WorkResult
 * ====================
 */
WorkResult::WorkResult(WorkResult::RESULTS resultType, Cost& results) : _resultType(resultType), _results(&results) {
}

WorkResult::WorkResult(WorkResult::RESULTS resultType) : _resultType(resultType), _results(NULL) {
}

WorkResult::~WorkResult() {
    delete _results; // delete NULL is ok
}

Cost& WorkResult::getResults() const {
    return *_results;
}

WorkResult::RESULTS WorkResult::getResultType() const {
    return _resultType;
}

/* 
 * ====================
 * Worker
 * ====================
 */
Worker::Worker(ConvNet& convNet) : _convNet(&convNet) {
}

/* 
 * ====================
 * DataWorker
 * ====================
 */
DataWorker::DataWorker(ConvNet& convNet, CPUData& data) : Worker(convNet), _data(&data) {
    _dp = &convNet.getDataProvider();
    _dp->setData(*_data);
}

DataWorker::~DataWorker() {
    _dp->clearData();
}

/* 
 * ====================
 * TrainingWorker
 * ====================
 */
TrainingWorker::TrainingWorker(ConvNet& convNet, CPUData& data, double progress, bool test)
    : DataWorker(convNet, data), _progress(progress), _test(test) {
}

// Need to setData here (as opposed to the constructor) because the constructor executes in
// the original CPU thread, which is not the one with GPU access.
void TrainingWorker::run() {
	_convNet->setTrainingProgress(_progress);
    Cost& batchCost = *new Cost(0);
    for (int i = 0; i < _dp->getNumMinibatches(); i++) {
        _convNet->fprop(i, _test ? PASS_TEST : PASS_TRAIN);
        _convNet->getCost(batchCost);
        
        if (!_test) {
            _convNet->bprop(PASS_TRAIN);
            _convNet->updateWeights();
        }
    }
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

/*
 * ====================
 * SyncWorker
 * ====================
 */
SyncWorker::SyncWorker(ConvNet& convNet) : Worker(convNet) {
}

void SyncWorker::run() {
    _convNet->copyToCPU();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::SYNC_DONE));
}

/* 
 * ====================
 * GradCheckWorker
 * ====================
 */
GradCheckWorker::GradCheckWorker(ConvNet& convNet, CPUData& data) 
    : DataWorker(convNet, data) {
}

void GradCheckWorker::run() {
    _convNet->checkGradients();
    exit(0);
}

/* 
 * ====================
 * MultiviewTestWorker
 * ====================
 */
MultiviewTestWorker::MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews, Matrix& cpuProbs, const char* logregName) 
    : DataWorker(convNet, data), _numViews(numViews), _cpuProbs(&cpuProbs), _logregName(logregName) {
    assert(_data->getNumCases() % _numViews == 0);
}

MultiviewTestWorker::MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews) 
    : DataWorker(convNet, data), _numViews(numViews), _cpuProbs(NULL), _logregName("") {
    assert(_data->getNumCases() % _numViews == 0);
}

MultiviewTestWorker::~MultiviewTestWorker() {
    delete _cpuProbs;
}

void MultiviewTestWorker::run() {
    int numCasesReal = _dp->getNumCases() / _numViews;
    int numMiniReal = DIVUP(numCasesReal, _dp->getMinibatchSize());
    
    Cost& batchCost = *new Cost(0);
    for (int i = 0; i < numMiniReal; i++) {
        for (int v = 0; v < _numViews; v++) {
            CPUData& mini = _dp->getDataSlice(v * numCasesReal + i * _dp->getMinibatchSize(),
                                              min((v + 1) * numCasesReal, v * numCasesReal + (i + 1) * _dp->getMinibatchSize()));
            _convNet->fprop(mini, v == 0 ? PASS_MULTIVIEW_TEST_START : v == _numViews - 1 ? PASS_MULTIVIEW_TEST_END : PASS_MULTIVIEW_TEST);
        }
        if (_cpuProbs != NULL) {
            LogregCostLayer& logregLayer = *dynamic_cast<LogregCostLayer*>(&_convNet->getLayer(_logregName));
            cudaSetDevice(logregLayer.getDeviceID());
            Matrix& miniProbs = _cpuProbs->sliceRows(i * _dp->getMinibatchSize(),
                                                     min(numCasesReal, (i + 1) * _dp->getMinibatchSize()));
            NVMatrix& acts = logregLayer.getProbsAccum();
            NVMatrix acts_T;
            acts.transpose(acts_T);
            acts_T.copyToHost(miniProbs);
            
            delete &miniProbs;
        }
        _convNet->getCost(batchCost);
    }
    cudaDeviceSynchronize();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

/* 
 * ====================
 * FeatureWorker
 * ====================
 */
FeatureWorker::FeatureWorker(ConvNet& convNet, CPUData& data, MatrixV& ftrs, stringv& layerNames)
    : DataWorker(convNet, data), _ftrs(&ftrs), _layerNames(&layerNames) {
    assert(layerNames.size() == ftrs.size());
    for (int i = 0; i < layerNames.size(); i++) {
        assert(ftrs[i]->getNumRows() == data.getNumCases());
        assert(!ftrs[i]->isTrans());
    }
}

FeatureWorker::~FeatureWorker() {
    for (int i = 0; i < _ftrs->size(); i++) {
        delete _ftrs->at(i);
    }
    delete _ftrs;
    delete _layerNames;
}

void FeatureWorker::run() {
    
    Cost& batchCost = *new Cost(0);
    
    for (int i = 0; i < _dp->getNumMinibatches(); i++) {
        _convNet->fprop(i, PASS_FEATURE_GEN);
        _convNet->getCost(batchCost);
        for (int f = 0; f < _layerNames->size(); f++) {
            Layer& ftrLayer = _convNet->getLayer(_layerNames->at(f));
            int d = ftrLayer.getDeviceID();
            cudaSetDevice(d);
            Matrix& miniFtrs = _ftrs->at(f)->sliceRows(i * _dp->getMinibatchSize(),
                                                       min(_dp->getNumCases(), (i + 1) * _dp->getMinibatchSize()));
            NVMatrix& acts = ftrLayer.getActs();
            NVMatrix acts_T;
            if (acts.isTrans()) {
                NVMatrix& soft_T = acts.getTranspose();
                soft_T.transpose(acts_T);
                delete &soft_T;
            } else {
                acts.transpose(acts_T);
            }
            acts_T.copyToHost(miniFtrs);
            delete &miniFtrs;
        }
    }
    cudaDeviceSynchronize();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}

/* 
 * ====================
 * DataGradWorker
 * ====================
 */
DataGradWorker::DataGradWorker(ConvNet& convNet, CPUData& data, Matrix& dataGrads, int dataLayerIdx, int softmaxLayerIdx)
    : DataWorker(convNet, data), _dataGrads(&dataGrads), _dataLayerIdx(dataLayerIdx), _softmaxLayerIdx(softmaxLayerIdx) {
    assert(dataGrads.getNumRows() == data.getNumCases());
    assert(!dataGrads.isTrans());
}

DataGradWorker::~DataGradWorker() {
    delete _dataGrads;
}

void DataGradWorker::run() {
//    DataLayer& dataLayer = *dynamic_cast<DataLayer*>(&_convNet->getLayer(_dataLayerIdx));
//    SoftmaxLayer& softmaxLayer = *dynamic_cast<SoftmaxLayer*>(&_convNet->getLayer(_softmaxLayerIdx));
//    softmaxLayer.setDoLogregGrad(false);
//    Cost& batchCost = *new Cost(0);
//    for (int i = 0; i < _dp->getNumMinibatches(); i++) {
//        _convNet->fprop(i, PASS_TEST);
//        _convNet->getCost(batchCost);
//        softmaxLayer.getActs().apply(NVMatrixOps::Log(), softmaxLayer.getActsGrad());
//        
//        softmaxLayer.getActsGrad().addScalar(1);
//        softmaxLayer.getActsGrad().scale(-1);
//        softmaxLayer.incRcvdBInputs();
//        softmaxLayer.bprop(PASS_TEST);
//        
//        Matrix& miniDataGrads = _dataGrads->sliceRows(i * _dp->getMinibatchSize(),
//                                                      min(_dp->getNumCases(), (i + 1) * _dp->getMinibatchSize()));
//        NVMatrix& grads = dataLayer.getActsGrad();
//        NVMatrix grads_T;
//        if (grads.isTrans()) {
//            NVMatrix& soft_T = grads.getTranspose();
//            soft_T.transpose(grads_T);
//            delete &soft_T;
//        } else {
//            grads.transpose(grads_T);
//        }
//        grads_T.copyToHost(miniDataGrads);
//        delete &miniDataGrads;
//        
//        _convNet->reset();
//    }
//    cudaThreadSynchronize();
//    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
}
