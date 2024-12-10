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

#ifndef CONVNET3
#define	CONVNET3

#include <vector>
#include <string>
#include <set>
#include <map>
#include <helper_cuda.h>
#include <time.h>
#include <queue.h>
#include <thread.h>
#include <math.h>
#include <sync.h>
#include <quantizer.cuh>
#include <messages.cuh>
#include <pipedispenser.cuh>

#include "layer.cuh"
#include "data.cuh"
#include "worker.cuh"
#include "weights.cuh"
#include "hostmem.cuh"

class Worker;
class WorkResult;
class Layer;
class DataLayer;
class CostLayer;
class ConvNetGPU;

class ConvNet : public Thread {
protected:
    std::map<std::string,Layer*> _layerMap;
    std::vector<DataLayer*> _dataLayers;
    std::vector<ConvNetGPU*> _convNetThreads; // List of convnet threads
    DataProvider* _dp;
    CPUData* _data;
    ThreadSynchronizer* _sync;
    PipeDispenser* _pd;
    intv* _deviceIDs;
    std::vector<intv*>* _deviceCPUs;
    
    Queue<Worker*> _workerQueue;
    Queue<WorkResult*> _resultQueue;
    Queue<Message*> _msgQueue;
    
    int _numFwdTerminal, _numBwdTerminal;
    int _weightUpdateFreq, _numBwdMiniPasses;
    // For gradient checking
    int _numFailures;
    int _numTests;
    // Training progress (between 0 and 1).
    // Used to determine learning rate based on LearningRateSchedule.
    double _trainingProgress;
    double _baseErr;
    
    void waitForTerminals(int numMsgs, MESSAGES msg);
    void sendMessage(MESSAGES msg, bool sync);
    void findBwdTerminal(Layer& l, std::set<std::string>& visited, std::set<std::string> &terminal);
    void* run();
public:
    ConvNet(PyObject* layerParams, intv& deviceIDs, std::vector<intv*>& deviceCPUs, int minibatchSize, int weightUpdateFreq);
    
    Queue<Message*>& getMessageQueue();
    Queue<Worker*>& getWorkerQueue();
    Queue<WorkResult*>& getResultQueue();
    DataProvider& getDataProvider();
    
    Layer& operator[](string& name);
    Layer& getLayer(string& name);
    void copyToCPU();
    void copyToGPU();
    void updateWeights();
    void reset();
    
    void bprop(PASS_TYPE passType);
    void fprop(PASS_TYPE passType);
    void fprop(int miniIdx, PASS_TYPE passType);
    void fprop(CPUData& data, PASS_TYPE passType);

    void setTrainingProgress(double progress);
    double getTrainingProgress() const;

    bool checkGradient(const std::string& name, float eps, Weights& weights); 
    void checkGradients();
    Cost& getCost();
    Cost& getCost(Cost& cost);
    double getCostValue();
    int getDeviceID(int gpuIdx);
    intv& getDeviceIDs();
    ThreadSynchronizer& getSync();
    void syncWithChildren();
    int getWeightUpdateFreq();
    int getNumBwdMiniPasses();
    int getMinibatchSize();
    PipeDispenser& getPipeDispenser();
};

class ConvNetGPU : public Thread {
protected:
    std::map<std::string,Layer*> _layerMap;
    std::vector<CostLayer*> _costs;
    ConvNet* _convNet;
    int _deviceID;
    Queue<Message*> _msgQueue;
    
    void initCuda();
    virtual void initLayer(PyObject* paramsDict);
    void* run();    
    
    void copyToCPU();
    void copyToGPU();
    void updateWeights();
    void reset();
public:
    ConvNetGPU(PyObject* layerList, int deviceID, intv& deviceCPUs, ConvNet* convNet);
    
    std::map<std::string, Layer*>& getLayerMap();
    
    void bprop(PASS_TYPE passType);
    void fprop(PASS_TYPE passType);
    void fprop(int miniIdx, PASS_TYPE passType);
    int getDeviceID();
    
    ConvNet& getConvNet();
    
    void enqueueMessage(Message* msg);
    Queue<Message*>& getMessageQueue();
    std::vector<CostLayer*>& getCostLayers();
    
    Cost& getCost(int numCases);
    Layer& operator[](string& name);
    Layer& getLayer(string& name);
};

#endif	/* CONVNET */

