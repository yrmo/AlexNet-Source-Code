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

#ifndef LAYER_CUH
#define	LAYER_CUH

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <assert.h>
#include <nvmatrix.cuh>
#include <multisoftmax.h>
#include <helper_timer.h>

#include "convnet.cuh"
#include "cost.cuh"
#include "weights.cuh"
#include "neuron.cuh"
#include "data.cuh"
#include "layer_kernels.cuh"
#include "hostmem.cuh"
#include "softmaxtree.cuh"
#include "pipedispenser.cuh"

class Cost;
class ConvNet;
class ConvNetGPU;
class CostLayer;
class DataLayer;
//class Message;
//class FpropMessage;

// The input matrix here is the squared norm.
// This replaces the squared norm with:
// 1 if it is below the threshold given by norm2
// norm/sqrt(a) otherwise -- i.e. the desired norm (not squared)
class WeightConstraintOperator {
private:
    float _norm, _norm2;
public:
    WeightConstraintOperator(float norm) : _norm(norm), _norm2(norm*norm) {
    }
    __device__ inline float operator()(const float a) const {
        return a > _norm2 ? __fdividef(_norm, sqrtf(a)) : 1.0f;
    }
};

class WeightContrastNormOperator {
private:
    float _min, _max, _scale;
public:
    WeightContrastNormOperator(float min, float max, float scale) : _min(min), _max(max), _scale(scale) {
    }
    __device__ inline float operator()(float a) const {
        a = sqrtf(a) * _scale;
        return a < _min ? __fdividef(_min, a) : a > _max ? __fdividef(_max, a) : 1.0f;
    }
};

/*
 * Abstract layer.
 */
class Layer {
protected:
    ConvNetGPU* _convNetGPU;
    std::vector<Layer*> _prev, _next;
    int _rcvdFInputs;
    std::map<int, int> _rcvdBInputs;
    int _rcvdBInputMsgs;
    int _numOutputs;
    NVMatrixV _inputs;
    std::map<int, NVMatrix*> _outputs;
    std::map<int, NVMatrix*> _actsGrad; // Layer activity gradients
    bool _gradConsumer, _foundGradConsumers, _trans;
    bool _conserveMem;
    bool _bwdTerminal;
    int _numGradProducersNext;
    int _actsTarget, _actsGradTarget;
    std::string _name, _type;
    int _deviceID;
    intv _nextDeviceIDs;
    HostNVMatrix _hostMemFwd, _hostMemBwd;
    Quantizer* _fwdQuantizer, *_bwdQuantizer;
    
    virtual void fpropNext(PASS_TYPE passType);
    virtual void truncBwdActs(); 
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) = 0;
    
    virtual void bpropCommon(NVMatrix& v, PASS_TYPE passType) {
        // Do nothing by default
    }
    virtual void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType) {
        assert(!isGradProducer()); // Only do nothing if not grad producer
    }
    void shuffle(intv& v);
public:
    static bool _saveActsGrad, _saveActs;
    
    Layer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool trans);
    
    virtual void fprop(PASS_TYPE passType);
    void fprop(NVMatrix& v, PASS_TYPE passType);
    virtual void fprop(NVMatrixV& v, PASS_TYPE passType);
    virtual void bprop(PASS_TYPE passType);
    virtual void bprop(NVMatrix& v, PASS_TYPE passType);
    virtual void reset();
    int getNumCases(NVMatrix& v);
    int incRcvdBInputs(int deviceID);
    int getRcvdFInputs();
    int getRcvdBInputs(int deviceID);
    int incRcvdBInputMsgs();
    bool isGradConsumer();
    bool hasGradProducerNext(std::string& layerName);
    // Does this layer produce a gradient for any layer?
    virtual bool isGradProducer();
    // Does this layer produce a gradient for layer of given name?
    virtual bool isGradProducer(std::string& layerName);
    std::string& getName();
    std::string& getType();
    void addNext(Layer* l);
    void addPrev(Layer* l);
    std::vector<Layer*>& getPrev();
    std::vector<Layer*>& getNext();
    virtual NVMatrix& getActs();
    virtual NVMatrix& getActs(int deviceID);
    virtual NVMatrix& getActsGrad(int deviceID);
    virtual NVMatrix& getActsGrad();
    virtual void postInit();
    int getDeviceID();
    ConvNetGPU& getConvNetGPU();
    ConvNet& getConvNet();
    PipeDispenser& getPipeDispenser();
    void setBwdTerminal(bool t);
    // Do nothing if this layer has no weights
    virtual bool updateWeights() {
        return false;
    }
    virtual void checkGradients() {
    }
    virtual void copyToCPU() {
    }
    virtual void copyToGPU()  {
    }
};

class NeuronLayer : public Layer {
protected:
    Neuron* _neuron;
    string _neuronType;
    
    virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    virtual void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    NeuronLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
    std::string& getNeuronType();
};

class WeightLayer : public Layer {
protected:
    WeightList _weights;
    Weights *_biases;
    float _wStep, _bStep;
    bool _gradComputed;
    
    void bpropCommon(NVMatrix& v, PASS_TYPE passType);
    virtual void bpropBiases(NVMatrix& v, PASS_TYPE passType) = 0;
    virtual void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) = 0;
    virtual void constrainWeights() = 0;
public:
    WeightLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool trans, bool useGrad, bool initWeights);
    virtual bool updateWeights();
    virtual void copyToCPU();
    virtual void copyToGPU();
    virtual void checkGradients();
    Weights& getWeights(int idx);
};

class FCLayer : public WeightLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
    virtual void constrainWeights();
public:
    FCLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool useGrad, bool initWeights);
    FCLayer();
};

class TreeFCLayer : public FCLayer {
protected:
    TreeWeights* _treeWeights;
    static void makeTree(PyObject* pyTree, SoftmaxNode& rootNode);
    void constrainWeights();
public:
    TreeFCLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void checkGradients();
};

class SoftmaxLayer : public Layer {
protected:
    bool _doLogregGrad;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    SoftmaxLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
    void setDoLogregGrad(bool b);
};

class ConcatenationLayer : public Layer {
protected:
    intv* _copyOffsets;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    ConcatenationLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
    void setDoLogregGrad(bool b);
};

class EltwiseSumLayer : public Layer {
protected:
    floatv* _coeffs;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    EltwiseSumLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class EltwiseMaxLayer : public Layer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    EltwiseMaxLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class DataLayer : public Layer {
protected:
    bool _useBuffer;
    int _dataIdx;
    int _bufferMinibatchIdx;
    std::map<int, NVMatrix*> _outputs2; // Buffer for copying data during computation
    CPUData* _bufferData;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void postInit();
    void copyData(CPUData& data, bool other);
    void fpropNext(PASS_TYPE passType);
public:
    DataLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
    NVMatrix& getActs(int deviceID);
    NVMatrix& getActs(int deviceID, bool other);
    bool isGradProducer();
    void fprop(PASS_TYPE passType);
    void fprop(NVMatrixV& data, PASS_TYPE passType);
    void setBuffer(CPUData& data, int minibatchIdx);
    void startFprop(CPUData& data, PASS_TYPE passType);
    void startFpropFromBuffer(PASS_TYPE passType);
    int getBufferMinibatchIdx();
    CPUData* getBufferData();
};

class LocalLayer : public WeightLayer {
protected:
    struct FilterConns {
        int* hFilterConns;
        int* dFilterConns;
    };
    vector<FilterConns>* _filterConns;
    
    intv* _padding, *_stride, *_filterSize, *_channels, *_imgSize, *_groups;
    intv* _imgPixels, *_filterPixels, *_filterChannels, *_overSample, *_randSparse;
    int _modulesX, _modules, _numFilters;

    void copyToGPU();
    
public:
    LocalLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool useGrad);
};

class ConvLayer : public LocalLayer {
protected:
    int _partialSum;
    bool _sharedBiases;
    floatv* _weightContrastNormMin, *_weightContrastNormMax;
    NVMatrix _weightGradTmp, _actGradTmp;

    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
    void truncBwdActs();
    void constrainWeights();

public:
    ConvLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
}; 

class LocalUnsharedLayer : public LocalLayer {
protected:
    NVMatrix _sexMask;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropBiases(NVMatrix& v, PASS_TYPE passType);
    void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
    void constrainWeights();
public:
    LocalUnsharedLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
}; 

class PoolLayer : public Layer {
protected:
    int _channels, _sizeX, _start, _stride, _outputsX;
    int _imgSize;
    string _pool;
public:
    PoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool trans);
    
    static PoolLayer& makePoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
}; 

class AvgPoolLayer : public PoolLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    AvgPoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
}; 

class MaxPoolLayer : public PoolLayer {
protected:
    bool _abs;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    MaxPoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool abs);
};

class RandomPoolLayer : public PoolLayer {
protected:
    bool _doMax;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    RandomPoolLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class RandomScaleLayer : public Layer {
protected:
    int _channels, _imgSize, _tgtSize, _minScaledSize;
    float _maxScale; // should be >= 1
    NVMatrix _rescaledActs;
    std::vector<double> _scaleProbs;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    
    RandomScaleLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class NailbedLayer : public Layer {
protected:
    int _channels, _start, _stride, _outputsX;
    int _imgSize;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    
    NailbedLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class GaussianBlurLayer : public Layer {
protected:
    int _channels;
    Matrix* _hFilter;
    NVMatrix _filter;
    NVMatrix _actGradsTmp;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void copyToGPU();
    
    GaussianBlurLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class HorizontalReflectionLayer : public Layer {
protected:
    int _channels, _imgSize;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    
    HorizontalReflectionLayer(ConvNetGPU* convNet, PyObject* paramsDict);
};

class ResizeLayer : public Layer {
protected:
    int _channels;
    float _scale;
    int _imgSize, _tgtSize;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);

    ResizeLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class HiddenSexLayer : public Layer {
protected:
    bool _enable;
    float _keep;
    NVMatrix _sexMask;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
    HiddenSexLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class RGBToYUVLayer : public Layer {
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);

    RGBToYUVLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class RGBToLABLayer : public Layer {
protected:
    bool _center;
public:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);

    RGBToLABLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class ResponseNormLayer : public Layer {
protected:
    int _channels, _size;
    float _scale, _pow;
    NVMatrix _denoms;

    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
public:
    ResponseNormLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
}; 

class CrossMapResponseNormLayer : public ResponseNormLayer {
protected:
    bool _blocked;
    float _minDiv;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    CrossMapResponseNormLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
}; 

class ContrastNormLayer : public ResponseNormLayer {
protected:
    int _imgSize;
    NVMatrix _meanDiffs;
    
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
    void truncBwdActs();
public:
    ContrastNormLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class CostLayer : public Layer {
protected:
    float _coeff;
    doublev _costv;
public:
    CostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict, bool trans);
    void bprop(NVMatrix& v, PASS_TYPE passType);
//    void bprop(PASS_TYPE passType); // Pure idiocy... it won't compile without this useless definition.
    void fprop(PASS_TYPE passType); 
    
    virtual doublev& getCost();
    float getCoeff();
    bool isGradProducer();
    void setSendTerminalMessages(bool send);
    
    static CostLayer& makeCostLayer(ConvNetGPU* convNetGPU, string& type, PyObject* paramsDict);
};

/*
 * Input 0: labels
 * Input 1: softmax outputs
 */
class CrossEntCostLayer : public CostLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    CrossEntCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

/*
 * Input 0: labels
 * Input 1: softmax outputs
 */
class LogregCostLayer : public CostLayer {
protected:
    NVMatrix _correctProbs, _topkProbs;
    NVMatrix _probsAccum;
    int _numAccumed;
    int _topk;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    LogregCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
    NVMatrix& getProbsAccum();
};

/*
 * Input 0: labels
 * Input 1: logistic outputs
 */
class CrossEnt2CostLayer : public CostLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    CrossEnt2CostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
    class CrossEntOperator {
    public:
        __device__ inline float operator()(const float t, const float y) const {
            return t * safelog(y) + (1.0f - t) * safelog(1.0f - y);
        }
    };
    // Only for use with non-logistic units
    class CrossEntGradientOperator {
    private:
        float _coeff;
    public:
        CrossEntGradientOperator(float coeff) : _coeff(coeff) {
            
        }
        __device__ inline float operator()(const float t, const float y) const {
            return _coeff * (__fdividef(t, y) + __fdividef(1.0f - t, 1.0f - y));
        }
    };
};

/*
 * Input 0: labels
 * Input 1: logistic outputs
 */
class RobustFlickrCost : public CostLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    RobustFlickrCost(ConvNetGPU* convNetGPU, PyObject* paramsDict);
    class RobustFlickrCostOperator {
    public:
        __device__ inline float operator()(const float t, const float y) const {
            const float d = (y-t) * (y-t);
            return __logf(1 + d);// - (t * safelog(y));
        }
    };
    // Only for use with non-logistic units
    class RobustFlickrCostGradientOperator {
    private:
        float _coeff;
    public:
        RobustFlickrCostGradientOperator(float coeff) : _coeff(coeff) {
        }
        __device__ inline float operator()(const float t, const float y) const {
            const float d = y - t;
            return -_coeff * (__fdividef(2.0f * d, 1.0f + d*d) /*- __fdividef(t, y)*/);
        }
    };
};

class SumOfSquaresCostLayer : public CostLayer {
protected:
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    SumOfSquaresCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

/*
 * Input 0: labels
 * Input 1: energies
 */
class MultiSoftmaxCostLayer : public CostLayer {
protected:
    NVMatrix _probsT;
    Matrix _cpuProbs, _cpuLabels, _energies_T_CPU;
    std::vector<Matrix*> B;
    int _setSize, _numOut, _threads;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    MultiSoftmaxCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
    void computeCost(bool useEnergies);
};

/*
 * input 0: gates
 * input 1: what to sum and square
 */
class GatedSumOfSquaresCostLayer : public CostLayer {
protected:
    NVMatrix _ungated;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    GatedSumOfSquaresCostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

class TICACostLayer : public CostLayer {
protected:
    int _sizeX, _channels;
    void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
    void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
    TICACostLayer(ConvNetGPU* convNetGPU, PyObject* paramsDict);
};

#endif	/* LAYER_CUH */

