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

#ifndef WEIGHTS_CUH
#define	WEIGHTS_CUH

#include <string>
#include <vector>
#include <iostream>
#include <helper_cuda.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>
#include "util.cuh"
#include "softmaxtree.cuh"
#include <lr.cuh>

using namespace std;

class Weights {
protected:
    Matrix* _hWeights, *_hWeightsInc;
    NVMatrix* _weights, *_weightsInc, *_weightsGrad;
    NVMatrix* _weightsGradAvg, *_weightsGrad2Avg;
    
    LearningRateSchedule* _lrs;

    float _wc, _mom, _wball, _superEps;
    bool _onGPU, _useGrad, _cleanup;
    int _numUpdates;
    
    // Non-NULL if these weights are really shared from some other layer
    Weights* _srcWeights;
public:
    
    class Grad2AvgOperator {
    private:
        float _mom;
    public:
        Grad2AvgOperator(float mom) : _mom(mom) {
        }
        __device__ inline float operator()(const float G2, const float g) const {
            return _mom * G2 + (1.0f - _mom) * g * g;
        }
    };
    
    NVMatrix& operator*() const;
    
    Weights(Weights& srcWeights, LearningRateSchedule& lrs);
    Weights(Matrix& hWeights, Matrix& hWeightsInc, LearningRateSchedule& lrs, float wc, float wball, float mom, float superEps, bool useGrad, bool cleanup=true);
        
    virtual ~Weights();

    virtual NVMatrix& getW() const;
    virtual NVMatrix& getInc() const;
    virtual NVMatrix& getGrad() const;
    virtual Matrix& getCPUW() const;
    virtual Matrix& getCPUWInc() const;
    virtual LearningRateSchedule& getLearningRateSchedule() const;
    virtual int getNumRows() const;
    virtual int getNumCols() const;
    virtual void copyToCPU();
    
    // This function is assumed to be called in the order in which the layers
    // were defined
    virtual void copyToGPU();
    
    virtual void update(float progress);
    int incNumUpdates();
    
    // Returns the number of times a gradient has been computed for this
    // weight matrix during the current pass (interval between two calls of update())
    // through the net. This number will only be greater than 1 if this weight matrix
    // is *shared* by multiple layers in the net.
    int getNumUpdates() const;
    float getEps(float progress) const;
    float getMom() const;
    float getWC() const;
    float getWBall() const;
    bool isUseGrad() const;
    bool isOwner() const;
    float getSuperEps() const;
};

class TreeWeights : public Weights {
protected:
    NVMatrix _effWeights;
    NVMatrix* _leafWeights, *_leafGrad, *_leafInc;
    SoftmaxTree* _tree;
    
public:
    void copyToGPU();
    void update(float progress);
    NVMatrix& getW() const;
    NVMatrix& getInc() const;
    NVMatrix& getGrad() const;
    NVMatrix& getAllW() const;
    NVMatrix& getAllInc() const;
    NVMatrix& getAllGrad() const;
    int getNumRows() const;
    
    void makeWeights();
    void distributeGradients();
    TreeWeights(SoftmaxTree& tree, Matrix& hWeights, Matrix& hWeightsInc, LearningRateSchedule& lrs, float wcBase, float mom);
};

class DummyWeights : public Weights {
public:
    DummyWeights(Matrix& hWeights, Matrix& hWeightsInc, NVMatrix& weights, NVMatrix& incs, NVMatrix& grads);
};

class WeightList {
private:
    std::vector<Weights*> _weightList;

public:
    Weights& operator[](const int idx) const;
    ~WeightList();
    WeightList();
    void addWeights(Weights& w);
    void update(float progress);
    void copyToCPU();
    void copyToGPU();
    int getSize() const;
};

#endif	/* WEIGHTS_CUH */
