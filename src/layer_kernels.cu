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

#include <assert.h>
#include <vector>
#include <layer_kernels.cuh>


//#define LOG(x) ((x) > 0.0 ? log(x) : -1000.0)

// Computes log(exp(x) + exp(y))
//#define LOGADD(x, y) ()

using namespace std;


/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * energies:        (numOut, numCases)
 * labels:          (1, numCases)
 * maxEnergies:     (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * top5Probs:       (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 * 
 * This routine uses energeis to determine top-1 score because they're more accurate than top-n 
 * probabilities, which have numerical errors in them.
 */
__global__ void kMultiSoftmaxCost_engs(float* probs, float* energies, float* labels, float* maxEnergies,
                                  float* labelLogProbs, float* correctProbs, float* top5Probs,
                                  const int numCases, const int numOut, const int setSize) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxe = maxEnergies[tx];
        const float labelp = probs[label * numCases + tx];
        const float labele = energies[label * numCases + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        int numBiggerEnergies = 0, numEqualsEnergies = 0;
        for (int i = 0; i < numOut; ++i) {
            numBiggerEnergies += energies[i * numCases + tx] > labele;
            numEqualsEnergies += energies[i * numCases + tx] == labele;
        }

        const int slotsLeft = setSize - numBiggerEnergies;
        
        top5Probs[tx] = slotsLeft <= 0 ? 0 : (numEqualsEnergies <= slotsLeft ? 1 : float(slotsLeft) / numEqualsEnergies);
//        if (numEqualsEnergies != 1) {
//            printf("numEqualsEnergies: %d, labelp: %e, maxp: %e\n", numEqualsEnergies, labelp, maxe);
//        }
        correctProbs[tx] = labele != maxe ? 0.0f : 1.0f / float(numEqualsEnergies);
    }
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxEnergies:     (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * top5Probs:       (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 * 
 */
__global__ void kMultiSoftmaxCost(float* probs, float* labels, float* maxProbs,
                                  float* labelLogProbs, float* correctProbs, float* top5Probs,
                                  const int numCases, const int numOut, const int setSize) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxp = maxProbs[tx];
        const float labelp = probs[label * numCases + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        int numBiggerProbs = 0, numEqualsProbs = 0;
        for (int i = 0; i < numOut; ++i) {
            numBiggerProbs += probs[i * numCases + tx] > labelp;
            numEqualsProbs += probs[i * numCases + tx] == labelp;
        }

        const int slotsLeft = setSize - numBiggerProbs;
        
        top5Probs[tx] = slotsLeft <= 0.0f ? 0.0f : (numEqualsProbs <= slotsLeft ? 1.0f : float(slotsLeft) / numEqualsProbs);
        correctProbs[tx] = labelp != maxp ? 0.0f : 1.0f / float(numEqualsProbs);
    }
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * top5Probs:       (1, numCases)   (*out)
 * 
 * target:          (1, numCases) == log(y_l[labels,:]
 */
void computeMultiSoftmaxCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& energies, NVMatrix& labelLogProbs_out,
                       NVMatrix& correctProbs_out, NVMatrix& top5Probs_out, int setSize, bool useEnergies) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    assert(energies.isContiguous());
    assert(energies.isSameDims(probs));
    
//    NVMatrix& maxProbs = probs.max(0);
    NVMatrix& maxPE = useEnergies ? energies.max(0) : probs.max(0);
    
    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    top5Probs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    
    if (useEnergies) {
        cudaFuncSetCacheConfig(kMultiSoftmaxCost_engs, cudaFuncCachePreferL1);
        kMultiSoftmaxCost_engs<<<blocks, threads>>>(probs.getDevData(), energies.getDevData(), labels.getDevData(), maxPE.getDevData(),
                                        labelLogProbs_out.getDevData(), correctProbs_out.getDevData(), top5Probs_out.getDevData(),
                                        numCases, numOut, setSize);
    } else {
        cudaFuncSetCacheConfig(kMultiSoftmaxCost, cudaFuncCachePreferL1);
        kMultiSoftmaxCost<<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), maxPE.getDevData(),
                                        labelLogProbs_out.getDevData(), correctProbs_out.getDevData(), top5Probs_out.getDevData(),
                                        numCases, numOut, setSize);
    }

    getLastCudaError("computeLogregCost: Kernel execution failed");
//    cudaThreadSynchronize();
    delete &maxPE;
}

/*
 * energies:    (numCases, numOut) (yes this is weird)
 * bLattice:    (numOut + 1, setSize, numCases) (*out)
 * 
 * This is intended to work for cases when setSize <= 32.
 * Block size (y, x) = (1, B_X)
 * 
 * NOTE: 
 *  B_X must be a multiple of 32
 */
template <int B_X>
__global__ void kMSMBackward(float* energies, float* bLattice, const int numCases, const int numOut, const int setSize) {
    extern __shared__ float shmem[];
    const int tidx = blockIdx.x * B_X + threadIdx.x;
    const int casesPerWarp = 32 / setSize;
    const int casesPerBlock = casesPerWarp * B_X / 32;
    const int numWorkersPerWarp = casesPerWarp * setSize;
    const int tidxInWarp = tidx % 32;
    const int warpIdx = tidx / 32;
    const int blockCaseIdx = blockIdx.x * casesPerBlock;
    const int caseIdxInBlock = threadIdx.x / setSize;
    const int caseIdx = warpIdx * casesPerWarp + tidxInWarp / setSize;
    const bool doWork = tidxInWarp < numWorkersPerWarp && caseIdx < numCases;
    
    const int bIdx = threadIdx.x + threadIdx.x/setSize + 1;
    volatile float* B = shmem;
    volatile float* shE = &shmem[B_X + casesPerBlock]; // Dimensions (casesPerBlock, 32 + 1)

    const int loadY = warpIdx;
    const int loadX = tidxInWarp;
    
    energies += (blockCaseIdx + loadY) * numOut + loadX;
    bLattice += tidx;
    if (blockIdx.x != 0) {
        return;
    }
    // The first row of the lattice has a 1 in the columns corresponding to 
    // zero set size, 0 elsewhere.
    for (int t = threadIdx.x; t < B_X + casesPerBlock; t += B_X) {
        B[t] = t % setSize == 0;
    }
    
    for (int l = 0; l < numOut / 32; ++l) { // Load 32 energies at a time for casesPerBlock cases
        __syncthreads();
        // Load energies into shmem
        for (int r = 0; r < casesPerBlock && blockCaseIdx + loadY + r < numCases; r += B_X / 32) {
            shE[(r + loadY) * (32 + 1) + loadX] = __expf(energies[r * numOut]);
            printf("%f\n", energies[r * numOut]);
        }
        __syncthreads();
        
        // Compute 32 rows of the lattice
        if (doWork) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                B[bIdx] = B[bIdx - 1] * shE[caseIdxInBlock * (32 + 1) + i] + B[bIdx];
                bLattice[i * numCases * setSize] = B[bIdx];
//                printf("thread %d wrote %d to idx %d\n", tidx, B[bIdx], bIdx);
            }
        }
        printf("thread %d made it\n", tidx);
        bLattice += 32 * numCases * setSize;
    }
//    if (numOut % 32 != 0) {
//        __syncthreads();
//        
//    }
}

/*
 * energies:    (numCases, numOut) (yes this is weird)
 * bLattice:    (numOut + 1, setSize, numCases) (*out)
 */
void MSMBackward(NVMatrix& energies, NVMatrix& bLattice, int setSize) {
    int numCases = energies.getNumRows(); 
    int numOut = energies.getNumCols();

    assert(!energies.isTrans());
    assert(!bLattice.isTrans());
    assert(energies.isContiguous());
    assert(energies.isContiguous());
    
    bLattice.resize((numOut + 1) * setSize, numCases);
    int B_X = 32;
    int casesPerBlock = B_X / setSize;
    int shmem = 4*(B_X + casesPerBlock + casesPerBlock * (32 + 1));
    dim3 threads(B_X, 1);
    dim3 blocks(DIVUP(numCases*setSize, B_X), 1);
    printf("allocating %d words of shmem\n", shmem/4);
    cudaFuncSetCacheConfig(kMSMBackward<32>, cudaFuncCachePreferShared);
    kMSMBackward<32><<<blocks, threads, shmem>>>(energies.getDevData(), bLattice.getDevData(), 
                                     numCases, numOut, setSize);
    getLastCudaError("kMSMBackward: Kernel execution failed");
}

/*
 * E = sum(p_l * log(y_l))
 * probs:           (numOut, numCases)
 * labels:          (numOut, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
__global__ void kCrossEntCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        probs += tx;
        labels += tx;
        maxProbs += tx;
        labelLogProbs += tx;
        correctProbs += tx;
        
        const float maxp = maxProbs[0];

        /*
         * Compute the probability of guessing the correct case if you take the most-probable label.
         * 
         * This is done like this:
         * 
         * - If the most probable label is not equal to the true label, then the probability is zero.
         * - Otherwise, the probability is 1 / (number of labels whose probability is equal to the maximum).
         * 
         * This is certainly overkill -- in practice, it's just about impossible for two labels to get assigned
         * maximum probability. But it's a safety measure to prevent over-estimating your accuracy.
         * Though it could never happen in reality. Well it could. But it wouldn't. Cool?
         */
        float crossEnt = 0.0f;
        int numMax = 0;
        bool correctLabel = false;
        for (int i = 0; i < numOut; i++) {
            const float label_prob = labels[i * numCases];
            const float model_prob = probs[i * numCases];
            numMax += model_prob == maxp;
            crossEnt += label_prob * safelog(model_prob);
            correctLabel |= model_prob == maxp && label_prob > 0.0f;
        }
        labelLogProbs[0] = crossEnt;
        if (!correctLabel) {
            correctProbs[0] = 0.0f;
        } else {
            correctProbs[0] = 1.0f / float(numMax);
        }
    }
}

/*
 * E = sum(p_l * log(y_l))
 * y_l:     (numOut, numCases)
 * labels:  (numOut, numCases)
 * 
 * dE_dy_l: (numOut, numCases)
 */
template <bool add>
__global__ void kCrossEntGrad(float* y_l, float* labels, float* dE_dy_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const float label_prob = labels[tidx];
        const float model_prob = y_l[tidx];
        const float v = gradCoeff * __fdividef(label_prob, model_prob);
        if (add) {
            dE_dy_l[tidx] += v;
        } else {
            dE_dy_l[tidx] = v;
        }
    }
}

/*
 * E = sum(p_l * log(y_l))
 * y_l:     (numOut, numCases)
 * labels:  (numOut, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kCrossEntSoftmaxGrad(float* y_l, float* labels, float* dE_dx_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        float v = 0;
        const float model_prob = y_l[tidx];
        for (int j = 0; j < numOut; j++) {
            const float label_prob = labels[j * numCases + tx];
            v += label_prob * ((j == ty) - model_prob);
        }
        v *= gradCoeff;
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}


/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
__global__ void kLogregCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxp = maxProbs[tx];
        const float labelp = probs[label * numCases + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        /*
         * Compute the probability of guessing the correct case if you take the most-probable label.
         * 
         * This is done like this:
         * 
         * - If the most probable label is not equal to the true label, then the probability is zero.
         * - Otherwise, the probability is 1 / (number of labels whose probability is equal to the maximum).
         * 
         * This is certainly overkill -- in practice, it's just about impossible for two labels to get assigned
         * maximum probability. But it's a safety measure to prevent over-estimating your accuracy.
         * Though it could never happen in reality. Well it could. But it wouldn't. Cool?
         */
        if (labelp != maxp) {
            correctProbs[tx] = 0;
        } else {
            int numMax = 0;
            for (int i = 0; i < numOut; i++) {
                numMax += probs[i * numCases + tx] == maxp;
            }
            correctProbs[tx] = 1.0f / float(numMax);
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dy_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregCostGrad(float* y_l, float* labels, float* dE_dy_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * (label == ty);
        v = __fdividef(v, y_l[tidx]);
        if (add) {
            dE_dy_l[tidx] += v;
        } else {
            dE_dy_l[tidx] = v;
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregSoftmaxGrad(float* y_l, float* labels, float* dE_dx_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * ((label == ty) - y_l[tidx]);
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

/*
 * dE_dy_l: (numOut, numCases)
 * y_l:     (numOut, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kSoftmaxGrad(float* dE_dy_l, float* y_l, float* dE_dx_l, const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        float v = 0;
        for (int j = 0; j < numOut; j++) {
            v += dE_dy_l[j * numCases + tx] * ((j == ty) - y_l[j * numCases + tx]);
        }
        v *= y_l[tidx];
        
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

template <int B_X, bool add>
__global__ void kEltwiseMaxGrad(float* actGrad, float* input, float* output, float* target,
                                const int numElements) {
    for (int i = B_X * blockIdx.x + threadIdx.x; i < numElements; i += B_X * gridDim.x) {
        if (add) {
            target[i] += actGrad[i] * (output[i] == input[i]);
        } else {
            target[i] = actGrad[i] * (output[i] == input[i]);
        }
    }
}

void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add) {
    assert(actGrad.isContiguous());
    assert(output.isContiguous());
    assert(input.isContiguous());
    assert(actGrad.isSameDims(input));
    assert(actGrad.isSameDims(output));
    
    dim3 blocks(DIVUP(actGrad.getNumElements(), 128));
    dim3 threads(128);
    if (add) {
        assert(actGrad.isSameDims(target));
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, true>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, true><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    } else {
        target.resize(actGrad);
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, false>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, false><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    }
    
    getLastCudaError("computeEltwiseMaxGrad: Kernel execution failed");
}

/*
 * E = sum_i{-p_i*log(y_i)}
 * probs:           (numOut, numCases)
 * labels:          (numOut, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
void computeCrossEntCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.isSameDims(probs));
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    
    NVMatrix& maxProbs = probs.max(0);
    
    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaFuncSetCacheConfig(kCrossEntCost, cudaFuncCachePreferL1);
    kCrossEntCost<<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     labelLogProbs_out.getDevData(), correctProbs_out.getDevData(),
                                     numCases, numOut);
    getLastCudaError("kCrossEntCost: Kernel execution failed");

    delete &maxProbs;
}

void computeCrossEntGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.isSameDims(probs));
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kCrossEntGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kCrossEntGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    getLastCudaError("kCrossEntGrad: Kernel execution failed");
}

void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, bool add) {
    int numCases = acts.getLeadingDim();
    int numOut = acts.getFollowingDim();

    assert(acts.isSameDims(actsGrad));
    assert(acts.isContiguous());
    assert(actsGrad.isContiguous());
    assert(target.isContiguous());
    assert(acts.isTrans());
    assert(actsGrad.isTrans());

    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(acts);
        kSoftmaxGrad<false><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    } else {
        kSoftmaxGrad<true><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    }
    getLastCudaError("computeSoftmaxGrad: Kernel execution failed");
}

void computeCrossEntSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getLeadingDim() == probs.getLeadingDim() && labels.getFollowingDim() == probs.getFollowingDim());
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(probs.isTrans());
    assert(!labels.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        cudaFuncSetCacheConfig(kCrossEntSoftmaxGrad<false>, cudaFuncCachePreferL1);
        kCrossEntSoftmaxGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                        numCases, numOut, coeff);
    } else {
        cudaFuncSetCacheConfig(kCrossEntSoftmaxGrad<true>, cudaFuncCachePreferL1);
        kCrossEntSoftmaxGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                        numCases, numOut, coeff);
    }
    getLastCudaError("kCrossEntSoftmaxGrad: Kernel execution failed");
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases) == log(y_l[labels,:]
 */
void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    
    NVMatrix& maxProbs = probs.max(0);
    
    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaFuncSetCacheConfig(kLogregCost, cudaFuncCachePreferL1);
    kLogregCost<<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     labelLogProbs_out.getDevData(), correctProbs_out.getDevData(),
                                     numCases, numOut);
    getLastCudaError("computeLogregCost: Kernel execution failed");
//    cudaThreadSynchronize();
    delete &maxProbs;
}

void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregCostGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregCostGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    getLastCudaError("computeLogregGrad: Kernel execution failed");
}

void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregSoftmaxGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregSoftmaxGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    getLastCudaError("computeLogregSoftmaxGrad: Kernel execution failed");
}
