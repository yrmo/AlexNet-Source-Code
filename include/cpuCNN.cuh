/* 
 * File:   cpuFuncs.h
 * Author: Alex Krizhevsky
 *
 * Created on September 10, 2012, 5:05 PM
 */

#ifndef CPUFUNCS_H
#define	CPUFUNCS_H
#include <helper_cuda.h>
#include <softmaxtree.cuh>
/*
 * weights: (numNodes, numFeatures)
 * nodes:   numNodesAtDepth-length array of ushort2 
 *          where x coordinate gives node idx and y coordinate gives parent idx
 * targets: (numNodes, numFeatures)
 * 
 */
void cpuSoftmaxTreeFwd(float* weights, float* targets, const int numFeatures, SoftmaxTree& tree);

/*
 * grads:   (numNodes, numFeatures)
 * 
 */
void cpuSoftmaxTreeBwd(float* grads, const int numFeatures, SoftmaxTree& tree);

void cpuSoftmaxTreeUpdateWeights(float* weights, float* weightsInc, float* weightsGrad,
                                 const int numFeatures, float eps, const float mom, float wc, SoftmaxTree& tree);

#endif	/* CPUFUNCS_H */

