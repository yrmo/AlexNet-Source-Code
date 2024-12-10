


#include "softmaxtree.cuh"
/*
 * weights: (numNodes, numFeatures)
 * targets: (numNodes, numFeatures)
 * 
 */
void cpuSoftmaxTreeFwd(float* weights, float* targets, const int numFeatures, SoftmaxTree& tree) {
    for (int d = 0; d <= tree.getDepth(); ++d) {
        for (SoftmaxNodeV::iterator it = tree.getNodesAtDepth(d).begin(); it!= tree.getNodesAtDepth(d).end(); ++it) {
            SoftmaxNode& node = **it;
            SoftmaxNode* parent = node.getParent();
            for (int f = 0; f < numFeatures; ++f) {
                targets[node.getLabel() * numFeatures + f] = weights[node.getLabel() * numFeatures + f]
                                                           + (parent == NULL ? 0 : targets[parent->getLabel() * numFeatures + f]);
            }
        }
    }
}

/*
 * grads:   (numNodes, numFeatures)
 * 
 */
void cpuSoftmaxTreeBwd(float* grads, const int numFeatures, SoftmaxTree& tree) {
    for (int h = 1; h <= tree.getHeight(); ++h) {
        for (SoftmaxNodeV::iterator it = tree.getNodesAtHeight(h).begin(); it!= tree.getNodesAtHeight(h).end(); ++it) {
            SoftmaxNode& node = **it;
            for (int f = 0; f < numFeatures; ++f) {
                grads[node.getLabel() * numFeatures + f] = 0;
            }
            for (SoftmaxNodeV::iterator itc = node.getChildren().begin(); itc!= node.getChildren().end(); ++itc) {
                SoftmaxNode& child = **itc;
                for (int f = 0; f < numFeatures; ++f) {
                    grads[node.getLabel() * numFeatures + f] += grads[child.getLabel() * numFeatures + f];
                }
            }          
        }
    }
}

/*
 * weights:     (numNodes, numFeatures)
 * weightsInc:  (numNodes, numFeatures)
 * weightsGrad: (numNodes, numFeatures)
 * nodeSizes:   numNodes-array whose ith element gives number of leaves under
 *              node with label i.
 */
void cpuSoftmaxTreeUpdateWeights(float* weights, float* weightsInc, float* weightsGrad,
                                 const int numFeatures, float eps, const float mom, float wc, SoftmaxTree& tree) {
    for (int d = 0; d <= tree.getDepth(); d++) {
        for (SoftmaxNodeV::iterator it = tree.getNodesAtDepth(d).begin(); it!= tree.getNodesAtDepth(d).end(); ++it) {
            SoftmaxNode& node = **it;
            float w = wc / node.getSize();
            float e = eps;// * sqrt(node.getSize());
            for (int f = 0; f < numFeatures; ++f) {
                weightsInc[node.getLabel() * numFeatures + f] = mom * weightsInc[node.getLabel() * numFeatures + f]
                                                              + e * (weightsGrad[node.getLabel() * numFeatures + f] - w * weights[node.getLabel() * numFeatures + f]);
                weights[node.getLabel() * numFeatures + f] += weightsInc[node.getLabel() * numFeatures + f];
            }
        }
    }
}
