/* 
 * File:   softmaxtree.h
 * Author: Alex Krizhevsky
 *
 * Created on September 9, 2012, 5:50 PM
 */

#ifndef SOFTMAXTREE_H
#define	SOFTMAXTREE_H

#include <helper_cuda.h>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <assert.h>

#include <nvmatrix.cuh>
#include <matrix.h>

class SoftmaxNode;
class SoftmaxTree;
typedef std::vector<SoftmaxNode*> SoftmaxNodeV;

class SoftmaxNode {
    friend class SoftmaxTree;
protected:
    SoftmaxNodeV _children;
    SoftmaxNode* _parent;
    int _depth, _height, _size;
    int _label;

    /*
     * Computes height for entire subtree rooted at this node and populates
     * given height->nodes map.
     */
    int setDistances(std::map<int, SoftmaxNodeV*>& nodeHeights,
                     std::map<int, SoftmaxNodeV*>& nodeDepths);
    
    void setNodeCounts(int &nodes, int& leaves);
    /*
     * Compute the number of leaves in this subtree, which is a good estimate
     * of the number of training cases it represents.
     */
    int setSizes(ushort* nodeSizes);
    
public:
    SoftmaxNode(SoftmaxNode* parent, int label);
    ~SoftmaxNode();
    SoftmaxNode& addChild(int label);
    
    int getDepth() const;
    int getHeight() const;
    int getLabel() const;
    int getSize() const;
    SoftmaxNode* getParent(); // Might be null, so must be pointer
    SoftmaxNodeV& getChildren();
};


/*
 * numLabels: the number of leaves in the tree (normally 1000)
 * numNodes: the total number of nodes in the tree
 */
class SoftmaxTree {
    friend class SoftmaxNode;
protected:
    SoftmaxNode* _root;
    std::map<int, SoftmaxNodeV*> _nodeHeights, _nodeDepths;
    /*
     * Map from depth --> ushort2[]
     * where each ushort2 gives the index and parent index
     * of a node at the given depth.
     */
    std::map<int, ushort2*> _nodeFwdMeta;
    /*
     * Map from height --> ushort2[]
     * where each ushort2 gives the index and number of children
     * of a node at the given height.
     */
    std::map<int, ushort2*> _nodeBwdMeta;
    /*
     * Map from height --> ushort[][]
     * where each ushort[] gives children of a given node at a given height.
     */
    std::map<int, ushort**> _nodeChildMeta;
    
    /*
     * An array of length numNodes with index i storing the number
     * of leaves in subtree rooted at node with label i.
     */
    ushort* _nodeSizes;
    int _numNodes, _numLeaves;
    void setDistances();
    void setNodeCounts();
    void setNodeSizes();
    void setFwdMeta();
    void setBwdMeta();
    void preprocess(NVMatrix& inp);
    void postprocess(NVMatrix& inp);
public:
    SoftmaxTree(int rootLabel);
    ~SoftmaxTree();
    
    void finalize();
    
    SoftmaxNode& getRoot();
    SoftmaxNodeV& getNodesAtHeight(int height);
    SoftmaxNodeV& getNodesAtDepth(int depth);
    int getHeight() const;
    int getDepth() const;
    int getNumLeaves() const;
    int getNumNodes() const;
    
    /*
     * offsets: (numNodes, numFeatures)
     * targets: (numNodes, numFeatures) 
     */
    void makeWeights(NVMatrix& offsets, NVMatrix& targets);
    
    /*
     * grads: (numNodes, numFeatures)
     * 
     * The idea is that grads contains gradients for the leaves 
     * (i.e. the first numLabels rows), so this routine will
     * distribute them up the tree.
     */
    void distributeGradients(NVMatrix& grads);
    
    /*
     * inc := mom * inc - wc * epsW * weight + epsW * grad
     * weight := weight + inc
     * 
     * weights: (numNodes, numFeatures)
     * incs:    (numNodes, numFeatures)
     * grads:   (numNodes , numFeatures)
     */
    void updateWeights(NVMatrix& weights, NVMatrix& incs, NVMatrix& grads, float epsWBase, float mom, float wcBase);
    
    
};

#endif	/* SOFTMAXTREE_H */

