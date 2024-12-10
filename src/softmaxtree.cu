#include <softmaxtree.cuh>

#include "layer.cuh"

using namespace std;

/*
 * This launches a series of blocks for every node at a given depth.
 * The "series" just spans the length of the weight vectors.
 * 
 * The operation performed is (loosely):
 *     targets[d] := weights[d] + targets[d-1]
 * 
 * Block size: (y, x) = (1, B_X)
 * Grid size:  (y, x) = (numNodesAtDepth, ceil(numFeatures/B_X))
 * 
 * weights: (numNodes, numFeatures)
 * nodes:   numNodesAtDepth-length array of ushort2 
 *          where x coordinate gives node idx and y coordinate gives parent idx
 * targets: (numNodes, numFeatures)
 * 
 */
template<int B_X,bool root>
__global__ void kSoftmaxTreeFwd(float* weights, ushort2* nodes, float* targets, const int numFeatures) {
    __shared__ ushort2 node; // node.x == node idx, node.y == parent node idx
    const int depthNodeIdx = blockIdx.y;
    const int featureOffset = blockIdx.x * B_X + threadIdx.x;
    if (threadIdx.x == 0) {
        node = nodes[depthNodeIdx];
    }
    __syncthreads();
    weights += featureOffset;
    targets += featureOffset;
    // No loops for now
    if (featureOffset < numFeatures) {
        if (root) {
            targets[node.x * numFeatures] = weights[numFeatures * node.x];
        } else {
            targets[node.x * numFeatures] = targets[node.y * numFeatures] + weights[numFeatures * node.x];
        }
    }
}

/*
 * This launches a series of blocks for every node at a given height.
 * The "series" just spans the length of the weight vectors.
 * 
 * The operation performed is (loosely):
 *     grads[h] := sum_d{grads[h-1]}
 * 
 * Block size: (y, x) = (1, B_X)
 * Grid size:  (y, x) = (numNodesAtHeight, ceil(numFeatures/B_X))
 * 
 * grads:   (numNodes, numFeatures)
 * nodes:   numNodesAtHeight-length array of ushort2 
 *          where x coordinate gives node idx and y coordinate gives NUMBER OF CHILDREN 
 *                                                                   ^ (note difference with kSoftmaxTreeFwd)
 * childrenPtrs: numNodesAtHeight-length array of pointers to children indices
 * 
 * The idea is to start one of these grids at each height, in sequence, starting
 * from height = 1.
 * 
 * The rows 0-numLabels-1 of grads must already have the correct softmax gradients (these
 * are the nodes at height = 0).
 * 
 */
template<int B_X>
__global__ void kSoftmaxTreeBwd(float* grads, ushort2* nodes, ushort** childrenPtrs, const int numFeatures) {
    __shared__ ushort2 node; // node.x == node idx, node.y == parent node idx
    __shared__ ushort* childrenPtr;
    __shared__ ushort children[B_X];
    const int heightNodeIdx = blockIdx.y;
    const int featureOffset = blockIdx.x * B_X + threadIdx.x;
    if (threadIdx.x == 0) {
        node = nodes[heightNodeIdx];
        childrenPtr = childrenPtrs[heightNodeIdx];
    }
    __syncthreads();
    
    grads += featureOffset;
    const int nodeIdx = node.x;
    const int numChildren = node.y;
    
    float nodeGrad = 0;
    for (int c = 0; c < numChildren; c += B_X) {
        
        if (c + threadIdx.x < numChildren) {
            children[threadIdx.x] = childrenPtr[c + threadIdx.x];
        }
        __syncthreads();
        if (featureOffset < numFeatures) {
            const int numChildrenLeft = min(B_X, numChildren - c);
            for  (int cc = 0; cc < numChildrenLeft; ++cc) {
                const int childIdx = children[cc];
                //const int childIdx = childrenPtr[c + cc];
                nodeGrad += grads[childIdx * numFeatures];
            }
        }
        __syncthreads();
    }
    if (featureOffset < numFeatures) {
        grads[nodeIdx * numFeatures] = nodeGrad;
    }
}

/*
 * 
 * Block size: (y, x) = (1, B_X)
 * Grid size:  (y, x) = (1, numNodes)
 * 
 * weights:     (numNodes, numFeatures)
 * weightsInc:  (numNodes, numFeatures)
 * weightsGrad: (numNodes, numFeatures)
 * nodeSizes:   numNodes-array whose ith element gives number of leaves under
 *              node with label i.
 * 
 * TODO: why did I make nodeSizes ushort? int would prolly be fine.
 */
template<int B_X>
__global__ void kSoftmaxTreeUpdateWeights(float* weights, float* weightsInc, float* weightsGrad,
                                          ushort* nodeSizes, const int numFeatures,
                                          float eps, const float mom, float wc) {
    __shared__ int nodeSize; // node.x == node idx, node.y == parent node idx
    const int nodeIdx = blockIdx.x;
    if (threadIdx.x == 0) {
        nodeSize = nodeSizes[nodeIdx];
    }
    __syncthreads();
    weights += nodeIdx * numFeatures;
    weightsInc += nodeIdx * numFeatures;
    weightsGrad += nodeIdx * numFeatures;
    
    // TODO: make these shared?
//    eps *= sqrtf(nodeSize);
    wc /= nodeSize;
    eps /= nodeSize; // larger epsw at the leaves
    
    for (int f = threadIdx.x; f < numFeatures; f += B_X) {
        const float inc = mom * weightsInc[f] + eps * (weightsGrad[f] - wc * weights[f]);
        weightsInc[f] = inc;
        weights[f] += inc;
    }
}

/*
 * ==================
 * SoftmaxNode
 * ==================
 */
int SoftmaxNode::setDistances(std::map<int, SoftmaxNodeV*>& nodeHeights,
                              std::map<int, SoftmaxNodeV*>& nodeDepths) {
    _height = 0;
    for (SoftmaxNodeV::iterator it = _children.begin(); it != _children.end(); ++it) {
        _height = max(_height, (*it)->setDistances(nodeHeights, nodeDepths));
    }
    _height += _children.size() > 0;
    if (nodeHeights.count(_height) == 0) {
        nodeHeights[_height] = new SoftmaxNodeV();
    }
    if (nodeDepths.count(_depth) == 0) {
        nodeDepths[_depth] = new SoftmaxNodeV();
    }

    nodeHeights[_height]->push_back(this);
    nodeDepths[_depth]->push_back(this);
    return _height;
}

void SoftmaxNode::setNodeCounts(int &nodes, int& leaves) {
    nodes++;
    leaves += _children.size() == 0;
    for (SoftmaxNodeV::iterator it = _children.begin(); it != _children.end(); ++it) {
        (*it)->setNodeCounts(nodes, leaves);
    }
}

int SoftmaxNode::setSizes(ushort* nodeSizes) {
    _size = _children.size() == 0;
    for (SoftmaxNodeV::iterator it = _children.begin(); it != _children.end(); ++it) {
        _size += (*it)->setSizes(nodeSizes);
    }
    nodeSizes[_label] = _size;
    return _size;
}

SoftmaxNode::SoftmaxNode(SoftmaxNode* parent, int label) 
    : _parent(parent), _label(label), _size(0), _height(0) {
    _depth = parent == NULL ? 0 : parent->getDepth() + 1;
}

SoftmaxNode::~SoftmaxNode() {
    for (SoftmaxNodeV::iterator it = _children.begin(); it != _children.end(); ++it) {
        delete *it;
    }
}

int SoftmaxNode::getDepth() const {
    return _depth;
}

int SoftmaxNode::getHeight() const {
    return _height;
}

int SoftmaxNode::getSize() const {
    return _size;
}

int SoftmaxNode::getLabel() const {
    return _label;
}

SoftmaxNode* SoftmaxNode::getParent() {
    return _parent;
}

SoftmaxNodeV& SoftmaxNode::getChildren() {
    return _children;
}

SoftmaxNode& SoftmaxNode::addChild(int label) {
    _children.push_back(new SoftmaxNode(this, label));
    return *_children.back();
}

/*
 * ==================
 * SoftmaxTree
 * ==================
 */
SoftmaxTree::SoftmaxTree(int rootLabel)  {
    _root = new SoftmaxNode(NULL, rootLabel);
    _nodeSizes = NULL;
    _numNodes = 0;
    _numLeaves = 0;
}

SoftmaxTree::~SoftmaxTree() {
    checkCudaErrors(cudaFreeHost(_nodeSizes));
    
    for (map<int, SoftmaxNodeV*>::iterator it = _nodeHeights.begin(); it != _nodeHeights.end(); ++it) {
        int height = it->first;
        SoftmaxNodeV& nodes = *it->second;
        for (int n = 0; n < nodes.size(); n++) {
            checkCudaErrors(cudaFreeHost(_nodeChildMeta[height][n]));
        }
        checkCudaErrors(cudaFreeHost(_nodeChildMeta[height]));
        checkCudaErrors(cudaFreeHost(_nodeChildMeta[height]));
        delete &nodes;
    }
    for (map<int, SoftmaxNodeV*>::iterator it = _nodeDepths.begin(); it != _nodeDepths.end(); ++it) {
        SoftmaxNodeV& nodes = *it->second;
        int depth = it->first;
        checkCudaErrors(cudaFreeHost(_nodeFwdMeta[depth]));
        delete &nodes;
    }
    
    delete _root;
}

void SoftmaxTree::setFwdMeta() {
    for (map<int, SoftmaxNodeV*>::iterator it = _nodeDepths.begin(); it != _nodeDepths.end(); ++it) {
        SoftmaxNodeV& nodes = *it->second;
        ushort2* meta;
        checkCudaErrors(cudaHostAlloc(&meta, sizeof(ushort2) * nodes.size(), cudaHostAllocPortable));
        int depth = it->first;
        _nodeFwdMeta[depth] = meta;
        for (int n = 0; n < nodes.size(); n++) {
            meta[n].x = nodes[n]->getLabel();
            // Setting the root to have parent 0 is ok because the fwd kernel won't
            // query this anyway when root == true.
            meta[n].y = nodes[n]->getParent() == NULL ? 0 : nodes[n]->getParent()->getLabel();
        }
    }
}

void SoftmaxTree::setBwdMeta() {
    for (map<int, SoftmaxNodeV*>::iterator it = _nodeHeights.begin(); it != _nodeHeights.end(); ++it) {
        SoftmaxNodeV& nodes = *it->second;
        ushort2* meta;
        ushort** childMeta;
        checkCudaErrors(cudaHostAlloc(&meta, sizeof(ushort2) * nodes.size(), cudaHostAllocPortable));
        checkCudaErrors(cudaHostAlloc(&childMeta, sizeof(ushort*) * nodes.size(), cudaHostAllocPortable));
        int height = it->first;
        _nodeBwdMeta[height] = meta;
        _nodeChildMeta[height] = childMeta;
        for (int n = 0; n < nodes.size(); n++) {
            checkCudaErrors(cudaHostAlloc(&childMeta[n], sizeof(ushort) * nodes[n]->getChildren().size(), cudaHostAllocPortable));
            for (int c = 0; c < nodes[n]->getChildren().size(); c++) {
                childMeta[n][c] = nodes[n]->getChildren()[c]->getLabel();
            }
            meta[n].x = nodes[n]->getLabel();
            meta[n].y = nodes[n]->getChildren().size();
        }
    }
}

void SoftmaxTree::setDistances() {
    _nodeHeights.clear();
    _nodeDepths.clear();
    _root->setDistances(_nodeHeights, _nodeDepths);
}

void SoftmaxTree::setNodeCounts() {
    _numNodes = 0;
    _numLeaves = 0;
    _root->setNodeCounts(_numNodes, _numLeaves);
}

void SoftmaxTree::setNodeSizes() {
    assert(_numLeaves > 0);
    checkCudaErrors(cudaHostAlloc(&_nodeSizes, sizeof(ushort) * _numNodes, cudaHostAllocPortable));
    _root->setSizes(_nodeSizes);
}

void SoftmaxTree::finalize() {
    setDistances();
    setNodeCounts();
    setNodeSizes();
    setFwdMeta();
    setBwdMeta();
}

SoftmaxNode& SoftmaxTree::getRoot() {
    return *_root;
}

SoftmaxNodeV& SoftmaxTree::getNodesAtHeight(int height) {
    return *_nodeHeights[height];
}

SoftmaxNodeV& SoftmaxTree::getNodesAtDepth(int depth) {
    return *_nodeDepths[depth];
}

int SoftmaxTree::getHeight() const {
    return _root->getHeight();
}

/*
 * A tree with only a root is taken to have depth 0.
 */
int SoftmaxTree::getDepth() const {
    return _nodeDepths.size() - 1;
}

int SoftmaxTree::getNumLeaves() const {
    return _numLeaves;
}

int SoftmaxTree::getNumNodes() const {
    return _numNodes;
}

/*
* offsets: (numNodes, numFeatures)
* targets: (numNodes, numFeatures) 
*/
void SoftmaxTree::makeWeights(NVMatrix& offsets, NVMatrix& targets) {
    preprocess(offsets);
    preprocess(targets);
    assert(offsets.getNumRows() == _numNodes);
    assert(targets.isSameDims(offsets));
    int numFeatures = offsets.getNumCols();
    dim3 threads = dim3(256); // 256 seems to work best on dummy binary tree
    dim3 blocks = dim3(DIVUP(numFeatures, 256), 1); // Only the root is at depth 0
    cudaFuncSetCacheConfig(kSoftmaxTreeFwd<256, true>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(kSoftmaxTreeFwd<256, false>, cudaFuncCachePreferL1);
    kSoftmaxTreeFwd<256, true><<<blocks, threads>>>(offsets.getDevData(), _nodeFwdMeta[0], targets.getDevData(), numFeatures);
    getLastCudaError("kSoftmaxTreeFwd: kernel execution failed");
    for (int d = 1; d <= getDepth(); d++) {
        blocks = dim3(DIVUP(numFeatures, 256), _nodeDepths[d]->size());
        kSoftmaxTreeFwd<256, false><<<blocks, threads>>>(offsets.getDevData(), _nodeFwdMeta[d], targets.getDevData(), numFeatures);
        getLastCudaError("kSoftmaxTreeFwd: kernel execution failed");
    }
    
    postprocess(offsets);
    postprocess(targets);
}

/*
* grads: (numNodes, numFeatures)
* 
* The idea is that grads contains gradients for the leaves 
* (i.e. the first numLabels rows), so this routine will
* distribute them up the tree. 
* 
*/
void SoftmaxTree::distributeGradients(NVMatrix& grads) {
    preprocess(grads);
    assert(grads.getNumRows() == _numNodes);
    int numFeatures = grads.getNumCols();
    // The leaves (nodes at height = 0) already have gradients computed.
    // So start at the nodes at height = 1.
    dim3 threads = dim3(512); // this block size works best :/
    cudaFuncSetCacheConfig(kSoftmaxTreeBwd<512>, cudaFuncCachePreferL1);
    for (int h = 1; h <= getHeight(); ++h) {
        dim3 blocks = dim3(DIVUP(numFeatures, 512), _nodeHeights[h]->size());
        kSoftmaxTreeBwd<512><<<blocks, threads>>>(grads.getDevData(), _nodeBwdMeta[h], _nodeChildMeta[h],  numFeatures);
        getLastCudaError("kSoftmaxTreeBwd: kernel execution failed");
    }
    postprocess(grads);
}

/*
 * inc := mom * inc - wc * epsW * weight + epsW * grad
 * weight := weight + inc
 * 
 * weights: (numNodes, numFeatures)
 * incs:    (numNodes, numFeatures)
 * grads:   (numNodes , numFeatures)
 */
void SoftmaxTree::updateWeights(NVMatrix& weights, NVMatrix& incs, NVMatrix& grads, float epsWBase, float mom, float wcBase) {
    preprocess(weights);
    preprocess(incs);
    preprocess(grads);
    
    assert(grads.getNumRows() == _numNodes);
    assert(grads.isSameDims(incs));
    assert(grads.isSameDims(weights));
    int numFeatures = grads.getNumCols();
    dim3 threads = dim3(512);
    dim3 blocks = dim3(_numNodes);
    cudaFuncSetCacheConfig(kSoftmaxTreeUpdateWeights<512>, cudaFuncCachePreferL1);
    kSoftmaxTreeUpdateWeights<512><<<blocks, threads>>>(weights.getDevData(), incs.getDevData(), grads.getDevData(),
                                                        _nodeSizes, numFeatures, epsWBase, mom, wcBase);
    getLastCudaError("kSoftmaxTreeUpdateWeights: kernel execution failed");
    weights.transpose();
    incs.transpose();
    grads.transpose();
}

void SoftmaxTree::preprocess(NVMatrix& inp) {
    inp.transpose();
    assert(!inp.isTrans());
    assert(inp.isContiguous());
}

void SoftmaxTree::postprocess(NVMatrix& inp) {
    inp.transpose();
}
