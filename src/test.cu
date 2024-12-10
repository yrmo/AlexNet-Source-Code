#include <iostream>
#include <stdlib.h>
#include <vector>
#include <set>
#include <test.cuh>
#include <layer_kernels.cuh>
#include <multisoftmax.h>
#include <cpuCNN.cuh>

static StopWatchInterface *timer = NULL;
using namespace std;
void init_tests(int boardNum) {
	cudaSetDevice(boardNum > -1 ? boardNum : 0);
//    cublasInit();
	NVMatrix::initCublas();
    NVMatrix::initRandom(7);
    sdkCreateTimer(&timer);
}

void compareResults(Matrix& cpu, NVMatrix& gpu, const char* matrixName) {
    Matrix gpuOnCPU(cpu);
    gpu.copyToHost(gpuOnCPU);
    gpuOnCPU.subtract(cpu);
    gpuOnCPU.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU matrices %s: %.6f\n", matrixName, gpuOnCPU.max());
}

void test_blattice() {
    printf("===============================\n");
    printf("test_blattice\n");
    printf("===============================\n");

    int numCases = 2;
    int numOut = 32;
    int setSize = 3;

    cout << "numCases: " << numCases << endl;
    cout << "numOut: " << numOut << endl;
    cout << "setSize: " << setSize << endl;
    NVMatrix nvEnergies(numCases, numOut);
    Matrix energies(numCases, numOut);
    Matrix bLattice(numOut, numCases * setSize);

    nvEnergies.randomizeUniform();
    nvEnergies.copyToHost(energies);
    //energies.randomizeUniform();
    bLattice.apply(Matrix::ZERO); // for now
    
    Matrix &enMax = energies.max(1);
    energies.addVector(enMax, -1);
    
    nvEnergies.copyFromHost(energies);
    NVMatrix nvBLattice(bLattice, true);

    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    
    MSMBackward(nvEnergies, nvBLattice, setSize);

    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    
    printf("Energies: \n");
    nvEnergies.print(10, 5);
    
    printf("GPU (partial) result:\n");
    nvBLattice.print(0, 5, 0, 5);
    printf("GPU time: %.6f msec\n", sdkGetTimerValue(&timer));
}

//void test_multiSoftmaxCPU() {
//    printf("===============================\n");
//    printf("test_multiSoftmaxCPU\n");
//    printf("===============================\n");
//
//    int numCases = 2;
//    int numOut = 5;
//    int setSize = 3;
//    
////    int numCases = 128;
////    int numOut = 1000;
////    int setSize = 5;
//
//    cout << "numCases: " << numCases << endl;
//    cout << "numOut: " << numOut << endl;
//    cout << "setSize: " << setSize << endl;
//    
//    Matrix energies(numCases, numOut);
//    Matrix B(numOut + 1, setSize + 1);
//    Matrix probs(energies);
//    energies.randomizeUniform();
//    probs.apply(Matrix::ZERO); // for now
//    
//    Matrix &enMax = energies.max(1);
//    energies.addVector(enMax, -1);
//    B.apply(Matrix::ZERO);
//
//    sdkResetTimer(&timer);
//    sdkStartTimer(&timer);
//
//    MultiSoftmaxCPU_T(energies, B, probs, setSize, -1);
//
//    cudaThreadSynchronize();
//    sdkStopTimer(&timer);
//    
//    printf("Energies: \n");
//    energies.print(10, 5);
//    
//    printf("CPU (partial) result:\n");
//    probs.print(0, 5, 0, 5);
//    printf("CPU time: %.6f msec\n", sdkGetTimerValue(&timer));
//}

void test_multiSoftmaxCPU_parallel() {
    printf("===============================\n");
    printf("test_multiSoftmaxCPU_parallel\n");
    printf("===============================\n");

    int workers = 8;
    
    int numCases = 2;
    int numOut = 5;
    int setSize = 2;
    
//    int numCases = 128;
//    int numOut = 1000;
//    int setSize = 5;

    cout << "workers: " << workers << endl;
    cout << "numCases: " << numCases << endl;
    cout << "numOut: " << numOut << endl;
    cout << "setSize: " << setSize << endl;
    
    NVMatrix nvEnergies(numCases, numOut);
    Matrix energies(numCases, numOut);
    vector<Matrix*> B;
    Matrix probs(energies);
    Matrix fixed(numCases, 1);
    nvEnergies.randomizeUniform();
    nvEnergies.copyToHost(energies);
    //energies.randomizeUniform();
    probs.apply(Matrix::ZERO); // for now
    
    Matrix &enMax = energies.max(1);
    energies.addVector(enMax, -1);
    
    fixed.apply(Matrix::ONE);
    fixed.scale(2);
    
    for (int i = 0; i < workers; i++) {
        B.push_back(new Matrix(numOut + 1, setSize + 1));
        B[i]->apply(Matrix::ONE);
        B[i]->scale(-INF);
    }

    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    MultiSoftmaxCPU_T_parallel(energies, B, probs, fixed, setSize, true);

    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    
    printf("Energies: \n");
    energies.print(10, 10);
    
    printf("CPU (partial) result:\n");
    probs.print(0, 5, 0, 10);
    printf("CPU time: %.6f msec\n", sdkGetTimerValue(&timer));
}

SoftmaxTree* makeDummyTree(int depth) {
    int numNodes = (1 << (depth + 1)) - 1;
    int numLeaves = (numNodes + 1) / 2;
    
    int idx = numNodes - 1;
    SoftmaxTree* tree = new SoftmaxTree(idx--);
    vector<SoftmaxNode*> prevLevel;
    
    prevLevel.push_back(&tree->getRoot());
    while (idx >= 0) {
        int sz = prevLevel.size();
        for (int i = 0; i < sz; i++) {
            SoftmaxNode& node = *prevLevel[0];
            SoftmaxNode& child1 = node.addChild(idx--);
            SoftmaxNode& child2 = node.addChild(idx--);
            prevLevel.push_back(&child1);
            prevLevel.push_back(&child2);
            prevLevel.erase(prevLevel.begin());
        }
    }
    tree->finalize();
    assert(tree->getNumLeaves() == numLeaves);
    assert(tree->getNumNodes() == numNodes);
    return tree;
}

void test_sftree_fwd() {
    printf("===============================\n");
    printf("test_sftree_fwd\n");
    printf("===============================\n");

    int numFeatures = 6*6*128;
    int depth = 10;
    SoftmaxTree* tree = makeDummyTree(depth);
    cout << "numFeatures: " << numFeatures << endl;
    cout << "depth: " << depth << endl;
    cout << "numNodes: " << tree->getNumNodes() << endl;
    cout << "numLabels: " << tree->getNumLeaves() << endl;
    
    Matrix weights(tree->getNumNodes(), numFeatures);
    Matrix targets(tree->getNumNodes(), numFeatures);
    NVMatrix nvWeights(tree->getNumNodes(), numFeatures);
    NVMatrix nvTargets(tree->getNumNodes(), numFeatures);

    weights.randomizeUniform();

    nvWeights.copyFromHost(weights);
    
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    cpuSoftmaxTreeFwd(weights.getData(), targets.getData(), numFeatures, *tree);

    sdkStopTimer(&timer);
    printf("CPU (partial) result:\n");
    targets.print(0, 7, 0, 5);
    printf("CPU time: %.6f msec\n", sdkGetTimerValue(&timer));

    sdkResetTimer(&timer);
    cudaDeviceSynchronize();
    
    nvWeights.transpose();
    nvTargets.transpose();
    sdkStartTimer(&timer);
    
    tree->makeWeights(nvWeights, nvTargets);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    
    nvWeights.transpose();
    nvTargets.transpose();
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 7, 0, 5);
    printf("GPU time: %.6f msec\n", sdkGetTimerValue(&timer));
    compareResults(targets, nvTargets, "targets");
}

void test_sftree_bwd() {
    printf("===============================\n");
    printf("test_sftree_bwd\n");
    printf("===============================\n");

    int numFeatures = 6*6*128;
    int depth = 10;
    SoftmaxTree* tree = makeDummyTree(depth);
    cout << "numFeatures: " << numFeatures << endl;
    cout << "depth: " << depth << endl;
    cout << "numNodes: " << tree->getNumNodes() << endl;
    cout << "numLabels: " << tree->getNumLeaves() << endl;
    
    Matrix grads(tree->getNumNodes(), numFeatures);
    NVMatrix nvGrads(tree->getNumNodes(), numFeatures);

    grads.randomizeUniform();

    nvGrads.copyFromHost(grads);
    
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    cpuSoftmaxTreeBwd(grads.getData(), numFeatures, *tree);

    sdkStopTimer(&timer);
    printf("CPU (partial) result:\n");
    grads.print(0, 7, 0, 5);
    printf("CPU time: %.6f msec\n", sdkGetTimerValue(&timer));

    sdkResetTimer(&timer);
    cudaDeviceSynchronize();
    
    nvGrads.transpose();
    sdkStartTimer(&timer);
    
    tree->distributeGradients(nvGrads);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    
    nvGrads.transpose();
    printf("GPU (partial) result:\n");
    nvGrads.print(0, 7, 0, 5);
    printf("GPU time: %.6f msec\n", sdkGetTimerValue(&timer));
    compareResults(grads, nvGrads, "grads");
}

void test_sftree_update() {
    printf("===============================\n");
    printf("test_sftree_update\n");
    printf("===============================\n");

    float eps = 0.001, wc = 0.005, mom = 0.9;
    int numFeatures = 6*6*128;
    int depth = 10;
    SoftmaxTree* tree = makeDummyTree(depth);
    cout << "numFeatures: " << numFeatures << endl;
    cout << "depth: " << depth << endl;
    cout << "numNodes: " << tree->getNumNodes() << endl;
    cout << "numLabels: " << tree->getNumLeaves() << endl;
    
    Matrix grads(tree->getNumNodes(), numFeatures);
    Matrix weights(tree->getNumNodes(), numFeatures);
    Matrix incs(tree->getNumNodes(), numFeatures);
    NVMatrix nvGrads(tree->getNumNodes(), numFeatures);
    NVMatrix nvWeights(tree->getNumNodes(), numFeatures);
    NVMatrix nvIncs(tree->getNumNodes(), numFeatures);

    grads.randomizeUniform();
    weights.randomizeUniform();
    incs.randomizeUniform();

    nvGrads.copyFromHost(grads);
    nvWeights.copyFromHost(weights);
    nvIncs.copyFromHost(incs);
    
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    cpuSoftmaxTreeUpdateWeights(weights.getData(), incs.getData(), grads.getData(), numFeatures, eps, mom, wc, *tree);

    sdkStopTimer(&timer);
    printf("CPU (partial) result:\n");
    weights.print(0, 7, 0, 5);
    printf("CPU time: %.6f msec\n", sdkGetTimerValue(&timer));

    sdkResetTimer(&timer);
    cudaDeviceSynchronize();
    
    nvGrads.transpose();
    nvWeights.transpose();
    nvIncs.transpose();
    sdkStartTimer(&timer);
    
    tree->updateWeights(nvWeights, nvIncs, nvGrads, eps, mom, wc);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    
    nvGrads.transpose();
    nvWeights.transpose();
    nvIncs.transpose();
    printf("GPU (partial) result:\n");
    nvWeights.print(0, 7, 0, 5);
    printf("GPU time: %.6f msec\n", sdkGetTimerValue(&timer));
    compareResults(weights, nvWeights, "weights");
    compareResults(incs, nvIncs, "incs");
}

int main(int argc, char** argv) {
    
    int boardNum = get_board_lock();
    if (boardNum == GPU_LOCK_NO_BOARD) {
        printf("No free GPU boards!\n");
        exit(EXIT_FAILURE);
    } else if(boardNum == GPU_LOCK_NO_SCRIPT) {
        printf("Running on default board.\n");
    } else {
        printf("Running on board %d\n", boardNum);
    }
    init_tests(boardNum);
    
//    test_blattice();
//    test_multiSoftmaxCPU();
//    test_multiSoftmaxCPU_parallel();
//    test_sftree_fwd();
//    test_sftree_bwd();
//    test_mdiag();
//    test_mdiagGrad();
    return 0;
}
