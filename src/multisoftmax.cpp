
#include <assert.h>
//#include <mathimf.h>
#include <multisoftmax.h>

using namespace std;

// Computes log(exp(x) + exp(y))
inline double logadd(const double x, const double y) {
    if (x <= -INF && y <= -INF) {
        return -INF;
    }
    const double M = max(x,y);
    const double m = min(x,y);
    const double diff = M - m;
//    return diff > 15 ? M : M + LOG(1.0 + EXP(-diff));
//    return m <= -INF ? M : M + LOG(1.0f + EXP(-diff));
    return diff > 15 ? M : (diff > 5 ? M + EXP(-diff) : M + LOG(1.0 + EXP(-diff)));
}

/*
 * elts:     (numCases, numOut) 
 * B:        (N + 1, size + 1) -- batckward lattice matrix, MUST BE initially -INF
 * fixed:    (numCases, 1)
 * probs:    (numCases, numOut) (*out)
 * 
 * double precision is much faster than single. :/
 */
void MultiSoftmaxCPU_T_logspace(Matrix& elts, Matrix& logB, Matrix& probs, Matrix& fixed, int size, bool nofix) {
    int numCases = elts.getNumRows();
    assert(probs.isSameDims(elts));
    assert(!elts.isTrans());
    assert(!logB.isTrans());
    assert(!probs.isTrans());
    assert(fixed.getNumRows() == numCases);
    assert(fixed.getNumCols() == 1);
    int N = elts.getNumCols();
    Matrix& logF = *new Matrix(size + 1, 1); // Forward column

    // Prepare logB
    logB(N, 0) = 0;
    
    for (int c = 0; c < numCases; ++c) {
        int fx = nofix ? -1 : int(fixed(c, 0));
        // Backward pass
        for (int i = N - 1; i >= 0; --i) {
            double elt = elts(c, i);
            logB(i, 0) = i <= fx ? -INF : 0.0f;
            for (int s = max(1, size - i); s < size + 1; ++s) {
                logB(i, s) = fx == i ? logB(i + 1, s - 1) + elt : logadd(logB(i + 1, s - 1) + elt, logB(i + 1, s));
            }
        }
        // Log partition function
        double logZ = logB(0, size);
        
        // Forward pass
        logF.apply(Matrix::ONE);
        logF.scale(-INF);
        logF(0, 0) = 0;
        
        for (int i = 1; i < N + 1; ++i) {
            double logy = -INF;
            double elt = elts(c, i - 1);
            for (int s = size; s >= 0; --s) {
                if (s < size) {
                    logy = logadd(logy, logF(s, 0) + logB(i, size - 1 - s));
                }
                if (s > 0) {
                    logF(s, 0) = fx == i - 1 ? logF(s - 1, 0) + elt : logadd(logF(s - 1, 0) + elt, logF(s, 0));
                } else if (fx == i - 1) {
                    logF(0, 0) = -INF;
                }
            }
            logy += elt - logZ;
            probs(c, i - 1) = EXP(logy) - (fx >= 0 ? probs(c, i - 1) : 0);
        }
    }
    
    delete &logF;
}

MultiSoftmaxWorker::MultiSoftmaxWorker(Matrix* elts, Matrix* B, Matrix* probs, Matrix* fixed, int size, bool nofix) 
    : Thread(true), _elts(elts), _B(B), _probs(probs), _fixed(fixed), _size(size), _nofix(nofix) {
    
}

MultiSoftmaxWorker::~MultiSoftmaxWorker() {
    delete _elts;
    delete _probs;
    delete _fixed;
}

void* MultiSoftmaxWorker::run() {
    MultiSoftmaxCPU_T_logspace(*_elts, *_B, *_probs, *_fixed, _size, _nofix);
    return NULL;
}

/*
 * elts:     (numCases, numOut) 
 * B:        vector of (N + 1, size + 1) -- batckward lattice matrix, should be initially zero
 * fixed:    (numCases, 1)
 * probs:    (numCases, numOut) (*out)
 * 
 * NOTE: remember to write a version of this for transposed matrices.
 * It may end up being significantly faster, which is important if 
 * I plan to use CPU for this.
 */
void MultiSoftmaxCPU_T_parallel(Matrix& elts, vector<Matrix*>& B, Matrix& probs, Matrix& fixed, int size, bool nofix) {
    int numCases = elts.getNumRows();
    int numWorkers = min(numCases, (int)B.size());
    probs.resize(elts);
    int casesPerWorker = DIVUP(numCases, B.size());
    numWorkers = min(numWorkers, DIVUP(numCases, casesPerWorker));
    vector<Thread*> workers;
    for (int i = 0; i < numWorkers; ++i) {
        Matrix* eltSlice = &elts.sliceRows(i * casesPerWorker, min(elts.getNumRows(), (long int)(i + 1) * casesPerWorker));
        Matrix* probSlice = &probs.sliceRows(i * casesPerWorker, min(elts.getNumRows(), (long int)(i + 1) * casesPerWorker));
        Matrix* fixedSlice = &fixed.sliceRows(i * casesPerWorker, min(elts.getNumRows(), (long int)(i + 1) * casesPerWorker));
        workers.push_back(new MultiSoftmaxWorker(eltSlice, B[i], probSlice, fixedSlice, size, nofix));
        workers[i]->start();
    }
    for (int i = 0; i < numWorkers; ++i) {
        workers[i]->join();
        delete workers[i];
    }
}