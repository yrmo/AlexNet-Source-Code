/* 
 * File:   multisoftmax.h
 * Author: Alex Krizhevsky
 *
 * Created on May 9, 2012, 5:36 PM
 */

#ifndef MULTISOFTMAX_H
#define	MULTISOFTMAX_H

#include <algorithm>
#include <thread.h>
#include <matrix.h>
#include <vector>

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define EXP exp
#define LOG log
#define INF 1e35f

class MultiSoftmaxWorker : public Thread {
protected:
    Matrix* _elts, *_B, *_probs, *_fixed;
    int _size;
    bool _nofix;
    void* run();
public:
    MultiSoftmaxWorker(Matrix* elts, Matrix* B, Matrix* probs, Matrix* _fixed, int size, bool nofix);
    virtual ~MultiSoftmaxWorker();
};

void MultiSoftmaxCPU_T_parallel(Matrix& elts, std::vector<Matrix*>& B, Matrix& probs, Matrix& fixed, int size, bool nofix);

#endif	/* MULTISOFTMAX_H */

