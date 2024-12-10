/*
 * quantizer.cuh
 *
 *  Created on: 2013-02-15
 *      Author: spoon
 */

#ifndef QUANTIZER_CUH_
#define QUANTIZER_CUH_

#include <Python.h>
#include <util.cuh>
#include <string>
#include <nvmatrix.cuh>
#include <conv_util.cuh>

class Quantizer {
protected:
    NVMatrix* _quantized;
    int _numRows, _numCols;
    bool _trans;
    virtual void _quantize(NVMatrix& src, NVMatrix& tgt);
    virtual void _dequantize(NVMatrix& tgt, float scaleTarget, float scaleOutput);
public:
    Quantizer();
    virtual ~Quantizer();
    void quantize(NVMatrix& src, NVMatrix& tgt);
    void dequantize(NVMatrix& tgt);
    void dequantize(NVMatrix& tgt, float scaleTarget, float scaleOutput);

    static Quantizer& make(PyObject* qDict);
};

class HalfQuantizer : public Quantizer {
protected:
    void _quantize(NVMatrix& src, NVMatrix& tgt);
    void _dequantize(NVMatrix& tgt, float scaleTarget, float scaleOutput);
public:
    HalfQuantizer();
};


#endif /* QUANTIZER_CUH_ */
