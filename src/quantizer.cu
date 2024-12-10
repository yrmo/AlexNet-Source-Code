#include <quantizer.cuh>

using namespace std;

/*=================
 * Quantizer
 * ================
 */

Quantizer& Quantizer::make(PyObject* lrsDict) {
    string type = pyDictGetString(lrsDict, "type");
    if (type == "default") {
        return *new Quantizer();
    } else if (type == "half") {
        return *new HalfQuantizer();
    }
    throw string("Unknown quantizer type ") + type;
}

Quantizer::Quantizer() : _numRows(0), _numCols(0), _trans(false) {
}

Quantizer::~Quantizer() {
}

void Quantizer::quantize(NVMatrix& src, NVMatrix& tgt) {
    _quantize(src, tgt);
    _quantized = &tgt;
    _numRows = src.getNumRows();
    _numCols = src.getNumCols();
    _trans = src.isTrans();
}

void Quantizer::dequantize(NVMatrix& tgt, float scaleTarget, float scaleOutput) {
    _dequantize(tgt, scaleTarget, scaleOutput);
    tgt.setTrans(_trans);
    tgt.reshape(_numRows, _numCols);
}

void Quantizer::dequantize(NVMatrix& tgt) {
    dequantize(tgt, 0, 1);
}

void Quantizer::_quantize(NVMatrix& src, NVMatrix& tgt) {
    src.copy(tgt);
}

void Quantizer::_dequantize(NVMatrix& tgt, float scaleTarget, float scaleOutput) {
    tgt.add(*_quantized, scaleTarget, scaleOutput);
}

/*=================
 * HalfQuantizer
 * ================
 */
HalfQuantizer::HalfQuantizer() : Quantizer() {
}

void HalfQuantizer::_quantize(NVMatrix& src, NVMatrix& tgt) {
    convQuantizeHalf(src, tgt);
}

void HalfQuantizer::_dequantize(NVMatrix& tgt, float scaleTarget, float scaleOutput) {
    convDequantizeHalf(*_quantized, tgt, _numRows * _numCols, scaleTarget, scaleOutput);
}
