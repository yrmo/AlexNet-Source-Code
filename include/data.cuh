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

#ifndef DATA_CUH
#define	DATA_CUH

#include <vector>
#include <algorithm>
#include "util.cuh"

class Data {
protected:
    MatrixV* _data;
    void assertDimensions() {
        assert(_data->size() > 0);
        for (int i = 1; i < _data->size(); i++) {
            assert(_data->at(i-1)->getNumCols() == _data->at(i)->getNumCols());
            assert(_data->at(i-1)->isTrans() == _data->at(i)->isTrans());
        }
        assert(_data->at(0)->getNumCols() > 0);
    }
public:
    typedef typename MatrixV::iterator T_iter;
    // Cases in columns, but array may be transposed
    // (so in memory they can really be in rows -- in which case the array is transposed
    //  during the copy to GPU).
    Data(PyObject* pyData) {
        _data = getMatrixV(pyData);
        assertDimensions();
    }
    
    Data(MatrixV* data) : _data(data) {
        assertDimensions();
    }

    ~Data() {
        for (T_iter it = _data->begin(); it != _data->end(); ++it) {
            delete *it;
        }
        delete _data;
    }
    
    Matrix& operator [](int idx) const {
        return *_data->at(idx);
    }
    
    int getSize() const {
        return _data->size();
    }
    
    MatrixV& getData() const {
        return *_data;
    }
    
    Matrix& getData(int i) const {
        return *_data->at(i);
    }
    
    bool isTrans() const {
        return _data->at(0)->isTrans();
    }

    int getNumCases() const {
        return _data->at(0)->getNumCols();
    }
};

typedef Data CPUData;

class DataProvider {
protected:
    CPUData* _hData;
    NVMatrixV _data;
    int _minibatchSize;
public:
    DataProvider(int minibatchSize);
    void setData(CPUData&);
    void clearData();
    CPUData& getMinibatch(int idx);
    CPUData& getDataSlice(int startCase, int endCase);
    int getNumMinibatches();
    int getMinibatchSize();
    int getNumCases();
    int getNumCasesInMinibatch(int idx);
};

#endif	/* DATA_CUH */

