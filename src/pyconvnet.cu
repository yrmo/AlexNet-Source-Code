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

#include <Python.h>
#include <arrayobject.h>
#include <assert.h>
#include <helper_cuda.h>
#include <cublas.h>
#include <time.h>
#include <vector>

#include <matrix.h>
#include <queue.h>
#include <worker.cuh>
#include <util.cuh>
#include <cost.cuh>

#include <pyconvnet.cuh>
#include <convnet.cuh>

using namespace std;
static ConvNet* model = NULL;

static PyMethodDef _ConvNetMethods[] = {{ "initModel",          initModel,              METH_VARARGS },
                                        { "startBatch",         startBatch,             METH_VARARGS },
                                        { "finishBatch",        finishBatch,            METH_VARARGS },
                                        { "checkGradients",     checkGradients,         METH_VARARGS },
                                        { "startMultiviewTest", startMultiviewTest,     METH_VARARGS },
                                        { "startFeatureWriter", startFeatureWriter,     METH_VARARGS },
                                        { "startDataGrad",      startDataGrad,          METH_VARARGS },
                                        { "syncWithHost",       syncWithHost,           METH_VARARGS },
                                        { NULL, NULL }
};

#if defined(_WIN64) || defined(_WIN32)
extern "C" __declspec(dllexport) void initpyconvnet() {
    (void) Py_InitModule("pyconvnet", _ConvNetMethods);
    import_array();
}
#else
void INITNAME() {
    (void) Py_InitModule(QUOTEME(MODELNAME), _ConvNetMethods);
    import_array();
}
#endif

PyObject* initModel(PyObject *self, PyObject *args) {
    assert(model == NULL);

    PyDictObject* pyLayerParams;
    PyListObject* pyDeviceIDs, *pyDeviceCPUs;
    int pyMinibatchSize;
    int pyWeightUpdateFreq;

    if (!PyArg_ParseTuple(args, "O!O!O!ii",
                          &PyDict_Type, &pyLayerParams,
                          &PyList_Type, &pyDeviceIDs,
                          &PyList_Type, &pyDeviceCPUs,
                          &pyMinibatchSize,
                          &pyWeightUpdateFreq)) {
        return NULL;
    }
    intv& deviceIDs = *getIntV((PyObject*)pyDeviceIDs);
    vector<intv*>& deviceCPUs = *new vector<intv*>();
    for (int i = 0; i < PyList_GET_SIZE(pyDeviceCPUs); i++) {
        intv* v = getIntV(PyList_GetItem((PyObject*)pyDeviceCPUs, i));
        deviceCPUs.push_back(v);
    }
    model = new ConvNet((PyObject*)pyLayerParams,
                        deviceIDs,
                        deviceCPUs,
                        pyMinibatchSize,
                        pyWeightUpdateFreq);

    model->start();
    return Py_BuildValue("i", 0);
}

/*
 * Starts training/testing on the given batch (asynchronous -- returns immediately).
 */
PyObject* startBatch(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    double progress;
    int test = 0;
    if (!PyArg_ParseTuple(args, "O!d|i",
        &PyList_Type, &data,
        &progress,
        &test)) {
        return NULL;
    }
    CPUData* cpuData = new CPUData((PyObject*)data);
    
    TrainingWorker* wr = new TrainingWorker(*model, *cpuData, progress, test);
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

/*
 * Starts testing on the given batch (asynchronous -- returns immediately).
 */
PyObject* startMultiviewTest(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    int numViews;
    PyArrayObject* pyProbs = NULL;
    char* logregName = NULL;
    if (!PyArg_ParseTuple(args, "O!i|O!s",
        &PyList_Type, &data,
        &numViews,
        &PyArray_Type, &pyProbs,
        &logregName)) {
        return NULL;
    }
    CPUData* cpuData = new CPUData((PyObject*)data);
    MultiviewTestWorker* wr = pyProbs == NULL ? new MultiviewTestWorker(*model, *cpuData, numViews)
                                              : new MultiviewTestWorker(*model, *cpuData, numViews, *new Matrix(pyProbs), logregName);
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

PyObject* startFeatureWriter(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    PyListObject* pyFtrs;
    PyListObject* pyLayerNames;
    if (!PyArg_ParseTuple(args, "O!O!O!",
        &PyList_Type, &data,
        &PyList_Type, &pyFtrs,
        &PyList_Type, &pyLayerNames)) {
        return NULL;
    }
    stringv* layerNames = getStringV((PyObject*)pyLayerNames);
    CPUData* cpuData = new CPUData((PyObject*)data);
    MatrixV* ftrs = getMatrixV((PyObject*)pyFtrs);
    
    FeatureWorker* wr = new FeatureWorker(*model, *cpuData, *ftrs, *layerNames);
    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

PyObject* startDataGrad(PyObject *self, PyObject *args) {
//    assert(model != NULL);
//    PyListObject* data;
//    int dataLayerIdx, softmaxLayerIdx;
//    if (!PyArg_ParseTuple(args, "O!ii",
//        &PyList_Type, &data,
//        &dataLayerIdx, &softmaxLayerIdx)) {
//        return NULL;
//    }
//    CPUData* cpuData = new CPUData((PyObject*)data);
//    Matrix& ftrs = *mvec.back();
//    mvec.pop_back();
//    
//    DataGradWorker* wr = new DataGradWorker(*model, *cpuData, ftrs, dataLayerIdx, softmaxLayerIdx);
//    model->getWorkerQueue().enqueue(wr);
    return Py_BuildValue("i", 0);
}

/*
 * Waits for the trainer to finish training on the batch given to startBatch.
 */
PyObject* finishBatch(PyObject *self, PyObject *args) {
    assert(model != NULL);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::BATCH_DONE);
    
    Cost& cost = res->getResults();
    PyObject* dict = PyDict_New();
    CostMap& costMap = cost.getCostMap();
    for (CostMap::const_iterator it = costMap.begin(); it != costMap.end(); ++it) {
        PyObject* v = PyList_New(0);
        for (vector<double>::const_iterator iv = it->second->begin(); iv != it->second->end(); ++iv) {
            PyObject* f = PyFloat_FromDouble(*iv);
            PyList_Append(v, f);
        }
        PyDict_SetItemString(dict, it->first.c_str(), v);
    }
    
    PyObject* retVal = Py_BuildValue("Ni", dict, cost.getNumCases());
    delete res; // Deletes cost too
    return retVal;
}

PyObject* checkGradients(PyObject *self, PyObject *args) {
    assert(model != NULL);
    PyListObject* data;
    if (!PyArg_ParseTuple(args, "O!",
        &PyList_Type, &data)) {
        return NULL;
    }
    CPUData* cpuData = new CPUData((PyObject*)data);
    
    GradCheckWorker* wr = new GradCheckWorker(*model, *cpuData);
    model->getWorkerQueue().enqueue(wr);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::BATCH_DONE);
    delete res;
    return Py_BuildValue("i", 0);
}

/*
 * Copies weight matrices from GPU to system memory.
 */
PyObject* syncWithHost(PyObject *self, PyObject *args) {
    assert(model != NULL);
    SyncWorker* wr = new SyncWorker(*model);
    model->getWorkerQueue().enqueue(wr);
    WorkResult* res = model->getResultQueue().dequeue();
    assert(res != NULL);
    assert(res->getResultType() == WorkResult::SYNC_DONE);
    
    delete res;
    return Py_BuildValue("i", 0);
}

