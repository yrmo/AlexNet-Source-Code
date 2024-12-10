/*
 * messages.cuh
 *
 *  Created on: 2013-02-25
 *      Author: spoon
 */

#ifndef MESSAGES_CUH_
#define MESSAGES_CUH_

#include <string>

enum MESSAGES { FPROP_TERMINAL,
                BPROP_TERMINAL,
                BPROP_READY,
                FPROP_READY,
                SYNC,
                COPY_TO_CPU,
                COPY_TO_GPU,
                UPDATE_WEIGHTS,
                RESET,
                COST_COMPUTED,
                BPROP_START,
//                COPY,
//                DEQUANTIZE,
                RUNME};

class Message {
protected:
    MESSAGES _messageType;
public:
    MESSAGES getMessageType() {
        return _messageType;
    }
    Message(MESSAGES messageType) : _messageType(messageType) {
    }
    virtual ~Message() {
    }
};

/*
 * A message that performs some simple function in its run method.
 */
class RunMeMessage : public Message {
public:
    RunMeMessage() : Message(RUNME) {
    }
    virtual void run() = 0;

    virtual ~RunMeMessage() {
    }
};

class CopyMessage : public RunMeMessage {
protected:
    NVMatrix* _src, *_tgt;
public:
    CopyMessage(NVMatrix* src, NVMatrix* tgt) : _src(src), _tgt(tgt), RunMeMessage() {
    }
    void run() {
        _src->copy(*_tgt);
    }
    ~CopyMessage() {
        assert(_src->isView());
        delete _src;
    }
};

class DequantizeMessage : public RunMeMessage {
protected:
    Quantizer* _q;
    NVMatrix *_tgt;
public:
    DequantizeMessage(Quantizer* q, NVMatrix* tgt) : _q(q), _tgt(tgt), RunMeMessage()  {
    }
    void run() {
        _q->dequantize(*_tgt);
    }
    ~DequantizeMessage() {
    }
};

class PropMessage : public Message {
protected:
    std::string _fromLayer, _toLayer;
    PASS_TYPE _passType;
public:
    std::string& getFromLayer() {
        return _fromLayer;
    }

    std::string& getToLayer() {
        return _toLayer;
    }

    PASS_TYPE getPassType() {
        return _passType;
    }
    PropMessage(std::string fromLayer, std::string toLayer, PASS_TYPE passType, MESSAGES msgType)
        : _fromLayer(fromLayer), _toLayer(toLayer), _passType(passType), Message(msgType) {
    }
};

class FpropMessage : public PropMessage {
public:
    FpropMessage(std::string fromLayer, std::string toLayer, PASS_TYPE passType)
        : PropMessage(fromLayer, toLayer, passType, FPROP_READY) {
    }
};

class BpropMessage : public PropMessage {
public:
    BpropMessage(std::string fromLayer, std::string toLayer, PASS_TYPE passType)
        : PropMessage(fromLayer, toLayer, passType, BPROP_READY) {
    }
};

class BpropStartMessage : public Message {
protected:
    PASS_TYPE _passType;
public:
    PASS_TYPE getPassType() {
        return _passType;
    }

    BpropStartMessage(PASS_TYPE passType)
        : _passType(passType), Message(BPROP_START) {
    }
};



#endif /* MESSAGES_CUH_ */
