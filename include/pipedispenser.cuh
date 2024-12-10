/*
 * pipedispenser.cuh
 *
 *  Created on: 2013-03-01
 *      Author: spoon
 */

#ifndef PIPEDISPENSER_CUH_
#define PIPEDISPENSER_CUH_

#include <pthread.h>
#include <set>
#include <algorithm>
#include <iterator>
#include <util.cuh>

class PipeDispenser {
protected:
    int _numPipes;
    seti _pipes;
    pthread_mutex_t *_mutex;
    void lock() {
        pthread_mutex_lock(_mutex);
    }

    void unlock() {
        pthread_mutex_unlock(_mutex);
    }
public:
    PipeDispenser(const seti& pipes) {
        _pipes.insert(pipes.begin(), pipes.end());
        _mutex = (pthread_mutex_t*)(malloc(sizeof (pthread_mutex_t)));
        pthread_mutex_init(_mutex, NULL);
    }

    virtual ~PipeDispenser() {
        pthread_mutex_destroy(_mutex);
        free(_mutex);
    }

    virtual int getPipe(const seti& interested) = 0;
    int getPipe(int interested) {
        seti tmp;
        tmp.insert(interested);
        return getPipe(tmp);
    }
    virtual void freePipe(int pipe) = 0;
};

/*
 * This one blocks until there is a free pipe to return.
 */
class PipeDispenserBlocking : public PipeDispenser {
protected:
    pthread_cond_t *_cv;

    void wait() {
        pthread_cond_wait(_cv, _mutex);
    }

    void broadcast() {
        pthread_cond_broadcast(_cv);
    }

    int getAvailablePipes(const seti& interested, intv& available) {
        available.clear();
        std::set_intersection(_pipes.begin(), _pipes.end(), interested.begin(), interested.end(), std::back_inserter(available));
        return available.size();
    }
public:
    PipeDispenserBlocking(const seti& pipes) : PipeDispenser(pipes) {
        _cv = (pthread_cond_t*)(malloc(sizeof (pthread_cond_t)));
        pthread_cond_init(_cv, NULL);
    }

    ~PipeDispenserBlocking() {
        pthread_cond_destroy(_cv);
        free(_cv);
    }

    int getPipe(const seti& interested) {
        lock();
        intv avail;
        while (getAvailablePipes(interested, avail) == 0) {
            wait();
        }
        int pipe = avail[0];
        _pipes.erase(pipe);
        unlock();
        return pipe;
    }

    void freePipe(int pipe) {
        lock();
        _pipes.insert(pipe);
        broadcast();
        unlock();
    }
};

/*
 * This one returns the least-occupied pipe.
 */
class PipeDispenserNonBlocking : public PipeDispenser  {
protected:
    std::map<int,int> _pipeUsers;

public:
    PipeDispenserNonBlocking(const seti& pipes) : PipeDispenser(pipes) {
        for (seti::iterator it = pipes.begin(); it != pipes.end(); ++it) {
            _pipeUsers[*it] = 0;
        }
    }

    int getPipe(const seti& interested) {
        lock();
        int pipe = -1, users = 1 << 30;
        for (seti::iterator it = _pipes.begin(); it != _pipes.end(); ++it) {
            if (interested.count(*it) > 0 && _pipeUsers[*it] < users) {
                pipe = *it;
                users = _pipeUsers[*it];
            }
        }
        if (pipe >= 0) {
            _pipeUsers[pipe]++;
        }
        unlock();
        return pipe;
    }

    void freePipe(int pipe) {
        lock();
        _pipeUsers[pipe]--;
        unlock();
    }
};


#endif /* PIPEDISPENSER_CUH_ */
