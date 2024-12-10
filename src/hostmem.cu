#include <hostmem.cuh>

PinnedHostMem::PinnedHostMem() : _numBytes(0), _data(NULL) {

}

PinnedHostMem::~PinnedHostMem() {
    if (_numBytes > 0) {
        checkCudaErrors(cudaFreeHost(_data));
    }
}

void PinnedHostMem::resize(uint bytes) {
    if (_numBytes != bytes) {
        if (_numBytes > 0) {
            checkCudaErrors(cudaFreeHost(_data));
        }
        checkCudaErrors(cudaHostAlloc(&_data, bytes, cudaHostAllocPortable));
        _numBytes = bytes;
    }
}

void PinnedHostMem::copyFrom(void* src, uint bytes) {
    resize(bytes);
    checkCudaErrors(cudaMemcpy(_data, src, bytes, cudaMemcpyDefault));
}

void PinnedHostMem::copyTo(void* dst) {
    checkCudaErrors(cudaMemcpy(dst, _data, _numBytes, cudaMemcpyDefault));
}

void* PinnedHostMem::getData() {
    return _data;
}
