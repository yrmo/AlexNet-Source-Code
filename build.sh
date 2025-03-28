#!/bin/sh

export NVMATRIX_K20X_INCLUDE=$HOME/AlexNet-Source-Code/cuda-convnet2/nvmatrix/include
export NVCONV2_K20X_INCLUDE=$HOME/AlexNet-Source-Code/cuda-convnet2/cudaconvnet/include
export NVMATRIX=$HOME/AlexNet-Source-Code/cuda-convnet2/nvmatrix
export CUDACONV=$HOME/AlexNet-Source-Code/cuda-convnet2/cudaconv3

rm -rf build
rm *.so
mkdir -p build

cp -r src build/
cp -r include build/
cp $NVMATRIX/src/nvmatrix.cu $NVMATRIX/src/nvmatrix_kernels.cu build/src
cp $NVMATRIX/include/nvmatrix.cuh $NVMATRIX/include/nvmatrix_kernels.cuh $NVMATRIX/include/nvmatrix_operators.cuh build/include
# cp $NVMATRIX/src/nvmatrix.cu $NVMATRIX/src/nvmatrix_kernels.cu $NVMATRIX/src/gpu_locking.cpp build/src
# cp $NVMATRIX/include/nvmatrix.cuh $NVMATRIX/include/nvmatrix_kernels.cuh $NVMATRIX/include/nvmatrix_operators.cuh $NVMATRIX/include/gpu_locking.h build/include
cp $CUDACONV/src/conv_util.cu $CUDACONV/src/filter_acts.cu $CUDACONV/src/weight_acts.cu $CUDACONV/src/img_acts.cu build/src
cp $CUDACONV/include/conv_util.cuh $CUDACONV/include/cudaconv2.cuh build/include
cp Makefile-distrib build/Makefile

cd build && make -j kepler=1 $* && cd ..
ln -fs build/*.so ./
