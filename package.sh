#!/bin/sh

DEST=./cuda-convnet/trunk
PYTHON_MODULES=/home/spoon/dev/python_modules

mkdir -p $DEST/src/common
mkdir -p $DEST/src/cudaconv2
mkdir -p $DEST/src/nvmatrix
mkdir -p $DEST/include/common
mkdir -p $DEST/include/cudaconv2
mkdir -p $DEST/include/nvmatrix
mkdir -p $DEST/example-layers

cp src/*.cu $DEST/src
cp include/*.cuh $DEST/include

cp ABOUT convdata.py convnet.py layer.py shownet.py $DEST/

cp $NVMATRIX_INCLUDE/*.cuh $DEST/include/nvmatrix
cp $NVMATRIX_INCLUDE/../src/nvmatrix*.cu $DEST/src/nvmatrix
cp $PYTHON_MODULES/util.py $PYTHON_MODULES/options.py $PYTHON_MODULES/ordereddict.py $PYTHON_MODULES/gpumodel.py $PYTHON_MODULES/data.py $DEST/
cp $MYCPP_LIBS_INCLUDE/matrix.h $MYCPP_LIBS_INCLUDE/matrix_funcs.h $MYCPP_LIBS_INCLUDE/queue.h $MYCPP_LIBS_INCLUDE/thread.h $DEST/include/common
cp $MYCPP_LIBS_INCLUDE/matrix.cpp $DEST/src/common
cp $NVCONV2_INCLUDE/conv_util.cuh $NVCONV2_INCLUDE/cudaconv2.cuh $DEST/include/cudaconv2
cp $NVCONV2_INCLUDE/../src/conv_util.cu $NVCONV2_INCLUDE/../src/filter_acts.cu $NVCONV2_INCLUDE/../src/img_acts.cu $NVCONV2_INCLUDE/../src/weight_acts.cu $DEST/src/cudaconv2

cp ./example-layers/*.cfg $DEST/example-layers
cp common-gcc-cuda-4.0.mk build.sh readme.html $DEST
cp Makefile-distrib $DEST/Makefile

