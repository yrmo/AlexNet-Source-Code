#!/bin/bash

# 3000..3146
for i in {3035..3146}; do
#for i in {2000..2047}; do
    echo "Testing on batch $i"
    python convnet.py -f /nobackup/kriz/tmp/ConvNet__2012-06-25_17.55.06 --test-only=1 --test-range="$i" --multiview-test=1
done
