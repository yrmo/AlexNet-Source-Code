#!/bin/sh

python convnet.py -f /ais/gobi3/u/kriz/tmp/ConvNet__2012-08-03_14.28.23 --epochs=22 >> logs/layers-120-4gpu.log
python convnet.py -f /ais/gobi3/u/kriz/tmp/ConvNet__2012-08-03_14.28.23 --layer-params=./layers/layer-params-120-4gpu-auto2.cfg --epochs=49 >> logs/layers-120-4gpu.log
python convnet.py -f /ais/gobi3/u/kriz/tmp/ConvNet__2012-08-03_14.28.23 --layer-params=./layers/layer-params-120-4gpu-auto3.cfg --epochs=66 >> logs/layers-120-4gpu.log
python convnet.py -f /ais/gobi3/u/kriz/tmp/ConvNet__2012-08-03_14.28.23 --layer-params=./layers/layer-params-120-4gpu-auto4.cfg --color-noise=0 --epochs=73 >> logs/layers-120-4gpu.log
python convnet.py -f /ais/gobi3/u/kriz/tmp/ConvNet__2012-08-03_14.28.23 --layer-params=./layers/layer-params-120-4gpu-auto5.cfg --epochs=81 >> logs/layers-120-4gpu.log
python convnet.py -f /ais/gobi3/u/kriz/tmp/ConvNet__2012-08-03_14.28.23 --layer-params=./layers/layer-params-120-4gpu-auto6.cfg --epochs=95 >> logs/layers-120-4gpu.log
