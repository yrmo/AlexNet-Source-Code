from gpumodel import *
import numpy as n
import numpy.random as nr

def get_src():
    src = IGPUModel.load_checkpoint('/nobackup/kriz/tmp/ConvNet__2012-09-19_23.29.04')
    return src['model_state']['layers']
    
def makew(name, idx, shapes, params):
    src, src_layer = get_src(), params[0]
    if name == 'localcombine' and idx == 2:
        return n.array(0.01 * nr.randn(shapes[0], shapes[1]), dtype=n.single, order='C')
    return src[src_layer]['weights'][idx]
    
def makeb(name, shapes, params):
    src, src_layer = get_src(), params[0]
    return src[src_layer]['biases']
    
def makec(name, idx, shapes, params):
    src, src_layer = get_src(), params[0]
    return src[src_layer]['filterConns'][idx]
