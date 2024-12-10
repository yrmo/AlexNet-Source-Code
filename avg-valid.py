from util import *
import os
import sys
import re
import random as r
import numpy.random as nr
from math import sqrt

#VALID_PATH = '/ais/gobi3/u/kriz/lsvrc-2012'
VALID_PATH = '/storage/lsvrc-2012'

def compute_top5(preds, labels):
    errs = 0
    for c in xrange(preds.shape[0]):
        err = True
        for i in xrange(5):
            top = preds[c,:].argmax()
            if top == labels[c]:
                err = False
                break
            preds[c, top] = -1
        errs += err
    return errs
    #top5 = [[k[0] for k in sorted(zip(xrange(preds.shape[1]), preds[c,:]), key=lambda x:x[1], reverse=True)[:5]] for c in xrange(preds.shape[0])]
    #return sum(l not in t for l,t in zip(labels, top5))

def do_avg(paths, coeffs, top5=False):
    #coeffs = [float(x) for x in sys.argv[2].split(',')]
    off = unpickle(os.path.join(VALID_PATH, 'batches.meta'))['label_offset']
    errs1, errs5, cases = 0, 0, 0
    for i,f in enumerate(sorted(os.listdir(paths[0]))):
        b = int(re.match('test_preds_(\d+)', f).group(1))
        dics = [unpickle(os.path.join(p, f)) for p in paths]
        dicv = unpickle(os.path.join(VALID_PATH, 'data_batch_%d' % b))
        labels = n.array([d[1]+off for d in dicv[2]])
        assert labels.min >= 0 and labels.max() < 1000 
        preds = sum(c * d['data'] for c,d in zip(coeffs, dics))
        assert preds.shape[1] == 1000
        err1 = sum(preds.argmax(1) != labels)
        err5 = compute_top5(preds, labels) if top5 else 0
        errs1 += err1
        errs5 += err5
        cases += preds.shape[0]

        #print "%.4f %.4f" % (float(err1) / preds.shape[0], float(err5) / preds.shape[0])
    return errs1 / float(cases), errs5 / float(cases)
    #print "Average error rate with coeffs %s: %.4f %.4f" % (", ".join("%.2f" % f for f in coeffs), errs1 / float(cases), errs5 / float(cases))

def find_coeffs(paths, passes=5, cmin=0.0, cmax=1.0, step=0.05):
    coeffs = [(cmax-cmin)/2 for i in xrange(len(paths))]
    #coeffs = [cmin + (r.random() * (cmax-cmin)) for i in xrange(len(paths))]
    best1 = do_avg(paths, coeffs, top5=True)[1]
    changed = -1
    for p in xrange(passes):
        print "Pass %d" % p
        for i in xrange(len(coeffs)):
            if changed == i:
                changed = -2
                break
            for c in [cmin + c * step for c in xrange(1+int((cmax-cmin)/step))]:
                oldc = coeffs[i]
                coeffs[i] = c
                err = do_avg(paths, coeffs, top5=True)[1]
                if err < best1:
                    best1 = err
                    changed = i
                else:
                    coeffs[i] = oldc
            print "Best error rate: %.4f, coeffs: [%s]" % (best1, ",".join("%.2f" % f for f in coeffs))
        if changed == -2:
            break
            
def find_coeffs2(paths, passes=50):
    #coeffs = n.array([r.random() for i in xrange(len(paths))])
    coeffs = n.array([0.5 for i in xrange(len(paths))])
    coeffs /= coeffs.sum()
    
    
    #crange = [[cmin + c * step for c in xrange(1+int((cmax-cmin)/step))] for i in xrange(len(paths))]
    for p in xrange(passes):
        print "Pass %d" % p
        for i in nr.permutation(range(coeffs.shape[0])):
            #bigger = r.randint(0,2) == 0
            #c = coeffs[i] + r.random() * (1 - coeffs[i]) if bigger else r.random() * coeffs[i]
            c = min(1, max(0, coeffs[i] + nr.randn() / (2*sqrt(1+p))))
            oldc = coeffs[i]
            coeffs[i] = c
            err = do_avg(paths, coeffs, top5=True)[1]
            changed = ""
            if err < best1:
                best1 = err
                changed = "*"
                #crange = [[cmin + x * step for x in xrange(1+int((cmax-cmin)/step))] for i in xrange(len(paths))]
            else:
                coeffs[i] = oldc
            coeffs /= coeffs.sum()
            #crange[i].remove(c)
            print "Best error rate: %.4f, coeffs: [%s]%s" % (best1, ",".join("%.4f" % f for f in coeffs), changed)


if __name__ == "__main__":
    paths = sys.argv[1].split(',')
    if len(sys.argv) == 2:
        find_coeffs(paths)
    else:
        coeffs = n.array([float(x) for x in sys.argv[2].split(',')])
        errs = do_avg(paths, coeffs, top5=True)
        print "Average error rate with coeffs %s: %.4f %.4f" % (", ".join("%.2f" % f for f in coeffs), errs[0], errs[1])
