from util import *
import os
import sys
import re
import random as r
import os

def do_avg(paths, tgtpath, coeffs):
    for i,f in enumerate(sorted(os.listdir(paths[0]))):
        b = int(re.match('test_preds_(\d+)', f).group(1))
        dics = [unpickle(os.path.join(p, f)) for p in paths]
        preds = sum(c * d['data'] for c,d in zip(coeffs, dics))
        pickle(os.path.join(tgtpath, 'test_preds_%d' % b), {'data': preds})
        print "Wrote batch %d" % b

if __name__ == "__main__":
    paths = sys.argv[1].split(',')
    tgtpath = sys.argv[2]
    if not os.path.exists(tgtpath):
        os.makedirs(tgtpath)
    coeffs = [float(x) for x in sys.argv[3].split(',')]
    do_avg(paths, tgtpath, coeffs)

