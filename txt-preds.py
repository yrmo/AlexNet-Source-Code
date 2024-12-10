from util import *
import os
import sys

if __name__ == "__main__":
    path = sys.argv[1]
    for f in sorted(os.listdir(path)):
        dic = unpickle(os.path.join(path, f))
        preds = dic['data']
        assert preds.shape[1] == 1000
        for c in xrange(preds.shape[0]): # loop over cases
            # Notice the +1 here to convert from 0-based indices to 1-based
            top5 = [x[0] + 1 for x in reversed(sorted(list(enumerate(preds[c,:])), key=lambda x:x[1])[-5:])]
            assert min(top5) >= 1 and max(top5) <= 1000
            print " ".join(str(x) for x in top5)
