import os
import sys
from PIL import Image
from StringIO import StringIO
from util import *

src = '/ais/gobi3/u/ilya/flickr_85/'
dst = '/ais/gobi3/u/kriz/flickr-85-1024/'

BATCH_SIZE = 2048

def save_batch(c_strings, c_sizes, c_labels, out_b):
    pickle(os.path.join(dst, 'data_batch_%d' % out_b), (c_strings, c_sizes, c_labels))

    return out_b + 1
if __name__ == "__main__":
    c_strings = []
    c_sizes = []
    c_labels = []
    out_b = 1
    for b in xrange(977):
        failed = 0
        strings, sizes, labels = unpickle(os.path.join(src, '%s' % b))
        for s,z,l in zip(strings, sizes, labels):
            try:
                im = Image.open(StringIO(s)).convert('RGB')
                c_strings += [s]
                c_sizes += [z]
                c_labels += [l]
                
                if len(c_strings) == BATCH_SIZE:
                    out_b = save_batch(c_strings, c_sizes, c_labels, out_b)
                    c_strings = []
                    c_sizes = []
                    c_labels = []
            except IOError,e:
                failed += 1
        print "Batch %d failed: %d" % (b, failed)
            
    if len(c_strings) > 0:
        save_batch(c_strings, c_sizes, c_labels, out_b)
