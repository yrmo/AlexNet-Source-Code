import sys
from util import *
import pylab as pl
import numpy as n
import numpy.random as nr
from PIL import Image
from StringIO import StringIO

def print_top5(preds, lnames):
    print preds
    for i in xrange(len(preds)):
        print "Label %d: %s" %(i, lnames[preds[i]])

if __name__ == "__main__":
    pred_path = sys.argv[1]
    data_path = sys.argv[2]
    batch = nr.randint(98) + 3000
    data = unpickle(os.path.join(data_path, 'data_batch_%d' % batch))[0]
    preds = [n.array([int(x) - 1 for x in l.split(' ')]) for l in open(pred_path).readlines()]

    img_idx = nr.randint(len(data))
    meta = unpickle(os.path.join(data_path, 'batches.meta'))
    lnames = meta['label_names']
    print "Batch: %d, img idx: %d" % (batch, img_idx)

    img = n.asarray(Image.open(StringIO(data[img_idx])).convert('RGB'))

    print_top5(preds[(batch - 3000) * 1024 + img_idx], lnames)
    
    pl.imshow(img)
    pl.show()
