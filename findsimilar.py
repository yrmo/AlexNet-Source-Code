import os
import sys
from getopt import getopt
import numpy as n
import numpy.random as nr
from time import time
from util import *
import pylab as pl
import gc

imnet_dir = '/storage2/imnet-contest'
ftr_dir = '/storage2/imnet-features-4096'

TEST_IMGS = 128
TOP_IMGS = 16
TEST_BATCH = 'data_batch_3000'

IMG_SIZE = 256
IMGS_PER_FIGURE = 16

def draw_fig(test_imgs, tops):
    for f in xrange(TEST_IMGS/IMGS_PER_FIGURE):
        
        pl.figure(f+1, figsize=(15,15))
        pl.clf()
        bigpic = n.zeros((3, (IMG_SIZE+1)*IMGS_PER_FIGURE - 1, (IMG_SIZE+1)*(1+TOP_IMGS) + 3), dtype=n.single)
        for i in xrange(IMGS_PER_FIGURE):
            img_idx = f * IMGS_PER_FIGURE + i
            bigpic[:, (IMG_SIZE+1) * i:(IMG_SIZE+1)*i+IMG_SIZE,:IMG_SIZE] = test_imgs[:,img_idx].reshape(3, IMG_SIZE, IMG_SIZE)
            for j in xrange(TOP_IMGS):
                if tops[img_idx][j]['img'] is not None:
                    bigpic[:, (IMG_SIZE+1) * i:(IMG_SIZE+1)*i+IMG_SIZE,IMG_SIZE + 4 + j*(IMG_SIZE+1):IMG_SIZE + 4 + j*(IMG_SIZE+1)+IMG_SIZE] = tops[img_idx][j]['img'].reshape(3, IMG_SIZE, IMG_SIZE)
        bigpic /= 255
        pl.imshow(bigpic.swapaxes(0,1).swapaxes(1,2), interpolation='lanczos')

if __name__ == "__main__":
    (options, args) = getopt(sys.argv[1:], "")
    options = dict(options)
    
    # Take 128 images from test batch
    dic = unpickle(os.path.join(ftr_dir, TEST_BATCH))
    p = nr.permutation(dic['data'].shape[0])[:TEST_IMGS]
    data = dic['data'][p,:]
    labels = dic['labels'][:,p]
    dicimgs = unpickle(os.path.join(imnet_dir, TEST_BATCH))
    test_imgs = dicimgs['data'][:,p]
    
    tops = [[{'dist': n.inf, 'batch': 0, 'idx': 0, 'img': None} for i in xrange(TOP_IMGS)] for j in xrange(TEST_IMGS)]
    
    pl.ion()
    for b in xrange(1, 1335):
        dic = unpickle(os.path.join(ftr_dir, 'data_batch_%d' % b))
        dicimgs = unpickle(os.path.join(imnet_dir, 'data_batch_%d' % b))
        t = time()
        dists = [n.sum((data[i,:] - dic['data'])**2, axis=1) for i in xrange(TEST_IMGS)]
        minidx = [d.argmin() for d in dists]
        print dists[0].shape
        for i, dist, midx, top in zip(xrange(TEST_IMGS), dists, minidx, tops):
            k = TOP_IMGS
            while k > 0 and dist[midx] < top[k - 1]['dist']:
                k -= 1
            if k < TOP_IMGS:
                top.insert(k, {'dist': dist[midx], 'batch': b, 'idx': midx, 'img': dicimgs['data'][:,midx].copy()})
                top.pop()
            #print top
        del dic
        del dicimgs
        del dists
        del minidx
        gc.collect()
        #print tops
        print "Finished training batch %d (%f sec)" % (b, time() - t)
        if b % 50 == 0:
            draw_fig(test_imgs, tops)
            pl.draw()
    pl.ioff()
    draw_fig(test_imgs, tops)
    pl.show()
