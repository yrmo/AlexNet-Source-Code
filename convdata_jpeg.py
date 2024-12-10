
from data import *
import numpy.random as nr
import numpy as n
import random as r
from time import time
from threading import Thread
from math import sqrt
import sys
from pylab import *
from PIL import Image
from StringIO import StringIO
from convdata import ImageNetDP
    
class JPEGBatchLoaderThread(Thread):
    def __init__(self, data_dir, path, data_mean, no_crop, label_offset, tgt, list_out):
        Thread.__init__(self)
        self.data_dir = data_dir
        self.path = path
        self.tgt = tgt
        self.list_out = list_out
        self.label_offset = label_offset
        self.data_mean = data_mean
        self.no_crop = no_crop
        #print "loading %d" % self.bnum
        
    @staticmethod
    def load_jpeg_batch((strings, orig_sizes, labels), data_mean, no_crop, label_offset, tgt):
        lab_arr = n.zeros((len(strings), 1), dtype=n.single)
        failed = 0
        img256 = n.zeros((256, 256, 3), dtype=n.uint8) if no_crop else None
        for k,(s,l) in enumerate(zip(strings, labels)):
            try:
                ima = n.asarray(Image.open(StringIO(s)).convert('RGB'))
                if no_crop:
                    off_y, off_x = (256 - ima.shape[0]) / 2, (256 - ima.shape[1]) / 2
                    img256[:] = data_mean
                    img256[off_y:ima.shape[0]+off_y,off_x:ima.shape[1]+off_x,:] = ima
                    tgt[k - failed,:] = img256.swapaxes(0,2).swapaxes(1,2).flatten()
                else:
                    tgt[k - failed,:] = ima.swapaxes(0,2).swapaxes(1,2).flatten()
                # For the 2012 test set, the labels will be None
                lab_arr[k - failed,0] = 0 if l[1] is None else l[1] + label_offset
            except IOError:
                failed += 1
        return {'data': tgt[:len(strings) - failed,:],
                'labels': lab_arr[:len(strings) - failed,:]}
    
    def run(self):
        p = JPEGBatchLoaderThread.load_jpeg_batch(unpickle(self.path),
                                                  self.data_mean,
                                                  self.no_crop,
                                                  self.label_offset,
                                                  self.tgt)
        self.list_out.append(p)
        
class ColorNoiseMakerThread(Thread):
    def __init__(self, pca_stdevs, pca_vecs, num_noise, list_out):
        Thread.__init__(self)
        self.pca_stdevs, self.pca_vecs = pca_stdevs, pca_vecs
        self.num_noise = num_noise
        self.list_out = list_out
        
    def run(self):
        noise = n.dot(nr.randn(self.num_noise, 3).astype(n.single) * self.pca_stdevs.T, self.pca_vecs.T)
        self.list_out.append(noise)

class JPEGCroppedImageNetDP(ImageNetDP):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        ImageNetDP.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.mini = dp_params['minibatch_size']
        self.border_size = dp_params['crop_border']
        self.inner_size = self.img_size - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.crop_chunk = 32 # This many images will be cropped in the same way
        self.batch_size = self.batch_meta['batch_size']
        self.label_offset = 0 if 'label_offset' not in self.batch_meta else self.batch_meta['label_offset']
        self.no_crop = False if 'no_crop' not in self.batch_meta else self.batch_meta['no_crop']
        self.scalar_mean = 'scalar_mean' in dp_params and dp_params['scalar_mean'] 
        # Maintain poitners to previously-returned data matrices so they don't get garbage collected.
        # I've never seen this happen but it's a safety measure.
        self.data = [None, None] # These are pointers to previously-returned data matrices
        # This is where I crop data into
        self.cropped_data = [n.zeros((0*self.data_mult, self.get_data_dims()), dtype=n.float32) for x in xrange(2)] 
        # This is where I load data into (jpeg --> uint8)
        self.orig_data = [n.zeros((self.batch_size, self.img_size**2*3), dtype=n.uint8) for x in xrange(1 if test else 2)] 
            
        self.loader_thread, self.color_noise_thread = None, None
        self.convnet = dp_params['convnet']
            
        self.num_noise = self.batch_size
        self.batches_generated, self.loaders_started = 0, 0
        self.data_mean_crop = self.data_mean.reshape((3,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((1,3*self.inner_size**2))
        if self.no_crop or self.scalar_mean:
            self.data_mean_crop = self.data_mean.mean()
            
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.inner_size**2 * 3
        return 1

    def start_loader(self, batch_idx):
        self.load_data = []
        #print "loading %d" % self.batch_range_perm[self.batch_idx]
        self.loader_thread = JPEGBatchLoaderThread(self.data_dir,
                                                   self.get_data_file_name(self.batch_range[batch_idx]),
                                                   self.data_mean_crop,
                                                   self.no_crop,
                                                   self.label_offset,
                                                   self.orig_data[self.loaders_started],
                                                   self.load_data)
        self.loader_thread.start()
        self.loaders_started = (self.loaders_started + 1) % 2
        
    def start_color_noise_maker(self):
        color_noise_list = []
        self.color_noise_thread = ColorNoiseMakerThread(self.color_stdevs, self.color_eig, self.num_noise, color_noise_list)
        self.color_noise_thread.start()
        return color_noise_list
         
    def get_labels(self, datadic):
        pass
    
    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        if self.test:
            epoch, batchnum, self.data[self.d_idx] = LabeledDataProvider.get_next_batch(self)
            self.data[self.d_idx] = JPEGBatchLoaderThread.load_jpeg_batch(self.data[self.d_idx],
                                                                          self.data_mean_crop,
                                                                          self.no_crop,
                                                                          self.label_offset,
                                                                          self.orig_data[0])
        else:
            epoch, batchnum = self.curr_epoch, self.curr_batchnum

            if self.loader_thread is None:
                self.start_loader(self.batch_idx)
                self.loader_thread.join()
                self.data[self.d_idx] = self.load_data[0]

                self.start_loader(self.get_next_batch_idx())
            else:
                # Set the argument to join to 0 to re-enable batch reuse
                self.loader_thread.join()
                if not self.loader_thread.is_alive():
                    self.data[self.d_idx] = self.load_data[0]
                    self.start_loader(self.get_next_batch_idx())
                #else:
                #    print "Re-using batch"
            self.advance_batch()
        
        cropped = self.get_cropped_data(self.data[self.d_idx])
        if self.color_noise_coeff > 0 and not self.test:
            # At this point the data already has 0 mean.
            # So I'm going to add noise to it, but I'm also going to scale down
            # the original data. This is so that the overall scale of the training
            # data doesn't become too different from the test data.
            s = cropped.shape
            cropped_size = self.get_data_dims(0) / 3
            ncases = s[0]

            if self.color_noise_thread is None:
                self.color_noise_list = self.start_color_noise_maker()
                self.color_noise_thread.join()
                self.color_noise = self.color_noise_list[0]
                self.color_noise_list = self.start_color_noise_maker()
            else:
                self.color_noise_thread.join(0)
                if not self.color_noise_thread.is_alive():
                    self.color_noise = self.color_noise_list[0]
                    self.color_noise_list = self.start_color_noise_maker()

            cropped = self.cropped_data[self.d_idx] = cropped.reshape((ncases*3, cropped_size))
            self.color_noise = self.color_noise[:ncases,:].reshape((3*ncases, 1))
            cropped += self.color_noise * self.color_noise_coeff
            cropped = self.cropped_data[self.d_idx] = cropped.reshape((ncases, 3* cropped_size))
            cropped /= (1.0 + self.color_noise_coeff)
            
        self.data[self.d_idx]['labels'] = self.get_labels(self.data[self.d_idx])
        self.data[self.d_idx]['data'] = cropped
        self.batches_generated += 1

        if False and not self.test:
            idx = 111
            cropped -= cropped.min()
            cropped /= cropped.max()
            label = int(self.data[self.d_idx]['labels'][idx,0])
            print label
            print self.batch_meta['label_names'][label]
            print cropped.max(), cropped.min()
            print self.data[self.d_idx]['labels']
            self.showimg(cropped[idx,:])
        
        # NOTE: It would be good to add some logic here to pad irregularly-sized
        # batches by duplicating training cases. 

        return epoch, batchnum, [self.data[self.d_idx]['data'].T, self.data[self.d_idx]['labels'].T]
       
    def get_cropped_data(self, data):
        cropped = self.cropped_data[self.d_idx]
        if cropped.shape[0] != data['data'].shape[0] * self.data_mult:
            cropped = self.cropped_data[self.d_idx] = n.zeros((data['data'].shape[0] * self.data_mult, cropped.shape[1]), dtype=n.float32)
        self.__trim_borders(data['data'], cropped)

        return self.subtract_mean(cropped)
        
    def subtract_mean(self,data):
        data -= self.data_mean_crop
        return data
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
        mean = self.data_mean_crop if data.flags.f_contiguous or self.scalar_mean else self.data_mean_crop.T
        return n.require((data + (mean if add_mean else 0)).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(x.shape[0], 3, self.img_size, self.img_size)

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1]]
                    target[i * x.shape[0]:(i+1)* x.shape[0],:] = pic.reshape((x.shape[0], self.get_data_dims()))
                    target[(self.num_views/2 + i) * x.shape[0]:(self.num_views/2 +i+1)* x.shape[0],:] = pic[:,:,:,::-1].reshape((x.shape[0],self.get_data_dims()))
            else:
                pic = y[:,:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size] # just take the center for now
                target[:,:] = pic.reshape((x.shape[0], self.get_data_dims()))
        else:
            for c in xrange(0, x.shape[0], self.crop_chunk): # loop over cases in chunks
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)

                endY, endX = startY + self.inner_size, startX + self.inner_size
                c_end = min(c + self.crop_chunk, x.shape[0])
                pic = y[c:c_end,:,startY:endY,startX:endX]
                if nr.randint(2) == 0: # also flip the images with 50% probability
                    pic = pic[:,:,:,::-1]
                target[c:c_end,:] = pic.reshape((c_end-c, self.get_data_dims()))
                
                # With 5% chance, replace this chunk with the average of this chunk and some future chunk
                #if c >= self.crop_chunk and nr.rand() < 0.05:
                    #r = nr.randint(0, c - self.crop_chunk + 1)
                    #r_end = r + self.crop_chunk
                    #target[c:c_end,:] = 0.75 * target[c:c_end,:] + 0.25 * target[r:r_end,:]
                    #print "faded in past batch (%d,%d) to batch (%d,%d)" % (r, r_end, c, c_end)
            #for c in xrange(0, x.shape[0]-self.crop_chunk, self.crop_chunk): # loop over cases in chunks
            #    if nr.rand() < 0.05:
            #        c_end = min(c + self.crop_chunk, x.shape[0])
            #        r = nr.randint(c, x.shape[0] - self.crop_chunk+1)
            #        r_end = r + self.crop_chunk
            #        target[c:c_end,:] = 0.75 * target[c:c_end,:] + 0.25 * target[r:r_end,:]
                    #print "faded in past batch (%d,%d) to batch (%d,%d)" % (r, r_end, c, c_end)
                    
            #target[:] = n.require(target[:,nr.permutation(x.shape[1])], requirements='C')
    
class JPEGCroppedImageNetLogRegDP(JPEGCroppedImageNetDP):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        JPEGCroppedImageNetDP.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
    def get_labels(self, data):
        return n.require(n.tile(n.array(data['labels'], dtype=n.single).reshape((data['data'].shape[0], 1)), (self.data_mult, 1)), requirements='C')
        
