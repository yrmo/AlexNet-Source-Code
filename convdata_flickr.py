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
    
class JPEGBatchLoaderThread(Thread):
    def __init__(self, data_dir, path, freq_to_id, tgt, tgt_labels, list_out):
        Thread.__init__(self)
        self.data_dir = data_dir
        self.path = path
        self.tgt = tgt
        self.tgt_labels = tgt_labels
        self.list_out = list_out
        self.freq_to_id = freq_to_id
        #print "loading %d" % self.bnum

    @staticmethod
    def raw_to_freq_id(raw_tags, freq_to_id):
        raw_tags = [''.join(t.lower().strip().split()) for t in raw_tags]
        return [freq_to_id[t] for t in raw_tags if t in freq_to_id]

    @staticmethod
    def load_jpeg_batch((strings, sizes, labels), freq_to_id, tgt, tgt_labels):
        tgt_labels[:] = 0
        for k,s in enumerate(strings):
            ima = n.asarray(Image.open(StringIO(s)).convert('RGB'))
            tgt[k,:] = ima.swapaxes(0,2).swapaxes(1,2).flatten()
            tgt_labels[k, JPEGBatchLoaderThread.raw_to_freq_id(labels[k], freq_to_id)] = 1

        return {'data': tgt[:len(strings),:],
                'labels': tgt_labels[:len(strings),:]}
    
    def run(self):
        p = self.load_jpeg_batch(unpickle(self.path),
                                 self.freq_to_id,
                                 self.tgt,
                                 self.tgt_labels)
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

class FlickrDP(LabeledDataProvider):
    MAX_PCA_COMPONENTS = 1024 # Use this many components for noise generation
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.init_commons(data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

    def init_commons(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        self.data_mean = self.batch_meta['data_mean'].astype(n.single)
        self.color_eig = self.batch_meta['color_pca'][1].astype(n.single)
        self.color_stdevs = n.c_[self.batch_meta['color_pca'][0].astype(n.single)]
        self.color_noise_coeff = dp_params['color_noise']
        self.pca_noise_coeff = dp_params['pca_noise']
        self.num_colors = 3
        self.img_size = int(sqrt(self.batch_meta['num_vis'] / self.num_colors))
        self.freq_to_id = self.batch_meta['freq_to_id']
        
    def get_labels(self, datadic):
        pass
    
    def showimg(self, img):
        pixels = img.shape[0] / 3
        size = int(sqrt(pixels))
        img = img.reshape((3,size,size)).swapaxes(0,2).swapaxes(0,1)
        imshow(img, interpolation='nearest')
        show()
        
    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledDataProvider.get_next_batch(self)
        # This takes about 1 sec per batch :(
        # If I don't convert both to single ahead of time, it takes even longer.
        data = n.require(datadic['data'] - self.data_mean, dtype=n.single, requirements='C')
        
        labels = self.get_labels(datadic)
        
        # Labels have to be in the range 0-(number of classes - 1)
        assert labels.max() < self.get_num_classes(), "Invalid labels!"
        assert labels.min() >= 0, "Invalid labels!"
        return epoch, batchnum, [data, labels]
    
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
        return n.require((data + (self.data_mean if add_mean else 0)).reshape(data.shape[0], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

class JPEGCroppedFlickrDP(FlickrDP):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.init_commons(data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
        self.img_size = int(sqrt(self.batch_meta['num_vis'] / self.num_colors))
        self.border_size = dp_params['crop_border']
        self.inner_size = self.img_size - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.crop_chunk = 32 # This many images will be cropped in the same way
        self.batch_size = self.batch_meta['batch_size']
        
        # Maintain poitners to previously-returned data matrices so they don't get garbage collected.
        # I've never seen this happen but it's a safety measure.
        self.data = [None, None]
        self.cropped_data = [n.zeros((0*self.data_mult, self.get_data_dims()), dtype=n.float32) for x in xrange(2)]
        if self.test:
            self.orig_data = [n.zeros((self.batch_size, self.img_size**2*3), dtype=n.uint8) for x in xrange(1)]
            self.orig_labels = [n.zeros((self.batch_size, self.get_num_classes()), dtype=n.float32) for x in xrange(2)]
        else:
            self.orig_data = [n.zeros((self.batch_size, self.img_size**2*3), dtype=n.uint8) for x in xrange(2)]
            # There have to be 3 copies of labels because this matrix actually gets used by the training code
            self.orig_labels = [n.zeros((self.batch_size, self.get_num_classes()), dtype=n.float32) for x in xrange(3)]
            
        self.loader_thread, self.color_noise_thread = None, None
        self.convnet = dp_params['convnet']
            
        self.num_noise = self.batch_size
        self.batches_generated, self.loaders_started = 0, 0
        self.data_mean_crop = self.data_mean.reshape((3,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((1,3*self.inner_size**2))

    def get_data_dims(self, idx=0):
        assert idx in (0,1), "Invalid index: %d" % idx
        if idx == 0:
            return self.inner_size**2 * 3
        return self.get_num_classes()

    def start_loader(self, batch_idx):
        self.load_data = []
        #print "loading %d" % self.batch_range_perm[self.batch_idx]
        self.loader_thread = JPEGBatchLoaderThread(self.data_dir, self.get_data_file_name(self.batch_range[batch_idx]), self.freq_to_id,
                                                   self.orig_data[self.loaders_started % 2], self.orig_labels[self.loaders_started % 3],
                                                   self.load_data)
        self.loader_thread.start()
        self.loaders_started += 1
        
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
            self.data[self.d_idx] = JPEGBatchLoaderThread.load_jpeg_batch(self.data[self.d_idx], self.freq_to_id, self.orig_data[0], self.orig_labels[self.d_idx])
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
#                else:
#                    print "Re-using batch"
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

#        idx = 1000
#        cropped -= cropped.min()
#        cropped /= cropped.max()
#        
#        print [self.batch_meta['label_names'][i] for i in n.where(self.data['labels'][idx,:]==1)[0]]
#        self.showimg(cropped[idx,:])
        #print cropped.shape
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
        return n.require((data.T + (self.data_mean_crop if add_mean else 0)).reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
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
            #target[:] = n.require(target[:,nr.permutation(x.shape[1])], requirements='C')
    
class JPEGCroppedFlickrCEDP(JPEGCroppedFlickrDP):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        JPEGCroppedFlickrDP.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
    def get_labels(self, data):
        return n.require(n.tile(data['labels'], (self.data_mult, 1)), requirements='C')
        
class DummyConvNetCEDP(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim, num_classes=16, num_cases=16)
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        dic['data'] = n.require(dic['data'].T, requirements='F')
        dic['labels'] = n.zeros((self.get_data_dims(idx=1), dic['data'].shape[1]), dtype=n.float32, order='F')
        for c in xrange(dic['labels'].shape[1]): # loop over cases
            r = nr.randint(0, dic['labels'].shape[0])
            dic['labels'][r,c] = 1
        
        return epoch, batchnum, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 16
