from data import *
import numpy.random as nr
import numpy as n
import random as r
from time import time
from threading import Thread
from math import sqrt
import sys
from pylab import *

class FlatMemoryDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean'].reshape((self.batch_meta['data_mean'].shape[0], 1))
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = d['labels'].astype(n.int)
            d['labelprobs'] = n.zeros((self.get_num_classes(), d['data'].shape[1]), dtype=n.single)
            for c in xrange(d['data'].shape[1]):
                d['labelprobs'][d['labels'][c],c] = 1.0
            
    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labelprobs']]
    
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else self.get_num_classes()

class ImageNetDP(LabeledDataProvider):
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
#        wordvecs = datadic['wordvecs']
        wordpres = datadic['wordpres']
        
        # Labels have to be in the range 0-(number of classes - 1)
        assert labels.max() < self.get_num_classes(), "Invalid labels!"
        assert labels.min() == 0, "Invalid labels!"
        return epoch, batchnum, [data, labels, wordpres]
    
        
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data, add_mean=True):
        return n.require((data + (self.data_mean if add_mean else 0)).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
class ImageNetLogRegDP(ImageNetDP):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        ImageNetDP.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
    def get_labels(self, datadic):
        return n.array(datadic['labels'], dtype=n.single).reshape((1, datadic['data'].shape[1]))
    
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.img_size**2 * self.num_colors
        if idx == 2:
            return 100
        return 1
    
class BatchLoaderThread(Thread):
    def __init__(self, data_dir, path, list_out):
        Thread.__init__(self)
        self.data_dir = data_dir
        self.path = path
        self.list_out = list_out
        #print "loading %d" % self.bnum
        
    def run(self):
        self.list_out.append(unpickle(self.path))
    
class ColorNoiseMakerThread(Thread):
    def __init__(self, pca_stdevs, pca_vecs, num_noise, list_out):
        Thread.__init__(self)
        self.pca_stdevs, self.pca_vecs = pca_stdevs, pca_vecs
        self.num_noise = num_noise
        self.list_out = list_out
        
    def run(self):
        noise = n.dot(self.pca_vecs, nr.randn(3, self.num_noise).astype(n.single) * self.pca_stdevs)
        self.list_out.append(noise)

class CroppedImageNetDP(ImageNetDP):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        ImageNetDP.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
        self.border_size = dp_params['crop_border']
        self.inner_size = self.img_size - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.crop_chunk = 32 # This many images will be cropped in the same way
        
        # Maintain poitners to previously-returned data matrices so they don't get garbage collected.
        # I've never seen this happen but it's a safety measure.
        self.data = [None, None]
        self.cropped_data = [n.zeros((self.get_data_dims(), 0*self.data_mult), dtype=n.single) for x in xrange(2)]
        
        self.loader_thread, self.color_noise_thread = None, None
        self.convnet = dp_params['convnet']

        self.num_noise = 1024
        self.batches_generated = 0
        self.data_mean_crop = self.data_mean.reshape((3,self.img_size,self.img_size))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((3*self.inner_size**2, 1))
        
    def get_data_dims(self, idx=0):
        if idx == 0:
            return self.inner_size**2 * 3
        return 1

    def start_color_noise_maker(self):
        color_noise_list = []
        self.color_noise_thread = ColorNoiseMakerThread(self.color_stdevs, self.color_eig, self.num_noise, color_noise_list)
        self.color_noise_thread.start()
        return color_noise_list
         
    def get_labels(self, datadic):
        pass
    
    def start_loader(self, batch_idx):
        self.load_data = []
        self.loader_thread = BatchLoaderThread(self.data_dir, self.get_data_file_name(self.batch_range[batch_idx]), self.load_data)
        self.loader_thread.start()

    def get_next_batch(self):
        self.d_idx = self.batches_generated % 2
        if self.test:
            epoch, batchnum, self.data[self.d_idx] = LabeledDataProvider.get_next_batch(self)
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
            self.advance_batch()

        cropped = self.get_cropped_data(self.data[self.d_idx])
        if self.color_noise_coeff > 0 and not self.test:
            # At this point the data already has 0 mean.
            # So I'm going to add noise to it, but I'm also going to scale down
            # the original data. This is so that the overall scale of the training
            # data doesn't become too different from the test data.
            s = cropped.shape
            cropped_size = self.get_data_dims(0) / 3
            ncases = s[1]

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
#                    print "Generated new noise"
#                else:
#                    print "Reusing old noise"
                # If the noise thread IS alive, then we'll just re-use the noise from the last run
            cropped = self.cropped_data[self.d_idx] = cropped.reshape((3, cropped_size, ncases)).swapaxes(0,1).reshape((cropped_size, ncases*3))
            self.color_noise = self.color_noise[:,:ncases].reshape((1, 3*ncases))
            cropped += self.color_noise * self.color_noise_coeff
            cropped = self.cropped_data[self.d_idx] = cropped.reshape((cropped_size, 3, ncases)).swapaxes(0,1).reshape(s)
            cropped /= 1.0 + self.color_noise_coeff
            
#        cropped -= cropped.min()
#        cropped /= cropped.max()
#        self.showimg(cropped[:,0])
        
        self.data[self.d_idx]['labels'] = self.get_labels(self.data[self.d_idx])
        self.data[self.d_idx]['data'] = cropped
        self.batches_generated += 1
        return epoch, batchnum, [self.data[self.d_idx]['data'], self.data[self.d_idx]['labels']]
       
    def get_cropped_data(self, data):
        cropped = self.cropped_data[self.d_idx]
        if cropped.shape[1] != data['data'].shape[1] * self.data_mult:
            cropped = self.cropped_data[self.d_idx] = n.zeros((cropped.shape[0], data['data'].shape[1] * self.data_mult), dtype=n.single)
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
        return n.require((data + (self.data_mean_crop if add_mean else 0)).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(3, self.img_size, self.img_size, x.shape[1])
        
        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(0, x.shape[1], self.crop_chunk): # loop over cases in chunks
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)

                endY, endX = startY + self.inner_size, startX + self.inner_size
                c_end = min(c + self.crop_chunk, x.shape[1])
                pic = y[:,startY:endY,startX:endX, c:c_end]
                if nr.randint(2) == 0: # also flip the images with 50% probability
                    pic = pic[:,:,::-1,:]
                target[:,c:c_end] = pic.reshape((self.get_data_dims(),c_end-c))
            #target[:] = n.require(target[:,nr.permutation(x.shape[1])], requirements='C')

class CroppedImageNetLogRegDP(CroppedImageNetDP):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedImageNetDP.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        
    def get_labels(self, datadic):
        return n.require(n.tile(n.array(datadic['labels'], dtype=n.single).reshape((1, datadic['data'].shape[1])), (1, self.data_mult)), requirements='C')
        
class RandomScaleImageNetLogRegDP(CroppedImageNetLogRegDP):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CroppedImageNetLogRegDP.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        del self.cropped_data
        self.data_mean_mean = self.data_mean.mean()
        
    def get_cropped_data(self):
        if self.test and self.multiview:
            x = self.data['data']
            y = x.reshape(3, self.img_size, self.img_size, x.shape[1])
            target = n.zeros((self.inner_size**2*3, self.data['data'].shape[1]*self.num_views), dtype=n.uint8)
            start_positions = [(0,0), (0, self.border_size), (0, self.border_size*2),
                               (self.border_size, 0), (self.border_size, self.border_size), (self.border_size, self.border_size*2),
                              (self.border_size*2, 0), (self.border_size*2, self.border_size), (self.border_size*2, self.border_size*2)]
            end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
            for i in xrange(self.num_views):
                target[:,i * x.shape[1]:(i+1)* x.shape[1]] = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:].reshape((self.inner_size**2*3,x.shape[1]))
            return self.subtract_mean(target)
        elif not self.test:
            # it should be ok to flip it into the same matrix
            # since if it ends up being reused, flips are invertible.
            self.reflect_data(self.data['data'], self.data['data'])
        return self.subtract_mean(self.data['data'])
    
    def reflect_data(self, x, target):
        y = x.reshape(3, self.img_size, self.img_size, x.shape[1])
        for c in xrange(0, x.shape[1], self.crop_chunk): # loop over cases in chunks
            c_end = min(c + self.crop_chunk, x.shape[1])
            pic = y[:,:,:, c:c_end]
            if nr.randint(2) == 0: # flip the images with 50% probability
                pic = pic[:,:,::-1,:]

            target[:,c:c_end] = pic.reshape((self.get_data_dims(),c_end-c))
    
    # Note that this variant subtracts the same scalar from each pixel
    def subtract_mean(self, data):
        return n.require(data - self.data_mean_mean, dtype=n.single, requirements='C') 
    
    def get_data_dims(self, idx=0):
        return self.img_size**2 * 3 if idx == 0 else 1
        
class DummyConvNetLogRegDP(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)
        self.batch_meta['tree'] = dict([(i, []) for i in xrange(self.num_classes)])
        self.batch_meta['tree'][10] = [0, 1, 2]
        self.batch_meta['tree'][11] = [3, 4, 5]
        self.batch_meta['tree'][12] = [6, 7]
        self.batch_meta['tree'][13] = [8, 9]
        self.batch_meta['tree'][14] = [10, 11]
        self.batch_meta['tree'][15] = [12, 13]
        self.batch_meta['tree'][16] = [14, 15]
        self.batch_meta['all_wnids'] = {'gproot': 16}
        self.img_size = int(sqrt(data_dim/3))
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        dic['data'] = n.require(dic['data'].T, requirements='C')
        dic['labels'] = n.require(dic['labels'].T, requirements='C')
        dic['gates'] = nr.rand(1, dic['data'].shape[1]).astype(n.single)
        
        return epoch, batchnum, [dic['data'], dic['labels'], dic['gates']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1
