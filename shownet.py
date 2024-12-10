
import numpy
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from options import *
#import pygame as pg
import Image
#from pygame.locals import *
from time import sleep
#from collections import Counter
import gc

#import cv
#this is important for capturing/displaying images
#from opencv import highgui as hg

try:
    import pylab as pl
except:
    print "This script requires the matplotlib python library (Ubuntu/Fedora package name python-matplotlib). Please install it."
    sys.exit(1)

class ShowNetError(Exception):
    pass

class ShowConvNet(ConvNet):
    def __init__(self, op, load_dic):
        ConvNet.__init__(self, op, load_dic)
    
    def get_gpus(self):
        self.need_gpu = self.op.get_value('show_preds') or self.op.get_value('write_features') \
                        or self.op.get_value('show_data_grad') or self.op.get_value('webcam') or self.op.get_value('top5') \
                        or self.op.get_value('show_maps')
        if self.need_gpu:
            ConvNet.get_gpus(self)
            
    def init_data_providers(self):
        class Dummy:
            def advance_batch(self):
                pass
        if self.need_gpu:
            ConvNet.init_data_providers(self)
        else:
            self.train_data_provider = self.test_data_provider = Dummy()
    
    def import_model(self):
        if self.need_gpu:
            ConvNet.import_model(self)
            
    def init_model_state(self):
        #ConvNet.init_model_state(self)
        if self.op.get_value('show_maps'):
            self.map_layer_idx = self.get_layer_idx(self.op.get_value('show_maps'))
        if self.op.get_value('show_preds') or self.op.get_value('webcam'):
            self.softmax_name = self.op.get_value('show_preds') or self.op.get_value('webcam')
        if self.op.get_value('write_features'):
            self.ftr_layer_name = self.op.get_value('write_features')
        if self.op.get_value('top5'):
            self.ftr_layer_idx = self.get_layer_idx(self.op.get_value('top5'))
        if self.op.get_value('show_data_grad'):
            self.data_layer_idx = self.get_layer_idx('data')
            self.softmax_idx = self.get_layer_idx('probs')
            for l in self.model_state['layers']:
                if l['name'] != 'labels':
                    l['actsGradTarget'] = -1
                    l['gradConsumer'] = True
                if l['name'] != 'data':
                    l['conserveMem'] = True
            
    def init_model_lib(self):
        if self.need_gpu:
            ConvNet.init_model_lib(self)

    def plot_cost(self):
        if self.show_cost not in self.train_outputs[0][0]:
            raise ShowNetError("Cost function with name '%s' not defined by given convnet." % self.show_cost)
        train_errors = [o[0][self.show_cost][self.cost_idx] for o in self.train_outputs]
        test_errors = [o[0][self.show_cost][self.cost_idx] for o in self.test_outputs]

        numbatches = len(self.train_batch_range)
        test_errors = numpy.row_stack(test_errors)
        test_errors = numpy.tile(test_errors, (1, self.testing_freq))
        test_errors = list(test_errors.flatten())
        test_errors += [test_errors[-1]] * max(0,len(train_errors) - len(test_errors))
        test_errors = test_errors[:len(train_errors)]

        numepochs = len(train_errors) / float(numbatches)
        pl.figure(1)
        x = range(0, len(train_errors))
        pl.plot(x, train_errors, 'k-', label='Training set')
        pl.plot(x, test_errors, 'r-', label='Test set')
        pl.legend()
        ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
        epoch_label_gran = int(ceil(numepochs / 20.)) # aim for about 20 labels
        epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
        ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))

        pl.xticks(ticklocs, ticklabels)
        pl.xlabel('Epoch')
#        pl.ylabel(self.show_cost)
        pl.title('%s[%d]' % (self.show_cost, self.cost_idx))
        
    def make_filter_fig(self, filters, filter_start, fignum, _title, num_filters, combine_chans, FILTERS_PER_ROW=16):
        MAX_ROWS = 24
        MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS
        num_colors = filters.shape[0]
        f_per_row = int(ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
        filter_end = min(filter_start+MAX_FILTERS, num_filters)
        filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
        filter_pixels = filters.shape[1]
        filter_size = int(sqrt(filters.shape[1]))
        fig = pl.figure(fignum)
        fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
        num_filters = filter_end - filter_start
        if not combine_chans:
            bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size*num_colors * f_per_row + f_per_row + 1), dtype=n.single)
        else:
            bigpic = n.zeros((3, filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)
    
        for m in xrange(filter_start,filter_end ):
            filter = filters[:,:,m]
            y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
            if not combine_chans:
                for c in xrange(num_colors):
                    filter_pic = filter[c,:].reshape((filter_size,filter_size))
                    bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                           1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
            else:
                filter_pic = filter.reshape((3, filter_size,filter_size))
                bigpic[:,
                       1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
        pl.xticks([])
        pl.yticks([])
        if not combine_chans:
            pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
        else:
            bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
            pl.imshow(bigpic, interpolation='nearest')        
        
    def plot_filters(self):
        FILTERS_PER_ROW = 16
        filter_start = 0 # First filter to show
        if self.show_filters not in self.layers:
            raise ShowNetError("Layer with name '%s' not defined by given convnet." % self.show_filters)
        layer = self.layers[self.show_filters]
        filters = layer['weights'][self.input_idx]
#        filters = filters - filters.min()
#        filters = filters / filters.max()
        if layer['type'] == 'fc': # Fully-connected layer
            num_filters = layer['outputs']
            channels = self.channels
            filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        elif layer['type'] in ('conv', 'local'): # Conv layer
            num_filters = layer['filters']
            channels = layer['filterChannels'][self.input_idx]
            if layer['type'] == 'local':
                filters = filters.reshape((layer['modules'], channels, layer['filterPixels'][self.input_idx], num_filters))
                filters = filters[:, :, :, self.local_plane] # first map for now (modules, channels, pixels)
                filters = filters.swapaxes(0,2).swapaxes(0,1)
                num_filters = layer['modules']
#                filters = filters.swapaxes(0,1).reshape(channels * layer['filterPixels'][self.input_idx], num_filters * layer['modules'])
#                num_filters *= layer['modules']
                FILTERS_PER_ROW = layer['modulesX']
            else:
                filters = filters.reshape(channels, filters.shape[0]/channels, filters.shape[1])
        
        
        # Convert YUV filters to RGB
        if self.yuv_to_rgb and channels == 3:
            R = filters[0,:,:] + 1.28033 * filters[2,:,:]
            G = filters[0,:,:] + -0.21482 * filters[1,:,:] + -0.38059 * filters[2,:,:]
            B = filters[0,:,:] + 2.12798 * filters[1,:,:]
            filters[0,:,:], filters[1,:,:], filters[2,:,:] = R, G, B
        combine_chans = not self.no_rgb and channels == 3
        
        # Make sure you don't modify the backing array itself here -- so no -= or /=
        if self.norm_filters:
            #print filters.shape
            filters = filters - n.tile(filters.reshape((filters.shape[0] * filters.shape[1], filters.shape[2])).mean(axis=0).reshape(1, 1, filters.shape[2]), (3, filters.shape[1], 1))
            filters = filters / n.sqrt(n.tile(filters.reshape((filters.shape[0] * filters.shape[1], filters.shape[2])).var(axis=0).reshape(1, 1, filters.shape[2]), (3, filters.shape[1], 1)))
            #filters = filters - n.tile(filters.min(axis=0).min(axis=0), (3, filters.shape[1], 1))
            #filters = filters / n.tile(filters.max(axis=0).max(axis=0), (3, filters.shape[1], 1))
        #else:
        filters = filters - filters.min()
        filters = filters / filters.max()

        self.make_filter_fig(filters, filter_start, 2, 'Layer %s' % self.show_filters, num_filters, combine_chans, FILTERS_PER_ROW=FILTERS_PER_ROW)
    
    def plot_predictions(self):
        data = self.get_next_batch(train=False)[2] # get a test batch
        num_classes = self.test_data_provider.get_num_classes()
        NUM_ROWS = 2
        NUM_COLS = 4
        NUM_IMGS = NUM_ROWS * NUM_COLS
        NUM_TOP_CLASSES = min(num_classes, 5) # show this many top labels
        
        label_names = [lab.split(',')[0] for lab in self.test_data_provider.batch_meta['label_names']]
        if self.only_errors:
            preds = n.zeros((data[0].shape[1], num_classes), dtype=n.single)
        else:
            preds = n.zeros((NUM_IMGS, num_classes), dtype=n.single)
            #rand_idx = nr.permutation(n.r_[n.arange(1), n.where(data[1] == 552)[1], n.where(data[1] == 795)[1], n.where(data[1] == 449)[1], n.where(data[1] == 274)[1]])[:NUM_IMGS]
            rand_idx = nr.randint(0, data[0].shape[1], NUM_IMGS)
            data[0] = n.require(data[0][:,rand_idx], requirements='C')
            data[1] = n.require(data[1][:,rand_idx], requirements='C')
#        data += [preds]
        # Run the model
        self.libmodel.startFeatureWriter(data, [preds], [self.softmax_name])
        self.finish_batch()
        
        fig = pl.figure(3, figsize=(12,9))
        fig.text(.4, .95, '%s test samples' % ('Mistaken' if self.only_errors else 'Random'))
        if self.only_errors:
            # what the net got wrong
            err_idx = [i for i,p in enumerate(preds.argmax(axis=1)) if p not in n.where(data[1][:,i] > 0)[0]]
            err_idx = r.sample(err_idx, min(len(err_idx), NUM_IMGS))
            data[0], data[1], preds = data[0][:,err_idx], data[1][:,err_idx], preds[err_idx,:]
            
        data[0] = self.test_data_provider.get_plottable_data(data[0])
        import matplotlib.gridspec as gridspec
        import matplotlib.colors as colors
        cconv = colors.ColorConverter()
        gs = gridspec.GridSpec(NUM_ROWS*2, NUM_COLS,
                               width_ratios=[1]*NUM_COLS, height_ratios=[2,1]*NUM_ROWS )
        #print data[1]
        for row in xrange(NUM_ROWS):
            for col in xrange(NUM_COLS):
                img_idx = row * NUM_COLS + col
                if data[0].shape[0] <= img_idx:
                    break
                pl.subplot(gs[(row * 2) * NUM_COLS + col])
                #pl.subplot(NUM_ROWS*2, NUM_COLS, row * 2 * NUM_COLS + col + 1)
                pl.xticks([])
                pl.yticks([])
                img = data[0][img_idx,:,:,:]
                pl.imshow(img, interpolation='lanczos')
                show_title = data[1].shape[0] == 1
                true_label = [int(data[1][0,img_idx])] if show_title else n.where(data[1][:,img_idx]==1)[0]
                #print true_label
                #print preds[img_idx,:].shape
                #print preds[img_idx,:].max()
                true_label_names = [label_names[i] for i in true_label]
                img_labels = sorted(zip(preds[img_idx,:], label_names), key=lambda x: x[0])[-NUM_TOP_CLASSES:]
                #print img_labels
                axes = pl.subplot(gs[(row * 2 + 1) * NUM_COLS + col])
                height = 0.5
                ylocs = n.array(range(NUM_TOP_CLASSES))*height
                pl.barh(ylocs, [l[0] for l in img_labels], height=height, \
                        color=['#ffaaaa' if l[1] in true_label_names else '#aaaaff' for l in img_labels])
                #pl.title(", ".join(true_labels))
                if show_title:
                    pl.title(", ".join(true_label_names), fontsize=15, fontweight='bold')
                else:
                    print true_label_names
                pl.yticks(ylocs + height/2, [l[1] for l in img_labels], x=1, backgroundcolor=cconv.to_rgba('0.65', alpha=0.5), weight='bold')
                for line in enumerate(axes.get_yticklines()): 
                    line[1].set_visible(False) 
                #pl.xticks([width], [''])
                #pl.yticks([])
                pl.xticks([])
                pl.ylim(0, ylocs[-1] + height)
                pl.xlim(0, 1)
    
    def do_write_features(self):
        if not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)
        next_data = self.get_next_batch(train=False)
        b1 = next_data[1]
        num_ftrs = self.layers[self.ftr_layer_name]['outputs']
        
#        def showimg(img):
#            pixels = img.shape[0] / 3
#            size = int(sqrt(pixels))
#            img = img.reshape((3,size,size)).swapaxes(0,2).swapaxes(0,1)
#            pl.imshow(img, interpolation='nearest')
#            pl.show()
        while True:
            batch = next_data[1]
            data = next_data[2]
            ftrs = n.zeros((data[0].shape[1], num_ftrs), dtype=n.single)
            self.libmodel.startFeatureWriter(data, [ftrs], [self.ftr_layer_name])
            
            # load the next batch while the current one is computing
            next_data = self.get_next_batch(train=False)
            self.finish_batch()
            path_out = os.path.join(self.feature_path, 'data_batch_%d' % batch)
#            print ftrs
#            ftrs += self.train_data_provider.batch_meta['data_mean'].mean()
#            ftrs /= 255
#            showimg(ftrs[1,:]); sys.exit(0)

            pickle(path_out, {'data': ftrs, 'labels': data[1]})
            print "Wrote feature file %s" % path_out
            if next_data[1] == b1:
                break
        pickle(os.path.join(self.feature_path, 'batches.meta'), {'source_model':self.load_file,
                                                                 'num_vis':num_ftrs})
        
    def do_top5(self):
        num_classes = self.test_data_provider.get_num_classes()
        nv = self.train_data_provider.num_views

        next_data = self.get_next_batch(train=False)
        batch = next_data[1]
        data = next_data[2]
        print data[0].shape
        num_cases = data[0].shape[1] / nv
        print "num cases: %d" % num_cases
        ftrs = [n.zeros((num_cases, num_classes), dtype=n.single) for i in xrange(2)]
        for v in xrange(self.train_data_provider.num_views):
            vdata = [d[:,v*num_cases:(v+1)*num_cases] for d in data] + [ftrs[1]]
            print [d.shape for d in vdata]
            self.libmodel.startFeatureWriter(vdata, self.ftr_layer_idx)
            self.finish_batch()
            ftrs[0] += ftrs[1]
        ftrs = ftrs[0]
        print ftrs.max()
        print "Batch %d top5 error: i dunno" % batch
        print ftrs
        labels = data[1][:,:num_cases].astype(n.int32)
        print labels, labels.shape
        v = 0
        for m in xrange(5):
            maxlocs = ftrs.argmax(axis=1)
            v += (maxlocs == labels).sum()
            ftrs[n.arange(ftrs.shape[0]),maxlocs] = 0
            print v
            
    # NOTE: THIS ROUTINE APPLIES RELU NONLINAERITY TO MAPS
    # Change this if you're not actually using relu units
    def do_showmaps(self):
        NUM_MAPS = 16
        NUM_IMGS = 12
        nr.seed(87213)
        data = self.get_next_batch(train=False)[2]
        rand_idx = nr.randint(0, data[0].shape[1], NUM_IMGS)
        data[0] = n.require(data[0][:,rand_idx], requirements='C')
        data[1] = n.require(data[1][:,rand_idx], requirements='C')
        cases = data[0].shape[1]
        ldic = dict([(l['name'], l) for l in self.layers])
        print ldic.keys()
        num_ftrs = self.layers[self.map_layer_idx]['outputs']
        map_size = self.layers[self.map_layer_idx]['modulesX'] if 'modulesX' in self.layers[self.map_layer_idx] else self.layers[self.map_layer_idx]['outputsX']
        num_maps = num_ftrs / map_size**2
        ftrs = n.zeros((data[0].shape[1], num_ftrs), dtype=n.single)
        
        self.libmodel.startFeatureWriter(data + [ftrs], self.map_layer_idx)
        self.finish_batch()
        
        fig = pl.figure(5)
        fig.text(.4, .95, 'Layer %s feature maps' % self.show_maps)

        data[0] = self.test_data_provider.get_plottable_data(data[0])
        # This map will have size (cases, num_maps, map_size, map_size)
        print ftrs.shape
        ftrs = ftrs.reshape(cases, num_maps, map_size, map_size)
        print ftrs.min(), ftrs.max()
        print ftrs.shape
        ftrs[ftrs<0] = 0
        ftrs -= ftrs.min()
        ftrs /= ftrs.max()
        rand_idx = nr.permutation(range(NUM_MAPS))[:ftrs.shape[1]]
        ftrs = ftrs[:,rand_idx,:,:]
#        ftrs = self.test_data_provider.get_plottable_data(ftrs.T, add_mean=False)

        for i in xrange(NUM_IMGS):
            pl.subplot(NUM_IMGS, NUM_MAPS + 1, i * (NUM_MAPS + 1) + 1)
            
            pl.xticks([])
            pl.yticks([])
            img = data[0][i,:,:,:]
            pl.imshow(img, interpolation='lanczos')
#            return
            for m in xrange(NUM_MAPS):
                pl.subplot(NUM_IMGS, NUM_MAPS + 1, i * (NUM_MAPS + 1) + m + 2)
                pl.xticks([])
                pl.yticks([])
                img = ftrs[i,m, :,:]
                pl.imshow(img, cmap=pl.cm.gray, interpolation='nearest')
        
    def do_show_data_grad(self):
        NUM_ROWS = 2
        NUM_COLS = 4
        NUM_IMGS = NUM_ROWS * NUM_COLS
        
        data = self.get_next_batch(train=False)[2]
        rand_idx = nr.randint(0, data[0].shape[1], NUM_IMGS)
        data[0] = n.require(data[0][:,rand_idx], requirements='C')
        data[1] = n.require(data[1][:,rand_idx], requirements='C')
        
        label_names = [lab.split(',')[0] for lab in self.test_data_provider.batch_meta['label_names']]
        data_dim = self.layers[self.data_layer_idx]['outputs']

        grads = n.zeros((data[0].shape[1], data_dim), dtype=n.single)
        self.libmodel.startDataGrad(data + [grads], self.data_layer_idx, self.softmax_idx)
        self.finish_batch()
        
        fig = pl.figure(4)
        fig.text(.4, .95, 'Data gradients')
        print grads.shape, data[0].shape
        
        grads = self.test_data_provider.get_plottable_data(grads.T, add_mean=False)
#        grads -= grads.min()
#        grads /= grads.max()
#        grads[grads<0] = 0;
#        grads[grads>0] = 0; grads = -grads;
        data[0] = self.test_data_provider.get_plottable_data(data[0])
        for row in xrange(NUM_ROWS):
            for col in xrange(NUM_COLS):
                img_idx = row * NUM_COLS + col
                if data[0].shape[0] <= img_idx:
                    break
                pl.subplot(NUM_ROWS*2, NUM_COLS, row * 2 * NUM_COLS + col + 1)
                pl.xticks([])
                pl.yticks([])
                img = data[0][img_idx,:,:,:]
                pl.imshow(img, interpolation='nearest')
                true_label = int(data[1][0,img_idx])
                #true_labels = set(label_names[l] for l in list(n.where(data[1][:,img_idx] > 0)[0]))
                
                pl.subplot(NUM_ROWS*2, NUM_COLS, (row * 2 + 1) * NUM_COLS + col + 1)

                #pl.title(", ".join(true_labels))
                pl.title(label_names[true_label])
                img = grads[img_idx,:]
                # Suppress small grads
                img -= img.mean()
                s = n.sqrt(img.var())
                img[n.abs(img)<3*s] = 0
                img -= img.min()
                img /= img.max()
                pl.imshow(img, interpolation='nearest')
                
    def do_webcam(self):

        num_classes = self.test_data_provider.get_num_classes()
        label_names = [lab.split(',')[0] for lab in self.test_data_provider.batch_meta['label_names']]
        camera = hg.cvCreateCameraCapture(1)
        #highgui.cvSetCaptureProperty(camera, highgui.CV_CAP_PROP_FRAME_WIDTH, 320 );
        #highgui.cvSetCaptureProperty(camera, highgui.CV_CAP_PROP_FRAME_HEIGHT, 240 );
        
        def get_image():
            im = hg.cvQueryFrame(camera)
            # Add the line below if you need it (Ubuntu 8.04+)
        #    im = cv.cvGetMat(im)
            #convert Ipl image to PIL image
            return cv.adaptors.Ipl2NumPy(im) 
        
#        fps = 30.0
        frames_per_run = 4
        frames = 0
        
        pg.init()
        pg.display.set_mode((224,224))
        pg.display.set_caption("WebCam Demo")
        screen = pg.display.get_surface()
        
        images = n.zeros((self.test_data_provider.get_data_dims(), 32), dtype=n.single)
        labels = n.zeros((1, 32), dtype=n.single) # dummy
        preds = [n.zeros((32, num_classes), dtype=n.single) for i in xrange(2)]
        preds_idx = 0
        while True:
            im = get_image()
            images[:,0:28] = images[:,4:]
            cropped = im[128:352,208:432,:]
            cropped_swapped = cropped.swapaxes(0,2).swapaxes(1,2) 
            images[:,28] = cropped_swapped.reshape((self.test_data_provider.get_data_dims(),))
            images[:,29] = cropped_swapped[:,:,::-1].reshape((self.test_data_provider.get_data_dims(),))
            
            cropped = im[16:464,96:544,:]
            im = cv.adaptors.NumPy2PIL(cropped)
            cropped = cv.adaptors.PIL2NumPy(im.resize((224,224)))
            cropped_swapped = cropped.swapaxes(0,2).swapaxes(1,2) 
            images[:,30] = cropped_swapped.reshape((self.test_data_provider.get_data_dims(),))
            images[:,31] = cropped_swapped[:,:,::-1].reshape((self.test_data_provider.get_data_dims(),))
            
            im = cv.adaptors.NumPy2PIL(cropped)
            pg_img = pg.image.frombuffer(im.tostring(), im.size, im.mode)
            screen.blit(pg_img, (0,0))
            pg.display.flip()
            
            images[:,28:] -= self.test_data_provider.data_mean_crop

            if frames % frames_per_run == 0 and frames >= 32: # Run convnet
                if frames - frames_per_run >= 32: # Wait for last batch to finish, if it hasn't yet
                    self.finish_batch()
                    p = preds[1 - preds_idx].mean(axis=0)
                    m = p.argmax()
#                    m = Counter(preds[1 - preds_idx].argmax(axis=1)).most_common(1)[0][0]
                    print "Label: %s (%.2f)" % (label_names[m] if p[m] > 0.0 else "<<none>>", p[m])
#                    ent = -(n.log(p) * p).sum(axis=0)
#                    print "Label: %s (entropy: %.2f)" % (label_names[m], ent)
#                    print "Label: %s " % (label_names[m])
                    
                # Run the model
                self.libmodel.startFeatureWriter([images, labels, preds[preds_idx]], self.softmax_idx)
                preds_idx = 1 - preds_idx
                

            frames += 1
#            sleep(1.0 / fps)
            

                
    def start(self):
        self.op.print_values()
        if self.show_cost:
            self.plot_cost()
        if self.show_filters:
            self.plot_filters()
        if self.show_preds:
            self.plot_predictions()
        if self.write_features:
            self.do_write_features()
        if self.show_data_grad:
            self.do_show_data_grad()
        if self.webcam:
            self.do_webcam()
        if self.top5:
            self.do_top5()
        if self.show_maps:
            self.do_showmaps()
        pl.show()
        sys.exit(0)
            
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        for option in list(op.options):
            if option not in ('gpu', 'load_file', 'train_batch_range', 'test_batch_range', 'multiview_test', 'data_path', 'logreg_name', 'pca_noise', 'scalar_mean'):
                op.delete_option(option)
        op.add_option("show-cost", "show_cost", StringOptionParser, "Show specified objective function", default="")
        op.add_option("show-filters", "show_filters", StringOptionParser, "Show learned filters in specified layer", default="")
        op.add_option("norm-filters", "norm_filters", BooleanOptionParser, "Individually normalize filters shown with --show-filters", default=0)
        op.add_option("input-idx", "input_idx", IntegerOptionParser, "Input index for layer given to --show-filters", default=0)
        op.add_option("cost-idx", "cost_idx", IntegerOptionParser, "Cost function return value index for --show-cost", default=0)
        op.add_option("no-rgb", "no_rgb", BooleanOptionParser, "Don't combine filter channels into RGB in layer given to --show-filters", default=False)
        op.add_option("yuv-to-rgb", "yuv_to_rgb", BooleanOptionParser, "Convert RGB filters to YUV in layer given to --show-filters", default=False)
        op.add_option("channels", "channels", IntegerOptionParser, "Number of channels in layer given to --show-filters (fully-connected layers only)", default=0)
        op.add_option("show-preds", "show_preds", StringOptionParser, "Show predictions made by given softmax on test set", default="")
        op.add_option("only-errors", "only_errors", BooleanOptionParser, "Show only mistaken predictions (to be used with --show-preds)", default=False, requires=['show_preds'])
        op.add_option("write-features", "write_features", StringOptionParser, "Write test data features from given layer", default="", requires=['feature-path'])
        op.add_option("feature-path", "feature_path", StringOptionParser, "Write test data features to this path (to be used with --write-features)", default="")
        op.add_option("show-data-grad", "show_data_grad", BooleanOptionParser, "Show data gradient in given data layer", default=False)
        op.add_option("webcam", "webcam", StringOptionParser, "Show webcam demo with given softmax layer's predictions", default="")
        op.add_option("local-plane", "local_plane", IntegerOptionParser, "Local plane to show", default=0)
        op.add_option("top5", "top5", StringOptionParser, "Compute top5 test error from given layer", default=False)
        op.add_option("show-maps", "show_maps", StringOptionParser, "Show feature maps in given layer", default="")


        op.options['load_file'].default = None
        return op
    
if __name__ == "__main__":
    #nr.seed(6)
    try:
        op = ShowConvNet.get_options_parser()
        op, load_dic = IGPUModel.parse_options(op)
        model = ShowConvNet(op, load_dic)
        model.start()
    except (UnpickleError, ShowNetError, opt.GetoptError), e:
        print "----------------"
        print "Error:"
        print e 

