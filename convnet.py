import numpy as n
import numpy.random as nr
import random as r
from util import *
from data import *
from options import *
from gpumodel import *
import sys
import math as m
import layer as lay
from convdata import *
from convdata_jpeg import JPEGCroppedImageNetLogRegDP
from convdata_flickr import JPEGCroppedFlickrCEDP, DummyConvNetCEDP
from convdata_cifar import CIFARDataProvider, CroppedCIFARDataProvider
from os import linesep as NL
import pylab as pl
import copy as cp

class ConvNet(IGPUModel):
    def __init__(self, op, load_dic, dp_params={}):
        filename_options = []
        for v in ('color_noise', 'pca_noise', 'multiview_test', 'crop_border', 'scalar_mean', 'minibatch_size'):
            dp_params[v] = op.get_value(v)

        IGPUModel.__init__(self, "ConvNet", op, load_dic, filename_options, dp_params=dp_params)
        self.writing_test = False
        
    def import_model(self):
        lib_name = "_ConvNet_k20x" if is_kepler_machine() else "_ConvNet"
        print "========================="
        print "Importing %s C++ module" % lib_name
        self.libmodel = __import__(lib_name) 
        
    def init_model_lib(self):
        self.libmodel.initModel(self.layers, self.device_ids, self.device_cpus, self.minibatch_size, self.wupdate_freq)
        
    def init_model_state(self):
        ms = self.model_state
        if self.load_file:
            ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self, ms['layers'])
        else:    
            ms['layers'] = lay.LayerParser.parse_layers(self.layer_def, self.layer_params, self)
        
        # Convert convolutional layers to local
        if len(self.op.get_value('conv_to_local')) > 0:
            for i, layer in enumerate(ms['layers']):
                if layer['type'] == 'conv' and layer['name'] in self.op.get_value('conv_to_local'):
                    lay.LocalLayerParser.conv_to_local(ms['layers'], i)
        # Decouple weight matrices
        if len(self.op.get_value('unshare_weights')) > 0:
            for name_str in self.op.get_value('unshare_weights'):
                if name_str:
                    name = lay.WeightLayerParser.get_layer_name(name_str)
                    if name is not None:
                        name, idx = name[0], name[1]
                        if name not in ms['layers']:
                            raise ModelStateException("Layer '%s' does not exist; unable to unshare" % name)
                        layer = ms['layers'][name]
                        lay.WeightLayerParser.unshare_weights(layer, ms['layers'], matrix_idx=idx)
                    else:
                        raise ModelStateException("Invalid layer name '%s'; unable to unshare." % name_str)
        self.op.set_value('conv_to_local', [], parse=False)
        self.op.set_value('unshare_weights', [], parse=False)
        self.writing_test = False
    
    def get_layer_idx(self, layer_name, check_type=[]):
        try:
            layer_idx = [l['name'] for l in self.model_state['layers']].index(layer_name)
            if check_type:
                layer_type = self.model_state['layers'][layer_idx]['type']
                if layer_type not in check_type:
                    raise ModelStateException("Layer with name '%s' has type '%s'; should be one of %s." % (layer_name, layer_type, ",".join("'%s'" %s for s in check_type)))
            return layer_idx
        except ValueError:
            raise ModelStateException("Layer with name '%s' not defined." % layer_name)

    def fill_excused_options(self):
        if self.op.get_value('check_grads'):
            self.op.set_value('save_path', '')
            self.op.set_value('train_batch_range', '0')
            self.op.set_value('test_batch_range', '0')
            self.op.set_value('data_path', '')
            
    # Make sure the data provider returned data in proper format
    def parse_batch_data(self, batch_data, train=True):
        if max(d.dtype != n.single for d in batch_data[2]):
            raise DataProviderException("All matrices returned by data provider must consist of single-precision floats.")
        return batch_data

    def start_batch(self, batch_data, train=True):
        data = batch_data[2]
        self.writing_test = False
        
        if self.check_grads:
            self.libmodel.checkGradients(data)
        elif not train and self.multiview_test:
            num_views = self.test_data_provider.num_views
            if self.test_out != "" and self.logreg_name != "":
                self.writing_test = True
                self.test_file_name = os.path.join(self.test_out, 'test_preds_%d' % batch_data[1])
                self.probs = n.zeros((data[0].shape[1]/num_views, self.test_data_provider.get_num_classes()), dtype=n.single)
                self.libmodel.startMultiviewTest(data, num_views, self.probs, self.logreg_name)
            else:
                self.libmodel.startMultiviewTest(data, num_views)
        else:
            num_batches_total = self.num_epochs * len(self.train_batch_range)
            progress = min(1.0, max(0.0, float(self.get_num_batches_done()-1) / num_batches_total))
            self.libmodel.startBatch(data, progress, not train)
            
    def finish_batch(self):
        ret = IGPUModel.finish_batch(self)
        if self.writing_test:
            if not os.path.exists(self.test_out):
                os.makedirs(self.test_out)
            pickle(self.test_file_name,  {'data': self.probs,
                                          'note': 'generated from %s' % self.save_file})
        return ret
    
    def print_iteration(self):
        print "%d.%d..." % (self.epoch, self.batchnum),
        
    def print_train_time(self, compute_time_py):
        print "(%.3f sec)" % (compute_time_py)
        
    def print_costs(self, cost_outputs):
        costs, num_cases = cost_outputs[0], cost_outputs[1]
        for errname in costs.keys():
            costs[errname] = [(v/num_cases) for v in costs[errname]]
            print "%s: " % errname,
            print ", ".join("%.6f" % v for v in costs[errname]),
            if sum(m.isnan(v) for v in costs[errname]) > 0 or sum(m.isinf(v) for v in costs[errname]):
                print "^ got nan or inf!"
                sys.exit(1)
        
    def print_train_results(self):
        self.print_costs(self.train_outputs[-1])
        
    def print_test_status(self):
        pass
        
    def print_test_results(self):
        print NL + "======================Test output======================"
        self.print_costs(self.test_outputs[-1])
        print NL + "----------------------Averages-------------------------"
        self.print_costs((self.aggregate_test_outputs(self.test_outputs[-len(self.test_batch_range):])[0], min(len(self.test_outputs), len(self.test_batch_range))))
        print NL + "-------------------------------------------------------",
        for name in sorted(self.layers.keys()): # This is kind of hacky but will do for now.
            l = self.layers[name]
            if 'weights' in l:
                if type(l['weights']) == n.ndarray:
                    print "%sLayer '%s' weights: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['weights'])), n.mean(n.abs(l['weightsInc']))),
                elif type(l['weights']) == list:
                    print ""
                    print NL.join("Layer '%s' weights[%d]: %e [%e]" % (l['name'], i, n.mean(n.abs(w)), n.mean(n.abs(wi))) for i,(w,wi) in enumerate(zip(l['weights'],l['weightsInc']))),
                print "%sLayer '%s' biases: %e [%e]" % (NL, l['name'], n.mean(n.abs(l['biases'])), n.mean(n.abs(l['biasesInc']))),
        print ""
        
    def conditional_save(self):
        self.save_state()
        print "-------------------------------------------------------"
        print "Saved checkpoint to %s" % os.path.join(self.save_path, self.save_file)
        print "=======================================================",
        
    def aggregate_test_outputs(self, test_outputs):
        test_outputs = cp.deepcopy(test_outputs)
        num_cases = sum(t[1] for t in test_outputs)
        for i in xrange(1 ,len(test_outputs)):
            for k,v in test_outputs[i][0].items():
                for j in xrange(len(v)):
                    test_outputs[0][0][k][j] += test_outputs[i][0][k][j]
        
        return (test_outputs[0][0], num_cases)
    
    @classmethod
    def get_options_parser(cls):
        op = IGPUModel.get_options_parser()
        op.add_option("mini", "minibatch_size", IntegerOptionParser, "Minibatch size", default=128)
        op.add_option("layer-def", "layer_def", StringOptionParser, "Layer definition file", set_once=True)
        op.add_option("layer-params", "layer_params", StringOptionParser, "Layer parameter file")
        op.add_option("check-grads", "check_grads", BooleanOptionParser, "Check gradients and quit?", default=0, excuses=['data_path','save_path','train_batch_range','test_batch_range'])
        op.add_option("multiview-test", "multiview_test", BooleanOptionParser, "Cropped DP: test on multiple patches?", default=0)
        op.add_option("crop-border", "crop_border", IntegerOptionParser, "Cropped DP: crop border size", default=4, set_once=True)
        op.add_option("conv-to-local", "conv_to_local", ListOptionParser(StringOptionParser), "Convert given conv layers to unshared local", default=[])
        op.add_option("unshare-weights", "unshare_weights", ListOptionParser(StringOptionParser), "Unshare weight matrices in given layers", default=[])
        op.add_option("conserve-mem", "conserve_mem", BooleanOptionParser, "Conserve GPU memory (slower)?", default=0)
        op.add_option("color-noise", "color_noise", FloatOptionParser, "Add PCA noise to color channels with given scale", default=0.0)
        op.add_option("test-out", "test_out", StringOptionParser, "Output test case predictions to given path", default="", requires=['logreg_name', 'multiview_test'])
        op.add_option("logreg-name", "logreg_name", StringOptionParser, "Logreg cost layer name (for --test-out)", default="")
        op.add_option("pca-noise", "pca_noise", FloatOptionParser, "Add PCA noise to pixels with given scale", default=0.0)
        op.add_option("scalar-mean", "scalar_mean", FloatOptionParser, "Subtract scalar pixel mean (as opposed to vector)?", default=False)
        op.add_option("wupdate-freq", "wupdate_freq", IntegerOptionParser, "Weight update (inverse) frequency, in minibatches (1 = every minibatch)", default=1)
        
        op.delete_option('max_test_err')
        op.options["max_filesize_mb"].default = 0
        op.options["testing_freq"].default = 50
        op.options["num_epochs"].default = 50000
        op.options['dp_type'].default = None

        DataProvider.register_data_provider('dummy-lr-n', 'Dummy ConvNet logistic regression', DummyConvNetLogRegDP)
        DataProvider.register_data_provider('inet-lr', 'ImageNet logistic regression', ImageNetLogRegDP)
        DataProvider.register_data_provider('inet-lr-cropped', 'ImageNet logistic regression cropped', CroppedImageNetLogRegDP)
        DataProvider.register_data_provider('inet-lr-cropped-jpeg', 'ImageNet logistic regression cropped JPEG', JPEGCroppedImageNetLogRegDP)
        DataProvider.register_data_provider('inet-rs-lr-cropped', 'Random scale cropped ImageNet logistic regression', RandomScaleImageNetLogRegDP)
        DataProvider.register_data_provider('flickr-ce-cropped', 'Flickr cross-entropy cropped', JPEGCroppedFlickrCEDP)
        DataProvider.register_data_provider('dummy-ce-n', 'Dummy cross-entropy', DummyConvNetCEDP)
        DataProvider.register_data_provider('flatmem', 'Flat memory', FlatMemoryDataProvider)
        
        DataProvider.register_data_provider('cifar', 'CIFAR', CIFARDataProvider)
        DataProvider.register_data_provider('cifar-cropped', 'Cropped CIFAR', CroppedCIFARDataProvider)
        return op
    
if __name__ == "__main__":
    #nr.seed(5)
    op = ConvNet.get_options_parser()

    op, load_dic = IGPUModel.parse_options(op)
    model = ConvNet(op, load_dic)
    model.start()
