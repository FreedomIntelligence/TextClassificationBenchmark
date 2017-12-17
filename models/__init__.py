# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np



from .LSTM import LSTMClassifier
from .CNN import CNN
from .CNNText import CNNText
from .CNN_Inception import CNNText_inception
from .FastText import FastText
def setup(opt):
    
    if opt.model == 'lstm':
        model = LSTMClassifier(opt)
    elif opt.model == 'cnn':
        model = CNN(opt)
    elif opt.model ==  'baisc_cnn':
        model = CNNText(opt) 
    elif opt.model ==  'cnn_inception':
        model = CNNText_inception(opt) 
    elif opt.model ==  'fasttext':
        model = FastText(opt) 

    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model
