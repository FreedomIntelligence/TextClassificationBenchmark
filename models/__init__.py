# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np



from .LSTM import LSTMClassifier
from .CNNBasic import BasicCNN1D,BasicCNN2D
from .CNNKim import KIMCNN1D,KIMCNN2D
from .CNNMultiLayer import MultiLayerCNN
from .CNNInception import InceptionCNN
from .FastText import FastText
def setup(opt):
    
    if opt.model == 'lstm':
        model = LSTMClassifier(opt)
    elif opt.model == 'baisc_cnn' or opt.model == "cnn":
        model = BasicCNN1D(opt)
    elif opt.model ==  'kim_cnn':
        model = KIMCNN1D(opt)
    elif opt.model ==  'multi_cnn':
        model = MultiLayerCNN(opt)
    elif opt.model ==  'inception_cnn':
        model = InceptionCNN(opt) 
    elif opt.model ==  'fasttext':
        model = FastText(opt) 

    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model
