# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np



from models.LSTM import LSTMClassifier
from models.CNNBasic import BasicCNN1D,BasicCNN2D
from models.CNNKim import KIMCNN1D,KIMCNN2D
from models.CNNMultiLayer import MultiLayerCNN
from models.CNNInception import InceptionCNN
from models.FastText import FastText
from models.Capsule import CapsuleNet
from models.RCNN import RCNN
from models.RNN_CNN import RNN_CNN
from models.LSTMBI import LSTMBI
from models.Transformer import AttentionIsAllYouNeed
from models.SelfAttention import SelfAttention
from models.LSTMwithAttention import LSTMAttention
from models.BERTFast import BERTFast
def setup(opt):
    
    if opt.model == 'lstm':
        model = LSTMClassifier(opt)
    elif opt.model == 'basic_cnn' or opt.model == "cnn":
        model = BasicCNN1D(opt)
    elif opt.model == 'baisc_cnn_2d' :
        model = BasicCNN2D(opt)
    elif opt.model == 'kim_cnn' :
        model = KIMCNN1D(opt)
    elif opt.model ==  'kim_cnn_2d':
        model = KIMCNN2D(opt)
    elif opt.model ==  'multi_cnn':
        model = MultiLayerCNN(opt)
    elif opt.model ==  'inception_cnn':
        model = InceptionCNN(opt) 
    elif opt.model ==  'fasttext':
        model = FastText(opt)
    elif opt.model ==  'capsule':
        model = CapsuleNet(opt)
    elif opt.model ==  'rnn_cnn':
        model = RNN_CNN(opt)
    elif opt.model ==  'rcnn':
        model = RCNN(opt)
    elif opt.model ==  'bilstm':
        model = LSTMBI(opt)
    elif opt.model == "transformer":
        model = AttentionIsAllYouNeed(opt)
    elif opt.model == "selfattention":
        model = SelfAttention(opt)
    elif opt.model == "lstm_attention":
        model =LSTMAttention(opt)
    elif opt.model == "bert":
        model =BERTFast(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model
