# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


from six.moves import cPickle

import opts
import models
import torch.nn as nn
import utils
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
opt = opts.parse_opt()
opt.model ='lstm'
from torch.nn.modules.loss import NLLLoss
def getDataIterator(opt):

    device = 1 if  torch.cuda.is_available()  else -1

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    train, test = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)    
    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())
    
    # make iterator for splits
    #train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=3, device=0)
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=opt.batch_size,device=device)
    # print batch information
#    batch = next(iter(train_iter))
#    print(batch.text)
#    print(batch.label)
    
    opt.label_size= len(LABEL.vocab)
    
    opt.vocab_size = len(TEXT.vocab)
    opt.embedding_dim= TEXT.vocab.vectors.size()[1]
    opt.embeddings = TEXT.vocab.vectors
    return train_iter, test_iter

opt.grad_clip=0.001
train_iter, test_iter = getDataIterator(opt)
model=models.setup(opt)
model.train()
optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
optimizer.zero_grad()
loss_fun = NLLLoss()
for batch in train_iter.__iter__():

#    
#    batch = next(iter(train_iter))

    predicted = model(batch.text[0])

    loss = loss_fun(predicted,batch.label)
    loss.backward()
    utils.clip_gradient(optimizer, opt.grad_clip)
    optimizer.step()
    print("loss : %.5f" % loss.data.numpy()[0])















    
    







