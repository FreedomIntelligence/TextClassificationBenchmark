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
from torch.nn.modules.loss import NLLLoss

opt = opts.parse_opt()
opt.model ='lstm'



train_iter, test_iter = utils.loadData(opt)
model=models.setup(opt)
if torch.cuda.is_available():
    model.cuda()
model.train()
optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
optimizer.zero_grad()
loss_fun = NLLLoss()

for batch in train_iter.__iter__():
    
    predicted = model(batch.text[0])

    loss = loss_fun(predicted,batch.label)
    loss.backward()
    utils.clip_gradient(optimizer, opt.grad_clip)
    optimizer.step()
    print("loss : %.5f" % loss.data.numpy()[0])
    
utils.evaluation(model,test_iter)


        














    
    







