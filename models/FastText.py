# -*- coding: utf-8 -*-

import torch as t

import numpy as np
from torch import nn
from collections import OrderedDict
class FastText(nn.Module):
    def __init__(self, opt ):
        super(FastText, self).__init__()
        self.model_name = 'FastText'
        
        linear_hidden_size=getattr(opt,"linear_hidden_size",2000)
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        if opt.__dict__.get("embeddings",None) is not None:
            print('load embedding')
            self.encoder.weight=nn.Parameter(opt.embeddings)
        
        
        self.content_fc = nn.Sequential(
            nn.Linear(opt.embedding_dim,linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            # nn.Linear(opt.linear_hidden_size,opt.linear_hidden_size),
            # nn.BatchNorm1d(opt.linear_hidden_size),
            # nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size,opt.label_size)
        )

 
    def forward(self,content):
       
        content_=t.mean(self.encoder(content),dim=1)


        out=self.content_fc(content_.view(content_.size(0),-1))

        return out