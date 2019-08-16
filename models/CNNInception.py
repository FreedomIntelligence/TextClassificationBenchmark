# -*- coding: utf-8 -*-


import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict

class Inception(nn.Module):
    def __init__(self,cin,co,relu=True,norm=True):
        super(Inception, self).__init__()
        assert(co%4==0)
        cos=[int(co/4)]*4
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.BatchNorm1d(co))
        if relu:self.activa.add_module('relu',nn.ReLU(True))
        self.branch1 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[0], 1,stride=1)),
            ])) 
        self.branch2 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[1], 1)),
            ('norm1', nn.BatchNorm1d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[1],cos[1], 3,stride=1,padding=1)),
            ]))
        self.branch3 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[2], 3,padding=1)),
            ('norm1', nn.BatchNorm1d(cos[2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[2],cos[2], 5,stride=1,padding=2)),
            ]))
        self.branch4 =nn.Sequential(OrderedDict([
            #('pool',nn.MaxPool1d(2)),
            ('conv3', nn.Conv1d(cin,cos[3], 3,stride=1,padding=1)),
            ]))
    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
        result=self.activa(torch.cat((branch1,branch2,branch3,branch4),1))
        return result
    
from models.BaseModel import BaseModel
class InceptionCNN(BaseModel):
    def __init__(self, opt ):
        super(InceptionCNN, self).__init__(opt)   
        incept_dim=getattr(opt,"inception_dim",512)
        self.model_name = 'CNNText_inception'
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)

        self.content_conv=nn.Sequential(
            Inception(opt.embedding_dim,incept_dim),#(batch_size,64,opt.content_seq_len)->(batch_size,64,(opt.content_seq_len)/2)
            #Inception(incept_dim,incept_dim),#(batch_size,64,opt.content_seq_len/2)->(batch_size,32,(opt.content_seq_len)/4)
            Inception(incept_dim,incept_dim),
            nn.MaxPool1d(opt.max_seq_len)
        )
        linear_hidden_size = getattr(opt,"linear_hidden_size",2000)
        self.fc = nn.Sequential(
            nn.Linear(incept_dim,linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(linear_hidden_size ,opt.label_size)
        )
        if opt.__dict__.get("embeddings",None) is not None:
            self.encoder.weight=nn.Parameter(opt.embeddings)
        self.properties.update(
                {"linear_hidden_size":linear_hidden_size,
                 "incept_dim":incept_dim,
                })
 
    def forward(self,content):
     
        content=self.encoder(content)
        if self.opt.embedding_type=="static":
            content=content.detach(0)

        content_out=self.content_conv(content.permute(0,2,1))
        out=content_out.view(content_out.size(0), -1)
        out=self.fc(out)
        return out
        
if __name__ == '__main__':
    import sys
    sys.path.append(r"..")
    import opts
    opt=opts.parse_opt()
    opt.vocab_size=2501
    opt.label_size=3
    m = CNNText_inception(opt)

    content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    o = m(content)
    print(o.size())        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        