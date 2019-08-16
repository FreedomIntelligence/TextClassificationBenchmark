# -*- coding: utf-8 -*-

import torch as t

import numpy as np
from torch import nn
from collections import OrderedDict
import os
class BaseModel(nn.Module):
    def __init__(self, opt ):
        super(BaseModel, self).__init__()
        self.model_name = 'BaseModel'
        self.opt=opt
        
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        if opt.__dict__.get("embeddings",None) is not None:
            self.encoder.weight=nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
        self.fc = nn.Linear(opt.embedding_dim, opt.label_size)
        

        self.properties = {"model_name":self.__class__.__name__,
#                "embedding_dim":self.opt.embedding_dim,
#                "embedding_training":self.opt.embedding_training,
#                "max_seq_len":self.opt.max_seq_len,
                "batch_size":self.opt.batch_size,
                "learning_rate":self.opt.learning_rate,
                "keep_dropout":self.opt.keep_dropout,
                }
 
    def forward(self,content):
        content_=t.mean(self.encoder(content),dim=1)
        out=self.fc(content_.view(content_.size(0),-1))
        return out
    

    
    def save(self,save_dir="saved_model",metric=None):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.model_info = "__".join([k+"_"+str(v) if type(v)!=list else k+"_"+str(v)[1:-1].replace(",","_").replace(",","")  for k,v in self.properties.items() ])
        if metric:
            path = os.path.join(save_dir, str(metric)[2:] +"_"+ self.model_info)
        else:
            path = os.path.join(save_dir,self.model_info)
        t.save(self,path)
        return path
    

        
if __name__ == '__main__':
    import sys
    sys.path.append(r"..")
    import opts
    opt=opts.parse_opt()
    opt.vocab_size=2501
    opt.embedding_dim=300
    opt.label_size=3
    m = BaseModel(opt)

    content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    o = m(content)
    print(o.size())
    path = m.save()