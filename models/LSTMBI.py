# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
#from memory_profiler import profile
from models.BaseModel import BaseModel
class LSTMBI(BaseModel):
    # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu
    def __init__(self,opt):
        super(LSTMBI, self).__init__(opt)


        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
#        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
  

        #self.bidirectional = True

        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim // 2, num_layers=self.opt.lstm_layers, dropout=self.opt.keep_dropout, bidirectional=self.opt.bidirectional)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.lsmt_reduce_by_mean = opt.__dict__.get("lstm_mean",True) 
        
        
        self.properties.update(
                {"hidden_dim":self.opt.hidden_dim,
                 "lstm_mean":self.lsmt_reduce_by_mean,
                 "lstm_layers":self.opt.lstm_layers,
#                 "bidirectional":str(self.opt.bidirectional)
                })

    def init_hidden(self,batch_size=None):
        if batch_size is None:
            batch_size= self.opt.batch_size
        
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(2*self.opt.lstm_layers, batch_size, self.opt.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(2*self.opt.lstm_layers, batch_size, self.opt.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2*self.opt.lstm_layers, batch_size, self.opt.hidden_dim // 2))
            c0 = Variable(torch.zeros(2*self.opt.lstm_layers, batch_size, self.opt.hidden_dim // 2))
        return (h0, c0)
#    @profile
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)

#        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        x=embeds.permute(1,0,2)  # we do this because the default parameter of lstm is False 
        self.hidden= self.init_hidden(sentence.size()[0]) #2x64x64
        lstm_out, self.hidden = self.bilstm(x, self.hidden)  #lstm_out:200x64x128
        if self.lsmt_reduce_by_mean=="mean":
            out = lstm_out.permute(1,0,2)
            final = torch.mean(out,1)
        else:
            final=lstm_out[-1]
        y  = self.hidden2label(final) #64x3  #lstm_out[-1]
        return y

