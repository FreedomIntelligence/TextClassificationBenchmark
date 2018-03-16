# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
#from memory_profiler import profile

class LSTMClassifier(nn.Module):
    # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu
    def __init__(self,opt):
        self.opt=opt
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
#        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.mean = opt.__dict__.get("lstm_mean",True) 

    def init_hidden(self,batch_size=None):
        if batch_size is None:
            batch_size= self.batch_size
        
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1,batch_size, self.hidden_dim))
        return (h0, c0)
#    @profile
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) #64x200x300

#        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        x=embeds.permute(1,0,2) #200x64x300
        self.hidden= self.init_hidden(sentence.size()[0]) #1x64x128
        lstm_out, self.hidden = self.lstm(x, self.hidden) #200x64x128
        if self.mean=="mean":
            out = lstm_out.permute(1,0,2)
            final = torch.mean(out,1)
        else:
            final=lstm_out[-1]
        y  = self.hidden2label(final)  #64x3
        return y
#    def forward1(self, sentence):
#       
#        return torch.zeros(sentence.size()[0], self.opt.label_size)
##    def __call__(self, **args):
##        self.forward(args)
#    def test():
#        
#        import numpy as np
#        
#        word_embeddings = nn.Embedding(10000, 300)
#        lstm = nn.LSTM(300, 100)
#        h0 = Variable(torch.zeros(1, 128, 100))
#        c0 = Variable(torch.zeros(1, 128, 100))
#        hidden=(h0, c0)
#        sentence = Variable(torch.LongTensor(np.zeros((128,30),dtype=np.int64)))
#        embeds = word_embeddings(sentence)
#        torch.tile(sentence)
#        sentence.size()[0]
#       
#        
#        
##        x= Variable(torch.zeros(30, 128, 300))
#        x = embeds.view(sentence.size()[1], self.batch_size, -1)
#        embeds=embeds.permute(1,0,2)
#        lstm_out, hidden = lstm(embeds, hidden)
##                  