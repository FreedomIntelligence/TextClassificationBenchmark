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
        self.word_embeddings.weight = nn.Parameter(opt.embeddings)
#        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)
#    @profile
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)

        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        self.hidden= self.init_hidden()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        return y
    def forward1(self, sentence):
       
        return torch.zeros(sentence.size()[0], self.opt.label_size)
#    def __call__(self, **args):
#        self.forward(args)
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
#       
#        
#        
##        x= Variable(torch.zeros(30, 128, 300))
#        embeds=embeds.view(sentence.size()[1],sentence.size()[0],-1)
#        lstm_out, hidden = lstm(embeds, hidden)
#                  