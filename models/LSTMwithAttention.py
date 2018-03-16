# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from sklearn.utils import shuffle
from torch.autograd import Variable

class LSTMAttention(torch.nn.Module):
    def __init__(self,opt):
        self.opt=opt
        super(LSTMAttention, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
#        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
  
        self.num_layers = opt.lstm_layers
        #self.bidirectional = True
        self.dropout = opt.keep_dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim // 2, batch_first=True,num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.mean = opt.__dict__.get("lstm_mean",True) 
        self.attn_fc = torch.nn.Linear(opt.embedding_dim, 1)
    def init_hidden(self,batch_size=None):
        if batch_size is None:
            batch_size= self.batch_size
        
        if self.use_gpu:
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim // 2))
        return (h0, c0)


    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state],1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)
    # end method attention
    

    def forward(self, X):
        embedded = self.word_embeddings(X)
        hidden= self.init_hidden(X.size()[0]) #
        rnn_out, hidden = self.bilstm(embedded, hidden)
        h_n, c_n = hidden
        attn_out = self.attention(rnn_out, h_n)
        logits = self.hidden2label(attn_out)
        return logits