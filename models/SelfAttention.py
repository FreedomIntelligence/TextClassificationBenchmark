# -*- coding: utf-8 -*-#
# https://arxiv.org/pdf/1703.03130.pdf
# A Structured Self-attentive Sentence Embedding
# https://github.com/nn116003/self-attention-classification/blob/master/imdb_attn.py

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
#from memory_profiler import profile

class SelfAttention(nn.Module):
    # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu
    def __init__(self,opt):
        self.opt=opt
        super(SelfAttention, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
#        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
  
        self.num_layers = 1
        #self.bidirectional = True
        self.dropout = opt.keep_dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim // 2, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.self_attention = nn.Sequential(
            nn.Linear(opt.hidden_dim, 24),
            nn.ReLU(True),
            nn.Linear(24,1)
            )
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
#    @profile
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)

#        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        x=embeds.permute(1,0,2)
        self.hidden= self.init_hidden(sentence.size()[0]) #2x64x64
        lstm_out, self.hidden = self.bilstm(x, self.hidden)  #lstm_out:200x64x128
        final =lstm_out.permute(1,0,2)#torch.mean(,1) 
        attn_ene = self.self_attention(final)
        attns =F.softmax(attn_ene.view(self.batch_size, -1))
        feats = (final * attns).sum(dim=1)
        y  = self.hidden2label(feats) #64x3
        
        return y