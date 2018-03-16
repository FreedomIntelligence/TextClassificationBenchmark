# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

# https://github.com/nmhkahn/MemN2N-pytorch/blob/master/memn2n/model.py

def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_dim+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class MemN2N(nn.Module):
    def __init__(self, opt):
        super(MemN2N, self).__init__()

        use_cuda = opt["use_cuda"]
        num_vocab = opt["num_vocab"]
        embedding_dim = opt["embedding_dim"]
        sentence_size = opt["sentence_size"]
        self.max_hops = opt["max_hops"]

        for hop in range(self.max_hops+1):
            C = nn.Embedding(num_vocab, embedding_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        self.softmax = nn.Softmax()
        self.encoding = Variable(torch.FloatTensor(
            position_encoding(sentence_size, embedding_dim)), requires_grad=False)

        if use_cuda:
            self.encoding = self.encoding.cuda()

    def forward(self, story, query):
        story_size = story.size()

        u = list()
        query_embed = self.C[0](query)
        # weired way to perform reduce_dot
        encoding = self.encoding.unsqueeze(0).expand_as(query_embed)
        u.append(torch.sum(query_embed*encoding, 1))
        
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.view(story.size(0), -1))
            embed_A = embed_A.view(story_size+(embed_A.size(-1),))
       
            encoding = self.encoding.unsqueeze(0).unsqueeze(1).expand_as(embed_A)
            m_A = torch.sum(embed_A*encoding, 2)
       
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob   = self.softmax(torch.sum(m_A*u_temp, 2))
        
            embed_C = self.C[hop+1](story.view(story.size(0), -1))
            embed_C = embed_C.view(story_size+(embed_C.size(-1),))
            m_C     = torch.sum(embed_C*encoding, 2)
       
            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
       
            u_k = u[-1] + o_k
            u.append(u_k)
       
        a_hat = u[-1]@self.C[self.max_hops].weight.transpose(0, 1)
        return a_hat, self.softmax(a_hat)