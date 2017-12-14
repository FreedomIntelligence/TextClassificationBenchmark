# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import numpy as np

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:       
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def loadData(opt):

    device = 0 if  torch.cuda.is_available()  else -1

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True,fix_length=100)
    LABEL = data.Field(sequential=False)

    train, test = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)    
    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=opt.batch_size,device=device)

    opt.label_size= len(LABEL.vocab)    
    opt.vocab_size = len(TEXT.vocab)
    opt.embedding_dim= TEXT.vocab.vectors.size()[1]
    opt.embeddings = TEXT.vocab.vectors
    
    return train_iter, test_iter

def evaluation(model,test_iter):
    accuracy=[]
    for batch in test_iter.__iter__():
        predicted = model(batch.text[0])
        percision = predicted == predicted
        accuracy.append(sum(percision)*1.0/len(percision) )
    return np.mean(accuracy)
    