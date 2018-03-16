# -*- coding: utf-8 -*-
import torch as t
import numpy as np
from torch import nn

class CNNText(nn.Module): 
    def __init__(self, opt ):
        super(CNNText, self).__init__()
        self.model_name = 'CNNText'
        self.opt=opt
        self.content_dim=opt.__dict__.get("content_dim",256)
        self.kernel_size=opt.__dict__.get("kernel_size",3)

        
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        if opt.__dict__.get("embeddings",None) is not None:
            self.encoder.weight=nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)


        self.content_conv = nn.Sequential(
            nn.Conv1d(in_channels = opt.embedding_dim,
                      out_channels = self.content_dim,
                      kernel_size = self.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = (opt.max_seq_len - self.kernel_size + 1))
#            nn.AdaptiveMaxPool1d()
        )

        self.fc = nn.Linear(self.content_dim, opt.label_size)


    def forward(self,  content):

        content = self.encoder(content)
        content_out = self.content_conv(content.permute(0,2,1))
        reshaped = content_out.view(content_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits

import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden_dim')   
    
    
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size')
    parser.add_argument('--embedding_dim', type=int, default=300,
                    help='embedding_dim')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning_rate')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                    help='grad_clip')
    parser.add_argument('--model', type=str, default="lstm",
                    help='model name')


#
    args = parser.parse_args()
    args.embedding_dim=300
    args.vocab_size=10000
    args.kernel_size=3
    args.num_classes=3
    args.content_dim=256
    args.max_seq_len=50
    
#
#    # Check if args are valid
#    assert args.rnn_size > 0, "rnn_size should be greater than 0"


    return args
 
if __name__ == '__main__':
    

    opt = parse_opt()
    m = CNNText(opt)
    content = t.autograd.Variable(t.arange(0,3200).view(-1,50)).long()
    o = m(content)
    print(o.size())

