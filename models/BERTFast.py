# -*- coding: utf-8 -*-
import torch as t
import numpy as np
from torch import nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class BERTFast(nn.Module): 
    def __init__(self, opt ):
        super(BERTFast, self).__init__()
        self.model_name = 'bert'
        self.opt=opt

        self.fc = nn.Linear(768, opt.label_size)

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')


    def forward(self,  content):
        encoded, _ = self.bert_model(content)
        encoded_doc = t.mean(encoded[-1],dim=1)
        logits = self.fc(encoded_doc)
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
    parser.add_argument('--label_size', type=str, default=2,
                    help='label_size')


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
    m = BERTFast(opt)
    content = t.autograd.Variable(t.arange(0,3200).view(-1,50)).long()
    o = m(content)
    print(o.size())

