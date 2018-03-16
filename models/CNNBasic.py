# -*- coding: utf-8 -*-
import torch as t
import numpy as np
from torch import nn

class BasicCNN1D(nn.Module): 
    def __init__(self, opt ):
        super(BasicCNN1D, self).__init__()
        self.model_name = 'CNNText'
        self.opt=opt
        self.content_dim=opt.__dict__.get("content_dim",256)
        self.kernel_size=opt.__dict__.get("kernel_size",3)

        
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        if opt.__dict__.get("embeddings",None) is not None:
            self.encoder.weight=nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)

        self.content_conv = nn.Sequential(
            nn.Conv1d(in_channels = opt.embedding_dim,
                      out_channels = self.content_dim, #256
                      kernel_size = self.kernel_size), #3
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = (opt.max_seq_len - self.kernel_size + 1))
#            nn.AdaptiveMaxPool1d()
        )
        self.fc = nn.Linear(self.content_dim, opt.label_size)

    def forward(self,  content):

        content = self.encoder(content) #64x200x300
        content_out = self.content_conv(content.permute(0,2,1)) #64x256x1
        reshaped = content_out.view(content_out.size(0), -1) #64x256
        logits = self.fc(reshaped) #64x3
        return logits
class BasicCNN2D(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, args):
        super(BasicCNN2D, self).__init__()
        self.opt = opt

        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.label_size = opt.label_size
        self.keep_dropout = opt.keep_dropout
        in_channel = 1
        self.kernel_nums = opt.kernel_nums
        self.kernel_sizes = opt.kernel_sizes

        self.embed = nn.Embedding(self.vocab_size+1, self.embedding_dim)
        
        if opt.__dict__.get("embeddings",None) is not None:
            self.embed.weight=nn.Parameter(opt.embeddings)
            
        self.conv = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, self.embedding_dim)) for K,out_channel in zip(self.kernel_sizes,self.kernel_nums)])

        self.dropout = nn.Dropout(self.keep_dropout)
        self.fc = nn.Linear(len(self.kernel_sizes) * self.out_channel, self.label_size)


    def forward(self, input_x):
        """
        :param input_x: a list size having the number of batch_size elements with the same length
        :return: batch_size X num_aspects tensor
        """
        # Embedding
        x = self.embed(input_x)  # dim: (batch_size, max_seq_len, embedding_size)

        if self.opt.static:
            x = F.Variable(input_x)

        # Conv & max pool
        x = x.unsqueeze(1)  # dim: (batch_size, 1, max_seq_len, embedding_size)

        # turns to be a list: [ti : i \in kernel_sizes] where ti: tensor of dim([batch, num_kernels, max_seq_len-i+1])
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]

        # dim: [(batch_size, num_kernels), ...]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        # Dropout & output
        x = self.dropout(x)  # (batch_size,len(kernel_sizes)*num_kernels)
        logit = F.log_softmax(self.fc(x))  # (batch_size, num_aspects)

        return logit
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

