# -*- coding: utf-8 -*-
# paper 


#



import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

BATCH_SIZE = 100

NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3

cuda = torch.cuda.is_available()

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)





class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS,padding=0):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules
        

        
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            prime=[3,5,7,9,11,13,17,19,23]
            sizes=prime[:self.num_capsules]
            self.capsules = nn.ModuleList(
                [nn.Conv1d(in_channels, out_channels, kernel_size=i, stride=2, padding=int((i-1)/2)) for i in sizes])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        
        if self.num_route_nodes != -1:
            priors =torch.matmul( x[None, :, :, None, :],self.route_weights[:, None, :, :, :])

            if torch.cuda.is_available():
                logits = torch.autograd.Variable(torch.zeros(priors.size())).cuda()
            else:
                logits = torch.autograd.Variable(torch.zeros(priors.size()))
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((torch.mul(probs , priors)).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (torch.mul(priors , outputs)).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self,opt):
        super(CapsuleNet, self).__init__()
        self.opt=opt    #300*300
        self.label_size=opt.label_size
        self.embed = nn.Embedding(opt.vocab_size+1, opt.embedding_dim)
        self.opt.cnn_dim = 1
        self.kernel_size = 3
        self.kernel_size_primary=3
        if opt.__dict__.get("embeddings",None) is not None:
            self.embed.weight=nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)

        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32)
        self.digit_capsules = CapsuleLayer(num_capsules=opt.label_size, num_route_nodes=int(32 * opt.max_seq_len/2), in_channels=8,
                                           out_channels=16)
        if self.opt.cnn_dim == 2:
            self.conv_2d = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(self.kernel_size,opt.embedding_dim), stride=(1,opt.embedding_dim),padding=(int((self.kernel_size-1)/2),0))
        else:
            self.conv_1d = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=opt.embedding_dim * self.kernel_size, stride=opt.embedding_dim, padding=opt.embedding_dim* int((self.kernel_size-1)/2) )

        self.decoder = nn.Sequential(
            nn.Linear(16 * self.label_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None,reconstruct=False):
        #x = next(iter(train_iter)).text[0]

        x= self.embed(x)
        if self.opt.cnn_dim == 1:
            x=x.view(x.size(0),1,x.size(-1)*x.size(-2))                
            x_conv = F.relu(self.conv_1d(x), inplace=True)
        else:
            
            x=x.unsqueeze(1)        
            x_conv = F.relu(self.conv_2d(x), inplace=True).squeeze(3)

        x = self.primary_capsules(x_conv)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        if not reconstruct:
            return classes
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            if torch.cuda.is_available():
                y = Variable(torch.sparse.torch.eye(self.label_size)).cuda().index_select(dim=0, index=max_length_indices.data)
            else:
                y = Variable(torch.sparse.torch.eye(self.label_size)).index_select(dim=0, index=max_length_indices.data)
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions
