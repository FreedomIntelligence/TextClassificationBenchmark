# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 22:40:58 2017

@author: lilianwang
"""

import torch

from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F

print("conv1d sample")

a=range(16)

x = Variable(torch.Tensor(a))

x=x.view(1,1,16)

print("x variable:", x)

b=torch.ones(3)

b[0]=0.1

b1=0.2

b2=0.3

weights = Variable(b)#torch.randn(1,1,2,2)) #out_channel*in_channel*H*W

weights=weights.view(1,1,3)

print ("weights:",weights)

y=F.conv1d(x, weights, padding=0)

print ("y:",y)


m = nn.Linear(20, 30)
input = Variable(torch.randn(128, 20))
output = m(input)
print(output.size()) ##torch.Size([128, 30])

m = nn.Conv1d(16, 33, 3, stride=2)
input = Variable(torch.randn(20, 16, 50))
output = m(input) #20x33x24