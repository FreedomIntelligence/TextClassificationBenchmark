# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


#https://github.com/zachAlbus/pyTorch-text-classification/blob/master/Zhang/model.py
class MultiLayerCNN(nn.Module):
    def __init__(self, opt):
        super(MultiLayerCNN, self).__init__()
        self.embed = nn.Embedding(opt.vocab_size + 1, opt.embedding_dim)
        
        if opt.__dict__.get("embeddings",None) is not None:
            self.embed.weight=nn.Parameter(opt.embeddings,requires_grad=opt.embedding_training)
            
        self.conv1 = nn.Sequential(
            nn.Conv1d(opt.max_seq_len, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc = nn.Linear(256*7, opt.label_size)

    def forward(self, x):
        # Embedding
        x = self.embed(x)  # dim: (batch_size, max_seq_len, embedding_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x)
