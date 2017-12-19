import torch
import torch.nn as nn
import torch.nn.functional as F


class KIMCNN1D(nn.Module):
    def __init__(self, opt):
        super(KIMCNN1D, self).__init__()

        self.embedding_type = opt.embedding_type
        self.batch_size = opt.batch_size
        self.max_seq_len = opt.max_seq_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.label_size = opt.label_size
        self.kernel_sizes = opt.kernel_sizes
        self.kernel_nums = opt.kernel_nums        
        self.keep_dropout = opt.keep_dropout
        self.in_channel = 1

        assert (len(self.kernel_sizes) == len(self.kernel_nums))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
        if self.embedding_type == "static" or self.embedding_type == "non-static" or self.embedding_type == "multichannel":
            self.embedding.weight=nn.Parameter(opt.embeddings)            
            if self.embedding_type == "static":
                self.embedding.weight.requires_grad = False
            elif self.embedding_type == "multichannel":
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
                self.embedding2.weight=nn.Parameter(opt.embeddings) 
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2
            else:
                pass
#
#        for i in range(len(self.kernel_sizes)):
#            conv = nn.Conv1d(self.in_channel, self.kernel_nums[i], self.embedding_dim * self.kernel_sizes[i], stride=self.embedding_dim)
#            setattr(self, 'conv_%d'%i, conv)
        self.convs = nn.ModuleList([nn.Conv1d(self.in_channel, num, self.embedding_dim * size, stride=self.embedding_dim) for size,num in zip(opt.kernel_sizes,opt.kernel_nums)])
        self.fc = nn.Linear(sum(self.kernel_nums), self.label_size)

    def get_conv(self, i):
        return getattr(self, 'conv_%d'%i)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.embedding_dim * self.max_seq_len)
        if self.embedding_type == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.embedding_dim * self.max_seq_len)
            x = torch.cat((x, x2), 1)

#        conv_results = [
#            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.max_seq_len - self.kernel_sizes[i] + 1)
#                .view(-1, self.kernel_nums[i])
#            for i in range(len(self.kernel_sizes))]
        conv_results = [
            F.max_pool1d(F.relu(self.convs[i](x)), self.max_seq_len - self.kernel_sizes[i] + 1)
                .view(-1, self.kernel_nums[i])
            for i in range(len(self.convs))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.keep_dropout, training=self.training)
        x = self.fc(x)
        return x



#https://github.com/zachAlbus/pyTorch-text-classification/blob/master/Yoon/model.py
class  KIMCNN2D(nn.Module):
    
    def __init__(self, opt):
        super(KIMCNN2D,self).__init__()
        self.opt = opt   
        self.embedding_type = opt.embedding_type
        self.batch_size = opt.batch_size
        self.max_seq_len = opt.max_seq_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.label_size = opt.label_size
        self.kernel_sizes = opt.kernel_sizes
        self.kernel_nums = opt.kernel_nums        
        self.keep_dropout = opt.keep_dropout
        
        self.embedding = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
        if self.embedding_type == "static" or self.embedding_type == "non-static" or self.embedding_type == "multichannel":
            self.embedding.weight=nn.Parameter(opt.embeddings)            
            if self.embedding_type == "static":
                self.embedding.weight.requires_grad = False
            elif self.embedding_type == "multichannel":
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
                self.embedding2.weight=nn.Parameter(opt.embeddings) 
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2
            else:
                pass
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num, (size, opt.embedding_dim)) for size,num in zip(opt.kernel_sizes,opt.kernel_nums)])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(opt.keep_dropout)
        self.fc = nn.Linear(sum(opt.kernel_nums), opt.label_size)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, x):
        x = self.embedding(x) # (N,W,D)
        


        x = x.unsqueeze(1) # (N,Ci,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)


        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc(x) # (N,C)
        return logit

