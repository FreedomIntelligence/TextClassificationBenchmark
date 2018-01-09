import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()

        self.embedding_type = opt.embedding_type
        self.batch_size = opt.batch_size
        self.max_sent_len = opt.max_sent_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.CLASS_SIZE = opt.label_size
        self.FILTERS = opt["FILTERS"]
        self.FILTER_NUM = opt["FILTER_NUM"]
        self.keep_dropout = opt.keep_dropout
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
        if self.embedding_type == "static" or self.embedding_type == "non-static" or self.embedding_type == "multichannel":
            self.WV_MATRIX = opt["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.embedding_type == "static":
                self.embedding.weight.requires_grad = False
            elif self.embedding_type == "multichannel":
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.embedding_dim * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, 'conv_%d'%i, conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.label_size)

    def get_conv(self, i):
        return getattr(self, 'conv_%d'%i)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.embedding_dim * self.max_sent_len)
        if self.embedding_type == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.embedding_dim * self.max_sent_len)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.max_sent_len - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.keep_dropout, training=self.training)
        x = self.fc(x)
        return x



#https://github.com/zachAlbus/pyTorch-text-classification/blob/master/Yoon/model.py
class  CNN1(nn.Module):
    
    def __init__(self, opt):
        super(CNN1,self).__init__()
        self.opt = opt
        
        V = opt.vocab_size
        D = opt.embedding_dim
        C = opt.label_size
        Ci = 1
        Co = opt.kernel_num
        Ks = opt.kernel_sizes

        self.embed = nn.Embedding(V, D)
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(opt.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


    def forward(self, x):
        x = self.embed(x) # (N,W,D)
        
        if self.args.static:
            x = Variable(x)

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
        logit = self.fc1(x) # (N,C)
        return logit

import torch.nn as nn


#https://github.com/zachAlbus/pyTorch-text-classification/blob/master/Zhang/model.py
class CNN2(nn.Module):
    def __init__(self, opt):
        super(CNN2, self).__init__()
        self.embed = nn.Embedding(opt.vocab_size + 1, opt.embedding_dim)

        self.conv1 = nn.Sequential(
            nn.Conv1d(opt.l0, 256, kernel_size=7, stride=1),
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

        self.fc = nn.Linear(256, opt.label_size)

    def forward(self, x_input):
        # Embedding
        x = self.embed(x_input)  # dim: (batch_size, max_seq_len, embedding_size)
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
class CNN3(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, args):
        super(CNN3, self).__init__()
        self.args = args

        embedding_dim = args.embed_dim
        embedding_num = args.num_features
        class_number = args.class_num
        in_channel = 1
        out_channel = args.kernel_num
        kernel_sizes = args.kernel_sizes

        self.embed = nn.Embedding(embedding_num+1, embedding_dim)
        self.conv = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in kernel_sizes])

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * out_channel, class_number)


    def forward(self, input_x):
        """
        :param input_x: a list size having the number of batch_size elements with the same length
        :return: batch_size X num_aspects tensor
        """
        # Embedding
        x = self.embed(input_x)  # dim: (batch_size, max_seq_len, embedding_size)

        if self.args.static:
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