import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
#from memory_profiler import profile

"""
Lai S, Xu L, Liu K, et al. Recurrent Convolutional Neural Networks for Text Classification[C]//AAAI. 2015, 333: 2267-2273.
"""

class RCNN(nn.Module):
    # embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu
    def __init__(self,opt):
        self.opt=opt
        super(RCNN, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings)
#        self.word_embeddings.weight.data.copy_(torch.from_numpy(opt.embeddings))
        
        self.num_layers = 1
        #self.bidirectional = True
        self.dropout = opt.keep_dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim // 2, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)

        ###self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        
        self.max_pooling = nn.MaxPool1d(kernel_size=3, stride=2)
        
        self.content_dim = 256
        #self.conv =  nn.Conv1d(opt.hidden_dim, self.content_dim, opt.hidden_dim * 2, stride=opt.embedding_dim)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)

    def init_hidden(self,batch_size=None):
        if batch_size is None:
            batch_size= self.batch_size
        
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(1,batch_size, self.hidden_dim // 2))
        return (h0, c0)
#    @profile
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) #64x200x300

#        x = embeds.view(sentence.size()[1], self.batch_size, -1)
        x=embeds.permute(1,0,2) #200x64x300
        self.hidden= self.init_hidden(sentence.size()[0]) #1x64x128
        lstm_out, self.hidden = self.bilstm(x, self.hidden) ###input (seq_len, batch, input_size) #Outupts:output, (h_n, c_n) output:(seq_len, batch, hidden_size * num_directions)
        #lstm_out 200x64x128
        
        c_lr =  lstm_out.permute(1,0,2)
        xi = torch.cat((c_lr[0:c_lr.size()[0]/2],embeds,c_lr[c_lr.size()[0]/2:]),1)
        yi = torch.tanh(xi)
        y = self.max_pooling(yi)
        
        ##y = self.conv(lstm_out.permute(1,2,0)) ###64x256x1

        y  = self.hidden2label(y.view(y.size()[0],-1)) #64x3
        return y