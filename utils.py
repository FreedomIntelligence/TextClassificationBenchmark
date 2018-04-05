# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import numpy as np
from functools import wraps
import time
import sys
import logging
import os

def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco  

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:       
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def loadData(opt):
    if not opt.from_torchtext:
        import dataHelper as helper
        return helper.loadData(opt)
    device = 0 if  torch.cuda.is_available()  else -1

    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True,fix_length=opt.max_seq_len)
    LABEL = data.Field(sequential=False)
    if opt.dataset=="imdb":
        train, test = datasets.IMDB.splits(TEXT, LABEL)
    elif opt.dataset=="sst":
        train, val, test = datasets.SST.splits( TEXT, LABEL, fine_grained=True, train_subtrees=True,
                                               filter_pred=lambda ex: ex.label != 'neutral')
    elif opt.dataset=="trec":
        train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)
    else:
        print("does not support this datset")
        
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)    
    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=opt.batch_size,device=device,repeat=False,shuffle=True)

    opt.label_size= len(LABEL.vocab)    
    opt.vocab_size = len(TEXT.vocab)
    opt.embedding_dim= TEXT.vocab.vectors.size()[1]
    opt.embeddings = TEXT.vocab.vectors
    
    return train_iter, test_iter


def evaluation(model,test_iter,from_torchtext=True):
    model.eval()
    accuracy=[]
#    batch= next(iter(test_iter))
    for index,batch in enumerate( test_iter):
        text = batch.text[0] if from_torchtext else batch.text
        predicted = model(text)
        prob, idx = torch.max(predicted, 1) 
        percision=(idx== batch.label).float().mean()
        
        if torch.cuda.is_available():
            accuracy.append(percision.data.cpu().numpy()[0] )
        else:
            accuracy.append(percision.data.numpy()[0] )
    model.train()
    return np.mean(accuracy)



def getOptimizer(params,name="adam",lr=None,momentum=None,scheduler=None):
    
    name = name.lower().strip()          
        
    if name=="adadelta":
        optimizer=torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0).param_groups()
    elif name == "adagrad":
        optimizer=torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)
    elif name == "sparseadam":        
        optimizer=torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    elif name =="adamax":
        optimizer=torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif name =="asgd":
        optimizer=torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    elif name == "lbfgs":
        optimizer=torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
    elif name == "rmsprop":
        optimizer=torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    elif name =="rprop":
        optimizer=torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    elif name =="sgd":
        optimizer=torch.optim.SGD(params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    else:
        print("undefined optimizer, use adam in default")
        optimizer=torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    if scheduler is not None:
        if scheduler == "lambdalr":
            lambda1 = lambda epoch: epoch // 30
            lambda2 = lambda epoch: 0.95 ** epoch
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        elif scheduler=="steplr":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler =="multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        elif scheduler =="reducelronplateau":
            return  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        else:
            pass

    else:
        return optimizer
    
    
    return 
def getLogger():

    now = int(time.time()) 
    timeArray = time.localtime(now)
    import random
    random_str = str(random.randint(1,10000))
    timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
    log_filename = "log/" +time.strftime("%Y%m%d", timeArray)+"_"+ random_str
    
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program) 
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists(log_filename):
        os.mkdir(log_filename)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',datefmt='%a, %d %b %Y %H:%M:%S',filename=log_filename+'/qa'+timeStamp+'.log',filemode='w')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    return logger
def is_writeable(path, check_parent=False):
    '''
    Check if a given path is writeable by the current user.
    :param path: The path to check
    :param check_parent: If the path to check does not exist, check for the
    ability to write to the parent directory instead
    :returns: True or False
    '''
    if os.access(path, os.F_OK) and os.access(path, os.W_OK):
    # The path exists and is writeable
        return True
    if os.access(path, os.F_OK) and not os.access(path, os.W_OK):
    # The path exists and is not writeable
        return False
    # The path does not exists or is not writeable
    if check_parent is False:
    # We're not allowed to check the parent directory of the provided path
        return False
    # Lets get the parent directory of the provided path
    parent_dir = os.path.dirname(path)
    if not os.access(parent_dir, os.F_OK):
    # Parent directory does not exit
        return False
    # Finally, return if we're allowed to write in the parent directory of the
    # provided path
    return os.access(parent_dir, os.W_OK)
def is_readable(path):
    '''
    Check if a given path is readable by the current user.
    :param path: The path to check
    :returns: True or False
    '''
    if os.access(path, os.F_OK) and os.access(path, os.R_OK):
    # The path exists and is readable
        return True
    # The path does not exist
    return False

