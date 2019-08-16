# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from six.moves import cPickle
import time,os,random
import itertools

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import NLLLoss,MultiLabelSoftMarginLoss,MultiLabelMarginLoss,BCELoss

import opts
import models
import utils


timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time()) ))
performance_log_file =  os.path.join("log","result"+timeStamp+ ".csv") 
if not os.path.exists(performance_log_file):
    with open(performance_log_file,"w") as f:
        f.write("argument\n")
        f.close() 
      
        
def train(opt,train_iter, test_iter,verbose=True):
    global_start= time.time()
    logger = utils.getLogger()
    model=models.setup(opt)
    if torch.cuda.is_available():
        model.cuda()
    params = [param for param in model.parameters() if param.requires_grad] #filter(lambda p: p.requires_grad, model.parameters())
    
    model_info =";".join( [str(k)+":"+ str(v)  for k,v in opt.__dict__.items() if type(v) in (str,int,float,list,bool)])  
    logger.info("# parameters:" + str(sum(param.numel() for param in params)))
    logger.info(model_info)
    
    
    model.train()
    optimizer = utils.getOptimizer(params,name=opt.optimizer, lr=opt.learning_rate,scheduler= utils.get_lr_scheduler(opt.lr_scheduler))
    optimizer.zero_grad()
    loss_fun = F.cross_entropy

    filename = None
    percisions=[]
    for i in range(opt.max_epoch):
        for epoch,batch in enumerate(train_iter):
            start= time.time()
            
            text = batch.text[0] if opt.from_torchtext else batch.text
            predicted = model(text)
    
            loss= loss_fun(predicted,batch.label)
    
            loss.backward()
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            
            if verbose:
                if  torch.cuda.is_available():
                    logger.info("%d iteration %d epoch with loss : %.5f in %.4f seconds" % (i,epoch,loss.cpu().data.numpy(),time.time()-start))
                else:
                    logger.info("%d iteration %d epoch with loss : %.5f in %.4f seconds" % (i,epoch,loss.data.numpy()[0],time.time()-start))
 
        percision=utils.evaluation(model,test_iter,opt.from_torchtext)
        if verbose:
            logger.info("%d iteration with percision %.4f" % (i,percision))
        if len(percisions)==0 or percision > max(percisions):
            if filename:
                os.remove(filename)
            filename = model.save(metric=percision)
        percisions.append(percision)
            
#    while(utils.is_writeable(performance_log_file)):
    df = pd.read_csv(performance_log_file,index_col=0,sep="\t")
    df.loc[model_info,opt.dataset] =  max(percisions) 
    df.to_csv(performance_log_file,sep="\t")    
    logger.info(model_info +" with time :"+ str( time.time()-global_start)+" ->" +str( max(percisions) ) )
    print(model_info +" with time :"+ str( time.time()-global_start)+" ->" +str( max(percisions) ) )

        
if __name__=="__main__": 
    parameter_pools = utils.parse_grid_parameters("config/grid_search_cnn.ini")
    
#    parameter_pools={
#            "model":["lstm","cnn","fasttext"],
#            "keep_dropout":[0.8,0.9,1.0],
#            "batch_size":[32,64,128],
#            "learning_rate":[100,10,1,1e-1,1e-2,1e-3],
#            "optimizer":["adam"],
#            "lr_scheduler":[None]            
#                        }    
    opt = opts.parse_opt()
    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] =opt.gpu
    train_iter, test_iter = utils.loadData(opt)
#    if from_torchtext:
#        train_iter, test_iter = utils.loadData(opt)
#    else:
#        import dataHelper 
#        train_iter, test_iter = dataHelper.loadData(opt)
    if False:
        model=models.setup(opt)
        print(opt.model)
        if torch.cuda.is_available():
            model.cuda()
        train(opt,train_iter, test_iter)
    else:
        
        pool =[ arg for arg in itertools.product(*parameter_pools.values())]
        random.shuffle(pool)
        args=[arg for i,arg in enumerate(pool) if i%opt.gpu_num==opt.gpu]
        
        for arg in args:
            olddataset = opt.dataset
            for k,v in zip(parameter_pools.keys(),arg):
                opt.__setattr__(k,v)
            if "dataset" in parameter_pools and olddataset != opt.dataset:
                train_iter, test_iter = utils.loadData(opt)
            train(opt,train_iter, test_iter,verbose=False)
   