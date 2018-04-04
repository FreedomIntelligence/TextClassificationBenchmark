# -*- coding: utf-8 -*-


from .imdb import IMDBDataset
from .mr import MRDataset
from .glove import Glove
from .sst import SSTDataset
from .ag import AGDataset

from .Dataset import Dataset
def getDataset(opt):
    if opt.dataset=="imdb":
        dataset = IMDBDataset(opt)
    elif opt.dataset=="mr":
        dataset = MRDataset(opt)
    elif opt.dataset=="sst":
        dataset =SSTDataset(opt)
    elif opt.dataset == "ag":
        dataset =AGDataset(opt)
    elif opt.dataset in ["cr","mpqa","mr","sst1","sst2","subj","trec"]:
        dataset =Dataset(opt)


    else:
        raise Exception("dataset not supported: {}".format(opt.dataset))
    return dataset

def getEmbedding(opt):
    if opt.embedding_file.startswith("glove"):
        assert len(opt.embedding_file.split(".")) ==3 , "embedding_type format wrong"
        _,corpus,dim=opt.embedding_file.split(".")
        return Glove(corpus,dim,opt)
    else:
        raise Exception("embedding not supported: {}".format(opt.embedding_type))

