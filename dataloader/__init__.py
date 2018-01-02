# -*- coding: utf-8 -*-


from .IMDB import IMDBDataset


def getDataset(opt):
    if opt.dataset=="imdb":
        dataset = IMDBDataset(opt)
        
    else:
        raise Exception("dataset not supported: {}".format(opt.dataset))
    return dataset

