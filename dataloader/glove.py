# -*- coding: utf-8 -*-
import os

from .Dataset import Dataset
class Glove(Dataset):
    def __init__(self,corpus,dim,opt=None,**kwargs):
        super(Glove,self).__init__(opt,**kwargs)

        self.root = ".vector_cache"
       
#        if not os.path.exists(self.root):
#            os.makedirs(self.root)
            
        embeding_urls = {
                '42b': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
                '840b': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
                'twitter.27b': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                '6b': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }
        

        self.urls= [ embeding_urls[corpus.lower()] ]
        print(self.urls)
        self.name = corpus

    
    def process(self):
       
        root=self.download()
        
        return root
    def getFilename(self):
        return self.process()

if __name__ =="__main__":
    import opts
    opt = opts.parse_opt()
    
               
    import dataloader
    glove=dataloader.getEmbedding(opt)
    print(glove.getFilename())

    