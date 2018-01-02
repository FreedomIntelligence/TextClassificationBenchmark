from .Dataset import Dataset

class IMDBDataset(Dataset):
    def __init__(self,opt=None,**kwargs):
        super(IMDBDataset,self).__init__(opt,**kwargs)
        self.urls=['http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz']
    
    
    def process(self):
        dirname=self.download()
        print("processing dirname: "+ dirname)        
        return dirname
        