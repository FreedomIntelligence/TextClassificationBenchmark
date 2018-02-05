from .Dataset import Dataset
import os
import pandas as pd
from codecs import open

class IMDBDataset(Dataset):
    def __init__(self,opt=None,**kwargs):
        super(IMDBDataset,self).__init__(opt,**kwargs)
        self.urls=['http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz']
        
    
    def process(self):
        
        root=self.download()
        root = os.path.join(root,"aclImdb")
        print("processing into: "+ root)
#        root = "D:\code\git\TextClassificationBenchmark\.data_waby\\imdb\\aclImdb"
        if not os.path.exists(self.saved_path):
            print("mkdir " + self.saved_path)
            os.makedirs(self.saved_path) # better than os.mkdir
            
        datafiles=[]
        
        for data_folder in  ("train","test"):
            data = []  
            for polarity in ("pos","neg"):
                diranme=os.path.join( os.path.join(root,data_folder), polarity)
                for rt, dirs, files in os.walk(diranme):
                    for f in files:
                        filename= os.path.join(rt,f)
                        data.append( {"text": open(filename,encoding="utf-8").read().strip(),"label":int(polarity=="pos")})
            df=pd.DataFrame(data)
            saved_filename=os.path.join(self.saved_path,data_folder+".csv")
            
            df[["text","label"]].to_csv(saved_filename,index=False,header=None,sep="\t",encoding="utf-8")
            print("finished %s"%saved_filename)
            datafiles.append(saved_filename)
        print("processing into formated files over")
        
        
        return datafiles
    
                                
                    
