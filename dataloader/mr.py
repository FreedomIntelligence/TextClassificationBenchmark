# -*- coding: utf-8 -*-

from .Dataset import Dataset
import os
import pandas as pd
import numpy as np
from codecs import open

class MRDataset(Dataset):
    def __init__(self,opt=None,**kwargs):
        super(MRDataset,self).__init__(opt,**kwargs)
        self.urls=['https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz']
        
    
    def process(self):
        
        root=self.download()
        root = os.path.join(root,"rt-polaritydata")
        print("processing into: "+ root)
##        root = "D:\code\git\TextClassificationBenchmark\.data_waby\\imdb\\aclImdb"
        if not os.path.exists(self.saved_path):
            print("mkdir " + self.saved_path)
            os.makedirs(self.saved_path) # better than os.mkdir
#            
        datas=[]
        for polarity in  ("neg","pos"):
            filename = os.path.join(root,"rt-polarity."+polarity) 
            records=[]
            with open(filename,encoding="utf-8",errors="replace") as f:
                for i,line in enumerate(f):
                    print(i)
                    print(line)
                    records.append({"text":line.strip(),"label": 1 if polarity == "pos" else 0})
            datas.append(pd.DataFrame(records))
        
            
           
        df = pd.concat(datas)
        from sklearn.utils import shuffle  
        df = shuffle(df).reset_index()
        
        split_index = [True] * int (len(df) *0.8) + [False] *(len(df)-int (len(df) *0.8))
#        train=df.sample(frac=0.8)
        train = df[split_index]
        test = df[~np.array(split_index)]
                     
        train_filename=os.path.join(self.saved_path,"train.csv")
        test_filename = os.path.join(self.saved_path,"test.csv")
        train[["text","label"]].to_csv(train_filename,encoding="utf-8",sep="\t",index=False,header=None)
        test[["text","label"]].to_csv(test_filename,encoding="utf-8",sep="\t",index=False,header=None)
            
        
#        
#        for data_folder in  ("train","test"):
#            data = []  
#            for polarity in ("pos","neg"):
#                diranme=os.path.join( os.path.join(root,data_folder), polarity)
#                for rt, dirs, files in os.walk(diranme):
#                    for f in files:
#                        filename= os.path.join(rt,f)
#                        data.append( {"text": open(filename,encoding="utf-8").read().strip(),"label":int(polarity=="pos")})
#            df=pd.DataFrame(data)
#            saved_filename=os.path.join(self.saved_path,data_folder+".csv")
#            
#            df[["text","label"]].to_csv(saved_filename,index=False,header=None,sep="\t",encoding="utf-8")
#            print("finished %s"%saved_filename)
        print("processing into formated files over")        
        
        return [train_filename,test_filename]

if __name__=="__main__":
    import opts
    opt = opts.parse_opt()
    opt.dataset="mr"
    import dataloader
    dataset= dataloader.getDataset(opt)
    dataset.process()
    

    