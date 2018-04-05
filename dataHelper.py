# -*- coding: utf-8 -*-

import os
import numpy as np
import string
from collections import Counter
import pandas as pd
from tqdm import tqdm
import random
import time
from utils import log_time_delta
from tqdm import tqdm
from dataloader import Dataset
import torch
from torch.autograd import Variable
from codecs import open
try:
    import cPickle as pickle
except ImportError:
    import pickle
class Alphabet(dict):
    def __init__(self, start_feature_id = 1, alphabet_type="text"):
        self.fid = start_feature_id
        if alphabet_type=="text":
            self.add('[PADDING]')
            self.add('[UNK]')
            self.add('[END]')
            self.unknow_token = self.get('[UNK]')
            self.end_token = self.get('[END]')
            self.padding_token = self.get('[PADDING]')

    def add(self, item):
        idx = self.get(item, None)
        if idx is None:
            idx = self.fid
            self[item] = idx
      # self[idx] = item
            self.fid += 1
        return idx
    
    def addAll(self,words):
        for word in words:
            self.add(word)
            
    def dump(self, fname,path="temp"):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path,fname), "w",encoding="utf-8") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))

class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
        self.allowDotting()
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()
            
class BucketIterator(object):
    def __init__(self,data,opt=None,batch_size=2,shuffle=True,test=False,position=False):
        self.shuffle=shuffle
        self.data=data
        self.batch_size=batch_size
        self.test=test        
        if opt is not None:
            self.setup(opt)
    def setup(self,opt):
        
        self.batch_size=opt.batch_size
        self.shuffle=opt.__dict__.get("shuffle",self.shuffle)
        self.position=opt.__dict__.get("position",False)
        self.padding_token =  opt.alphabet.padding_token
    
    def transform(self,data):
        if torch.cuda.is_available():
            data=data.reset_index()
            text= Variable(torch.LongTensor(data.text).cuda())
            label= Variable(torch.LongTensor([int(i) for i in data.label.tolist()]).cuda())                
        else:
            data=data.reset_index()
            text= Variable(torch.LongTensor(data.text))
            label= Variable(torch.LongTensor(data.label.tolist()))
        if self.position:
            position_tensor = self.get_position(data.text)
            return DottableDict({"text":(text,position_tensor),"label":label})
        return DottableDict({"text":text,"label":label})
    
    def get_position(self,inst_data):
        inst_position = np.array([[pos_i+1 if w_i != self.padding_token else 0 for pos_i, w_i in enumerate(inst)] for inst in inst_data])
        inst_position_tensor = Variable( torch.LongTensor(inst_position), volatile=self.test) 
        if torch.cuda.is_available():
            inst_position_tensor=inst_position_tensor.cuda()
        return inst_position_tensor

    def __iter__(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        batch_nums = int(len(self.data)/self.batch_size)
        for  i in range(batch_nums):
            yield self.transform(self.data[i*self.batch_size:(i+1)*self.batch_size])
        yield self.transform(self.data[-1*self.batch_size:])
    

        
                
@log_time_delta
def vectors_lookup(vectors,vocab,dim):
    embedding = np.zeros((len(vocab),dim))
    count = 1
    for word in vocab:
        if word in vectors:
            count += 1
            embedding[vocab[word]]= vectors[word]
        else:
            embedding[vocab[word]]= np.random.uniform(-0.5,+0.5,dim)#vectors['[UNKNOW]'] #.tolist()
    print( 'word in embedding',count)
    return embedding

@log_time_delta
def load_text_vec(alphabet,filename="",embedding_size=-1):
    vectors = {}
    with open(filename,encoding='utf-8') as f:
        for line in tqdm(f):
            items = line.strip().split(' ')
            if len(items) == 2:
                vocab_size, embedding_size= items[0],items[1]
                print( 'embedding_size',embedding_size)
                print( 'vocab_size in pretrained embedding',vocab_size)                
            else:
                word = items[0]
                if word in alphabet:
                    vectors[word] = items[1:]
    print( 'words need to be found ',len(alphabet))
    print( 'words found in wor2vec embedding ',len(vectors.keys()))
    
    if embedding_size==-1:
        embedding_size = len(vectors[list(vectors.keys())[0]])
    return vectors,embedding_size

def getEmbeddingFile(opt):
    #"glove"  "w2v"
    embedding_name = opt.__dict__.get("embedding","glove_6b_300")
    if embedding_name.startswith("glove"):
        return os.path.join( ".vector_cache","glove.6B.300d.txt")
    else:
        return opt.embedding_dir
    # please refer to   https://pypi.python.org/pypi/torchwordemb/0.0.7
    return 
@log_time_delta
def getSubVectors(opt,alphabet):
    pickle_filename = "temp/"+opt.dataset+".vec"
    if not os.path.exists(pickle_filename) or opt.debug:    
        glove_file = getEmbeddingFile(opt)
        wordset= set(alphabet.keys())   # python 2.7
        loaded_vectors,embedding_size = load_text_vec(wordset,glove_file) 
        
        vectors = vectors_lookup(loaded_vectors,alphabet,embedding_size)
        if opt.debug:
            if not os.path.exists("temp"):
                os.mkdir("temp")
            with open("temp/oov.txt","w","utf-8") as f:
                unknown_set = set(alphabet.keys()) - set(loaded_vectors.keys())
                f.write("\n".join( unknown_set))
        if  opt.debug:
            pickle.dump(vectors,open(pickle_filename,"wb"))
        return vectors
    else:
        print("load cache for SubVector")
        return pickle.load(open(pickle_filename,"rb"))
    
def getDataSet(opt):
    import dataloader
    dataset= dataloader.getDataset(opt)
#    files=[os.path.join(data_dir,data_name)   for data_name in ['train.txt','test.txt','dev.txt']]
    
    return dataset.getFormatedData()
    
    #data_dir = os.path.join(".data/clean",opt.dataset)
    #if not os.path.exists(data_dir):
    #     import dataloader
    #     dataset= dataloader.getDataset(opt)
    #     return dataset.getFormatedData()
    #else:
    #     for root, dirs, files in os.walk(data_dir):
    #         for file in files:
    #             yield os.path.join(root,file)
         
    
#    files=[os.path.join(data_dir,data_name)   for data_name in ['train.txt','test.txt','dev.txt']]
    
import re
def clean(text):
#    text="'tycoon.<br'"
    for token in ["<br/>","<br>","<br"]:
         text = re.sub(token," ",text)
    text = re.sub("[\s+\.\!\/_,$%^*()\(\)<>+\"\[\]\-\?;:\'{}`]+|[+——！，。？、~@#￥%……&*（）]+", " ",text)

#    print("%s $$$$$ %s" %(pre,text))     

    return text.lower().split()
@log_time_delta
def get_clean_datas(opt):
    pickle_filename = "temp/"+opt.dataset+".data"
    if not os.path.exists(pickle_filename) or opt.debug: 
        datas = [] 
        for filename in getDataSet(opt):
            df = pd.read_csv(filename,header = None,sep="\t",names=["text","label"]).fillna('0')
    
        #        df["text"]= df["text"].apply(clean).str.lower().str.split() #replace("[\",:#]"," ")
            df["text"]= df["text"].apply(clean)
            datas.append(df)
        if  opt.debug:
            if not os.path.exists("temp"):
                os.mkdir("temp")
            pickle.dump(datas,open(pickle_filename,"wb"))
        return datas
    else:
        print("load cache for data")
        return pickle.load(open(pickle_filename,"rb"))
    

def loadData(opt,embedding=True):
    if embedding==False:
        return loadDataWithoutEmbedding(opt)
    
    datas =get_clean_datas(opt)
    
    alphabet = Alphabet(start_feature_id = 0)
    label_alphabet= Alphabet(start_feature_id = 0,alphabet_type="label") 

    
    df=pd.concat(datas)   
    df.to_csv("demo.text",sep="\t",index=False)
    label_set = set(df["label"])
    label_alphabet.addAll(label_set)
    
    word_set=set()
    [word_set.add(word)  for l in df["text"] if l is not None for word in l ]
#    from functools import reduce
#    word_set=set(reduce(lambda x,y :x+y,df["text"]))            
   
    alphabet.addAll(word_set)

    vectors = getSubVectors(opt,alphabet)  
    
    if opt.max_seq_len==-1:
        opt.max_seq_len = df.apply(lambda row: row["text"].__len__(),axis=1).max()
    opt.vocab_size= len(alphabet)    
    opt.label_size= len(label_alphabet)
    opt.embedding_dim= vectors.shape[-1]
    opt.embeddings = torch.FloatTensor(vectors)
    opt.alphabet=alphabet
#    alphabet.dump(opt.dataset+".alphabet")     
    for data in datas:
        data["text"]= data["text"].apply(lambda text: [alphabet.get(word,alphabet.unknow_token)  for word in text[:opt.max_seq_len]] + [alphabet.padding_token] *int(opt.max_seq_len-len(text)) )
        data["label"]=data["label"].apply(lambda text: label_alphabet.get(text)) 
        
    return map(lambda x:BucketIterator(x,opt),datas)#map(BucketIterator,datas)  #

def loadDataWithoutEmbedding(opt):
    datas=[]
    for filename in getDataSet(opt):
        df = pd.read_csv(filename,header = None,sep="\t",names=["text","label"]).fillna('0')
        df["text"]= df["text"].str.lower()
        datas.append((df["text"],df["label"]))
    return datas
    


    

if __name__ =="__main__":
    import opts
    opt = opts.parse_opt()
    opt.max_seq_len=-1
    import dataloader
    dataset= dataloader.getDataset(opt)
    datas=loadData(opt)
    

