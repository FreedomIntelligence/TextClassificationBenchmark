# -*- coding: utf-8 -*-

import os
import numpy as np
import string
from collections import Counter
import pandas as pd
from tqdm import tqdm
import random
import time
import pickle
from utils import log_time_delta
from tqdm import tqdm
from dataloader import Dataset

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
        with open(os.path.join(path,fname), "w") as out:
            for k in sorted(self.keys()):
                out.write("{}\t{}\n".format(k, self[k]))

class BucketIterator(object):
    def __init__(self,data,opt=None,batch_size=2,shuffle=True):
        self.shuffle=shuffle
        self.data=data
        self.batch_size=batch_size
        if opt is not None:
            self.setup(opt)
    def setup(self,opt):
        self.data=opt.data
        self.batch_size=opt.batch_size
        self.shuffle=opt.__dict__.get("shuffle",self.shuffle)
    def __iter__(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        batch_nums = int(len(self.data)/self.batch_size)
        for  i in range(batch_nums):
            yield self.data[i*self.batch_size:(i+1)*self.batch_size]
        yield self.data[-1*self.batch_size:]
        
                
@log_time_delta
def getSubVectors(vectors,vocab,dim):
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

def getEmbeddingFile(name):
    #"glove"  "w2v"
    
    return "D:\dataset\glove\glove.6B.300d.txt"

def getDataSet(dataset):
    
    data_dir = ".data/clean/demo"
    files=[os.path.join(data_dir,data_name)   for data_name in ['train.txt','test.txt','dev.txt']]

        
    return files
    

def loadData(opt):
    datas = []
   
    alphabet = Alphabet(start_feature_id = 0)
    label_alphabet= Alphabet(start_feature_id = 0,alphabet_type="label")   
    for filename in getDataSet(opt.dataset):
        df = pd.read_csv(filename,header = None,sep="\t",names=["text","label"]).fillna('0')
        df["text"]= df["text"].str.lower().str.split()
        datas.append(df)
        
    df=pd.concat(datas)
    
    from functools import reduce 
    word_set=reduce(lambda x,y : set(x)|set(y),df["text"])
    alphabet.addAll(word_set)
    label_set = set(df["label"])
    label_alphabet.addAll(label_set)
    
    if opt.max_seq_len==-1:
        opt.max_seq_len = df.apply(lambda row: row["text"].__len__(),axis=1).max()
    
    for data in datas:
        data["text"]= data["text"].apply(lambda text: [alphabet.get(word,alphabet.unknow_token)  for word in text] + [alphabet.padding_token] *int(opt.max_seq_len-len(text)) )
        data["label"]=data["label"].apply(lambda text: label_alphabet.get(text))

        
    glove_file = getEmbeddingFile(opt.__dict__.get("embedding","glove_6b_300"))
    loaded_vectors,embedding_size = load_text_vec(alphabet,glove_file)
    vocab = [v for k,v in alphabet.items()]
    vectors = getSubVectors(loaded_vectors,vocab,embedding_size)
    
    opt.label_size= len(alphabet)    
    opt.vocab_size = len(label_alphabet)
    opt.embedding_dim= embedding_size
    opt.embeddings = vectors
   
    alphabet.dump(opt.dataset+".alphabet")              
    return map(BucketIterator,datas)  #map(lambda x:BucketIterator(x),datas)
    

if __name__ =="__main__":
    import opts
    opt = opts.parse_opt()
    opt.max_seq_len=-1
    import dataloader
    dataset= dataloader.getDataset(opt)
#    datas=loadData(opt)
    

