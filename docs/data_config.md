# 数据配置


##第一步先支持[torchtext](https://github.com/pytorch/text)本来支持的数据集合


The datasets module currently contains:

- Sentiment analysis: SST and IMDb
- Question classification: TREC
- Entailment: SNLI
- Language modeling: WikiText-2
- Machine translation: Multi30k, IWSLT, WMT14

Others are planned or a work in progress:

- Question answering: SQuAD

目前需要配置的数据集合

###Glove的下载到项目的根目录 ..vector_cache文件夹下

- [42B](http://nlp.stanford.edu/data/glove.42B.300d.zip)
- [840B](http://nlp.stanford.edu/data/glove.840B.300d.zip)
- [twitter.27B](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
- [6B](http://nlp.stanford.edu/data/glove.6B.zip)

###分类数据集下载配置

- [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)数据集下载到 .data/imdb
- [SST](http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip)数据集下载到.data/sst
- TREC [1](http://cogcomp.org/Data/QA/QC/train_5500.label) [2](http://cogcomp.org/Data/QA/QC/TREC_10.label) 问题分类数据集下载到.data/imdb

###文件结构示例如下

- TextClassificationBenchmark
	- .data
		- imdb
			- aclImdb_v1.tar.gz
		- sst
			- trainDevTestTrees_PTB.zip
		- trec
			- train_5500.label
			- TREC_10.label
	- .vector_cache
		- glove.42B.300d.zip
		- glove.840B.300d.zip
		- glove.twitter.27B.zip
		- glove.6B.zip

	

##更多的数据集请等待我们进一步更新