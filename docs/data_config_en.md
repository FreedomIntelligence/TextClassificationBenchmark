# Data configuration

**Install [torchtext](https://github.com/pytorch/text) for data processing**

The datasets module currently contains:

- Sentiment analysis: SST and IMDb
- Question classification: TREC
- Entailment: SNLI
- Language modeling: WikiText-2
- Machine translation: Multi30k, IWSLT, WMT14

Others are planned or a work in progress:

- Question answering: SQuAD

The current need to configure the data collection

### Glove 

Download to the project's root directory under the folder vector_cache

- [42B](http://nlp.stanford.edu/data/glove.42B.300d.zip)
- [840B](http://nlp.stanford.edu/data/glove.840B.300d.zip)
- [twitter.27B](http://nlp.stanford.edu/data/glove.twitter.27B.zip)
- [6B](http://nlp.stanford.edu/data/glove.6B.zip)

### Classification Datasets

- Download [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) dataset to .data/imdb
- Download [SST](http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip) dataset to .data/sst
- Download TREC [Question Classification ](http://cogcomp.org/Data/QA/QC/train_5500.label) [2](http://cogcomp.org/Data/QA/QC/TREC_10.label) dataset to .data/imdb

### File Structure

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

	

## More datasets and updates coming soon, please wait for us to update further
