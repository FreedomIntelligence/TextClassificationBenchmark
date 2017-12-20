# Text Classification Benchmark
A Benchmark of Text Classification in PyTorch


## Motivation

We are trying to build a Benchmark for Text Classification including


>Many Text Classification  **DataSet**, including Sentiment/Topic Classfication, popular language(e.g. English and Chinese). Meanwhile, a basic word embedding is provided.

>Implment many popular and state-of-art **Models**, especially in deep neural network.

## Have done
We have done some dataset and models
### Dataset done
- IMDB
- SST 
- Trec

### Models done
- BasicCNN
- KimCNN
- MultiLayerCNN
- InceptionCNN
- LSTM
- FastText


## Libary

You should have install [these librarys](docs/windows_torch.md)
<pre>
python3
torch
torchtext
</pre>

## Dataset 
Dataset will be automatically configured in current path, or download manually your data in [Dataset](docs/data_config.md),  step-by step.

including
<pre>
Glove embeding
Sentiment classfication dataset IMDB
</pre>
## How to run

Run in default  setting
<pre><code>python main.py</code></pre>

CNN 
<pre><code>python main.py -model cnn</code></pre>

LSTM
<pre><code>python main.py -model lstm</code></pre>

## Contributor
-	[@Allenzhai](https://github.com/zhaizheng)
-	[@JaredWei](https://github.com/jacobwei)
-	[@AlexMeng](https://github.com/EdwardLorenz)
-	[@Lilianwang](https://github.com/WangLilian)
-	[@ZhanSu](https://github.com/shuishen112)
-	[@Wabywang](https://github.com/Wabyking)

Welcome your issues and contribution!!!
