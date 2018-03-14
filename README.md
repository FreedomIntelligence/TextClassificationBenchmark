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
- FastText
- BasicCNN (KimCNN,MultiLayerCNN, Multi-perspective CNN)
- InceptionCNN
- LSTM (BILSTM, StackLSTM)
- LSTM with Attention (Self Attention / Quantum Attention)
- Hybrids between CNN and RNN (RCNN, C-LSTM)
- Transformer - Attention is all you need
- ConS2S
- Capsule
- Quantum-inspired NN

## Libary

You should have install [these librarys](docs/windows_torch_en.md)
<pre>
python3
torch
torchtext (optional)
</pre>

## Dataset 
Dataset will be automatically configured in current path, or download manually your data in [Dataset](docs/data_config_en.md),  step-by step.

including
<pre>
Glove embeding
Sentiment classfication dataset IMDB
</pre>


## usage


Run in default  setting
<pre><code>python main.py</code></pre>

CNN 
<pre><code>python main.py --model cnn</code></pre>

LSTM
<pre><code>python main.py --model lstm</code></pre>

## Road Map
- [X] Data preprossing framework
- [X] Models modules
- [ ] Loss, Estimator and hyper-paramter tuning.
- [ ] Test modules
- [ ] More Dataset
- [ ] More models



## Organisation of the repository
The core of this repository is models and dataset.


* ```dataloader/```: loading all dataset such as ```IMDB```, ```SST```

* ```models/```: creating all models such as ```FastText```, ```LSTM```,```CNN```,```Capsule```,```QuantumCNN``` ,```Multi-Head Attention```

* ```opts.py```: Parameter and config info.

* ```utils.py```: tools.

* ```dataHelper```: data helper




## Contributor
-	[@Allenzhai](https://github.com/zhaizheng)
-	[@JaredWei](https://github.com/jacobwei)
-	[@AlexMeng](https://github.com/EdwardLorenz)
-	[@Lilianwang](https://github.com/WangLilian)
-	[@ZhanSu](https://github.com/shuishen112)
-	[@Wabywang](https://github.com/Wabyking)

Welcome your issues and contribution!!!

