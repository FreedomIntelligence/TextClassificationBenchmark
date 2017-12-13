# TextClassificationBenchmark
A Benchmark of Text Classification in PyTorch

我们这个项目的主要目标是实现一些文本的baseline，主要从两个方面来做

>1.收集一些主要的文本分类的数据集，中文和英文，最好还能够提供一个基础的embedding向量

>2.实现一些state-of-art的文本分类模型，包括基础的机器学习方法，朴素贝叶斯+TFIDF和一些基于CNN/RNN的文本分类方法


在这样一个benchmark上做一些基础方法的比较

首先你可能需要安装一些基础的库 [安装库](docs/windows_torch.md)
<pre>
python3
torch
torchtext
</pre>

第二你可能需要把数据配置好，[数据配置](docs/data_config.md)
包括
<pre>
Glove词向量
情感文本分类数据集IMDB
</pre>
跑默认配置
<pre><code>python main.py</code></pre>

CNN 
<pre><code>python main.py -model cnn</code></pre>

LSTM
<pre><code>python main.py -model lstm</code></pre>

###Contributor
-	[@Allenzhai](https://github.com/zhaizheng)
-	[@JareWei](https://github.com/jacobwei)
-	[@AlexMeng](https://github.com/EdwardLorenz)
-	[@Lilianwang](https://github.com/WangLilian)
-	[@Wabywang](https://github.com/Wabyking)

Welcome your issues and contribution!!!
