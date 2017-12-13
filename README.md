# TextClassificationBenchmark
A Benchmark of Text Classification in PyTorch

我们这个项目的主要目标是实现一些文本的baseline，主要从两个方面来做

>1.收集一些主要的文本分类的数据集，中文和英文，最好还能够提供一个基础的embedding向量

>2.实现一些state-of-art的文本分类模型，包括基础的机器学习方法，朴素贝叶斯+TFIDF和一些基于CNN/RNN的文本分类方法


在这样一个benchmark上做一些基础方法的比较

跑默认配置
<pre><code>python main.py</code></pre>

CNN 
<pre><code>python main.py -model cnn</code></pre>

LSTM
<pre><code>python main.py -model lstm</code></pre>
