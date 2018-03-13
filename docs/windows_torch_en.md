# Windows Platform Installation for PyTorch

If Linux, Mac directly use pytorch from [homepage](http://pytorch.org/), and reinstall TorchText

## Python installation
Please install anaconda directly: [installation package](https://repo.continuum.io/archive/Anaconda3-5.0.1-Windows-x86_64.exe)

## Pytorch installation
In[Baidu Network Disk](https://pan.baidu.com/s/1dF6ayLr#list/path=%2Fpytorch) download offline, Version 0.3 or 0.2 wheels
<pre><code>pip install torch0.3XXX.whl</code></pre>

If it is a conda installation environment
<pre><code>conda install --offline  torch0.3XXX.tar.bz</code></pre>

## TorchText installation

The assumption is that you have git and pip, if you don't, you need to download git and put it in the Path environment variable.
<pre><code>pip install git+https://github.com/pytorch/text.git </code></pre>

If you need a proxy, 
<pre><code>pip install git+https://github.com/pytorch/text.git --proxy proxy.xx.com:8080 </code></pre>


Reference Link:
https://zhuanlan.zhihu.com/p/31747695
