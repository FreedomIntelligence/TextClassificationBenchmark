# Windows 平台安装 PyTorch

如果是Linux，Mac安装直接移步pytorch[主页](http://pytorch.org/), 再安装TorchText

## Python安装
建议直接安装anaconda的[安装包](https://repo.continuum.io/archive/Anaconda3-5.0.1-Windows-x86_64.exe)

## Pytorch安装
在[百度网盘](https://pan.baidu.com/s/1dF6ayLr#list/path=%2Fpytorch)下载一个 离线安装包 , 0.3版本或者是0.2版本均可
如果是whl安装包
<pre><code>pip install torch0.3XXX.whl</code></pre>
如果是一个conda安装包（压缩文件后缀）
<pre><code>conda install --offline  torch0.3XXX.tar.bz</code></pre>

## TorchText 安装

前提是有git和pip，如果没有需要下载git，并将其放到Path环境变量里
<pre><code>pip install git+https://github.com/pytorch/text.git </code></pre>

还需要有代理的话



<pre><code>pip install git+https://github.com/pytorch/text.git --proxy proxy.xx.com:8080 </code></pre>


参考链接
https://zhuanlan.zhihu.com/p/31747695
