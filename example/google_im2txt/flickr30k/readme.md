##  flickr30k

### 数据集划分

不同于 __flickr8k__ 提供了标准划分文件, __flickr30k__ 没有提供标准划分,

这个数据集的的划分就按照google等随机抽成
 ``train : val : test = 29783 : 1000 : 1000`` 的比例随机划分.
 
### 处理过程代码验证信息
```
processing 158915 captions for 31783 images in /dataset/flickr30k/flickr30k_captions
Creating vocabulary...
Total words: 19829
words in vocabulary:  8411
```