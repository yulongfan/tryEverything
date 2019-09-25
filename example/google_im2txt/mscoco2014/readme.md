## mscoco2014


### google im2txt in mscoco
INFO:tensorflow:Perplexity = 9.109227 (28 sec)
INFO:tensorflow:Perplexity = 9.109629 (33 sec)
INFO:tensorflow:Perplexity = 9.109629 (28 sec)


## google NIC 测评结果
```
mscoco2014 val:
computing Bleu score...
{'reflen': 37856, 'guess': [37878, 33827, 29776, 25725], 'testlen': 37878, 'correct': [25997, 12834, 6019, 2958]}
Bleu_1: 0.686
Bleu_2: 0.510
Bleu_3: 0.375
Bleu_4: 0.279
METEOR: 0.236
ROUGE_L: 0.511
CIDEr: 0.864
SPICE: 0.164
```
```
oxford_102flowers:
computing Bleu score...
{'reflen': 9393, 'guess': [9422, 8603, 7784, 6965], 'testlen': 9422, 'correct': [8868, 7245, 5679, 4612]}
```

```
cub200
computing Bleu score...
{'reflen': 14362, 'guess': [14392, 13201, 12010, 10819], 'testlen': 14392, 'correct': [13301, 9619, 6678, 4592]}
ratio: 1.00208884556
Bleu_1: 0.924
Bleu_2: 0.821
Bleu_3: 0.721
Bleu_4: 0.631
METEOR: 0.382
ROUGE_L: 0.734
CIDEr: 0.701
SPICE: 0.166
```

```
flickr8k
computing Bleu score...
{'reflen': 9371, 'guess': [9360, 8360, 7360, 6360], 'testlen': 9360, 'correct': [5896, 2597, 1079, 428]}
ratio: 0.998826165831
Bleu_1: 0.629
Bleu_2: 0.442
Bleu_3: 0.306
Bleu_4: 0.209
METEOR: 0.194
ROUGE_L: 0.458
CIDEr: 0.502
SPICE: 0.138
```

```
flickr30k
computing Bleu score...
{'reflen': 10571, 'guess': [10751, 9751, 8751, 7751], 'testlen': 10751, 'correct': [6622, 2924, 1213, 504]}
ratio: 1.01702771734
Bleu_1: 0.616
Bleu_2: 0.430
Bleu_3: 0.295
Bleu_4: 0.202
METEOR: 0.177
ROUGE_L: 0.436
CIDEr: 0.394
SPICE: 0.117
```



15w步在mscoco测试集上的评分结果:

index/model     |   Bleu_1  |   Bleu_2  |   Bleu_3  |   Bleu_4  |   CIDEr   |   ROUGE_L |   METEOR  |   SPICE   |   Perplexity
---             |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---
tsmc_im2txt     |   0.664   |   0.480   |   0.340   |   0.244   |   0.783   |   0.490   |   0.224   |   0.151   |   9.666
NIC             |   0.686   |   0.510   |   0.375   |   0.279   |   0.864   |   0.511   |   0.236   |   0.164   |   9.109
ablation_exp    |   0.421   |   0.217   |   0.122   |   0.073   |   0.138   |   0.340   |   0.120   |   0.052   |   12.463


2w步在Oxford_102flowers测试集上的评分结果:
index/model     |   Bleu_1  |   Bleu_2  |   Bleu_3  |   Bleu_4  |   CIDEr   |   ROUGE_L |   METEOR  |   SPICE   |   Perplexity
---             |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---
tsmc_im2txt     |   0.934   |   0.870   |   0.800   |   0.743   |   0.718   |   0.819   |   0.451   |   0.211   |   ....
NIC             |   0.941   |   0.890   |   0.833   |   0.787   |   0.805   |   0.848   |   0.482   |   0.215   |   ....
ablation_exp    |   0.894   |   0.817   |   0.734   |   0.671   |   0.426   |   0.771   |   0.392   |   0.203   |   ....


3w步在cub_200_2011测试集上的评分结果:
index/model     |   Bleu_1  |   Bleu_2  |   Bleu_3  |   Bleu_4  |   CIDEr   |   ROUGE_L |   METEOR  |   SPICE   |   Perplexity
---             |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---
tsmc_im2txt     |   0.914   |   0.814   |   0.715   |   0.622   |   0.642   |   0.726   |   0.375   |   0.164   |   ....
NIC             |   0.924   |   0.821   |   0.721   |   0.631   |   0.701   |   0.734   |   0.382   |   0.166   |   ....
ablation_exp    |   0.914   |   0.814   |   0.718   |   0.627   |   0.613   |   0.728   |   0.374   |   0.167   |   ....


1w步在flickr8k测试集上的评分结果:
index/model     |   Bleu_1  |   Bleu_2  |   Bleu_3  |   Bleu_4  |   CIDEr   |   ROUGE_L |   METEOR  |   SPICE   |   Perplexity
---             |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---
tsmc_im2txt     |   0.570   |   0.381   |   0.248   |   0.159   |   0.434   |   0.425   |   0.181   |   0.130   |   ....
NIC             |   0.629   |   0.442   |   0.306   |   0.209   |   0.502   |   0.458   |   0.194   |   0.138   |   ....
ablation_exp    |   0.556   |   0.365   |   0.234   |   0.151   |   0.399   |   0.414   |   0.173   |   0.118   |   ....


4w步在flickr30k测试集上的评分结果:
index/model     |   Bleu_1  |   Bleu_2  |   Bleu_3  |   Bleu_4  |   CIDEr   |   ROUGE_L |   METEOR  |   SPICE   |   Perplexity
---             |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---     |   ---
tsmc_im2txt     |   0.606   |   0.412   |   0.273   |   0.182   |   0.364   |   0.418   |   0.169   |   0.107   |   ....
NIC             |   0.616   |   0.430   |   0.295   |   0.202   |   0.394   |   0.436   |   0.177   |   0.117   |   ....
ablation_exp    |   0.547   |   0.352   |   0.223   |   0.146   |   0.276   |   0.394   |   0.154   |   0.091   |   ....

