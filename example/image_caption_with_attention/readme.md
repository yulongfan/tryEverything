### branch_r9.1
### 问题及任务内容

分支任务[2018/09/08]：

  * 训练与测试准确率极低，不到1%，需要核查数据流是否正确，GRU的dynamic loop实现逻辑是否有误
  * 兼顾完成注意力可视化工作
  * 快速验证差分进化算法的结合



### 实验部分备忘录：

  * 预训练图像描述模型，待基本收敛后，再执行差分进化算法
  * 注意GRU的隐状态初始化值是由annotation vectors的均值经由初始化函数(一个MLP)得到的，传入GRU时记得变为tuple
  * 注意预训练后再加差分进化算法时，学习率的设置(可能会设置到1e-4之下) ，尽可能参照文章方法
  * 注意目标函数，show attend and tell中是对注意力加了正则化的，另需要参考自适应注意力文章中空间注意力模型，他的效果会更好，
  时间允许则引入视觉哨兵，实现自适应注意力
  * 差分进化算法解空间的上下界定为[0,1],即Cost(i,j)值域为[0,1]
  * 思考清晰YOLO-9000分组方式的引入


### 关于gru初始化部分实现中问题解决方案

在外部用fetch抓取这个均值

```python
import tensorflow as tf
state = tf.reduce_mean(input_tensor=self.annotation_features, axis=1)  # ==> shape (batch, embedding_size)
print("[mean_annotation_features] mean_annotation_features: {}".format(state))

# # attention: state pass to gru is a tuple
state = tf.layers.dense(inputs=state, units=self.hidden_units, activation=tf.nn.relu, name="f_init")
state = (state,)
print("[init_state] init_state: {}".format(state))

tf.identity(state, name="init_annot_state")
```


### 注意力可视化中发现的问题

cpu与gpu执行测试绘制的注意力可视化效果不一样，cpu的成像是像素块，gpu比较光滑


### 截断过长的描述

实验对长度大于20的caption截断。实验发现，可能是因为受更少的`<pad>`影响，准确率提升明显几乎是2~3倍
```log
step 4150, Loss 1.626552, acc 0.212798
step 4160, Loss 1.494716, acc 0.211310
step 4170, Loss 1.470697, acc 0.203869
step 4180, Loss 1.667941, acc 0.205357
step 4190, Loss 1.436548, acc 0.205357
step 4200, Loss 1.570939, acc 0.261905
step 4200, Loss 1.570939, acc 0.261905
step 4210, Loss 1.804192, acc 0.206845
```
备注： 现在程序里的截断操作是在建好词表之后，所以，需要注意词表中有些词会在对序列采取截断后用不上，__但仍然占据词表位置__

```python
# we truncate captions longer than 20 words for COCO
truncate_train_seqs = []
for line in train_seqs:
    if len(line) > 22:
        line = line[:22]    # don't need to add '<end>' symbol
    truncate_train_seqs.append(line)
```

### 核查预训练模型输出的特征

20180913核查报告，通过对比特征矩阵是否相等，验证得知不相等，故认为特征输出正确，问题可能出在tf.while循环实现上