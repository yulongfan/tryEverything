# -*- coding: utf-8 -*-
# @File    : tryEverything/generate_chinese_text.py
# @Info    : @ TSMC-SIGGRAPH, 2018/6/20
# @Desc    : only support python3, tensorflow >=1.4
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os
from collections import Counter
from time import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Configration(object):
    def __init__(self):
        self.start_word = "<s>"
        self.end_word = "</s>"
        self.min_word_count = 4
        self.word_counts_output_file = "word_counts.vocab"  # text file

        # param setting for training
        self.batch_size = 64  # Sequences per batch
        self.num_steps = 50  # Number of sequence steps per batch
        self.lstm_size = 256  # Size of hidden layers in LSTMs
        self.num_layers = 2  # Number of LSTM layers
        self.learning_rate = 1e-4  # Learning rate
        self.keep_prob = 0.5  # Dropout keep probability
        self.epoch = 100
        self.save_frequence = 5  # save var every epoch
        self.ckpt = './ckpt'


cfg = Configration()


def load_metadata(filename="./dataset/chinese_gen"):
    # load meta dataset
    with open(filename, 'r', encoding='utf-8') as f:
        _input_text = f.readlines()
    lines = list()
    for line in _input_text:
        lines.append([word for word in line.strip()])

    return lines


def convert_text(vocabulary, filename="./dataset/chinese_gen"):
    """
    :param vocabulary: an instance of class Vocabulary
    :param filename: origin train_data
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    # conver text to corresponding number
    encoded = np.array([vocabulary.word_to_id(c) for c in text], dtype=np.int32)
    return encoded


class Vocabulary(object):
    def __init__(self, vocab, unk_id, unk_word="<UNK>"):
        """initializes the vocabulary.
        :arg vocab: a dictionary of word to word_id.
        unk_id: id of the special 'unknown' word."""
        assert type(vocab) == dict
        self._vocab = vocab
        self._unk_id = unk_id
        self._unk_word = unk_word
        self._reverse_vocab = dict(zip(self._vocab.values(), self._vocab.keys()))

    def word_to_id(self, word):
        """:returns the integer id of a word string"""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self._reverse_vocab):
            return self._unk_word
        else:
            return self._reverse_vocab[word_id]

    def get_vocab_len(self):
        return len(self._vocab)


def _create_vocab(captions):
    """creates the vocabulary of word to word_id.
    The vocabulary is saved to disk in a text file of word counts. the id of each word
    in the file is its corresponding 0-based line number.

    >   counter=Counter()
    >   for c in [['this','is','a','young','girl'],['tonight','we','are','young']]:
    ...     counter.update(c)
    >   counter
    Counter({'young': 2, 'this': 1, 'is': 1, 'a': 1, 'girl': 1, 'tonight': 1, 'we': 1, 'are': 1})

    :arg captions: a list of lists of strings.
    :returns a vocabulary object.
    """
    print("Creating vocabulary...")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words:", len(counter))

    # filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= cfg.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("words in vocabulary: ", len(word_counts))

    # write out the word counts file.
    with tf.gfile.FastGFile(cfg.word_counts_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("wrote vocabulary file:", cfg.word_counts_output_file)

    # create the vocabulary dictionary.
    # Side note: Be careful with 0's in vocabulary,
    # padding tensors with 0's may not be able to tell the difference between 0-padding
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    # enumerate a range of numbers starting at 1
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab, start=0)])
    vocab = Vocabulary(vocab_dict, unk_id)
    return vocab


def get_batches(arr, n_seqs, n_steps):
    """对已有的数组进行mini-batch分割. 注意,我们在拆分序列时,将会对序列长度加1,以便构建右移一位的目标序列
    arr: 待分割的数组
    n_seqs: 一个batch中的序列个数
    n_steps: 单个序列长度
    returns: note that, owing to the target sequence is the input sequence right-shifted by 1, output shape is (batch,n_steps)
    """
    batch_size = n_seqs * (n_steps + 1)
    n_batches = int(len(arr) / batch_size)

    # discarding not divisible part
    arr = arr[:batch_size * n_batches]

    arr = arr.reshape((n_seqs, -1))
    print(arr.shape)

    for n in range(0, arr.shape[1], n_steps + 1):
        x = arr[:, n:n + n_steps + 1]
        # note that: compare with inputs `x`, targets `y` will delay a character
        # y = np.zeros_like(x)
        # y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        y, x_hat = x[:, 1:], x[:, :-1]
        # attention: output shape is (batch,n_steps)
        yield x_hat, y


###################
#   build model   #
###################


# input_layer
def build_inputs(num_seqs, num_steps):
    """注意,get_batch中获得的形状是(batch,n_steps)
    :param num_seqs: 每个batch中的序列个数
    :param num_steps:  每个序列包含的字符数
    :return:
    """
    inputs = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(num_seqs, num_steps), name='targets')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob


# LSTM layers
def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    """
    :param lstm_size: nodes in lstm hidden layers
    :param num_layers:  the number of lstm hidden layers
    :param batch_size: batch_size
    :param keep_prob: dropout
    :return: a group of RNN cell composed sequentially of a number of RNNCells. and the initial_state
    """
    stack_drop = list()
    for _ in range(num_layers):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)  # Activation function of the inner states default is `tanh`
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        stack_drop.append(drop)
    group_cell = tf.nn.rnn_cell.MultiRNNCell(stack_drop)

    initial_state = group_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    return group_cell, initial_state


#   output_layer
def build_output(lstm_output, in_size, out_size):
    """
    :param lstm_output: results of lstm output
    :param in_size: the size of reshape lstm output layer
    :param out_size: the size of softmax layer
    :return:
    """
    # 将lstm的输出按列concate
    seq_output = tf.concat(lstm_output, 1)
    x = tf.reshape(seq_output, [-1, in_size])

    with tf.variable_scope('softmax'):
        softmax_w = tf.get_variable('softmax_w', [in_size, out_size], tf.float32, tf.truncated_normal_initializer(stddev=0.1))
        softmax_b = tf.get_variable('softmax_b', [out_size], tf.float32, tf.zeros_initializer)

    logits = tf.matmul(x, softmax_w) + softmax_b

    # softmax layer return prob distribution
    out = tf.nn.softmax(logits=logits, name='predictions')
    return out, logits


# loss_layer
def build_loss(logits, targets, num_classes):
    """
    :param logits: 全连接层的净输出结果(softmax激活前的值)
    :param targets: labels
    :param num_classes: vocab size
    :return:
    """
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    return tf.reduce_mean(loss)


# optimizer
# 通过设置一个阈值, 当gradients超过这个阈值时,就将它重置为阈值大小防止梯度爆炸

def build_optimizer(loss, learning_rate, grad_clip):
    # using clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer


#####################
#   combine model   #
#####################

class WordRNN(object):
    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=256, num_layers=2, learning_rate=5e-4, grad_clip=5,
                 sampling=False):
        """
        :param num_classes: vocab size
        :param batch_size:
        :param num_steps: the number of words in each sequence
        :param lstm_size:
        :param num_layers:
        :param learning_rate:
        :param grad_clip:
        :param sampling: False means for training process, True for test process.
        """
        if sampling:
            self.batch_size, self.num_steps = 1, 1
        else:
            self.batch_size, self.num_steps = batch_size, num_steps
        tf.reset_default_graph()  # convenient for switch train/test??

        self.inputs, self.targets, self.keep_prob = build_inputs(self.batch_size, self.num_steps)
        self.cell, self.initial_state = build_lstm(lstm_size, num_layers, self.batch_size, self.keep_prob)

        # now self.inputs shape: (1, 1), x_one_hot.get_shape()=(1, 1, 3177)
        x_one_hot = tf.one_hot(self.inputs, num_classes)  # note: 这是一个改进点,one_hot表达应该进一步变为embedding

        # run rnn
        self.outputs, self.state = tf.nn.dynamic_rnn(self.cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = self.state

        self.prediction, self.logits = build_output(self.outputs, lstm_size, num_classes)

        # loss and optimizer ( with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


def trainer(encoded, vocab):
    model = WordRNN(vocab.get_vocab_len(), cfg.batch_size, cfg.num_steps, cfg.lstm_size, cfg.num_layers, cfg.learning_rate)
    saver = tf.train.Saver(max_to_keep=cfg.epoch)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in tqdm(range(cfg.epoch)):
            new_state = sess.run(model.initial_state)
            counter = 0
            for x, y in tqdm(get_batches(encoded, cfg.batch_size, cfg.num_steps)):
                start = time()

                feed = {model.inputs: x, model.targets: y, model.keep_prob: cfg.keep_prob, model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.loss, model.final_state, model.optimizer], feed_dict=feed)

                end = time()
                counter = counter + 1
                # control the print lines
                if counter % 100 == 0:
                    print('epoch: {}/{}... '.format(e + 1, cfg.epoch),
                          'iters: {}... '.format(counter),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

            if e > 40 and e % cfg.save_frequence == 0:
                path = mkdir(cfg.ckpt)
                saver.save(sess, os.path.join(path, "{}_{}.ckpt".format(e, cfg.lstm_size)))
        path = mkdir(cfg.ckpt)
        saver.save(sess, os.path.join(path, "{}_{}.ckpt".format(cfg.epoch, cfg.lstm_size)))


def pick_top_k(preds, vocab_size, top_k=5):
    """ 演示用, 对于paper应该考虑用Beamsearch这样的搜索策略
    :param preds: 预测结果(结果的概率分布)
    :param vocab_size: 词汇表大小
    :param top_k: 最终预测值的随机选择范围
    :return:
    """
    p = np.squeeze(preds)
    # 将top_k之下的预测值都置0
    p[np.argsort(p)[:-top_k]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # random choice
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, vocab, prime_words="失败"):
    """
    :param checkpoint: weight param file
    :param n_samples: the number of words for generating txt
    :param vocab:
    :param prime_words:
    :return:
    """
    assert isinstance(vocab, Vocabulary)
    assert len(prime_words) > 0
    samples = [word for word in prime_words]
    # todo: 传入的词表大小可能不对, 因为词表不包含unk_id,故应该注意是否应使用vocab.get_vocab_len()+1
    model = WordRNN(vocab.get_vocab_len(), cfg.batch_size, cfg.num_steps,
                    cfg.lstm_size, cfg.num_layers, cfg.learning_rate, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)

        preds = None
        x = np.zeros((1, 1))  # because we have set sampling=True, means batch_size,n_seqs=1,1
        for word in prime_words:
            # 输入单个字符
            x[0, 0] = vocab.word_to_id(word)
            feed = {model.inputs: x, model.keep_prob: 1., model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], feed_dict=feed)
        word_id = pick_top_k(preds, vocab.get_vocab_len())
        samples.append(vocab.id_to_word(word_id))
        # 不断生成字符,直到达到指定数目
        for i in range(n_samples):
            x[0, 0] = word_id
            feed = {model.inputs: x, model.keep_prob: 1., model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], feed_dict=feed)
            word_id = pick_top_k(preds, vocab.get_vocab_len())
            samples.append(vocab.id_to_word(word_id))
    return ''.join(samples)

tf.flags.DEFINE_boolean("is_training", False, "is mode training?.")
FLAGS = tf.flags.FLAGS


if __name__ == "__main__":
    """test call"""
    lines = load_metadata()
    vocab = _create_vocab(lines)
    encoded = convert_text(vocab, "./dataset/chinese_gen")

    is_training = FLAGS.is_training
    # is_training = True
    if is_training:
        trainer(encoded, vocab)
    else:
        checkpoint = tf.train.latest_checkpoint(cfg.ckpt)
        samp = sample(checkpoint, 100, vocab, "只听得一声大叫")
        print(samp)
