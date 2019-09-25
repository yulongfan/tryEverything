# -*- coding: utf-8 -*-
# @File    : tryEverything/get_flower_tfrecord_data.py
# @Info    : @ TSMC-SIGGRAPH, 2018/6/13
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os

import scipy.misc
import tensorflow as tf

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3


class DataBase(object):
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.train_file_list = self._get_file('train')
        self.val_file_list = self._get_file('val')
        self.test_file_list = self._get_file('test')

    def _get_file(self, mode):
        file_list = os.listdir(self.file_dir)
        file_list = list(filter(lambda x: x.startswith(mode), file_list))
        file_list = list(map(lambda x: os.path.join(self.file_dir, x), file_list))
        return file_list

    def _parse_sequence_example(self, example):
        # Define how to parse the example
        context_features = {
            'image/data': tf.FixedLenFeature([], tf.string)
        }
        sequence_features = {
            'image/caption_ids': tf.FixedLenSequenceFeature([], tf.int64)
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=example,
            context_features=context_features,
            sequence_features=sequence_features
        )
        # image_decoded = tf.decode_raw(context_parsed['image/dataset'], tf.uint8)
        # Use decode_jpeg instead of decode_raw
        image_decoded = tf.image.decode_jpeg(context_parsed['image/data'], channels=3)
        image_resized = tf.image.resize_images(image_decoded,
                                               size=[224, 224],
                                               method=tf.image.ResizeMethod.BILINEAR)
        # image_resized = tf.reshape(image_decoded, [224, 224, 3])
        image_resized = tf.cast(image_resized, tf.float32) * (1. / 255.)
        # caption_ids: A 1-D int64 Tensor with dynamically specified length.
        caption_ids_decoded = tf.cast(sequence_parsed['image/caption_ids'], tf.int64)
        image_caption_id_pair = [image_resized, caption_ids_decoded]
        # del image_resized, caption_ids_decoded, image_decoded
        return image_caption_id_pair

    def get_dataset(self, batch_size, num_epochs, mode='train'):
        if mode == 'train':
            file_list = self.train_file_list
        elif mode == 'val':
            file_list = self.val_file_list
        elif mode == 'test':
            file_list = self.test_file_list
        else:
            raise Exception('No such file')

        dataset = tf.data.TFRecordDataset(file_list)
        dataset = dataset.map(self._parse_sequence_example)
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        return dataset


class Vocabulary(object):
    def __init__(self, vocab, unk_id):
        """initializes the vocabulary.
        :arg vocab: a dictionary of word to word_id.
        unk_id: id of the special 'unknown' word."""
        self._vocab = vocab
        self._unk_id = unk_id

        self._vocab_flip = dict(zip(self._vocab.values(), self._vocab.keys()))

    def word_to_id(self, word):
        """:returns the integer id of a word string"""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

    def id_to_word(self, id):
        """:returns the word"""
        if id in self._vocab_flip:
            return self._vocab_flip[id]
        else:
            return "<UNK>"


# fixme: to read word_counts.txt, build vocabulary
def build_vocab(word_counts):
    # create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab, start=0)])
    vocab = Vocabulary(vocab_dict, unk_id)
    return vocab


if __name__ == "__main__":
    with open('/dataset/102flowers_tfrecord/word_counts.txt', 'r') as f:
        text = f.readlines()
    word_counts = [line.strip().split() for line in text]
    vocab = build_vocab(word_counts)

    test = DataBase("/dataset/102flowers_tfrecord")
    # q&a:the resolution could be allow dispatching images of different sizes by dataset object, if batch size is equal to 1.
    dataset = test.get_dataset(1, 1, "val")
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    # test output
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        for i in range(100):
            image_caption_pair = sess.run(next_element)
            if i > 50 and i % 5 == 0:
                image, caption_ids = image_caption_pair
                scipy.misc.imsave("/result/%02d.jpg" % i, image[0])

                assert image.shape == (1, 224, 224, 3)
                # print(caption_ids[0])
                sentence = list()
                for index in caption_ids[0]:
                    sentence.append(vocab.id_to_word(index))
                print("%d). " % i, " ".join(sentence))
                sentence.clear()
