# -*- coding: utf-8 -*-
# @File    : tryEverything/build_oxford_102flower_data.py
# @Info    : @ TSMC-SIGGRAPH, 2018/6/8
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os
import random
import sys
import threading
from collections import namedtuple, Counter
from datetime import datetime

import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize

tf.flags.DEFINE_string("train_image_dir", "/dataset/oxford_102flowers/jpg",
                       "Training image directory.")

tf.flags.DEFINE_string("train_captions_file", "/dataset/cvpr2016_flowers/flower_captions",
                       "Training captions text file.")

tf.flags.DEFINE_string("output_dir", "/dataset/102flowers_tfrecord", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 128,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "/dataset/102flowers_tfrecord/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["filename", "captions"])


def _process_caption(caption):
    """Processes a caption string into a list of tonenized words.
    Args:
      caption: A string caption.
    Returns:
      A list of strings; the tokenized caption.
    """
    tokenized_caption = [FLAGS.start_word]
    tokenized_caption.extend(word_tokenize(caption.lower()))
    tokenized_caption.append(FLAGS.end_word)
    return tokenized_caption


class Vocabulary(object):
    def __init__(self, vocab, unk_id):
        """initializes the vocabulary.
        :arg vocab: a dictionary of word to word_id.
        unk_id: id of the special 'unknown' word."""
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """:returns the integer id of a word string"""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id


class ImageDecoder(object):
    """helper class for decoding images in tensorflow"""

    def __init__(self):
        """create a single tensorflow sessiion for all image decoding calls"""
        self._sess = tf.Session()

        # tensorflow ops for JPEG decoding
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg, feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
    # fixed typeError 'has type str, but expected one of: bytes'. for python3 compatibility
    # value = np.array(value).tobytes()
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(image, decoder, vocab):
    """builds a sequenceExample proto for an image-caption pair.
    :arg image: an ImageMetadata object.
    decoder: an ImageDecoder object.
    vocab: a Vocabulary object.
    :returns a SequenceExample proto.in which containing image dataset and integer captions"""
    with tf.gfile.FastGFile(image.filename, "rb") as f:
        encoded_image = f.read()
    # # in order to open the image, and I removed the try, except image decoder check that follows reading the image.
    # # see: [https://github.com/tensorflow/models/issues/827], just for supporting python3
    # try:
    #     encoded_image = decoder.decode_jpeg(encoded_image)
    # except (tf.errors.InvalidArgumentError, AssertionError):
    #     print("skipping file with invalid JPEG dataset: %s" % image.filename)
    #     return

    context = tf.train.Features(feature={
        "image/data": _bytes_feature(encoded_image),
        "image/filename": _bytes_feature(os.path.basename(image.filename).encode())
    })

    #  although there are 10 captions for each image in oxford-102flower dataset, we have assign image for each caption
    assert len(image.captions) == 1
    caption = image.captions[0]
    caption_ids = [vocab.word_to_id(word) for word in caption]

    feature_lists = tf.train.FeatureLists(feature_list={
        "image/caption_ids": _int64_feature_list(caption_ids)
    })

    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    return sequence_example


def _create_vocab(captions):
    """creates the vocabulary of word to word_id.
    The vocabulary is saved to disk in a text file of word ccunts. the id of each word
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
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("words in vocabulary: ", len(word_counts))

    # write out the word counts file.
    with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("wrote vocabulary file:", FLAGS.word_counts_output_file)

    # create the vocabulary dictionary.
    # note: Be careful with 0's in vocabulary,
    # padding tensors with 0's may not be able to tell the difference between 0-padding
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    # enumerate a range of numbers starting at 0
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab, start=0)])
    vocab = Vocabulary(vocab_dict, unk_id)
    return vocab


def _select_shorter_captions(captions, top_n):
    """
    :param captions: a list of lists of string, such as [['a','b'],['c','d','e']]
    :param top_n: an integer
    :return: a list with top_n shortest length of the lists of string,
    """
    assert top_n <= 10
    lengths = [[x, len(y)] for x, y in enumerate(captions)]
    # note: python3 works well, python2 unknown
    lengths.sort(key=lambda elem: elem[1])
    hit_elem = lengths[:top_n]
    top_n_sentences = [captions[id_len[0]] for id_len in hit_elem]
    return top_n_sentences


def _load_and_process_metadata(captions_file, image_dir):
    """loads image metadata for disk and processes the captions.
        return one elem of a list of ImageMetadata like follows:
        >   [ImageMetadata(filename='/dataset/102flowers/jpg_resized/image_01074.jpg',
            captions=[['<S>', 'this', 'flower', 'has', 'nice', 'yellow', 'petals', 'with', 'white', 'ovule', '.', '</S>'],
            ...
            ['<S>', 'this', 'is', 'a', 'flower', 'with', 'yellow', 'petals', 'and', 'a', 'white', 'stigma', '.', '</S>']])]
    :arg captions_file: text file containing image filename and caption annotations pairs.
                        image_xxxxx.jpg#caption in each line
    image_dir: directory containing the image files.
    :returns a list of ImageMetadata."""
    with tf.gfile.FastGFile(captions_file, "r") as f:
        captions_data = f.readlines()

    # extract the filenames. hint: image_xxxxx.jpg#caption in each line
    # extract the captions, each image is associated with multiple captions.
    img_to_captions = dict()
    for line in captions_data:
        _img_filename, _annotation = line.strip().split("#")
        img_to_captions.setdefault(_img_filename, [])
        img_to_captions[_img_filename].append(_annotation)

    img_filenames = set(img_to_captions.keys())
    assert len(img_to_captions) == len(img_filenames)

    # Process the captions and combine the dataset into a list of ImageMetadata.
    print("processing captions.")
    image_metadata = []
    num_captions = 0
    for base_filename in img_filenames:
        filename = os.path.join(image_dir, base_filename)
        captions = [_process_caption(c) for c in img_to_captions[base_filename]]
        # # hint: Select the five shortest sentences
        # captions = _select_shorter_captions(captions, 5)
        image_metadata.append(ImageMetadata(filename, captions))
        num_captions += len(captions)
    print("finished processing %d captions for %d images in %s" % (num_captions, len(img_filenames), captions_file))
    return image_metadata


def _process_image_files(thread_index, ranges, name, images, decoder, vocab, num_shards):
    """processes and saves a subset of images as TFRecord files in one thread.
    :arg thread_index:Integer thread identifire within [0,len(ranges)].
    ranges: a list of piars of integers secifying the ranges of the dataset to process in parallel.
    name: unique identifire specifying the dataset.
    images: list of ImageMetadata.
    decoder: an ImageDecoder object.
    vocab: a Vocabulary object.
    num_shards: Integer number of shards for the output files."""
    # each thread produces N shards where N = num_shards/num_threads, for instance, if
    # num_shards =128, and num_threads =2, then the first thread would produce shards [0, 64)
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards // num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # generate a sharded version of the file name, e.g. `train-00002-of-00010`
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, decoder, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()
        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


# todo
def _process_dataset(name, images, vocab, num_shards):
    """processes a complete dataset set and saves it as a TFRecord.
    :arg name: unique identifier specifying the dataset
    images: list of ImageMetadata.
    vocab: a Vocabulary object.
    num_shards: Integer number of shards for the output files."""
    # hint: break up each image into a separate entity for each cation.
    images = [ImageMetadata(image.filename, [caption]) for image in images for caption in image.captions]
    # now length of images.cations is one

    # shuffle the ordering of images. make the randomizatiion repeatable.
    random.seed(12345)
    random.shuffle(images)

    # break the images into num_threads batches. batch i is defined as images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = list()
    threads = list()
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in dataset set '%s'." %
          (datetime.now(), len(images), name))


def main(_):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
    assert _is_valid_num_shards(FLAGS.test_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load image metadata from caption files.
    flower_dataset = _load_and_process_metadata(FLAGS.train_captions_file,
                                                FLAGS.train_image_dir)

    # Redistribute the oxford-102flower dataset as follows:
    #   train_dataset = 85% of flower_dataset.
    #   val_dataset = 5% of flower_dataset (for validation during training).
    #   test_dataset = 10% of flower_dataset (for final evaluation).
    train_cutoff = int(0.85 * len(flower_dataset))
    val_cutoff = int(0.90 * len(flower_dataset))
    train_dataset = flower_dataset[0:train_cutoff]
    val_dataset = flower_dataset[train_cutoff:val_cutoff]
    test_dataset = flower_dataset[val_cutoff:]

    # Create vocabulary from the training captions.
    train_captions = [c for image in train_dataset for c in image.captions]
    vocab = _create_vocab(train_captions)

    _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
    _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)
    _process_dataset("test", test_dataset, vocab, FLAGS.test_shards)


#################
#   test call   #
#################

if __name__ == "__main__":
    # words = _text2words()
    # print(len(words))
    # print(list(words)[-20:])

    # # test func _load_and_process_metadata
    # im = _load_and_process_metadata("/dataset/cvpr2016_flowers/flower_captions", "/dataset/102flowers/jpg_resized")
    # print(im[:3])
    tf.app.run()
