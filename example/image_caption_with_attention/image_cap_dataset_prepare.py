# -*- coding: utf-8 -*-
# @File    : image_caption_with_attention_r9.1/image_cap_dataset_prepare.py
# @Info    : @ TSMC-SIGGRAPH, 2018/9/26
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import json
import os

import numpy as np
import tensorflow as tf
# # Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from configuration import ModelConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # only /gpu:gpu_id is visible


class ImageCapDataBase(object):
    """Image Caption Data Base"""

    def __init__(self, config: ModelConfig):
        """
        :param config: an instance of class ModelConfig
        """
        self.config = config
        self.annotation_file = '/dataset/cocodataset/annotations/captions_train2014.json'
        self.img_metadata_dir = os.path.join('/dataset/cocodataset', 'train2014')
        self.cache_dir = '/dataset/cocodataset/inceptionv3_conv_feature'

        self.train_captions = None
        self.img_name_vector = None

        # holding the preprocessed and tokenize captions.
        self.tokenizer = None
        # mapping ( index -> word )
        self.index_word = None

        # pad_sequence
        self.cap_vector = None

        # pre-trained model
        self.image_features_extract_model = None

        # Split the data into training and testing
        self.img_name_train, self.img_name_val, self.cap_train, self.cap_val = None, None, None, None

        # batch dataset for training
        self.dataset_train = None

        # images full path
        self.batch_path = None
        self.batch_features = None

    def load_json_file(self):
        # read the json file
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)

        # storing the captions and the image name in vectors
        all_captions = []
        all_img_name_vector = []

        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = os.path.join(self.img_metadata_dir, 'COCO_train2014_' + '%012d.jpg' % image_id)

            all_img_name_vector.append(full_coco_image_path)
            all_captions.append(caption)

        # shuffling the captions and image_names together
        # setting a random state,  guaranteed to have the same random sequence if you use the same seed.
        train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

        self.train_captions = train_captions[:self.config.limit_num_examples]
        self.img_name_vector = img_name_vector[:self.config.limit_num_examples]
        print("[load_json_file] (captions,img) tuples random select ", len(self.train_captions), len(self.img_name_vector))

    @staticmethod
    def load_image(image_path):
        """convert the images into the format inceptionV3 expects
        resizing the image to (299,299),using the `preprocess_input` method to place the pixels in the range of -1 to 1
        :param image_path: image path, a `Tensor` of type `string`.
        :return:
        """
        img = tf.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize_images(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    def build_inception_model(self):
        """Initialize InceptionV3 and load the pretrained Imagenet weights
        :return:
        """
        # we'll create a tf.keras model where the output layer is the last convolutional layer in the InceptionV3 architecture.
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        image_model.trainable = False
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        # hidden_layer = image_model.get_layer("mixed10").output
        self.image_features_extract_model = tf.keras.models.Model(inputs=new_input, outputs=hidden_layer)

    def caching_inception_feature(self):
        """
        After all the images are passed through the network, we pickle the dictionary and save it to disk.
        Caching the features extracted from InceptionV3
        """
        # getting the unique images
        encode_train = sorted(set(self.img_name_vector))
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train).map(self.load_image).batch(self.config.batch_size)
        # follow snippets need eager execution
        for img, path in tqdm(image_dataset):
            batch_features = self.image_features_extract_model(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
            for bf, p in zip(batch_features, path):
                path_of_img = p.numpy().decode('utf-8')
                filename = os.path.basename(path_of_img)
                path_of_feature = os.path.join(self.cache_dir, filename)
                np.save(path_of_feature, bf.numpy())

        print(image_dataset.output_types)
        print(image_dataset.output_shapes)

    @staticmethod
    def calc_max_length(tensor):
        """
        find the maxinum length of any caption in our dataset
        :param tensor:  an array with shape (n,m)
        :return:
        """
        return max(len(t) for t in tensor)

    def map_func(self, img_name, cap):
        filename = os.path.basename(img_name.decode('utf-8'))
        filename = os.path.join(self.cache_dir, filename + '.npy')
        img_tensor = np.load(filename)
        return img_tensor, cap

    def process_captions(self):
        """
        Preprocess and tokenize the captions.
        1). First, we'll tokenize the captions (e.g., by splitting on spaces).
        This will give us a vocabulary of all the unique words in the data (e.g., "surfing", "football", etc).

        2). Next, we'll limit the vocabulary size to the top 5,000 words to save memory.
        We'll replace all other words with the token "UNK" (for unknown).

        3). Finally, we create a word --> index mapping and vice-versa.
        We will then pad all sequences to the be same length as the longest one.
        :return:
        """
        # The steps above is a general process of dealing with text processing
        # choosing the top k words from the vocabulary
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.config.top_k,
                                                               filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',
                                                               oov_token="<unk>")
        self.tokenizer.fit_on_texts(self.train_captions)

        # choose the top_k words
        self.tokenizer.word_index = {key: value for key, value in self.tokenizer.word_index.items()
                                     if value <= self.config.top_k}
        # putting <unk> token in the word2idx dictionary
        self.tokenizer.word_index[self.tokenizer.oov_token] = self.config.top_k + 1
        self.tokenizer.word_index['<pad>'] = 0

        self.config.vocab_size = len(self.tokenizer.word_index)

        # creating the tokenized vectors
        train_seqs = self.tokenizer.texts_to_sequences(self.train_captions)

        # we truncate captions longer than 20 words for COCO
        truncate_train_seqs = []
        for line in train_seqs:
            if len(line) > 22:
                line = line[:22]    # don't need to add '<end>' symbol
            truncate_train_seqs.append(line)

        # creating a reverse mapping ( index -> word )
        self.index_word = {value: key for key, value in self.tokenizer.word_index.items()}

        # padding each vector to the max_length of the captions
        # if the max_length parameter is not provided, pad_sequences calculates that automatically.
        self.cap_vector = tf.keras.preprocessing.sequence.pad_sequences(truncate_train_seqs, padding='post')

        # calculating the max_length
        # used to store the attention weights
        max_length = self.calc_max_length(truncate_train_seqs)
        print("[process_captions] the longest captions length: {}".format(max_length))

    def build_dataset(self):
        # Split the data into training and testing using 0.9/0.1 split
        self.img_name_train, self.img_name_val, self.cap_train, self.cap_val = train_test_split(self.img_name_vector,
                                                                                                self.cap_vector,
                                                                                                test_size=0.1,
                                                                                                random_state=0)
        print("[build_dataset] train examples: {}, val examples : {}".format(len(self.img_name_train), len(self.img_name_val)))

        dataset_train = tf.data.Dataset.from_tensor_slices((self.img_name_train, self.cap_train))

        # using map to load the numpy file in parallel
        # https://www.tensorflow.org/api_docs/python/tf/py_func
        # Given a python function func, which takes numpy arrays as its arguments and returns numpy arrays as its outputs,
        # wrap this function as an operation in a TensorFlow graph.
        dataset_train = dataset_train.map(lambda item1, item2:
                                          tf.py_func(self.map_func, [item1, item2], [tf.float32, tf.int32]),
                                          num_parallel_calls=2)

        # shuffling and batching
        dataset_train = dataset_train.shuffle(self.config.buffer_size)
        dataset_train = dataset_train.repeat(self.config.epoch)
        self.dataset_train = dataset_train.batch(self.config.batch_size)
        # self.dataset_train = dataset_train.prefetch(1)

    def build(self):
        self.load_json_file()
        self.build_inception_model()
        # self.caching_inception_feature()
        self.process_captions()

