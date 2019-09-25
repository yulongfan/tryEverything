# -*- coding: utf-8 -*-
# @File    : image_caption_with_attention_r9.1/configuration.py
# @Info    : @ TSMC-SIGGRAPH, 2018/9/6
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


class ModelConfig(object):
    def __init__(self):
        self.batch_size = 32
        self.epoch = 5
        self.buffer_size = 1000

        self.embedding_dim = 256  # input img/sequence embedding dimension
        self.units = 512        # hidden units
        self.vocab_size = None  # the size of the caption's vocabulary

        # shape of the vector extracted from InceptionV3 is (64, 2048)
        # these two variables represent that
        self.attention_features_shape = 64
        self.features_shape = 2048

        # (Inference) The maximum caption length before stopping the search.
        self.max_caption_length = 20

        # limit the size of the training set for faster training, selecting the first 3w captions from the shuffled set
        self.limit_num_examples = 30000

        # choosing the top k words from the vocabulary
        self.top_k = 5000
