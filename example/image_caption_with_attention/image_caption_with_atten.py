# -*- coding: utf-8 -*-
# @File    : image_caption_with_attention/image_caption_with_atten.py
# @Info    : @ TSMC-SIGGRAPH, 2018/8/23
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 
""" refer to:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/
image_captioning_with_attention.ipynb

note: in ssh, Matplotlib chooses Xwindows backend by default. You need to set matplotlib to not use the Xwindows backend.
Add this code to the start of your script (before importing pyplot) and try again:

import matplotlib
matplotlib.use('Agg')

"""
import json
import os
import time
import matplotlib
matplotlib.use('Agg')    # Add this code to the start of your script (before importing pyplot)
# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt
import numpy as np
# import pickle
import tensorflow as tf
# import re
# from glob import glob
from PIL import Image
# # Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # only /gpu:gpu_id is visible

tf.enable_eager_execution()


class ModelConfig(object):
    def __init__(self):
        self.batch_size = 16
        self.epoch = 2
        self.buffer_size = 1000
        self.embedding_dim = 256
        self.units = 512
        self.vocab_size = None
        # shape of the vector extracted from InceptionV3 is (64, 2048)
        # these two variables represent that
        self.attention_features_shape = 64
        self.features_shape = 2048
        
        # (Inference) The maximum caption length before stopping the search.
        self.max_caption_length = 20


modelconfig = ModelConfig()

################################################################################
#   Caution: large download ahead. Download and prepare the MS-COCO dataset    #
################################################################################
# annotation_zip=tf.keras.utils.get_file('captions.zip',cache_subdir=os.path.abspath('.'),
#                                        origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
#                                        extract=True)
#
# annotation_file=os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

# name_of_zip = 'train2014.zip'
# if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
#     image_zip = tf.keras.utils.get_file(name_of_zip,
#                                         cache_subdir=os.path.abspath('.'),
#                                         origin='http://images.cocodataset.org/zips/train2014.zip',
#                                         extract=True)
#     PATH = os.path.dirname(image_zip) + '/train2014/'
# else:
#     PATH = os.path.abspath('.') + '/train2014/'

annotation_file = '/dataset/cocodataset/annotations/captions_train2014.json'
PATH = os.path.join('/dataset/cocodataset', 'train2014')

#########################################################################
#   Optionally, limit the size of the training set for faster training  #
#########################################################################

# read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# storing the captions and the image name in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = os.path.join(PATH, 'COCO_train2014_' + '%012d.jpg' % image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# shuffling the captions and image_names together
# setting a random state
train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

# selecting the first 3w captions from the shuffled set
num_examples = 30000
# num_examples = 300
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]
print("promote: ==> (captions,img) tuples random select ", len(train_captions), len(img_name_vector))


#############################################
#   Preprocess the images using InceptionV3 #
#############################################


def load_image(image_path):
    """convert the images into the format inceptionV3 expects
    resizing the image to (299,299),using the `preprocess_input` method to place the pixels in the range of -1 to 1
    :param image_path: image path
    :return:
    """
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


#####################################################################
#   Initialize InceptionV3 and load the pretrained Imagenet weights #
#####################################################################
# we'll create a tf.keras model where the output layer is the last convolutional layer in the InceptionV3 architecture.
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
#image_model = tf.keras.applications.InceptionV3(include_top=False)
#image_model._initial_weights = '/dataset/pre_trained_model/keras/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# image_model.trainable = False
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
# hidden_layer = image_model.get_layer("mixed10").output
image_features_extract_model = tf.keras.models.Model(inputs=new_input, outputs=hidden_layer)

# After all the images are passed through the network, we pickle the dictionary and save it to disk.
#####################################################
#   Caching the features extracted from InceptionV3 #
#####################################################
# getting th unique images
#encode_train = sorted(set(img_name_vector))
#image_dataset = tf.data.Dataset.from_tensor_slices(encode_train).map(load_image).batch(modelconfig.batch_size)
# follow snippets need eager execution
#for img, path in tqdm(image_dataset):
#    batch_features = image_features_extract_model(img)
#    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
#    for bf, p in zip(batch_features, path):
#        path_of_img = p.numpy().decode('utf-8')
#        filename = os.path.basename(path_of_img)
#        path_of_feature = os.path.join('/dataset/cocodataset/inceptionv3_conv_feature', filename)
#        np.save(path_of_feature, bf.numpy())
#
#print(image_dataset.output_types)
#print(image_dataset.output_shapes)
#
# iterator = image_dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# batch_img, batch_path = next_element
# batch_features = image_features_extract_model(batch_img)
# print(batch_img, batch_features)
# batch_features = tf.reshape(batch_features, [tf.shape(batch_features)[0], -1, tf.shape(batch_features)[3]])
#
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     total_steps = num_examples // batch_size
# #     counter = 0
# #     while True:
# #         try:
# #             batch_features_np, path_np = sess.run([batch_features, batch_path])
# #             print(path_np)
# #             for bf, p in zip(batch_features_np, path_np):
# #                 path_of_img = p.decode('utf-8')
# #                 filename = os.path.basename(path_of_img)
# #                 # path_of_feature = os.path.join('/dataset/cocodataset/inceptionv3_conv_feature', filename)
# #                 path_of_feature = os.path.join('/home', filename)
# #                 np.save(path_of_feature, bf)
# #             counter += 1
# #             if counter % 5 == 0:
# #                 print("{:.2f}, process: {}/{}".format(counter / total_steps, counter, total_steps))
# #         except tf.errors.OutOfRangeError:
# #             print("End of dataset")
# #             break

#############################################
#   Preprocess and tokennize the captions   #
#############################################
"""
1). First, we'll tokenize the captions (e.g., by splitting on spaces).
This will give us a vocabulary of all the unique words in the data (e.g., "surfing", "football", etc).

2). Next, we'll limit the vocabulary size to the top 5,000 words to save memory.
We'll replace all other words with the token "UNK" (for unknown).

3). Finally, we create a word --> index mapping and vice-versa.
We will then pad all sequences to the be same length as the longest one.
"""


def calc_max_length(tensor):
    """
    find the maxinum length of any caption in our dataset
    :param tensor:  an array with shape (n,m)
    :return:
    """
    return max(len(t) for t in tensor)


# The steps above is a general process of dealing with text processing
# choosing the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ', oov_token="<unk>")
tokenizer.fit_on_texts(train_captions)

tokenizer.word_index = {key: value for key, value in tokenizer.word_index.items() if value <= top_k}  # choose the top_k words
# putting <unk> token in the word2idx dictionary
tokenizer.word_index[tokenizer.oov_token] = top_k + 1
tokenizer.word_index['<pad>'] = 0

modelconfig.vocab_size = len(tokenizer.word_index)

# creating the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# creating a reverse mapping ( index -> word )
index_word = {value: key for key, value in tokenizer.word_index.items()}

# padding each vector to the max_length of the captions
# if the max_length parameter is not provided, pad_sequences calculates that automatically.
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# calculating the max_length
# used to store the attention weights
max_length = calc_max_length(train_seqs)

#################################################
#   Split the data into training and testing    #
#################################################
# create training and validation sets using 0.8/0.2 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, cap_vector, test_size=0.2, random_state=0)
print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))


def map_func(img_name, cap):
    filename = os.path.basename(img_name.decode('utf-8'))
    filename = os.path.join('/dataset/cocodataset/inceptionv3_conv_feature', filename + '.npy')
    img_tensor = np.load(filename)
    return img_tensor, cap


dataset_train = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# using map to load the numpy file in parallel
# https://www.tensorflow.org/api_docs/python/tf/py_func
# Given a python function func, which takes numpy arrays as its arguments and returns numpy arrays as its outputs,
# wrap this function as an operation in a TensorFlow graph.
dataset_train = dataset_train.map(lambda item1, item2: tf.py_func(map_func, [item1, item2], [tf.float32, tf.int32]),
                                  num_parallel_calls=8)

# shuffling and batching
dataset_train = dataset_train.shuffle(modelconfig.buffer_size)
dataset_train = dataset_train.batch(modelconfig.batch_size)
dataset_train = dataset_train.prefetch(1)


#############
#   Model   #
#############

class BahdanauAttention(tf.keras.Model):
    """using mlp as atten func"""

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)  # The weight alpha_i of each annotation vector a_i

    def call(self, inputs, hidden):
        """
        :param inputs: a tensor with shape (batch,64,embedding_size), in which including 64 annotation vectors
        :param hidden: rnn hidden state,type is tensor, with shape (batch, hidden_units)
        :return:
            context_vector: represent corresponding to a part of the image
            attention_weights: attention weights in current time step
        """
        # inputs(i.e. CNN_encoder output) shape == (batch_size, 64, embedding_size)

        # hidden shape==(batch_size, hidden_size)
        hidden_with_time_axis = tf.expand_dims(input=hidden, axis=1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(inputs) + self.W2(hidden_with_time_axis))  # trailing time_dims broadcast elem-wise add.

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, inputs):
        x = self.fc(inputs)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = self.gru_cell()
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, inputs, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention.call(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(inputs)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def gru_cell(self):
        return tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


encoder = CNN_Encoder(modelconfig.embedding_dim)
decoder = RNN_Decoder(modelconfig.embedding_dim, modelconfig.units, modelconfig.vocab_size)

optimizer = tf.train.AdamOptimizer()


# We are masking the loss calculated for padding
def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


#################
#   Training    #
#################

# adding this in a separate cell because if you run the training cell many times, the loss_plot array will be reset
loss_plot = []

start = time.time()
total_loss = 0
for epoch in range(modelconfig.epoch):
    for (step, (img_tensor, target)) in enumerate(dataset_train):
        loss = 0
        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * modelconfig.batch_size, 1)

        with tf.GradientTape() as tape:
            features = encoder.call(img_tensor)
            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder.call(dec_input, features, hidden)
                loss += loss_function(target[:, i], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss += (loss / int(target.shape[1]))
        variables = encoder.variables + decoder.variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

        if step % 50 == 0:
            print('Epoch {}/{} Step {} Loss {:.4f}'.format(epoch + 1, modelconfig.epoch,
                                                           step, loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / len(cap_vector))

    print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss / len(cap_vector)))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
# plt.show()
plt.savefig('loss_plot.png', bbox_inches='tight')


def evaluate(image):
    attention_plot = np.zeros((modelconfig.max_caption_length, modelconfig.attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder.call(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(modelconfig.max_caption_length):
        predictions, hidden, attention_weights = decoder.call(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
        result.append(index_word[predicted_id])

        if index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot, savename="plot_attention"):
    temp_image = np.array(Image.open(image))

    #fig = plt.figure(figsize=(10, 10))
    fig = plt.figure(figsize=(8.267, 11.692), dpi=96)

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    # plt.show()
    plt.savefig("{}.png".format(savename), bbox_inches='tight')
    plt.clf()


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)
# opening the image
Image.open(img_name_val[rid])

# #################################
# #   try it on your own images   #
# #################################
# image_url = 'https://tensorflow.org/images/surf.jpg'
# image_extension = image_url[-4:]
# image_path = tf.keras.utils.get_file('image'+image_extension,
#                                      origin=image_url)
#
for image_path in ["surf.jpg", "dog.jpg", "cat.jpg", "elephant.jpg", "tennis.jpg"]:
    result, attention_plot = evaluate(image_path)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(image_path, result, attention_plot,"{}_attention".format(image_path))
    # opening the image
    Image.open(image_path)


