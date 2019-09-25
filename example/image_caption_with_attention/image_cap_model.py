# -*- coding: utf-8 -*-
# @File    : image_caption_with_attention/image_cap_model.py
# @Info    : @ TSMC-SIGGRAPH, 2018/8/26
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

import os

import tensorflow as tf

from configuration import ModelConfig
from image_cap_dataset_prepare import ImageCapDataBase

os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # only /gpu:gpu_id is visible


class BahdanauAttention(object):
    """using mlp as atten func"""

    def __init__(self, units):
        """
        :param units: Integer or Long, dimensionality of the output space.
        """
        self.w1_units = units
        self.w2_units = units
        self.v_units = 1  # The weight alpha_i of each annotation vector a_i is a scalar

    def __call__(self, inputs, hidden):
        """
        :param inputs: a tensor with shape (batch,64,embedding_size), in which including 64 annotation vectors
        :param hidden: rnn hidden state,type is tensor, with shape (batch, hidden_units)
        :return:
            context_vector: represent corresponding to a part of the image
            attention_weights: attention weights in current time step
        """
        # inputs(i.e. CNN_encoder output) shape == (batch_size, 64, embedding_size)

        # hidden shape==(batch_size, hidden_size)
        hidden_with_time_axis = tf.expand_dims(input=hidden, axis=1)  # ==> shape (batch, 1, hidden_size)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(tf.layers.dense(inputs, self.w1_units) +
                           tf.layers.dense(hidden_with_time_axis, self.w2_units))  # trailing time_dims broadcast elem-wise add.

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(tf.layers.dense(score, self.v_units), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class CNN_Encoder(object):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        self.embedding_size = embedding_dim

    def __call__(self, inputs):
        """shape after fc == (batch_size, 64, embedding_dim)
        :param inputs: a tensor with shape (batch, 64, 2048)
        :return: a tensor with shape (batch, 64, embedding_dim)
        """
        # attention: this is a emergence mean
        inputs = tf.reshape(inputs, (-1, 64, 2048))
        # inputs = tf.reshape(inputs, (inputs_shape[0], inputs_shape[1], inputs_shape[2]))
        x = tf.nn.relu(tf.layers.dense(inputs, self.embedding_size))
        return x


class GRU_Decoder(object):
    def __init__(self, hidden_units, is_training=False):
        """
        :param hidden_units: the number of GRU Cell hidden units.
        """
        self.hidden_units = hidden_units
        self.is_training = is_training

        self.stacked_rnn_cell = self.build_grucell()
        self.max_time_steps = None
        self.input_ta = None

        # annotation features extract from CNN
        self.annotation_features = None

        self.attention = BahdanauAttention(self.hidden_units)

    def __call__(self, annotation_features, input_seqs, hidden_state):
        """
        :param annotation_features: annotation features extract from CNN, with shape (batch, feature_dim)
        :param input_seqs: input sequence with shape (batch,sequence_length), array like [[1,3,23,124,2,0],...]
        :param hidden_state: gru hidden state, a tuple with shape ((batch, hidden_units),)
        :return:
        """
        with tf.variable_scope("GRU_Decoder", reuse=tf.AUTO_REUSE):
            self.annotation_features = annotation_features

            input_seqs = tf.transpose(input_seqs, [1, 0, 2])  # (batch,time_steps, embedding) ==> (time_steps,batch,embedding)

            input_seqs_shape = tf.shape(input_seqs)
            self.max_time_steps = input_seqs_shape[0]
            # runing_batch_size = input_seqs_shape[1]
            output_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            attention_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            input_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            self.input_ta = input_ta.unstack(input_seqs)

            time_step = 0
            # hidden_state must be given
            # attention: state pass to gru is a tuple
            time_final, output_ta_final, state_final, attention_ta_final = tf.while_loop(
                cond=self.condition,
                body=self.body,
                loop_vars=(time_step, output_ta, hidden_state, attention_ta))

            output_final = output_ta_final.stack()  # ==> shape (time_steps, batch, hidden_size)
            output_final = tf.transpose(output_final, [1, 0, 2])  # ==> shape (batch, time_steps, hidden_size)

            attention_final = attention_ta_final.stack()  # ==> shape (time_steps, batch, 64, 1)
            attention_final = tf.squeeze(attention_final, [3])  # ==> shape (time_steps, batch, 64)
            attention_final = tf.transpose(attention_final, [1, 0, 2])  # ==> shape (batch, time_steps, 64)
            print("[attention_weights] attention_weights: {}".format(attention_final))
            return output_final, state_final, attention_final

    def body(self, time_step, output_ta_t, state, atten_ta):
        """ note: gru output hidden state is a tuple
        :param time_step: an integer denotes current time_step.
        :param output_ta_t: a tensor array record the recurrent network outputs.
        :param state: previous recurrent network's hidden state. == (batch_size, hidden_size).
        :param atten_ta: a tensor array for recording the attention_weights of each time_step.
        :return: next time_step, outputs of each time_step so far,
                current hidden state, attention_weights of each time_step so far.
        """
        # assert self.input_ta is not None
        xt = self.input_ta.read(time_step)  # xt shape is (batch_size, embedding_dim)

        # defining attention as a separate model
        context_vector, attention_weights = self.attention(self.annotation_features, state[0])
        # x shape after concatenation == (batch_size, embedding_dim + hidden_size)
        xt = tf.concat(values=[context_vector, xt], axis=-1)
        # passing the concatenated vector to the GRU
        new_output, new_state = self.stacked_rnn_cell(xt, state)
        output_ta_t = output_ta_t.write(time_step, new_output)
        atten_ta = atten_ta.write(time_step, attention_weights)
        return time_step + 1, output_ta_t, new_state, atten_ta

    def condition(self, time_step, *args):
        return time_step < self.max_time_steps

    def build_grucell(self):
        """Am i need to add dropout here?"""
        grucell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_units)
        if self.is_training:
            grucell = tf.nn.rnn_cell.DropoutWrapper(cell=grucell,
                                                    input_keep_prob=0.5,
                                                    output_keep_prob=0.5)
        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([grucell])
        return stacked_rnn_cell


class ImageCapModel(object):
    def __init__(self, config: ModelConfig, mode):
        """Basic setup
        Args:
            config: an instance of class ModelConfig containing configuration parameters.
            mode: "train", "eval" or "inference".
        """
        self.config = config
        assert mode in ["train", "eval", "inference"]
        self.mode = mode

        self.cap_model_database = ImageCapDataBase(self.config)
        self.cap_model_database.build()  # assign modelconfig.vocab_size
        assert self.config.vocab_size is not None
        self.encoder = CNN_Encoder(self.config.embedding_dim)
        self.decoder = GRU_Decoder(self.config.units, self.is_training())

        # initialize all variables with a truncated random normal initializer.
        self.initializer = tf.truncated_normal_initializer(stddev=0.1)

        self.loss = None
        self.accuracy = None

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.image = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An float32 Tensor with shape [batch_size, 64, embedding_size]
        self.annotation_features = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # Global step Tensor.
        self.global_step = None

    def is_training(self):
        """returns true if the model is built for training mode"""
        return self.mode == "train"

    @staticmethod
    def loss_function(real, pred):
        """We are masking the loss calculated for padding
        :param real: the ground truth label, shape is (n,), in generally, n is batch_size or number of examples.
        :param pred: prediction outputs, with shape (n,classes), in generally, n is batch_size or number of examples.
        :return:    loss_op
        """
        # mask = 1 - np.equal(real, 0)
        mask = 1.0 - tf.cast(tf.equal(real, tf.zeros(tf.shape(real)[0], tf.int32)), tf.float32)
        print("[loss_function] mask: ", mask)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask

        # compute accuracy
        probs = tf.nn.softmax(pred)
        class_idx = tf.argmax(probs, axis=-1)
        hit = tf.equal(class_idx, tf.to_int64(real))
        hit = tf.cast(hit, tf.float32)
        hit = hit * mask
        accuracy = tf.reduce_mean(hit)
        print("[loss_function] accuracy: {}".format(accuracy))

        return tf.reduce_mean(loss_), accuracy

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.

        Outputs:
          self.images
          self.input_seqs
          self.target_seqs (training and eval only)
        """
        if self.mode == "inference":
            # In inference mode, images and inputs are fed via placeholders.
            image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            input_feed = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed")
            # Process image and insert batch dimensions.
            temp_input = tf.expand_dims(self.cap_model_database.load_image(image_path=image_feed)[0], 0)
            img_tensor = self.cap_model_database.image_features_extract_model(temp_input)
            image = tf.reshape(img_tensor, (tf.shape(img_tensor)[0], -1, tf.shape(img_tensor)[-1]))
            # batch_cap = tf.expand_dims([self.cap_model_database.tokenizer.word_index['<start>']], 0)
            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
        else:
            self.cap_model_database.build_dataset()
            iterator = self.cap_model_database.dataset_train.make_one_shot_iterator()
            next_element = iterator.get_next()
            batch_img, batch_cap = next_element  # batch_img.shape=(batch, 64, 2048), batch_cap.shape=(batch,max_len_of_cap)
            # # attention: this is a emergence mean
            image = tf.reshape(batch_img, (-1, 64, 2048))

            batch_cap = tf.reshape(batch_cap, (tf.shape(batch_cap)[0], tf.shape(batch_cap)[-1]))
            num_captions = tf.shape(batch_cap)[0]
            caption_length = tf.shape(batch_cap)[1]
            input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

            input_seqs = tf.slice(batch_cap, [0, 0], [num_captions, input_length[0]])
            # the target sequence is the input sequence right-shifted by 1
            target_seqs = tf.slice(batch_cap, [0, 1], [num_captions, input_length[0]])

        self.image = image
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings
        Inputs:
            self.images: a string denotes the image path
        Outputs:
            self.image_embeddings: A tensor
        """
        self.annotation_features = self.encoder(self.image)

    def build_seq_embeddings(self):
        """ Builds the input sequence embeddings.
        Inputs:
            self.input_seqs: a list with shape (batch_size, seq_length)
        Outputs:
            self.seq_embeddings: A tensor with shape (batch_size, time_steps, embedding_size)
        """
        embedding_lookup = tf.keras.layers.Embedding(input_dim=self.config.vocab_size, output_dim=self.config.embedding_dim)
        self.seq_embeddings = embedding_lookup(self.input_seqs)
        print("[build_seq_embeddings] embedding lookup output `input_seqs` is : ", self.seq_embeddings)

    def build_model(self):
        """
        Inputs:
            self.image_embeddings
            self.seq_embeddings
            self.target_seqs (taining and eval only)
            self.input_mask (training and eval only)
        Outputs:
            self.loss (training and eval only)
        """
        with tf.variable_scope("imgcap", initializer=self.initializer, reuse=tf.AUTO_REUSE) as imgcap_scope:
            mean_annotation_features = tf.reduce_mean(input_tensor=self.annotation_features, axis=1)
            # ==> shape (batch, embedding_size)
            print("[mean_annotation_features] mean_annotation_features: {}".format(mean_annotation_features))

            # # attention: state pass to gru is a tuple
            mean_annotation_features = tf.layers.dense(inputs=mean_annotation_features, units=self.config.units,
                                                       activation=tf.nn.relu, name="f_init")
            init_annot_state = (mean_annotation_features,)
            print("[init_state] init_state: {}".format(init_annot_state))

            # Feed the image embeddings to set the initial GRU state.
            if self.mode == "inference":
                tf.identity(self.annotation_features, name="annotation_features")
                tf.identity(mean_annotation_features, name="mean_annotation_features")
                annotation_features = tf.placeholder(dtype=tf.float32,
                                                     shape=[None, self.config.attention_features_shape,
                                                            self.config.embedding_dim],
                                                     name="annotation_features_feed")
                print("[inference] annotation_features_feed: {}".format(annotation_features))

                # Placeholder for feeding a batch of concatenated states.
                pre_state_feed = tf.placeholder(dtype=tf.float32,
                                                shape=[None, sum(self.decoder.stacked_rnn_cell.state_size)],
                                                name="state_feed")
                print("[inference] pre_state_feed: {}".format(pre_state_feed))
                # if using LSTM, in inference mode, use concatenated states for convenient feeding and fetching
                # pre_state_tuple = tf.split(value=pre_state_feed, num_or_size_splits=2, axis=1) # suit for LSTM
                pre_state_tuple = (pre_state_feed,)
                print("[inference] pre_state_tuple: {}".format(pre_state_tuple))
                gru_outputs, gru_state_tuple, attention_weights = self.decoder(annotation_features=annotation_features,
                                                                               input_seqs=self.seq_embeddings,
                                                                               hidden_state=pre_state_tuple)
                # In inference mode, use concatenated states for convenient feeding and fetching.
                # Concatenate the resulting state.
                tf.concat(axis=1, values=gru_state_tuple, name="state")
                # tf.identity(input=attention_weights, name="attention_weights")
                tf.squeeze(input=attention_weights, name="attention_weights")  # a tensor with shape (64,)
            else:
                # gru_outputs.shape=(batch_size, time_steps, hidden_units),
                # gru_state_tuple.shape = (1,).   In which gru_state_tuple[0].shape= (batch_size, hidden_units)
                # attention_weights.shape=(batch_size, time_steps, 64, 1)
                gru_outputs, gru_state_tuple, attention_weights = self.decoder(self.annotation_features,
                                                                               self.seq_embeddings, init_annot_state)

        with tf.variable_scope("logits") as logits_scope:
            print("[build_model] gru_outputs: {}\n\t\t gru_state_tuple: {}\n\t\t attention_weights: {}".format(
                gru_outputs, gru_state_tuple, attention_weights))
            logits = tf.layers.dense(inputs=gru_outputs, units=self.config.vocab_size,
                                     kernel_initializer=self.initializer,
                                     name=logits_scope.name)  # ==> (batch_size, time_steps, vocab_size)

        predict_outputs = tf.reshape(logits, (-1, tf.shape(logits)[-1]))
        print("[build_model] predict_outputs: ", predict_outputs)

        if self.mode == "inference":
            tf.nn.softmax(predict_outputs, name="softmax")
        else:
            # compute losses.
            label = tf.reshape(self.target_seqs, (-1,))
            print("[build_model] label: ", label)
            print("[build_model] predict_outputs: ", predict_outputs)
            self.loss, self.accuracy = self.loss_function(label, predict_outputs)

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    def build(self):
        """Creates all ops for training and evaluation"""
        self.build_inputs()
        self.build_image_embeddings()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_global_step()
