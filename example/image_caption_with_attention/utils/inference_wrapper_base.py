# -*- coding: utf-8 -*-
# @File    : derain_reinfore/inference_wrapper_base.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/12
# @Desc    : refer to google's im2txt
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os.path

import tensorflow as tf


# pylint: disable=unused-argument


class InferenceWrapperBase(object):
    """Base wrapper class for performing inference with an image-to-text model."""

    def __init__(self):
        pass

    def build_model(self, model_config):
        """Builds the model for inference.
        Args:
          model_config: Object containing configuration for building the model.
        Returns:
          model: The model object.
        """
        tf.logging.fatal("Please implement build_model in subclass")

    def _create_restore_fn(self, checkpoint_path, saver):
        """Creates a function that restores a model from checkpoint.
        Args:
          checkpoint_path: Checkpoint file or a directory containing a checkpoint
            file.
          saver: Saver for restoring variables from the checkpoint file.
        Returns:
          restore_fn: A function such that restore_fn(sess) loads model variables
            from the checkpoint file.
        Raises:
          ValueError: If checkpoint_path does not refer to a checkpoint file or a
            directory containing a checkpoint file.
        """
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if not checkpoint_path:
                raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

        def _restore_fn(sess):
            tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
            saver.restore(sess, checkpoint_path)
            tf.logging.info("Successfully loaded checkpoint: %s",
                            os.path.basename(checkpoint_path))
            print("Successfully loaded checkpoint: ", os.path.basename(checkpoint_path))

        return _restore_fn

    def build_graph_from_config(self, model_config, checkpoint_path):
        """Builds the inference graph from a configuration object.
        Args:
          model_config: Object containing configuration for building the model.
          checkpoint_path: Checkpoint file or a directory containing a checkpoint
            file.
        Returns:
          restore_fn: A function such that restore_fn(sess) loads model variables
            from the checkpoint file.
        """
        tf.logging.info("Building model.")
        self.build_model(model_config)
        saver = tf.train.Saver()

        return self._create_restore_fn(checkpoint_path, saver)

    def feed_image(self, sess, encoded_image):
        """Feeds an image and returns the initial model state.

        See comments at the top of file.

        Args:
          sess: TensorFlow Session object.
          encoded_image: An encoded image string.

        Returns:
          state: A numpy array of shape [1, state_size].
        """
        tf.logging.fatal("Please implement feed_image in subclass")

    def inference_step(self, sess, input_feed, state_feed, annotation_features_feed):
        """Runs one step of inference.
        Args:
          sess: TensorFlow Session object.
          input_feed: A numpy array of shape [batch_size].
          state_feed: A numpy array of shape [batch_size, state_size].
          annotation_features_feed: A numpy array of shape [batch_size, 64, embedding_size]

        Returns:
          softmax_output: A numpy array of shape [batch_size, vocab_size].
          new_state: A numpy array of shape [batch_size, state_size].
        """
        tf.logging.fatal("Please implement inference_step in subclass")
