# -*- coding: utf-8 -*-
# @File    : google_im2txt/validation.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/11
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary
from utils.emb_json import store_json_file

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # only /gpu:gpu_id is visible

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   FLAGS.checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    # q&a: understand follow snippets
    # filenames = []
    # for file_pattern in FLAGS.input_files.split(","):
    #     # tf.gfile.Glob(pattern) Returns a list of files that match the given pattern(s)
    #     filenames.extend(tf.gfile.Glob(file_pattern))
    # note: assert FLAGS.input_files == 'utils/test_file_abspath_flickr30k'
    with open(FLAGS.input_files, 'r') as f:
        filenames = f.readlines()
    filenames = [filename.strip() for filename in filenames]
    tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), FLAGS.input_files)

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    with tf.Session(graph=g, config=session_config) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)

        json_file = list()
        for count, filename in enumerate(filenames):
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)  # 返回的是beam_size个caption
            # print("Captions for image %s:" % os.path.basename(filename))

            for i, caption in enumerate(captions):
                img_caption_dict = {}
                img_caption_dict['filename'] = os.path.basename(filename)
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                img_caption_dict['caption'] = sentence
                json_file.append(img_caption_dict)
            if count % 50 == 0:
                print("counter: %d" % count)

        store_json_file("im2txt_flickr30k_cap_google.json", json_file)


if __name__ == "__main__":
    tf.app.run()
