# -*- coding: utf-8 -*-
# @File    : image_caption/run_inference.py
# @Info    : @ TSMC-SIGGRAPH, 2018/8/27
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 
"""
note: in ssh, Matplotlib chooses Xwindows backend by default. You need to set matplotlib to not use the Xwindows backend.
Add this code to the start of your script (before importing pyplot) and try again:

import matplotlib
matplotlib.use('Agg')
"""
import math
import os

import tensorflow as tf
from tqdm import tqdm

# We'll generate plots of attention_weights in order to see which parts of an image our model focuses on during captioning
import inference_wrapper
from configuration import ModelConfig
from utils.caption_generator import CaptionGenerator
from attention_visualization import plot_attention
# import matplotlib
# matplotlib.use('Agg')  # Add this code to the start of your script (before importing pyplot)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # only /gpu:gpu_id is visible

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "ckpt_dir",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("input_files_dir", "img",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")


def main(_):
    # Build the inference graph
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(ModelConfig(), FLAGS.checkpoint_path)
    g.finalize()

    filenames = list(filter(lambda x: x.endswith('.jpg'), os.listdir(FLAGS.input_files_dir)))
    filenames = [os.path.join(FLAGS.input_files_dir, filename) for filename in filenames]
    print("Running de-rain infer on %d files from directory: %s" % (len(filenames), FLAGS.input_files_dir))
    print(filenames)

    index_word = model.index_word

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=g, config=config) as sess:
        # Load the model from checkpoint
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default parameters.
        generator = CaptionGenerator(model)

        if not os.path.exists("plot_attention"):
            os.makedirs("plot_attention")

        for i, filename in tqdm(enumerate(filenames)):
            # with tf.gfile.GFile(filename, "rb") as f:
            #     image = f.read()
            captions = generator.beam_search(sess, filename)  # return beam_size captions
            print("Captions for image %s:" % os.path.basename(filename))
            for j, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence_list = [index_word[w] for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence_list)
                print("  %d) %s (p=%f)" % (j, sentence, math.exp(caption.logprob)))
                if j == 1:
                    print(len(caption.attenplot))
                    plot_attention(filename, sentence_list, caption.attenplot)


if __name__ == '__main__':
    tf.app.run()
