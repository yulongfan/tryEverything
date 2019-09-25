# -*- coding: utf-8 -*-
# @File    : image_caption/attention_visualization.py
# @Info    : @ TSMC-SIGGRAPH, 2018/8/31
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 
"""
note: in ssh, Matplotlib chooses Xwindows backend by default. You need to set matplotlib to not use the Xwindows backend.
Add this code to the start of your script (before importing pyplot) and try again:

import matplotlib
matplotlib.use('Agg')

"""
import os
from math import ceil

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')  # Add this code to the start of your script (before importing pyplot)

import matplotlib.pyplot as plt


def evaluate():
    attention_plot = np.zeros((20, 64))
    text = "A caption text embedded into an image itself is of course not accessible to Google."
    result = text.split()
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


if not os.path.exists("plot_attention"):
    os.makedirs("plot_attention")


def plot_attention(image_path, captions, attention_plot):
    """
    :param image_path: a string, denotes image path
    :param captions: a list, denotes captions
    :param attention_plot: numpy ndarray
    :return:
    """
    basename = os.path.basename(image_path)

    temp_image = np.array(Image.open(image_path))

    fig = plt.figure(figsize=(23.384, 16.534), dpi=96)  # fig = plt.figure(figsize=(11.692, 8.267), dpi=300)

    len_result = len(captions)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        if len_result > 2:
            ax = fig.add_subplot(ceil(len_result / 2), ceil(len_result / 2), l + 1)
        else:  # captions too short!
            ax = fig.add_subplot(1, 2, l + 1)
        ax.set_title(captions[l])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.savefig(os.path.join("plot_attention", "{}.png".format(basename.split('.')[0])), bbox_inches='tight')
    plt.clf()


#################################
#   try it on your own images   #
#################################
def captions_on_my_own_imgs(img_dir="img"):
    """captions on your own images
    :param img_dir: images directory, where including jpg images
    :return:
    """
    images = os.listdir(img_dir)
    for image_name in images:
        image_path = os.path.join("img", image_name)
        result, attention_plot = evaluate()
        print('Prediction Caption:', ' '.join(result))
        plot_attention(image_path, result, attention_plot)


# attention_weight visualization : test demo
if __name__ == '__main__':
    captions_on_my_own_imgs()
