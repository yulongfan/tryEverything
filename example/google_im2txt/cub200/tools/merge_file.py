# -*- coding: utf-8 -*-
# @File    : google_im2txt/__init__.py.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/7
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.

import os
import shutil

base_path = '/dataset'

cub_merge_data = os.path.join(base_path, 'cub_merge_data', 'meta')
# cub_merge_data = os.path.join(cub_data_dir, 'meta')
if not os.path.exists(cub_merge_data):
    os.mkdir(cub_merge_data)
else:
    shutil.rmtree(cub_merge_data)
    os.mkdir(cub_merge_data)

# copy image dir to target dir
image_dir = os.path.join(base_path, 'CUB_200_2011/CUB_200_2011', 'images')
all_image_file = os.listdir(image_dir)
for image in all_image_file:
    print(image)
    shutil.copytree(os.path.join(image_dir, image),
                    os.path.join(cub_merge_data, image))

word_dir = os.path.join(base_path, 'cvpr2016_cub', 'text_c10')
all_word_file = os.listdir(word_dir)
all_word_file = filter(lambda x: not x.endswith('.t7'), all_word_file)
for word in all_word_file:
    word_file = os.listdir(os.path.join(word_dir, word))
    word_file = filter(lambda x: x.endswith('.txt'), word_file)
    for w in word_file:
        shutil.copy(os.path.join(word_dir, word, w),
                    os.path.join(cub_merge_data, word, w))
