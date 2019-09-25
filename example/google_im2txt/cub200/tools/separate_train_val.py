# -*- coding: utf-8 -*-
# @File    : google_im2txt/separate_train_val.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/7
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os.path
import random
import shutil

basedir = "/dataset/cub_merge_data"
metadir = os.path.join(basedir, 'meta')
subdir = os.listdir(metadir)

train_list = list()
val_list = list()
test_list = list()

for sdir in subdir:
    fullsdir = os.path.join(metadir, sdir)
    files = os.listdir(fullsdir)
    filenames = list(set([file.split(".")[0] for file in files]))
    # Shuffle the ordering of files.
    random.seed(12345)
    random.shuffle(filenames)
    train_cutoff = int(0.85 * len(filenames))
    val_cutoff = int(0.9 * len(filenames))
    print(len(filenames), train_cutoff, val_cutoff)
    for prefix_name in filenames[:train_cutoff]:
        train_list.append(os.path.join(fullsdir, prefix_name))
    for prefix_name in filenames[train_cutoff:val_cutoff]:
        val_list.append(os.path.join(fullsdir, prefix_name))
    for prefix_name in filenames[val_cutoff:]:
        test_list.append(os.path.join(fullsdir, prefix_name))

train_dir = os.path.join(basedir, 'train')
val_dir = os.path.join(basedir, 'val')
test_dir = os.path.join(basedir, 'test')

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

train_caption_list = list()
for i in train_list:
    basename = os.path.basename(i)
    img_name = basename + ".jpg"
    shutil.copyfile(i + ".jpg", os.path.join(train_dir, img_name))

    caption_file = i + ".txt"
    with open(caption_file, 'r') as f:
        captions = f.readlines()
    for line in captions:
        train_caption_list.append(img_name + "#" + line.strip())

print("copy training images finished!")

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

val_caption_list = list()
for i in val_list:
    basename = os.path.basename(i)
    img_name = basename + ".jpg"
    shutil.copyfile(i + ".jpg", os.path.join(val_dir, img_name))

    caption_file = i + ".txt"
    with open(caption_file, 'r') as f:
        captions = f.readlines()
    for line in captions:
        val_caption_list.append(img_name + "#" + line.strip())

print("copy validating images finished!")

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

test_caption_list = list()
for i in test_list:
    basename = os.path.basename(i)
    img_name = basename + ".jpg"
    shutil.copyfile(i + ".jpg", os.path.join(test_dir, img_name))

    caption_file = i + ".txt"
    with open(caption_file, 'r') as f:
        captions = f.readlines()
    for line in captions:
        test_caption_list.append(img_name + "#" + line.strip())

print("copy testing images finished!")

with open(os.path.join(basedir, "cub_train_captions"), 'w') as f:
    f.write('\n'.join(train_caption_list))
with open(os.path.join(basedir, "cub_val_captions"), 'w') as f:
    f.write('\n'.join(val_caption_list))
with open(os.path.join(basedir, "cub_test_captions"), 'w') as f:
    f.write('\n'.join(test_caption_list))

print("write captions files finished!")
