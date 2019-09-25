# -*- coding: utf-8 -*-
# @File    : google_im2txt/dataset_starndard_partition.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/9
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os.path
import shutil

baseDir = "/dataset/flickr8k"
imgDir = os.path.join(baseDir, 'flickr8k_images')


def data_partition(partition_file, obj_sub_dir, caption_dict, captions):
    """
    :param partition_file:  data partition file
    :param obj_sub_dir: one of strings "train","val","test"
    :param caption_dict:  a dict map image name to captions index
    :param captions:  a file including all captions
    :return:
    """
    with open(os.path.join(baseDir, partition_file), 'r') as f:
        filenames = f.readlines()

    fullnames = list()
    fetch_captions = list()
    for name in filenames:
        fullname = os.path.join(imgDir, name.strip())
        fullnames.append(fullname)
        for i in caption_dict[name.strip()]:
            fetch_captions.append(captions[i].strip())

    with open(os.path.join(baseDir, "flickr8k_%s_captions" % obj_sub_dir), 'w') as f:
        f.write('\n'.join(fetch_captions))

    obj_dir = os.path.join(baseDir, obj_sub_dir)
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)

    for file in fullnames:
        basename = os.path.basename(file)
        shutil.copyfile(file, os.path.join(obj_dir, basename))


if __name__ == '__main__':
    with open(os.path.join(baseDir, 'flickr8k_captions'), 'r') as f:
        _captions = f.readlines()
    _caption_dict = dict()
    for ids, caption in enumerate(_captions):
        _img_filename = caption.strip().split('#')[0]
        _caption_dict.setdefault(_img_filename, [])
        _caption_dict[_img_filename].append(ids)

    data_partition('Flickr_8k.trainImages.txt', 'train', _caption_dict, _captions)
    data_partition('Flickr_8k.devImages.txt', 'val', _caption_dict, _captions)
    data_partition('Flickr_8k.testImages.txt', 'test', _caption_dict, _captions)
