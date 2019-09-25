# -*- coding: utf-8 -*-
# @File    : google_im2txt/get_uni_caption.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/9
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import re

from utils.emb_json import load_json_file, store_json_file

data = load_json_file("google_im2txt_cap.json")

pattern = re.compile(r'COCO_val2014_(\d+).jpg')  # find integer number

# json_file = list()
# for item in data:
#     img_cap_dict = {}
#     image_id = pattern.findall(item['filename'])
#     img_cap_dict['image_id'] = int(image_id[0])
#     img_cap_dict['caption'] = item['caption']
#     json_file.append(img_cap_dict)
#
# store_json_file("val_filter_version.json", json_file)

json_file_uni = list()
for i, item in enumerate(data):
    if i % 3 == 0:
        img_cap_dict = {}
        image_id = pattern.findall(item['filename'])
        img_cap_dict['image_id'] = int(image_id[0])
        img_cap_dict['caption'] = item['caption']
        json_file_uni.append(img_cap_dict)

store_json_file("google_im2txt_test_uni_version.json", json_file_uni)
