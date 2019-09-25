# -*- coding: utf-8 -*-
# @File    : google_im2txt/emb_json.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/10
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import json


def store_json_file(filename, data):
    with open(filename, 'w') as json_file:
        json_file.write(json.dumps(data, indent=4))


def load_json_file(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
        return data
