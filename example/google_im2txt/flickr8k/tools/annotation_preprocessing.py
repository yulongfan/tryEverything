# -*- coding: utf-8 -*-
# @File    : google_im2txt/annotation_preprocessing.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/9
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os.path
import re

baseDir = "/dataset/flickr8k"
file = os.path.join(baseDir, "Flickr8k.token.txt")
assert os.path.exists(file)
with open(file, 'r') as f:
    lines = f.readlines()

newlines = list()
for line in lines:
    strinfo = re.compile('#\d{1}\s')
    newlines.append(strinfo.sub('#', line.strip()))

with open(os.path.join(baseDir, "flickr8k_captions"), 'w') as f:
    f.write('\n'.join(newlines))

