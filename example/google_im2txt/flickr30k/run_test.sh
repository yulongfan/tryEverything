#!/bin/bash
python3 run_inference.py --checkpoint_path=ckpt --vocab_file=/dataset/102flowers_tfrecord/word_counts.txt --input_files=/dataset/oxford_102flowers/jpg/image_00001.jpg
