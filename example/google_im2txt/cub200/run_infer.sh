#!/bin/bash
python3 run_inference.py --checkpoint_path=ckpt --vocab_file=/dataset/cub200_tfrecord/word_counts.txt --input_files=/dataset/cub_merge_data/American_Goldfinch_0017_32272.jpg
