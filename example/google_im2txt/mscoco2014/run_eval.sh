#!/usr/bin/env bash
python3 evaluate.py --input_file_pattern="/dataset/mscoco2014/test-?????-of-00008" --checkpoint_dir="./ckpt" --eval_dir="./eval"