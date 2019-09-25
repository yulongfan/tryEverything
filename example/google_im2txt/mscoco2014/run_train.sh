#!/usr/bin/env bash
# Run the training script.
python3 train.py \
    --input_file_pattern="/dataset/mscoco2014/train-?????-of-00256" \
    --inception_checkpoint_file="/dataset/inception_v3.ckpt" \
    --train_dir="./ckpt" \
    --train_inception=false \
    --number_of_steps=150000 \
    --log_every_n_steps=500
