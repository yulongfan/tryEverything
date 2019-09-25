#!/usr/bin/env bash
# Run the training script.
python3 train.py \
    --input_file_pattern="/dataset/flickr30k_tfrecord/train-?????-of-00256" \
    --inception_checkpoint_file="/dataset/inception_v3.ckpt" \
    --train_dir="./ckpt" \
    --train_inception=false \
    --number_of_steps=40000 \
    --log_every_n_steps=100
