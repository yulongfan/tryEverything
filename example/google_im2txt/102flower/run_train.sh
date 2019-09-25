#!/usr/bin/env bash
# Run the training script.
python3 train.py \
    --input_file_pattern="/dataset/102flowers_tfrecord/train-?????-of-00128" \
    --inception_checkpoint_file="/dataset/inception_v3.ckpt" \
    --train_dir="./ckpt" \
    --train_inception=false \
    --number_of_steps=20000 \
    --log_every_n_steps=100
