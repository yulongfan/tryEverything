# -*- coding: utf-8 -*-
# @File    : image_caption/train_model.py
# @Info    : @ TSMC-SIGGRAPH, 2018/8/27
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

import os
import time

import tensorflow as tf
from image_cap_model import ImageCapModel, ModelConfig

os.environ["CUDA_VISIBLE_DEVICES"] = '3'  # only /gpu:gpu_id is visible


def main(_):
    modelconfig = ModelConfig()
    cap_model = ImageCapModel(modelconfig, "train")
    cap_model.build()

    optimizer = tf.train.AdamOptimizer()
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cap_model.loss, cap_model.global_step)

    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    with tf.Session(config=configproto) as sess:
        sess.run(tf.global_variables_initializer())
        start = time.time()
        saver = tf.train.Saver(max_to_keep=1)

        if not os.path.exists("ckpt_dir"):
            os.makedirs("ckpt_dir")

        while True:
            try:
                step = tf.train.global_step(sess, cap_model.global_step)
                ins, ops, loss_print, acc, _ = sess.run([cap_model.input_seqs, cap_model.target_seqs,
                                                         cap_model.loss, cap_model.accuracy, train_op])
                if step % 10 == 0:
                    print("step {}, Loss {:.6f}, acc {:.6f}".format(step, loss_print, acc))
                if step > 0 and step % 100 == 0:
                    print("step {}, Loss {:.6f}, acc {:.6f}".format(step, loss_print, acc))
                    saver.save(sess, os.path.join("ckpt_dir", 'model.ckpt'), step)
            except tf.errors.OutOfRangeError:
                print("  --- arriving at the end of data ---  ")
                break
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    tf.app.run()
