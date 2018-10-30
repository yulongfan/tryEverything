# -*- coding: utf-8 -*-
# @File    : tv_denoise_batch_rgb.py
# @Info    : @ TSMC-SIGGRAPH, 2018/10/25
# @Desc    : refer to https://github.com/WanglifuCV/TotalVariationAlgorithms/blob/master/TVdenoise.m
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.

"""
Denoising/smoothing a given image y with the isotropic total variation.
The iterative algorithm converges to the unique image x minimizing

||x-y||_2^2/2 + lambda.TV(x)
TV(x)=||Dx||_1,2, where D maps an image to its gradient field.

total variation algorithm tends to be over-smoothed with missing fine image details
"""

import numpy as np
from scipy.misc import imsave
from PIL import Image
from tqdm import tqdm


def TVdenoising(img, lamb=0.1, tau=0.01, num_iter=100):
    """
    :param img: Image to be processed
    :param lamb: regularization parameter controlling the amount of denoising
    :param tau: proximal parameter > 0; influences the convergence speed
    :param num_iter: Number of iterations. If omitted or empty defaults to 100
    :return:
    """
    rho = 1.99  # relaxation parameter, in [1,2)
    sigma = 1 / tau / 8
    assert len(img.shape) == 4
    batch, height, width, channel = img.shape

    def opd(x):
        """
        :param x:
        :return:
        """
        wx = np.expand_dims(np.concatenate([np.diff(x, 1, 1), np.zeros((batch, 1, width, channel))], 1), 3)  # (N,H,W,1,C)
        hx = np.expand_dims(np.concatenate([np.diff(x, 1, 2), np.zeros((batch, height, 1, channel))], 2), 3)  # (N,H,W,1,C)
        return np.concatenate([wx, hx], 3)  # (N,H,W,2,C)

    def opdadj(u):
        """
        :param u:
        :return:
        """
        da = - np.concatenate([np.expand_dims(u[:, 0, :, 0, ...], 1), np.diff(u[:, :, :, 0, ...], 1, 1)], 1)  # (N,H,W,C)
        dj = np.concatenate([np.expand_dims(u[:, :, 0, 1, ...], 2), np.diff(u[:, :, :, 1, ...], 1, 2)], 2)  # (N,H,W,C)
        return da - dj  # (N,H,W,C)

    def prox_tau_f(x):
        """
        :param x:
        :return:
        """
        return (x + tau * img) / (1 + tau)

    def prox_sigma_g_conj(u):
        """
        :param u:
        :return:
        """
        max_u = np.sqrt(np.sum(np.square(u), axis=3, keepdims=True)) / lamb + 1e-8
        return u / max_u

    x2 = img  # initialization of the solution
    u2 = prox_sigma_g_conj(opd(x2))  # initialization of the dual solution
    x = None
    for i in tqdm(range(num_iter)):
        x = prox_tau_f(x2 - tau * opdadj(u2))
        u = prox_sigma_g_conj(u2 + sigma * opd(2 * x - x2))
        x2 = x2 + rho * (x - x2)
        u2 = u2 + rho * (u - u2)

    return x.astype(np.float32)


def denoiser(img):
    return TVdenoising(img, num_iter=200)


if __name__ == '__main__':
    inputs = np.asarray(Image.open("case1.jpg"), np.float32) / 255  # initial image
    inputs = np.expand_dims(inputs, 0)
    img = denoiser(inputs)
    print(img.shape)
    imsave('case1_derain.jpg', img[0])
