# -*- coding: utf-8 -*-
# @File    : tryEverything/kmeans_plus_plus.py
# @Info    : @ TSMC-SIGGRAPH, 2018/7/17
# @Desc    : refer to https://www.cnblogs.com/nocml/p/5150756.html
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import random
import string
from copy import copy
from random import choice

import matplotlib.pyplot as plt
import numpy as np

FLOAT_MAX = 1e100


class Point(object):
    __slots__ = ["word", "vector", "group", "distance"]  # only allow a fixed set of attributes.

    def __init__(self, word="<UNK>", vector=None, group=0, distance=0):
        self.word, self.vector, self.group, self.distance = word, vector, group, distance


def generate_points(num_points):
    """
    :param num_points: an integer, the number of data points
    :return:
    """
    points = [Point() for _ in range(num_points)]
    vmin, vmax = 100, 0
    for p in points:
        vector = np.random.choice(list(range(1000)), 2)
        word = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
        p.vector = vector
        p.word = word

        # # attention: if vmax - vmin=0, then (vector-vmin)/(vmax-vmin)=Nan
        vmin_hat, vmax_hat = vector.min(), vector.max()  # get min/max value
        if vmin_hat < vmin:
            vmin = vmin_hat
        if vmax_hat > vmax:
            vmax = vmax_hat

    for p in points:
        p.vector = np.subtract(p.vector, vmin) / np.subtract(vmax, vmin)

    return points


def nearest_cluster_center(point, cluster_centers):
    """Distance and index of the closest cluster center
    :param point: an instance of Point Object
    :param cluster_centers: a list including all of the cluster center point
    :return: a tuple, in which, the 1-th and 2-th elem denotes index and distance of the closest cluster center, respectively.
    """

    def dist_eclud(p_a, p_b):
        return np.sqrt(np.sum(np.square(np.subtract(p_a.vector, p_b.vector))))

    min_index = point.group
    min_dist = FLOAT_MAX

    for i, cc in enumerate(cluster_centers):
        d = dist_eclud(cc, point)
        if min_dist > d:
            min_dist = d
            min_index = i
            point.distance = round(d, 3)
    min_index_dist = (min_index, min_dist)
    return min_index_dist


def kpp(points, cluster_centers):
    cluster_centers[0] = copy(choice(points))  # hint: random choice one point as cluster centers
    distance = [0.0 for _ in range(len(points))]  # initialize all distance

    for i in range(1, len(cluster_centers)):
        sum = 0
        for j, point in enumerate(points):
            distance[j] = nearest_cluster_center(point, cluster_centers[:i])[1]
            sum += distance[j]

        sum *= random.random()

        for j, dist in enumerate(distance):
            sum -= dist
            if sum > 0:
                continue
            cluster_centers[i] = copy(points[j])
            break

    # gather all points to the nearest cluster centers
    for p in points:
        p.group = nearest_cluster_center(p, cluster_centers)[0]
    for i, cc in enumerate(cluster_centers):
        cc.group = i

    # for p in points:
    #     print(p.word, p.vector, p.group)
    #
    # print("--*--" * 10)
    # for cc in cluster_centers:
    #     print(cc.word, cc.vector, cc.group)
    # print("--+--" * 10)

    return points, cluster_centers


def kmeans(points, nclusters):
    cluster_centers = [Point() for _ in range(nclusters)]

    # call k++ init
    points, cluster_centers = kpp(points, cluster_centers)

    lenpts10 = len(points) >> 10

    changed = 0
    while True:
        print("iteration...")
        # group element for centroids are used as counters
        for cc in cluster_centers:
            cc.vector = [0, 0]
            cc.group = 0

        for p in points:
            cluster_centers[p.group].group += 1
            cluster_centers[p.group].vector += p.vector

        for cc in cluster_centers:
            cc.vector /= cc.group

        # find closest centroid of each PointPtr
        changed = 0
        for p in points:
            min_i = nearest_cluster_center(p, cluster_centers)[0]
            if min_i != p.group:
                changed += 1
                p.group = min_i

        # stop when 99.9% of points are good
        if changed <= lenpts10:
            break

    for i, cc in enumerate(cluster_centers):
        cc.group = i

    return points, cluster_centers


if __name__ == '__main__':
    points = generate_points(500)
    points, cluster_centers = kmeans(points, 5)

    # for p in points:
    #     print(p.word, p.vector, p.group, p.distance)

    print("--##--" * 10)
    x = list()
    y = list()
    for cc in cluster_centers:
        print(cc.word, cc.vector, cc.group)
        px_list = list()
        py_list = list()
        for p in points:
            if p.group == cc.group:
                px_list.append(p.vector[0])
                py_list.append(p.vector[1])
        x.append(px_list)
        y.append(py_list)

    print(len(x), len(y))
    print(x[0][0])
    color_list = ['red', 'blue', 'green', 'gray', 'orange', 'purple']
    for i, coord in enumerate(zip(x, y)):
        # refer to @Suranyi, [https://www.zhihu.com/question/37146648/answer/299029958]
        plt.scatter(coord[0], coord[1], s=200, label='$%d$' % i, c=color_list[i], marker='.', alpha=None, edgecolors='white')
        plt.legend()

    plt.show()
