# -*- coding: utf-8 -*-
# @File    : image_caption/differential_evolution_algorithm.py
# @Info    : @ TSMC-SIGGRAPH, 2018/9/2
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 
"""copy from @wlh"""

import numpy as np
from tqdm import tqdm


# N=100   # 种群大小
# G=100   # 最大迭代次数
# F = 0.5 # 突变算子
# Cr = 0.3    # 交叉概率
# D = 2   # 解空间的维度
# 我有点迷惑,上界和下界是个向量啊, 这个大小是怎么确定的,向量大小啊....
def de(n=4, m_size=20, f=0.5, cr=0.3, iterate_times=100, x_l=np.array([0, 1, 0, 2]), x_u=np.array([5, 6, 8, 4])):
    """
    :param n: 解空间的维度
    :param m_size: 种群大小(即个体总数)
    :param f: 变异系数
    :param cr: crossover probability, 交叉概率
    :param iterate_times: 总迭代次数
    :param x_l: 优化空间下界
    :param x_u: 优化空间上界
    :return:
    """
    # 初始化
    x_all = np.zeros((iterate_times, m_size, n))
    for i in range(m_size):  # 种群大小
        x_all[0][i] = x_l + np.random.random() * (x_u - x_l)    # 初始化第0次迭代值
    print('差分进化种群初始化完成！')
    print('解空间维度为{}, 优化空间c_min={}, c_max={}'.format(n, x_l, x_u))
    for g in tqdm(range(iterate_times - 1)):
        # print('当前第{}代'.format(g))
        for i in range(m_size):
            # 变异操作,对第g代随机抽取三个组成一个新的个体,对于第i个新个体来说,原本的第i个个体与它没有关系
            x_g_without_i = np.delete(arr=x_all[g], obj=i, axis=0)  # 删掉了当前的旧个体
            np.random.shuffle(x_g_without_i)  # 通过乱序实现随机抽取
            h_i = x_g_without_i[1] + f * (x_g_without_i[2] - x_g_without_i[3])  # 新的当前个体
            # 变异操作后,h_i个体可能会超过上下限区间,为了保证在区间以内对超过区间外的值赋值为相邻的边界值
            # 先处理上边界, 如果h_i[item]>上限,则取上限; 若小于上限, 则维持h_i[item]
            h_i = [h_i[item] if h_i[item] < x_u[item] else x_u[item] for item in range(n)]
            # 再处理下边界, 道理与上述相同
            h_i = [h_i[item] if h_i[item] > x_l[item] else x_u[item] for item in range(n)]
            # 交叉操作, 对变异后的个体, 根据交叉概率与适应度来确定最后的个体
            # print(h_i)
            # 交叉的时候,是逐条染色体进行交叉的, 如果随机数大于交叉概率, 则使用原始个体的染色体作为后代染色体,反之使用突变中间体的染色体
            v_i = np.array([x_all[g][i][j] if (np.random.random() > cr) else h_i[j] for j in range(n)])
            # 根据评估函数来确定是否更新新的个体
            if evaluate_func2(x_all[g][i]) > evaluate_func2(v_i):
                x_all[g + 1][i] = v_i
            else:
                x_all[g + 1][i] = x_all[g][i]
    evalute_result = [evaluate_func2(x_all[iterate_times - 1][i]) for i in range(m_size)]
    best_x_i = x_all[iterate_times - 1][np.argmin(evalute_result)]

    print("evalute_result: {}".format(evalute_result))
    print("min evalute_result idx: {}".format(np.argmin(evalute_result)))
    print("best individual: {}".format(best_x_i))

    print("x_all[init_generation]: {}".format(x_all[0]))
    print("x_all[last_generation]: {}".format(x_all[-1]))


def evaluate_func(x):
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    return 4 * a ** 2 - 3 * b + 5 * c ** 3 - 6 * d  # 4a^2 - 3b + 5c^3 - 6d


def evaluate_func2(x):
    """a - b + 2a^2 + 2ab + b^2
    :param x: numpy ndarray,with shape (2,)
    :return:
    """
    a = x[0]
    b = x[1]
    return a - b + 2 * a ** 2 + 2 * a * b + b ** 2  # 该函数需要求的是最小值,所以适应度在挑选的时候自然是越小越好argmin


if __name__ == '__main__':
    # de()
    # de(n=2, m_size=5, f=0.5, cr=0.3, iterate_times=5000, x_l=np.array([-10, -10]), x_u=np.array([5, 6]))
    de(n=2, m_size=5, f=0.5, cr=0.3, iterate_times=5000, x_l=np.array([1, 0]), x_u=np.array([5, 6]))
