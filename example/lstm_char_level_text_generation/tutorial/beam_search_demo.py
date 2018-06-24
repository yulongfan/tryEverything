# -*- coding: utf-8 -*-
# @File    : tryEverything/beam_search_demo.py
# @Info    : @ TSMC-SIGGRAPH, 2018/6/20
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


from math import log

from numpy import argmax
from numpy import array


# greedy decoder
def greedy_decoder(data):
    # index for largest probability each row
    return [argmax(s) for s in data]


# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


# define a sequence of 10 words over a vocab of 5 words
# dataset = [[0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.5, 0.4, 0.3, 0.2, 0.1],
#         [0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.5, 0.4, 0.3, 0.2, 0.1],
#         [0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.5, 0.4, 0.3, 0.2, 0.1],
#         [0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.5, 0.4, 0.3, 0.2, 0.1],
#         [0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.5, 0.4, 0.3, 0.2, 0.1]]
data = [[0.1, 0.25, 0.3, 0.4, 0.45],
        [0.45, 0.4, 0.3, 0.2, 0.15],
        [0.25, 0.3, 0.25, 0.4, 0.35],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]]

data = array(data)

# decode sequence
result = beam_search_decoder(data, 3)
# print result
for seq in result:
    print(seq)

print("-*-"*20)
# greedy decode sequence
greedy_result = greedy_decoder(data)
print(greedy_result)
