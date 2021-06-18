"""Misc helper functions
"""
import random
import numpy as np


def random_pair(n, n_swap):
    list_a = [random.choice([True, False]) for i in range(n)]
    # swap n_swap
    to_swap = np.random.choice(n, n_swap, replace=False)
    list_b = list(list_a)
    for v in to_swap:
        list_b[v] = not list_b[v]
    list_a = mask_to_idx(list_a)
    list_b = mask_to_idx(list_b)
    return list_a, list_b


def binary_bool_list(n):
    if n == 0:
        return []
    total = [[bool(int(j)) for j in '{:0{}b}'.format(i, n)] for i in range(2**n)]
    if n > 1:
        # delete all True and all False
        total = total[1:-1]
    return total


def mask_to_idx(in_list):
    out = []
    for i, boolean in enumerate(in_list):
        if boolean:
            out.append(i)
    return out