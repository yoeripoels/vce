"""Generation of explanations using representation models.

We define a [set of methods/class] to generate explanations for models following the REPR interface.
"""
import numpy as np


def get_increasing(perm_array):
    """From an array of 1s and 0s, return all possible 'increasing' arrays, i.e., arrays of the same length with
    0s swapped to 1s
    """
    n = len(perm_array)
    for i in range(len(perm_array)):
        assert perm_array[i] == 0 or perm_array[i] == 1

    cur = [[0], [1]] if perm_array[-1] == 0 else [[1]]

    for i in reversed(range(0, n - 1)):
        new_cur = []
        for c in cur:
            new_cur.append([1] + c)
            if perm_array[i] == 0:
                new_cur.append([0] + c)
        cur = new_cur
    cur = [np.array(c) for c in cur]
    return cur


# interpolate from x_a -> x_b, and from x_a -> most likely alternative class
# selection of candidates (threshold)
# then linear interpolation, swap dimensions at once, and explanation generator
# for explanation generator -> build graph. fill graph efficiently. dijkstra on graph.
