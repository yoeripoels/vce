"""Computing the explanation alignment cost (eac).
"""
import data.synthetic.structure as structure
import numpy as np
import dtw
from itertools import permutations, product
import math

'''
make each evaluation a class instance. first ->
  1. compute eac, we have class_a, class_b, lines, mod_a, mod_b
      --> compute_exp() and find_closest_explanation_preprocess() in explainmetric.py
'''

# also, re-implement plot_solution


def explanation_add_remove(all_lines, line_class_a, line_class_b, sp: structure.ShapeParser=None):
    """Creates a shape-interpolation between class_a -> class_b by removing all elements that are in a but not in b, then
    adding all elements that are in b but not in a.
    If a ShapeParser is supplied, create images, if not, return shapes.
    """
    to_remove = [i for i in range(len(all_lines)) if i in line_class_a and i not in line_class_b]
    to_add = [i for i in range(len(all_lines)) if i not in line_class_a and i in line_class_b]
    state = line_class_a.copy()
    states = [state.copy()]
    for i in to_remove:
        state.remove(i)
        states.append(state.copy())
    for i in to_add:
        state.append(i)
        states.append(state.copy())
    shapes = []
    for i in range(len(states)):
        lines = [all_lines[j] for j in states[i]]
        shapes.append(structure.lines_to_shape(lines))
    if sp is None:
        return shapes
    else:
        return sp.random_modification_multi(shapes)


def l2_dist_eps(a, b, eps=0.001):
    return np.linalg.norm(a-b) + eps


def create_all_explanations(lines, class_a, class_b):
    """Given all lines and class_a -> class_b, create a list of all possible explanations, where each
    such explanation is a list of lines.
    """
    # get all lines we must change to go from a->b
    to_swap = [i for i in range(len(lines)) if
               (i in class_a and i not in class_b) or (i in class_b and i not in class_a)]
    n_dif = len(to_swap)
    all_swaps = list(permutations(list(range(n_dif))))
    explanations = []
    # create an explanation for each order of swaps
    for i in range(len(all_swaps)):
        states = [class_a.copy()]
        state = class_a.copy()
        for j in range(n_dif):
            line_idx = to_swap[all_swaps[i][j]]
            if line_idx in state:  # must be removed
                state.remove(line_idx)
            else:  # it was not in: should be added
                state.append(line_idx)
            states.append(state.copy())
        explanations.append(states)
    return explanations


def create_all_intermediate_states(lines, class_a, class_b):
    """Given all lines and class_a -> class_b, create a list of configurations, where each configuration indicates
    which lines have been changed from a -> b
    """
    to_swap = [i for i in range(len(lines)) if
               (i in class_a and i not in class_b) or (i in class_b and i not in class_a)]
    n_dif = len(to_swap)
    common_state = set(class_a).intersection(set(class_b))  # these lines are always present
    swap_conf = list(product(range(2), repeat=n_dif))  # which lines have we swapped
    states = []
    for conf in swap_conf:  # for each possible combination of lines we have swapped
        state = [to_swap[i] for i in range(n_dif) if conf[i] == 1]
        state = list(set(state).union(common_state))
        state.sort()
        states.append(tuple(state))  # append
    return states


def dtw_to_solution(dtw):
    """From a dtw solution object, give the explanation alignment + cost
    """
    query_path = dtw.index1
    index_path = dtw.index2
    sol = []
    for i in range(len(query_path)):
        sol.append((index_path[i], query_path[i]))
    return sol, dtw.distance


def closest_explanation(explanations, lines, class_a, class_b, mod, sp, metric=l2_dist_eps, window=4):
    """Given a list of candidate explanations, compute each of their alignment w.r.t. some optimal explanation that
    changes 1 factor at a time.

    Returns the costs of these alignments, the identified closest ground-truth explanations, and the mappings between
    the candidate and ground-truth explanations.
    """
    # set up output
    ne = len(explanations)
    closest_dist = [math.inf for _ in range(ne)]
    closest_expl = [[] for _ in range(ne)]
    closest_sol = [[] for _ in range(ne)]

    # flatten our inputs
    shape_original = explanations[0][0].shape
    input_expl = [[x.flatten() for x in expl] for expl in explanations]

    # set up all expl paths we can take
    expl_paths = create_all_explanations(lines, class_a, class_b)

    # initialize window sizes for dtw
    window_size = [max(window, abs(len(expl_paths[0]) - len(expl))) for expl in explanations]

    # preprocess images
    states = create_all_intermediate_states(lines, class_a, class_b)
    state_to_image = {}
    for s in states:
        shape = structure.lines_to_shape([lines[i] for i in s])
        # sort our indices, so we can hash them
        s = list(s)
        s.sort()
        s = tuple(s)
        state_to_image[s] = sp.apply_random_modification(shape, *mod).flatten()

    # go through all explanation paths to find the bast aligning
    for i, expl in enumerate(expl_paths):
        expl_images = []
        for state in expl:  # create explanation from cached images
            state.sort()  # sort as dict is sorted
            expl_images.append(state_to_image[tuple(state)])

        for j in range(ne):  # for each explanation, check alignment to the ground truth explanation
            align = dtw.dtw(input_expl[j], expl_images, dist_method=metric, keep_internals=True,
                            step_pattern=dtw.symmetric1,
                            window_type='sakoechiba', window_args={'window_size': window_size[j]})
            sol, dist = dtw_to_solution(align)
            if dist < closest_dist[j]:  # check if this ground truth explanation is best for our candidate
                closest_dist[j] = dist
                closest_expl[j] = expl_images
                closest_sol[j] = sol

    # reshape our explanation images to the original images' shape
    closest_expl_r = [[x.reshape(shape_original) for x in c_e] for c_e in closest_expl]
    return closest_dist, closest_expl_r, closest_sol


def compute_eac(data_explanation, lines, classes, pair, pair_modification, sp, mod_from='a'):
    """Compute the eac given a list of explanations. Note that for each 'true' explanation we can provide
    multiple candidate explanations (e.g., from different interpolation methods), which are evaluated at once.

    Assume data_explanation is a list of candidate-lists, where each candidate list is a list of datapoints.
    --> e.g, data_explanations = [ [[x000, x001, x002], [x010, x011, x012, x013]], [[x100, x101], [x110, x111, x112]] ]
        where x_ijk -> i = the explained pair, j = the candidates for this pair, k = the datapoint in the explanation
    """

    # if we only supplied a list of explanations, make each explanation a list of 1 element
    if not isinstance(data_explanation[0][0], list):
        data_explanation = [[x] for x in data_explanation]
    num_candidate = len(data_explanation[0])

    eac = [[] for _ in range(num_candidate)]   # for each set of candidates, keep track of total cost
    solutions_expl = [[] for _ in range(num_candidate)]  # same for the closest identified explanations
    solutions_map = [[] for _ in range(num_candidate)]  # and the mapping from candidate<->target explanation

    for explanations, (a, b), (mod_a, mod_b) in zip(data_explanation, pair, pair_modification):
        mod = mod_a if mod_from == 'a' else mod_b
        costs, solution_expl, solution_map = closest_explanation(explanations, lines, classes[a], classes[b], mod, sp)
        for i, c in enumerate(costs):
            eac[i].append(c)
        for i, s_e in enumerate(solution_expl):
            solutions_expl.append(s_e)
        for i, s_m in enumerate(solution_map):
            solutions_map.append(s_m)

    return eac, solutions_expl, solutions_map
