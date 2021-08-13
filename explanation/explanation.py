"""Generation of explanations using representation models.

We define a [set of methods/class] to generate explanations for models following the REPR interface.
"""
import numpy as np
from model.r_model import VAECE
from model.base import REPR
import math
from dijkstra import Graph, DijkstraSPF


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


def get_encoding(model: REPR, data_a, data_b, x_from='a'):
    data_a = np.array([data_a])
    data_b = np.array([data_b])
    z_y_a = model.encode_y(data_a).numpy()
    z_y_b = model.encode_y(data_b).numpy()
    if x_from == 'a':
        z_x = model.encode_x(data_a).numpy()
    elif x_from == 'b':
        z_x = model.encode_x(data_b).numpy()
    else:
        raise ValueError('Incorrect setting for x_from: {}'.format(x_from))
    return z_y_a, z_y_b, z_x


def initialize_node_state(disc_dict, chg_disc_dict, z_y_a, z_y_b, src, dest):
    """Initialize the latent state in discriminator/change discriminator dictionaries, so we can efficiently process.
    """
    src_ar = np.array(src)
    dest_ar = np.array(dest)

    before_z = np.zeros_like(z_y_a)
    before_z[0] = z_y_a[0] * (1 - src_ar) + z_y_b[0] * src_ar

    after_z = np.zeros_like(z_y_a)
    after_z[0] = z_y_a[0] * (1 - dest_ar) + z_y_b[0] * dest_ar

    if src not in disc_dict:
        disc_dict[src] = before_z
    if dest not in disc_dict:
        disc_dict[dest] = after_z
    if (src, dest) not in chg_disc_dict:
        chg_disc_dict[(src, dest)] = (before_z, after_z)


def get_explanation_weights(model: VAECE, data_a, data_b, x_from='a', batch_size=1000):
    """Generate the weights of the explanation graph (to take shortest path through) for an image pair.
    """
    dim = model._dim_y
    nodes = get_increasing(np.zeros(dim))  # nodes in the transition graph

    # get encodings of before/after
    z_y_a, z_y_b, z_x = get_encoding(model, data_a, data_b, x_from=x_from)

    ########
    # initialize all intermediate latent states
    ########
    disc_dict = {}
    chg_disc_dict = {}
    for src in nodes:  # for each node / state
        all_connections = get_increasing(src)  # get all outgoing states / edges
        all_connections = [tuple(x) for x in all_connections]
        src = tuple(src)
        all_connections.remove(src)  # remove self-edge
        for dest in all_connections:
            initialize_node_state(disc_dict, chg_disc_dict, z_y_a, z_y_b, src, dest)
    ########
    # process latent states using discriminator and change discriminator
    ########
    # create arrays with all states
    disc_in = np.zeros((len(disc_dict), *z_y_a[0].shape))
    chg_disc_in_a = np.zeros((len(chg_disc_dict), *z_y_a[0].shape))
    chg_disc_in_b = np.zeros((len(chg_disc_dict), *z_y_a[0].shape))
    disc_map = {}
    chg_disc_map = {}
    for i, (ratio, z) in enumerate(disc_dict.items()):  # fill all discriminator states
        disc_in[i] = z
        disc_map[ratio] = i
    for i, ((ratio_a, ratio_b), (z_a, z_b)) in enumerate(chg_disc_dict.items()):  # fill all change discriminator states
        chg_disc_in_a[i] = z_a
        chg_disc_in_b[i] = z_b
        chg_disc_map[(ratio_a, ratio_b)] = i
    # z_x is shared throughout - tile such that the size matches
    z_x_disc = np.tile(z_x[0], (len(disc_dict), 1))
    z_x_chg_disc = np.tile(z_x[0], (len(chg_disc_dict), 1))

    # propagate through model to get scores, in batches
    split = int(math.ceil(len(disc_dict)/batch_size))
    disc_images = []
    disc_scores = []
    for i in range(split):
        s, e = i*batch_size, min((i+1)*batch_size, len(disc_dict))
        b_disc_images = model.decode(disc_in[s:e], z_x_disc[s:e])
        b_disc_scores = model.discriminator(b_disc_images)
        disc_images.extend(b_disc_images)
        disc_scores.extend(b_disc_scores)
    disc_images = np.array(disc_images)
    disc_scores = np.array(disc_scores)

    split = int(math.ceil(len(chg_disc_dict)/batch_size))
    chg_disc_scores = []
    for i in range(split):
        s, e = i * batch_size, min((i + 1) * batch_size, len(chg_disc_dict))
        b_chg_disc_in_a, b_chg_disc_in_b = model.decode(chg_disc_in_a[s:e], z_x_chg_disc[s:e]), \
                                           model.decode(chg_disc_in_b[s:e], z_x_chg_disc[s:e])
        chg_disc_scores.extend(model.change_discriminator.discriminate(b_chg_disc_in_a, b_chg_disc_in_b))
    chg_disc_scores = np.array(chg_disc_scores)
    ########
    # assemble graph according to these scores
    ########
    graph = {}
    nodes = get_increasing(np.zeros(dim))
    thresh_pixel, penalty = 3, 5
    for src in nodes:
        all_connections = get_increasing(src)
        all_connections = [tuple(x) for x in all_connections]
        src = tuple(src)
        all_connections.remove(src)  # remove self-edge
        # initialize weight arrays
        out_edges = []
        for dest in all_connections:
            gan_score = 1 - disc_scores[disc_map[dest]][1]
            if np.sum(disc_images[disc_map[dest]]) < thresh_pixel:  # extra penalty for going to empty images
                gan_score += penalty
            chg_disc_score = 1 - chg_disc_scores[chg_disc_map[(src, dest)]][1]
            num_change = sum(list(dest)) - sum(list(src))
            out_edges.append((dest, (gan_score, chg_disc_score, num_change)))  # (dest, scores)
        # set weights in graph
        graph[src] = out_edges
    return graph


def create_graph(graph_weights, w_disc=0.5, w_chg_disc=1, w_len=1):
    graph = Graph()
    for s in graph_weights:
        for d, weight in graph_weights[s]:
            disc_score, chg_disc_score, num_change = weight
            total_weight = (disc_score * w_disc + chg_disc_score * w_chg_disc) * (math.pow(num_change, w_len))
            graph.add_edge(s, d, total_weight)
    return graph


def get_changes_from_path(path):
    start_p = path[0]
    total_changes = []
    for i in range(1, len(path)):
        end_p = path[i]
        changes = []
        for j in range(len(start_p)):
            if start_p[j] < end_p[j]:
                changes.append(j)
        start_p = end_p
        total_changes.append(changes)
    return total_changes


def graph_explanation(model: VAECE, data_a, data_b, x_from='a', return_order=False, batch_size=1000, **kwargs):
    """Compute a graph-based explanation for VAE-CE.
    """
    # build explanation graph
    graph_weights = get_explanation_weights(model, data_a, data_b, x_from=x_from, batch_size=batch_size)
    ex_graph = create_graph(graph_weights, **kwargs)
    s = tuple(np.zeros(model._dim_y))
    e = tuple(np.ones(model._dim_y))

    # calculate shortest paths
    dijkstra = DijkstraSPF(ex_graph, s)

    # and get changes from this path to our dest
    change_order = get_changes_from_path(dijkstra.get_path(e))

    # decode this path and return as explanation
    z_y_a, z_y_b, z_x = get_encoding(model, data_a, data_b, x_from=x_from)

    start = model.decode(z_y_a, z_x)[0]
    data_out = [start]
    for changes in change_order:
        for c in changes:
            z_y_a[0][c] = z_y_b[0][c]
        step = model.decode(z_y_a, z_x)[0]
        data_out.append(step)
    data_out = [x.numpy() for x in data_out]  # tensor -> np
    if return_order:
        return data_out, change_order
    return data_out


def interpolation_explanation(model: REPR, data_a, data_b, x_from='a', num_step=5):
    """Linearly interpolate over all dimensions at once.
    """
    z_y_a, z_y_b, z_x = get_encoding(model, data_a, data_b, x_from=x_from)
    dif = z_y_b - z_y_a
    step = dif / (num_step - 1)
    state = np.copy(z_y_a)
    data_out = [model.decode(state, z_x).numpy()[0]]
    for i in range(num_step-1):
        state += step
        data_out.append(model.decode(state, z_x).numpy()[0])
    return data_out


def dimension_swap_explanation(model: REPR, data_a, data_b, x_from='a', t=1):
    """Interpolate by changing entire dimensions at a time. All dimensions with a difference smaller than t are
    changed in the first step. The order of dimensions is effectively random (order of dimension indices).
    """
    z_y_a, z_y_b, z_x = get_encoding(model, data_a, data_b, x_from=x_from)

    # compute which dimensions are insignificant / changed instantly
    insignificant = set()
    for i in range(model._dim_y):
        if abs(z_y_a[0][i] - z_y_b[0][i]) < t:
            insignificant.add(i)
    significant = set(range(model._dim_y)) - insignificant

    # create interpolation from this
    state = np.copy(z_y_a)
    data_out = [model.decode(state, z_x).numpy()[0]]

    # change insignificant dimensions
    for i in insignificant:
        state[0][i] = z_y_b[0][i]

    if len(significant) == 0:  # in case there are no more changes to make, append only these changes
        data_out.append(model.decode(state, z_x).numpy()[0])

    for i in significant:
        state[0][i] = z_y_b[0][i]
        data_out.append(model.decode(state, z_x).numpy()[0])
    return data_out
