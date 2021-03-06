"""Functions for post-processing image data
"""
import numpy as np
import re
import os
import random


def parse_data_x(data, h=None, w=None, d=None):
    if h is None:
        h = data.shape[1]
    if w is None:
        w = data.shape[2]
    if d is None:
        if len(data.shape) > 3:
            d = data.shape[3]
        else:
            d = 1
    data = data.reshape(data.shape[0], h, w, d)
    if issubclass(data.dtype.type, np.integer):
        data = data / 255
    data = data.astype('float32')
    return data


def parse_data_y(data, num_class):
    if len(data.shape) == 2 and data.shape[1] == num_class:
        return data.astype('float32')
    data = data.astype('int32')
    data = np.eye(num_class)[data]
    data = data.astype('float32')
    return data


def create_data(data_dir, data_names, filename='output', w=32, h=32, n=10000, data_line_indices=None, num_class=None,
                num_feature=None):
    x = np.zeros((n, h, w))
    u_class = num_class is not None
    u_feature = num_feature is not None

    if u_class:
        y = np.zeros(n)
    if u_feature:
        y_feature = [np.zeros(n) for _ in range(num_feature)]
        data_iterate = zip(data_names, data_line_indices)
    else:
        data_iterate = data_names

    cur_idx = 0
    for data_step in data_iterate:
        if u_feature:
            data_name, data_lines = data_step
        else:
            data_name = data_step
        # load data file
        data_file = os.path.join(data_dir, '{}_{}.npy'.format(data_name, filename))
        if not os.path.isfile(data_file):  # some classes do not necessarily have all data
            continue
        data = np.load(data_file)
        data_n = data.shape[0]  # how much to change

        # insert into large arrays
        x[cur_idx:cur_idx + data_n] = data[:]
        if u_class:
            data_class = int(
                re.sub('[^0-9]', '', data_name))  # grab idx, we assume name always includes class idx as only integer
            y[cur_idx:cur_idx + data_n] = data_class
        if u_feature:
            for line_idx in data_lines:
                y_feature[line_idx][cur_idx:cur_idx + data_n] = 1  # set to 1 for lines that are in this class
        # increase idx
        cur_idx += data_n

    if cur_idx < n:  # if we could not parse all, remove the remaining cols
        x = x[:cur_idx]
        if u_class:
            y = y[:cur_idx]
        if u_feature:
            y_feature = [fy[:cur_idx] for fy in y_feature]
    # post process all
    x = parse_data_x(x)
    if u_class:
        y = parse_data_y(y, num_class)
    if u_feature:
        y_feature = [parse_data_y(fy, 2) for fy in y_feature]
    return_val = [x]
    if u_class:
        return_val.append(y)
    if u_feature:
        return_val.append(y_feature)
    return return_val if len(return_val) > 1 else return_val[0]  # if we only have x, don't wrap in list


def create_adverse_data(data_a, data_b, n=None, adverse_use_same=True):
    """Create pairs consisting of two randomly selected 'change-pair' images
    """
    if n is None:
        n = data_a.shape[0]
    data_shape = (n, *data_a.shape[1:])

    adverse_pairs_a = np.zeros(data_shape)
    adverse_pairs_b = np.zeros(data_shape)
    idx_ord_a = np.random.permutation(n)
    idx_ord_b = np.random.permutation(n)
    if adverse_use_same:
        pick_random_choice = [True, False]
    else:
        pick_random_choice = [True]  # we only pick different-image pairs
    for i in range(n):
        if random.choice(pick_random_choice):  # pick a pair of two random elements, not matching idx
            adverse_pairs_a[i][:] = data_a[idx_ord_a[i]][:]
            idx_b = idx_ord_b[i]
            if random.choice([True, False]):  # randomly pick from either before or after
                adverse_pairs_b[i][:] = data_a[idx_b][:]
            else:
                adverse_pairs_b[i][:] = data_b[idx_b][:]
        else:  # pick one element, insert in both
            if random.choice([True, False]):  # randomly pick from either before or after
                adverse_pairs_a[i][:] = data_a[idx_ord_a[i]][:]
                adverse_pairs_b[i][:] = data_a[idx_ord_a[i]][:]
            else:
                adverse_pairs_a[i][:] = data_b[idx_ord_a[i]][:]
                adverse_pairs_b[i][:] = data_b[idx_ord_a[i]][:]
    return adverse_pairs_a, adverse_pairs_b


def create_change_pairs(cpair_p_a, cpair_p_b, cpair_n_a=None, cpair_n_b=None, adverse_use_same=True,
                        generate_adverse=True):
    """From positive change pairs (size = N/2) and (optionally) negative change pairs (generally size = N/4, or N/2 if
    we do not generate extra adverse pairs), create full change pair arrays / labels.
    """
    assert cpair_p_a.shape[0] == cpair_p_b.shape[0]
    if generate_adverse:
        # first, create extra negative pairs by also combining random images as bad pairs
        # (opposed to only pairs consisting of images with 2+ factors changed)
        cpair_adv_a, cpair_adv_b = create_adverse_data(cpair_p_a, cpair_p_b, adverse_use_same=adverse_use_same)
    else:
        # do not create extra negative pairs, only use those supplied
        # assert these have the correct dimension
        assert cpair_n_a is not None and cpair_n_b is not None
        assert cpair_n_a.shape[0] == cpair_n_b.shape[0] == cpair_p_a.shape[0]
        cpair_adv_a = cpair_n_a
        cpair_adv_b = cpair_n_b
    if cpair_n_a is not None and cpair_n_b is not None:
        n_negative = cpair_n_a.shape[0]
        cpair_adv_a[0:n_negative] = cpair_n_a[:]
        cpair_adv_b[0:n_negative] = cpair_n_b[:]

    # now that we have equal sized positive and negative pairs, construct final data
    n_half, h, w, d = cpair_p_a.shape[0:4]

    x_pair_full_a = np.zeros((n_half * 2, h, w, d))
    x_pair_full_b = np.zeros((n_half * 2, h, w, d))

    # add positive
    x_pair_full_a[0:n_half] = cpair_p_a[:]
    x_pair_full_b[0:n_half] = cpair_p_b[:]

    x_pair_full_a[n_half:] = cpair_adv_a[:]
    x_pair_full_b[n_half:] = cpair_adv_b[:]

    y_pair = np.zeros((n_half * 2))  # 0 or 1, negative = 0, positive = 1
    y_pair[0:n_half] = 1  # set positive
    x_pair_full_a = parse_data_x(x_pair_full_a)
    x_pair_full_b = parse_data_x(x_pair_full_b)
    y_pair = parse_data_y(y_pair, 2)

    return x_pair_full_a, x_pair_full_b, y_pair


def create_feature_pairs(x, y, y_feature, n_pair=None):
    """Creates pairs of datapoints where each datapoint share at least one feature, with this shared feature
    labeled by y_shared. Pick a similar number of 'positive' and 'negative' features to share.
    """
    #######
    # First, create lists per feature, where each such list contains datapoints with/without that feature
    #######
    feature_to_data = []
    nfeature_to_data = []
    idx_to_feature = []
    num_feature = len(y_feature)
    for i in range(num_feature):
        feature_to_data.append([])
        nfeature_to_data.append([])

    n = x.shape[0]
    for i in range(n):
        features = []
        non_features = []
        for j in range(num_feature):
            if y_feature[j][i][1] == 1:  # feature exists
                features.append(j)
            else:
                non_features.append(j)
        # positive feature, y_f=1
        to_add = random.choice(features)
        feature_to_data[to_add].append((x[i], y[i]))
        # negative feature, y_f=0
        to_add = random.choice(non_features)
        nfeature_to_data[to_add].append((x[i], y[i]))
        idx_to_feature.append(to_add)

    #######
    # From these feature lists, create pairs of data with (at least) 1 feature in common, and label this feature
    #######
    if n_pair is None:
        n_pair = n
    _, h, w = x.shape[0:3]
    num_class_y = y.shape[1]
    x_fpair_a = np.zeros((n_pair, h, w, 1))
    x_fpair_b = np.zeros((n_pair, h, w, 1))
    y_fpair = np.zeros((n_pair, 1))
    y_fpair_a = np.zeros((n_pair, num_class_y))
    y_fpair_b = np.zeros((n_pair, num_class_y))
    for i in range(n_pair):
        f = np.random.randint(num_feature)
        if random.choice([True, False]):
            # pick from positive feature
            idx_a, idx_b = np.random.randint(len(feature_to_data[f]), size=2)
            x_fpair_a[i][:] = feature_to_data[f][idx_a][0][:]
            x_fpair_b[i][:] = feature_to_data[f][idx_b][0][:]
            y_fpair_a[i][:] = feature_to_data[f][idx_a][1][:]
            y_fpair_b[i][:] = feature_to_data[f][idx_b][1][:]
        else:
            # pick from negative feature
            idx_a, idx_b = np.random.randint(len(nfeature_to_data[f]), size=2)
            x_fpair_a[i][:] = nfeature_to_data[f][idx_a][0][:]
            x_fpair_b[i][:] = nfeature_to_data[f][idx_b][0][:]
            y_fpair_a[i][:] = nfeature_to_data[f][idx_a][1][:]
            y_fpair_b[i][:] = nfeature_to_data[f][idx_b][1][:]
        y_fpair[i][0] = f
    y_fpair = y_fpair.astype('float32')
    return x_fpair_a, x_fpair_b, y_fpair, y_fpair_a, y_fpair_b


def create_feature_pairs_cat(x, y, y_feature, n_pair=None):
    """Creates pairs of datapoints where each datapoint share at least one feature, with this shared feature
    labeled by y_shared. Features are considered categorical, i.e., pick evenly from all shared features.
    """
    #######
    # First, create lists per feature, where each such list contains datapoints with/without that feature
    #######
    feature_to_data = []
    nfeature_to_data = []
    idx_to_feature = []
    num_feature = len(y_feature)
    for i in range(num_feature):
        feature_to_data.append([])
        nfeature_to_data.append([])

    n = x.shape[0]
    for i in range(n):
        features = []
        non_features = []
        for j in range(num_feature):
            if y_feature[j][i][1] == 1:  # feature exists
                features.append(j)
            else:
                non_features.append(j)
        # positive feature, y_f=1
        to_add = random.choice(features)
        feature_to_data[to_add].append((x[i], y[i]))
        # negative feature, y_f=0
        to_add = random.choice(non_features)
        nfeature_to_data[to_add].append((x[i], y[i]))
        idx_to_feature.append(to_add)

    #######
    # From these feature lists, create pairs of data with (at least) 1 feature in common, and label this feature
    #######
    if n_pair is None:
        n_pair = n
    _, h, w = x.shape[0:3]
    num_class_y = y.shape[1]
    x_fpair_a = np.zeros((n_pair, h, w, 1))
    x_fpair_b = np.zeros((n_pair, h, w, 1))
    y_fpair = np.zeros((n_pair, 1))
    y_fpair_a = np.zeros((n_pair, num_class_y))
    y_fpair_b = np.zeros((n_pair, num_class_y))
    for i in range(n_pair):
        f = np.random.randint(num_feature)
        if random.choice([True, False]):
            # pick from positive feature
            idx_a, idx_b = np.random.randint(len(feature_to_data[f]), size=2)
            x_fpair_a[i][:] = feature_to_data[f][idx_a][0][:]
            x_fpair_b[i][:] = feature_to_data[f][idx_b][0][:]
            y_fpair_a[i][:] = feature_to_data[f][idx_a][1][:]
            y_fpair_b[i][:] = feature_to_data[f][idx_b][1][:]
        else:
            # pick from negative feature
            idx_a, idx_b = np.random.randint(len(nfeature_to_data[f]), size=2)
            x_fpair_a[i][:] = nfeature_to_data[f][idx_a][0][:]
            x_fpair_b[i][:] = nfeature_to_data[f][idx_b][0][:]
            y_fpair_a[i][:] = nfeature_to_data[f][idx_a][1][:]
            y_fpair_b[i][:] = nfeature_to_data[f][idx_b][1][:]
        y_fpair[i][0] = f
    y_fpair = y_fpair.astype('float32')
    return x_fpair_a, x_fpair_b, y_fpair, y_fpair_a, y_fpair_b