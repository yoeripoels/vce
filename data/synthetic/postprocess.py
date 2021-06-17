"""Functions for post-processing image data


"""
import numpy as np
import re
import os

def parse_data_x(data, h=32, w=32, d=1):
    data = data.reshape(data.shape[0], h, w, d)
    data = data.astype('float32')
    return data


def parse_data_y(data, num_class):
    data = data.astype('int32')
    data = np.eye(num_class)[data]
    data = data.astype('float32')
    return data


def create_data(data_dir, data_names, filename='output', w=32, h=32, N=10000, data_line_indices=None, num_class=None,
                num_feature=None):
    x = np.zeros((N, h, w))
    u_class = num_class is not None
    u_feature = num_feature is not None

    if u_class:
        y = np.zeros(N)
    if u_feature:
        y_feature = [np.zeros(N) for _ in range(num_feature)]
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
        data = np.load(data_file)
        data_N = data.shape[0]  # how much to change

        # insert into large arrays
        x[cur_idx:cur_idx + data_N] = data[:]
        if u_class:
            data_class = int(
                re.sub('[^0-9]', '', data_name))  # grab idx, we assume name always includes class idx as only integer
            y[cur_idx:cur_idx + data_N] = data_class
        if u_feature:
            for line_idx in data_lines:
                y_feature[line_idx][cur_idx:cur_idx + data_N] = 1  # set to 1 for lines that are in this class
        # increase idx
        cur_idx += data_N

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
    return return_val
