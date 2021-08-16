"""Functions to generate input for CUDA-accelerated coda.
"""
import numpy as np
import random
import os
import data.synthetic.structure as structure
from util.helper import random_pair


def generate_cuda_input_c(shape_input_line, num_samples, pair_offset=-1, pair_offset_rand=False, max_points=5):
    sp = structure.ShapeParser(w=32, h=32)
    change = pair_offset > -1  # if <=0, only create single images
    total_shape = structure.lines_to_shape(shape_input_line, 'shape')
    shape_lines = len(total_shape.lines)
    num_lines = len(shape_input_line)
    if pair_offset > num_lines:
        raise Exception("Can't swap more lines than lines exist")
    cuda_input_shape = np.zeros((num_samples, shape_lines, 4), dtype=np.float64)  # x1, y1, x2, y2
    if change:
        cuda_input_shape_b = np.zeros((num_samples, shape_lines, 4), dtype=np.float64)  # x1, y1, x2, y2
    cuda_input_shape_sf = np.zeros((num_samples, max_points, 5), dtype=np.float64) # x1, y1, s, t, exp
    cuda_input_shape_sf_base = np.zeros((num_samples, 2), dtype=np.float64) # base_stroke, base_weight
    for i in range(num_samples):
        if not change:
            shape_dist, sf = sp.random_modification(total_shape, no_image=True)
        else:
            if pair_offset_rand:
                to_swap = np.random.randint(pair_offset, num_lines+1)
            else:
                to_swap = pair_offset
            sel_a, sel_b = random_pair(num_lines, to_swap)
            shape_a = []
            shape_b = []
            for j in range(len(shape_input_line)):
                if j in sel_a:
                    shape_a.append(shape_input_line[j])
                if j in sel_b:
                    shape_b.append(shape_input_line[j])
            shape_a = structure.lines_to_shape(shape_a, 'shape')
            shape_b = structure.lines_to_shape(shape_b, 'shape')
            (shape_dist, shape_dist_b), sf = sp.random_modification_multi([shape_a, shape_b], no_image=True)
        # put into array
        total_lines_s = len(shape_dist.lines)
        for j in range(total_lines_s): # handle normal
            cuda_input_shape[i][j] = shape_dist.lines[j].return_data()
        for j in range(total_lines_s, shape_lines):
            cuda_input_shape[i][j] = -1 # set end data
        if change:
            total_lines_s_b = len(shape_dist_b.lines)
            for j in range(total_lines_s_b): # handle normal
                cuda_input_shape_b[i][j] = shape_dist_b.lines[j].return_data()
            for j in range(total_lines_s_b, shape_lines):
                cuda_input_shape_b[i][j] = -1 # set end data
        # handle sf
        for j in range(max_points):
            if j >= len(sf.points):
                cuda_input_shape_sf[i][j][0] = -1  # wildcard, means no more points
            else:
                cuda_input_shape_sf[i][j] = sf.points[j].return_data()
        cuda_input_shape_sf_base[i] = sf.base_stroke, sf.base_weight
    if not change:
        return cuda_input_shape, cuda_input_shape_sf, cuda_input_shape_sf_base
    else:
        return cuda_input_shape, cuda_input_shape_b, cuda_input_shape_sf, cuda_input_shape_sf_base


def generate_cuda(shape, N):
    return generate_cuda_input_c(shape, N)


def generate_cuda_change(shape, N):
    return generate_cuda_input_c(shape, N, pair_offset=1)


def generate_cuda_change_adv(shape, N):
    return generate_cuda_input_c(shape, N, pair_offset=2, pair_offset_rand=True)


def generate_cuda_all(DIR, shape_list, name_list, N_regular, N_change, N_adv, subdirs=None):
    # create dirs if they do not already exist
    if subdirs is None:
        subdirs = ['r', 'c', 'adv']
    else:
        assert len(subdirs) == 3
    for subdir in subdirs:
        if not os.path.exists(os.path.join(DIR, subdir)):
            os.makedirs(os.path.join(DIR, subdir))

    for lines, name in zip(shape_list, name_list):
        print('Generating CUDA input for {} in {}'.format(name, DIR))
        # regular samples
        shape_in, strokefield_in, strokefield_base = generate_cuda(lines, N_regular)
        np.save(os.path.join(DIR, subdirs[0], '{}_shape_input.npy'.format(name)), shape_in)
        np.save(os.path.join(DIR, subdirs[0], '{}_sf_input.npy'.format(name)), strokefield_in)
        np.save(os.path.join(DIR, subdirs[0], '{}_sf_base_input.npy'.format(name)), strokefield_base)

        # (single/positive) change pairs
        shape_in, shape_b_in, strokefield_in, strokefield_base = generate_cuda_change(lines, N_change)
        np.save(os.path.join(DIR, subdirs[1], '{}_shape_input.npy'.format(name)), shape_in)
        np.save(os.path.join(DIR, subdirs[1], '{}_shape_b_input.npy'.format(name)), shape_b_in)
        np.save(os.path.join(DIR, subdirs[1], '{}_sf_input.npy'.format(name)), strokefield_in)
        np.save(os.path.join(DIR, subdirs[1], '{}_sf_base_input.npy'.format(name)), strokefield_base)

        # adverse pairs -- only generate if possible for this shape
        if len(lines) >= 2:
            shape_in, shape_b_in, strokefield_in, strokefield_base = generate_cuda_change_adv(lines, N_adv)
            np.save(os.path.join(DIR, subdirs[2], '{}_shape_input.npy'.format(name)), shape_in)
            np.save(os.path.join(DIR, subdirs[2], '{}_shape_b_input.npy'.format(name)), shape_b_in)
            np.save(os.path.join(DIR, subdirs[2], '{}_sf_input.npy'.format(name)), strokefield_in)
            np.save(os.path.join(DIR, subdirs[2], '{}_sf_base_input.npy'.format(name)), strokefield_base)
