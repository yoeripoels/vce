"""Generation of synthetic data.
This synthetic data constitutes the inputs necessary to train each explanation model.
"""
import data.synthetic.structure as structure
import data.synthetic.cuda.generate_input as cuda_input
import data.synthetic.cuda.process as cuda_process
import data.synthetic.postprocess as postprocess
import data.data as data
import numpy as np
import util.visualization as vis
import os


def create_line_interp(p0, p1, num_point):
    x0, y0 = p0
    x1, y1 = p1
    line = []
    for i in range(num_point):
        ratio = i / (num_point-1)
        line.append((x0*(1-ratio) + x1 * ratio, y0*(1-ratio) + y1 * ratio))
    return line


def idx_to_lines(lines, idx):
    return [lines[i] for i in idx]


def idx_to_shape(lines, idx):
    l = [lines[i] for i in idx]
    return structure.lines_to_shape(l)


def shuffle_data(dataset):
    N = dataset[0].shape[0]
    n_data = len(dataset)
    for i in range(n_data):
        assert dataset[i].shape[0] == N

    p = np.random.permutation(N)
    return [x[p] for x in dataset]


if __name__ == '__main__':
    #############################
    # CREATE LINE/SHAPE STRUCTURE
    #############################
    w, h = 32, 32
    sp = structure.ShapeParser(w=w, h=h)

    # create the base lines we will use to construct shapes/class-examples
    line_left = create_line_interp((0.2, 0.2), (0.2, 0.8), 20)
    line_right = create_line_interp((0.8, 0.2), (0.8, 0.8), 20)
    line_top = create_line_interp((0.2, 0.2), (0.8, 0.2), 20)
    line_bottom = create_line_interp((0.2, 0.8), (0.8, 0.8), 20)
    cross_a = create_line_interp((0.2, 0.2), (0.8, 0.8), 20)
    cross_b = create_line_interp((0.2, 0.8), (0.8, 0.2), 20)
    hor = create_line_interp((0.2, 0.5), (0.8, 0.5), 20)
    ver = create_line_interp((0.5, 0.2), (0.5, 0.8), 20)

    # line:      0          1           2         3            4        5        6    7
    all_lines = [line_left, line_right, line_top, line_bottom, cross_a, cross_b, hor, ver]

    # specify which lines belong to which class
    classes = [[6, 7], [6], [0, 1, 2, 3], [2, 3, 4, 5], [0, 1, 4, 5], [2, 3, 6], [0, 1, 7], [0, 2, 5], [1, 3, 5]]
    # we treat class 9 separately, as it has 2 variants
    class_9 = [[2, 3, 4], [2, 3, 5]]

    # get line coordinates for each class
    classes_lines = [idx_to_lines(all_lines, classes[i]) for i in range(len(classes))]
    class_9_lines = [idx_to_lines(all_lines, class_9[i]) for i in range(len(class_9))]

    # convert these to shapes
    classes_shape = [structure.lines_to_shape(l) for l in classes_lines]
    class_9_shape = [structure.lines_to_shape(l) for l in class_9_lines]

    # generate some images to get an idea of our data
    images_clean = []
    for i in range(len(classes)):
        images_clean.append(sp.shape_to_image(classes_shape[i], edge_fade=0.04))
    for i in range(len(class_9)):
        images_clean.append(sp.shape_to_image(class_9_shape[i], edge_fade=0.04))

    # and generate images with a random distortion, as will be used in the dataset
    images_distort = []
    for i in range(len(classes)):
        images_distort.append(sp.random_modification(classes_shape[i]))
    for i in range(len(class_9)):
        s = idx_to_shape(all_lines, class_9[i])
        images_distort.append(sp.random_modification(class_9_shape[i]))

    vis.plot_images([images_clean, images_distort], filename=os.path.join('out', 'dataset-sample.png'))

    #############################
    # CREATE DATA FROM SHAPES, CONVERT TO IMAGES
    #############################
    # note that data (cuda input + cuda output) is saved to disk rather than all kept in memory

    # set tmp dirs to use
    cuda_input_dir = os.path.join('tmp', 'cuda_process')
    subdirs = ['r', 'c', 'adv']
    # first, generate for classes 0-8
    n_regular = 100#00
    n_change_pair = n_regular // 2  # 50% positive pairs
    n_adv_change_pair = n_regular // 4  # 25% negative pairs - other 25% will be arbitrary image-combinations
    names = [str(i) for i in range(len(classes_lines))]  # classes 0-8
    cuda_input.generate_cuda_all(cuda_input_dir, classes_lines, names, n_regular, n_change_pair, n_adv_change_pair,
                                 subdirs=subdirs)
    # class 9 separately, as we only need half of the datapoints as there are 2 variants
    names_9 = ['9a', '9b']
    cuda_input.generate_cuda_all(cuda_input_dir, class_9_lines, names_9,
                                 n_regular//2, n_change_pair//2, n_adv_change_pair//2, subdirs=subdirs)
    all_names = names + names_9
    cuda_process.process_all(cuda_input_dir, subdirs, all_names, w=w, h=h)  # transform images and write them to disk

    #############################
    # POSTPROCESS IMAGES/LABELS FOR ALL MODELS
    #############################
    num_class = 10
    num_feature = 8
    output_dir = 'out'
    all_lines = classes_lines + class_9_lines
    all_line_indices = classes + class_9
    x, y, y_feature = postprocess.create_data(os.path.join(cuda_input_dir, subdirs[0]), all_names, filename='output',
                                              w=w, h=h, N=n_regular * num_class,
                                              data_line_indices=all_line_indices, num_feature=num_feature,
                                              num_class=num_class)

    name_list = ['x', 'y', *['y_f' + str(i) for i in range(num_feature)]]
    data.split_write(output_dir, [x, y, *y_feature], name_list, num_split=20)
    '''
    dvae = [x, y]
    cd = [x_pair_full_a, x_pair_full_b, y_pair, x, y]
    vaece = [x, x_p, y, y_p, x_pair_full_a]
    gvae = [x_fpair_a, x_fpair_b, y_fpair, x, y]
    lvae = [x, y, y_f0-y_f7]
    adagvae = [x_pos_pair_a, x_pos_pair_b, x, y]
    check gen_syn.ipynb for details
    '''
    # POSTPROCESS AND WRITE TO /OUT, MODEL BY MODEL. FOR CHANGE PAIRS, MAKE SURE WE SWAP HANDLE 0/1 CORRECTLY
