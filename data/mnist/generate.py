"""Generation of augmented MNIST.

We augment MNIST such that we can split images into (some approximation of) lines, which we
assume to be the underlying type of feature that determines the digit.

This augmentation class can be used to generate inputs necessary for methods that assume some sort of
'feature-difference' as supervision (VAE-CE and ADA-GVAE), as we can create such image-groups using these split digits.
"""
import data.mnist.augment as augment
import data.data as data
import data.postprocess as postprocess
import data.mnist.visualization as mnist_vis
import util.visualization as vis
import numpy as np
from util.helper import random_pair
from tensorflow.keras.datasets import mnist
import os


def parse_data_x_mnist(dataset):
    h, w, d = 28, 28, 1
    dataset = dataset.reshape(dataset.shape[0], h, w, d)
    dataset = np.pad(dataset, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    dataset = dataset.astype('float32')
    dataset /= 255
    return dataset


def augment_dataset(dataset, n=None, max_per_image=3, random_per_image=True):
    """Processes a (line-like) dataset.
    Returns positive change pairs and negative change pairs.
    """
    out_pair_a = []
    out_pair_b = []
    out_adv_pair_a = []
    out_adv_pair_b = []
    if n is None:
        n = dataset.shape[0]
    else:
        n = min(n, dataset.shape[0])
    for i in range(n):
        img = dataset[i]
        stroke_map = augment.split_digit(img)  # split our image
        num_stroke = np.amax(stroke_map) + 1
        to_select = min(max_per_image, num_stroke)
        for j in range(to_select):
            show_a, show_b = random_pair(num_stroke, 1)
            out_pair_a.append(augment.highlight_strokes(img, stroke_map, show_a))
            out_pair_b.append(augment.highlight_strokes(img, stroke_map, show_b))
            if num_stroke > 2:
                show_a_adv, show_b_adv = random_pair(num_stroke, np.random.randint(2, num_stroke))
                out_adv_pair_a.append(augment.highlight_strokes(img, stroke_map, show_a_adv))
                out_adv_pair_b.append(augment.highlight_strokes(img, stroke_map, show_b_adv))
        if (i+1) % 100 == 0:
            print('Done handling {}/{} images ({:.2f}%)'.format(i+1, n, (i+1)/n*100))
    out_pair_a = np.array(out_pair_a, dtype=np.uint8)
    out_pair_b = np.array(out_pair_b, dtype=np.uint8)
    out_adv_pair_a = np.array(out_adv_pair_a, dtype=np.uint8)
    out_adv_pair_b = np.array(out_adv_pair_b, dtype=np.uint8)
    return out_pair_a, out_pair_b, out_adv_pair_a, out_adv_pair_b


if __name__ == '__main__':
    # output dirs
    path = os.path.dirname(os.path.realpath(__file__))
    augment_out_dir = os.path.join(path, 'tmp', 'augment_full')
    output_dir = os.path.join(path, 'out')
    for d in [augment_out_dir, output_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # download MNIST
    (mnist_train, mnist_train_label), (mnist_test, mnist_test_label) = mnist.load_data()
    data_x, data_y = mnist_train, mnist_train_label  # can change this if we wish to augment test data

    # test and show 4 samples
    vis.plot_images([mnist_vis.visualize_process(img) for img in data_x[0:4]],
                    filename=os.path.join(output_dir, 'split-sample.png'))

    # create positive/negative pairs
    cpair_p_a, cpair_p_b, cpair_n_a, cpair_n_b = augment_dataset(data_x)

    # write temp result to disk
    for datafile, filename in zip([ cpair_p_a,   cpair_p_b,   cpair_n_a,   cpair_n_b],
                                  ['cpair_p_a', 'cpair_p_b', 'cpair_n_a', 'cpair_n_b']):
        np.save(os.path.join(augment_out_dir, filename), datafile)

    # parse / pad / reshape all data for creating the final dataset
    cpair_p_a, cpair_p_b, cpair_n_a, cpair_n_b, data_x = [parse_data_x_mnist(d) for d in
                                                          [cpair_p_a, cpair_p_b, cpair_n_a, cpair_n_b, data_x]]
    data_y = postprocess.parse_data_y(data_y, num_class=10)

    n = 100000
    x_pos_pair_a, x_pos_pair_b = data.shuffle_reduce([cpair_p_a, cpair_p_b], n_max=min(n, data_x.shape[0]))  # ADA-GVAE
    cpair_p_a, cpair_p_b = data.shuffle_reduce([cpair_p_a, cpair_p_b], n_max=n//2)  # VAE-CE / CD
    cpair_n_a, cpair_n_b = data.shuffle_reduce([cpair_n_a, cpair_n_b], n_max=n//4)  # half the adverse pairs

    # create full dataset
    x_pair_full_a, x_pair_full_b, y_pair = postprocess.create_change_pairs(cpair_p_a, cpair_p_b, cpair_n_a, cpair_n_b)

    # split/write all to disk
    num_split = 20

    data_list = [data_x, data_y]
    name_list = ['x', 'y']
    data.split_write(output_dir, data_list, name_list, num_split=num_split)  # for all models

    data_list = [data_x, data_y]
    name_list = ['x_p', 'y_p']
    data.split_write(output_dir, data_list, name_list, num_split=num_split)  # for VAECE -> shuffle x/y as 2nd sample

    data_list = [x_pair_full_a, x_pair_full_b, y_pair]
    name_list = ['x_pair_full_a', 'x_pair_full_b', 'y_pair']
    data.split_write(output_dir, data_list, name_list, num_split=num_split)  # for CD

    data_list = [x_pos_pair_a, x_pos_pair_b]
    name_list = ['x_pos_pair_a', 'x_pos_pair_b']
    data.split_write(output_dir, data_list, name_list, num_split=num_split)  # for ADA-GVAE -> only uses positive pairs
                                                                # x_pos_pair_a is also used for VAE-CE discriminator

    print('#####\n# Done augmenting MNIST\n#####')
