"""Load preprocessed data and create images using GPU-accelerated functions.
"""
from numba import cuda
import numpy as np
import data.synthetic.cuda.gpu as gpu
import os


def generate_images(data_dir, shape_name, w, h, pair=False):
    try:
        shape_input = np.load(os.path.join(data_dir, '{}_shape_input.npy'.format(shape_name)))
        sf_input = np.load(os.path.join(data_dir, '{}_sf_input.npy'.format(shape_name)))
        sf_base_input = np.load(os.path.join(data_dir, '{}_sf_base_input.npy'.format(shape_name)))
    except FileNotFoundError:
        print('Could not find files for {}'.format(os.path.join(data_dir, shape_name)))
        return False
    # pass to gpu
    c_shape_input = cuda.to_device(shape_input)
    c_sf_input = cuda.to_device(sf_input)
    c_sf_base_input = cuda.to_device(sf_base_input)

    # create output and share with gpu
    N = shape_input.shape[0]
    c_out = cuda.to_device(np.zeros((N, h, w)))

    # create the images
    griddimension = (32, 16)
    blockdimension = (32, 8)
    print('(CUDA) Processing images for {}'.format(os.path.join(data_dir, shape_name)))
    gpu.create_images[griddimension, blockdimension](c_shape_input, c_sf_input, c_sf_base_input, c_out)

    # retrieve output images
    c_out.copy_to_host()

    if pair:  # do this for shape_b / paired images as well
        shape_b_input = np.load(os.path.join(data_dir, '{}_shape_b_input.npy'.format(shape_name)))
        c_shape_b_input = cuda.to_device(shape_b_input)
        c_out_b = cuda.to_device(np.zeros((N, h, w)))
        print('(CUDA) Processing paired images for {}'.format(os.path.join(data_dir, shape_name)))
        gpu.create_images[griddimension, blockdimension](c_shape_b_input, c_sf_input, c_sf_base_input, c_out_b)

        c_out_b.copy_to_host()
        return c_out, c_out_b
    else:
        return c_out


def process_all(data_dir, subdirs, shape_names, w, h):
    for name in shape_names:
        images = generate_images(os.path.join(data_dir, subdirs[0]), name, w, h)
        np.save(os.path.join(data_dir, subdirs[0], '{}_output.npy'.format(name)), images)
        images, images_b = generate_images(os.path.join(data_dir, subdirs[1]), name, w, h, pair=True)
        np.save(os.path.join(data_dir, subdirs[1], '{}_output.npy'.format(name)), images)
        np.save(os.path.join(data_dir, subdirs[1], '{}_output_b.npy'.format(name)), images_b)
        return_data = generate_images(os.path.join(data_dir, subdirs[2]), name, w, h, pair=True)
        if return_data is not False:  # adv pairs can't be made for each dataset, so check whether we got a result
            images, images_b = return_data
            np.save(os.path.join(data_dir, subdirs[2], '{}_output.npy'.format(name)), images)
            np.save(os.path.join(data_dir, subdirs[2], '{}_output_b.npy'.format(name)), images_b)
