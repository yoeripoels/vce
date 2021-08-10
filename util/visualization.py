"""Visualization related functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_images(image_data, title='', title_end='', fontsize=15, size_w=20, size_h=20, thresh=0, adaptive=True,
                adj=None, force_scale=True, filename=None):
    """Plots images. image_data can be a single image, a list of images (shown horizontally), or a list of lists
    of images (each list being shown as a row).
    """
    if not isinstance(image_data, list):
        image_data = [[image_data]]
    elif not isinstance(image_data[0], list):
        image_data = [image_data]  # make multi dimensional
    num_w = max([len(x) for x in image_data])
    num_h = len(image_data)
    fig, axes = plt.subplots(num_h, num_w)
    if adaptive:
        fig.set_size_inches(5*num_w, 5*num_h)
    else:
        fig.set_size_inches(size_w, size_h)
    if num_h == 1:
        axes = [axes]
    if num_w == 1:
        axes = [[ax] for ax in axes]
    for i in range(num_h):
        for j in range(num_w):
            axes[i][j].get_xaxis().set_visible(False)
            axes[i][j].get_yaxis().set_visible(False)

            if j >= len(image_data[i]):
                continue  # no more images in this row
            img = image_data[i][j]
            if isinstance(img, tf.Tensor):
                img = img.numpy()
            image_kwargs = {}
            if not (len(img.shape) > 2 and img.shape[2] != 1):
                image_kwargs['cmap'] = 'gray'  # only set colourmap to gray if we have a grayscale image
                image_shape = img.shape[0:2]
            else:
                image_shape = img.shape[0:3]
            if thresh > 0:
                img[img < thresh] = 0
            if force_scale:
                axes[i][j].imshow(img.reshape(image_shape), vmin=0, vmax=1, **image_kwargs)
            else:
                axes[i][j].imshow(img.reshape(image_shape), **image_kwargs)
    if title != '':
        axes[0][0].set_title(title, fontsize=fontsize)
    if title_end != '':
        axes[-1][-1].set_title(title_end, fontsize=fontsize)
    if adj is not None:
        wspace, hspace = adj
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def traverse_dim(model, z_y, z_x, dim, dim_range=3., num_img=7):
    if dim[0] == 0:
        space_y = True
    else:
        space_y = False
    dim = dim[1]
    offset = np.linspace(-dim_range, dim_range, num_img)
    images = []
    for i in offset:
        if space_y:
            orig = z_y[0][dim]
            z_y[0][dim] = i
            image = model.decode(z_y, z_x)[0]
            images.append(image)
            z_y[0][dim] = orig
        else:
            orig = z_x[0][dim]
            z_x[0][dim] = i
            image = model.decode(z_y, z_x)[0]
            images.append(image)
            z_x[0][dim] = orig
    return images
