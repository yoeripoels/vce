"""Visualization related functions.
"""
import matplotlib.pyplot as plt


def plot_images(image_data, title='', title_end='', fontsize=15, size_w=20, size_h=20, thresh=0, adaptive=True,
                force_scale=True, filename=None):
    """Plots images. image_data can be a single image, a list of images (shown horizontally), or a list of lists
    of images (each list being shown as a row).
    """
    if not isinstance(image_data, list):
        image_data = [[image_data]]
    elif not isinstance(image_data[0], list):
        image_data = [image_data]  # make multi dimensional
    num_w = max([len(x) for x in image_data])
    num_h = len(image_data)
    h, w = image_data[0][0].shape[0:2]
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
            image = image_data[i][j]
            if thresh > 0:
                image[image < thresh] = 0
            if force_scale:
                axes[i][j].imshow(image.reshape((h, w)), cmap='gray', vmin=0, vmax=1)
            else:
                axes[i][j].imshow(image.reshape((h, w)), cmap='gray')
    if title != '':
        axes[0][0].set_title(title, fontsize=fontsize)
    if title_end != '':
        axes[-1][-1].set_title(title_end, fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show(bbox_inches='tight')