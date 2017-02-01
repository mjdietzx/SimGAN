"""
Module to plot generated images and save the plots to disc as we train our GAN.

"""

import os

import matplotlib
import numpy as np

matplotlib.use('Agg')  # b/c matplotlib is such a great piece of software ;) -- needed to work on ubuntu
from matplotlib import pyplot as plt


def plot_batch(synthetic_image_batch, refined_image_batch, figure_path):
    """
    Generate a plot of `batch_size` refined and synthetic images and save it to disc.
    Refined image and the synthetic image it was generated from are side-by-side
    (i.e. refined_image_0, synthetic_image_0, ..., refined_image_n, synthetic_image_n).

    :param synthetic_image_batch: Batch of synthetic images used to generate the refined images.
    :param refined_image_batch: Corresponding batch of refined images.
    :param figure_path: Full path of file name the plot will be saved as.
    """
    batch_size = synthetic_image_batch.shape[0]
    img_height = synthetic_image_batch.shape[1]
    img_width = synthetic_image_batch.shape[2]

    synthetic_image_batch = np.reshape(synthetic_image_batch, newshape=(-1, img_height, img_width))
    refined_image_batch = np.reshape(refined_image_batch, newshape=(-1, img_height, img_width))

    image_batch = np.concatenate((refined_image_batch, synthetic_image_batch))

    nb_rows = batch_size // 10 + 1
    nb_columns = 10 * 2

    _, ax = plt.subplots(nb_rows, nb_columns, sharex=True, sharey=True)

    for i in range(nb_rows):
        for j in range(0, nb_columns, 2):
            try:
                # pre-processing function, applications.xception.preprocess_input => [0.0, 1.0]
                ax[i][j].imshow((image_batch[i * nb_columns + j] / 2.0 + 0.5))
                ax[i][j + 1].imshow((image_batch[i * nb_columns + j + batch_size] / 2.0 + 0.5))
            except IndexError:
                pass
            ax[i][j].set_axis_off()
    plt.savefig(os.path.join(figure_path), dpi=600)
    plt.close()
