"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).

Note: Only Python 3 support currently.
"""
import os
import sys

from keras import applications
from keras import layers
from keras import models
from keras.preprocessing import image
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('Agg')
from matplotlib import pyplot as plt

#
# directories
#
path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')

#
# Temporary workarounds
#
DISC_SOFTMAX_OUTPUT_DIM = 252  # FIXME: is this correct?

#
# image dimensions
#
img_width = 55
img_height = 35
img_channels = 1

#
# training params
#
nb_steps = 10000
batch_size = 32
k_d = 1  # number of discriminator updates per step
k_g = 2  # number of generative network updates per step
log_interval = 100

#
# refined image history buffer
#
refined_image_history_buffer = np.zeros(shape=(0, img_height, img_width, img_channels))
history_buffer_max_size = batch_size * 1000  # TODO: what should the size of this buffer be?


def add_half_batch_to_image_history(half_batch_generated_images):
    global refined_image_history_buffer
    assert len(half_batch_generated_images) == batch_size / 2

    if len(refined_image_history_buffer) < history_buffer_max_size:
        refined_image_history_buffer = np.concatenate((refined_image_history_buffer, half_batch_generated_images))
    elif len(refined_image_history_buffer) == history_buffer_max_size:
        refined_image_history_buffer[:batch_size // 2] = half_batch_generated_images
    else:
        assert False

    np.random.shuffle(refined_image_history_buffer)


def get_half_batch_from_image_history():
    global refined_image_history_buffer

    try:
        return refined_image_history_buffer[:batch_size // 2]
    except IndexError:
        return np.zeros(shape=(0, img_height, img_width, img_channels))


def plot_batch(synthetic_image_batch, refined_image_batch, figure_name):
    """
    Generate a plot of `batch_size` refined/synthetic images (refined_image_0, synthetic_image_0, ..., refined_image_n).

    :param synthetic_image_batch: Batch of synthetic images used to generate the refined images.
    :param refined_image_batch: Corresponding batch of refined images.
    :param figure_name: Name that plot will be saved with.
    """
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
    plt.savefig(os.path.join(cache_dir, '{}.png'.format(figure_name)), dpi=600)
    plt.close()


def refiner_network(input_image_tensor):
    """
    The refiner network, Rθ, is a residual network (ResNet). It modifies the synthetic image on a pixel level, rather
    than holistically modifying the image content, preserving the global structure and annotations.

    :param input_image_tensor: Input tensor that corresponds to a synthetic image.
    :return: Output tensor that corresponds to a refined synthetic image.
    """
    def resnet_block(input_features, nb_features=64, nb_kernel_rows=3, nb_kernel_cols=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.

        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.

        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(input_features)
        y = layers.Activation('relu')(y)
        y = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(y)

        y = layers.merge([input_features, y], mode='sum')
        return layers.Activation('relu')(y)

    # an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
    x = layers.Convolution2D(64, 3, 3, border_mode='same')(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    for i in range(4):
        x = resnet_block(x)

    # the output of the last ResNet block is passed to a 1 × 1 convolutional layer producing 1 feature map
    # corresponding to the refined synthetic image
    return layers.Convolution2D(1, 1, 1, border_mode='same')(x)


def discriminator_network(input_image_tensor):
    """
    The discriminator network, Dφ, contains 5 convolution layers and 2 max-pooling layers.

    :param input_image_tensor: Input tensor corresponding to an image, either real or refined.
    :return: Output tensor that corresponds to the probability of whether an image is real or refined.
    """
    x = layers.Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2))(input_image_tensor)
    x = layers.Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(x)
    x = layers.Convolution2D(32, 3, 3, border_mode='same', subsample=(1, 1))(x)
    x = layers.Convolution2D(32, 1, 1, border_mode='same', subsample=(1, 1))(x)
    x = layers.Convolution2D(2, 1, 1, border_mode='same', subsample=(1, 1))(x)

    x = layers.Reshape((DISC_SOFTMAX_OUTPUT_DIM, ))(x)
    return layers.Activation('softmax', name='disc_softmax')(x)


def adversarial_training(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path=None, discriminator_model_path=None):
    """Adversarial training of refiner network Rθ and discriminator network Dφ."""
    #
    # define model input and output tensors
    #
    synthetic_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    refined_image_tensor = refiner_network(synthetic_image_tensor)

    refined_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    discriminator_output = discriminator_network(refined_or_real_image_tensor)

    combined_output = discriminator_network(refiner_network(synthetic_image_tensor))

    #
    # define models
    #
    refiner_model = models.Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')
    discriminator_model = models.Model(input=refined_or_real_image_tensor, output=discriminator_output,
                                       name='discriminator')
    # combined must output the refined image along w/ the disc's classification of it for the refiner's self-reg loss
    combined_model = models.Model(input=synthetic_image_tensor, output=[refined_image_tensor, combined_output],
                                  name='combined')

    #
    # define custom l1 loss function for the refiner
    #
    def self_regularization_loss(y_true, y_pred):
        delta = 0.001  # FIXME: need to find ideal value for this
        return tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))

    #
    # compile models
    #
    refiner_model.compile(optimizer='adam', loss=self_regularization_loss)
    discriminator_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    discriminator_model.trainable = False
    # TODO: add accuracy metric for `combined_output`
    combined_model.compile(optimizer='adam', loss=[self_regularization_loss, 'categorical_crossentropy'])

    #
    # data generators
    #
    datagen = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,
        dim_ordering='tf')

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                  'class_mode': None,
                                  'batch_size': batch_size}

    synthetic_generator = datagen.flow_from_directory(
        directory=synthesis_eyes_dir,
        **flow_from_directory_params
    )

    real_generator = datagen.flow_from_directory(
        directory=mpii_gaze_dir,
        **flow_from_directory_params
    )

    def get_image_batch(generator):
        """keras generators may generate an incomplete batch for the last batch"""
        img_batch = generator.next()
        if len(img_batch) != batch_size:
            img_batch = generator.next()

        assert len(img_batch) == batch_size

        return img_batch

    # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (refined)
    y_real = np.zeros(shape=(batch_size, DISC_SOFTMAX_OUTPUT_DIM))
    y_refined = np.ones(shape=(batch_size, DISC_SOFTMAX_OUTPUT_DIM))

    if not refiner_model_path:
        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print('pre-training the refiner network...')
        gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

        for i in range(1000):
            synthetic_image_batch = get_image_batch(synthetic_generator)
            gen_loss = np.add(refiner_model.train_on_batch(synthetic_image_batch, synthetic_image_batch), gen_loss)

            # log every `log_interval` steps
            if not i % log_interval:
                figure_name = 'refined_image_batch_pre_train_step_{}'.format(i)
                print('Saving batch of refined images during pre-training at step: {}.'.format(i))

                synthetic_image_batch = get_image_batch(synthetic_generator)
                plot_batch(synthetic_image_batch, refiner_model.predict(synthetic_image_batch), figure_name)

                print('Refiner model self regularization loss: {}.'.format(gen_loss / log_interval))
                gen_loss = np.zeros(shape=len(refiner_model.metrics_names))

        refiner_model.save(os.path.join(cache_dir, 'refiner_model_pre_trained.h5'))
    else:
        refiner_model.load_weights(refiner_model_path)

    if not discriminator_model_path:
        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print('pre-training the discriminator network...')
        disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

        for i in range(100):
            real_image_batch = get_image_batch(real_generator)
            disc_loss = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss)

            synthetic_image_batch = get_image_batch(synthetic_generator)
            refined_image_batch = refiner_model.predict(synthetic_image_batch)
            disc_loss = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined), disc_loss)

        discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))

        # hard-coded for now
        print('Discriminator model loss: {}.'.format(disc_loss / (100 * 2)))
    else:
        discriminator_model.load_weights(discriminator_model_path)

    combined_loss = np.zeros(shape=len(combined_model.metrics_names))
    disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

    # see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the refiner
        for _ in range(k_g * 2):
            # sample a mini-batch of synthetic images
            synthetic_image_batch = get_image_batch(synthetic_generator)

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            combined_loss = np.add(combined_model.train_on_batch(synthetic_image_batch,
                                                                 [synthetic_image_batch, y_real]), combined_loss)

        for _ in range(k_d):
            # sample a mini-batch of synthetic and real images
            synthetic_image_batch = get_image_batch(synthetic_generator)
            real_image_batch = get_image_batch(real_generator)

            # refine the synthetic images w/ the current refiner
            refined_image_batch = refiner_model.predict(synthetic_image_batch)

            # use a history of refined images
            half_batch_from_image_history = get_half_batch_from_image_history()
            add_half_batch_to_image_history(refined_image_batch[:batch_size // 2])

            try:
                refined_image_batch[:batch_size // 2] = half_batch_from_image_history[:batch_size // 2]
            except IndexError and ValueError as e:
                print(e)
                pass

            # update φ by taking an SGD step on mini-batch loss LD(φ)
            disc_loss = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss)
            disc_loss = np.add(discriminator_model.train_on_batch(refined_image_batch, y_refined), disc_loss)

        if not i % log_interval:
            # plot batch of refined images w/ current refiner
            figure_name = 'refined_image_batch_step_{}'.format(i)
            print('Saving batch of refined images at adversarial step: {}.'.format(i))

            synthetic_image_batch = get_image_batch(synthetic_generator)
            plot_batch(synthetic_image_batch, refiner_model.predict(synthetic_image_batch), figure_name)

            # log loss summary
            print('Refiner model loss: {}.'.format(combined_loss / (log_interval * k_g * 2)))
            print('Discriminator model loss: {}.'.format(disc_loss / (log_interval * k_d * 2)))

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_step_{}')
            refiner_model.save(model_checkpoint_base_name.format('refiner', i))
            discriminator_model.save(model_checkpoint_base_name.format('discriminator', i))


def main(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path, discriminator_model_path):
    adversarial_training(synthesis_eyes_dir, mpii_gaze_dir, refiner_model_path, discriminator_model_path)


if __name__ == '__main__':
    # TODO: if pre-trained models are passed in, we don't take the steps they've been trained for into account
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
