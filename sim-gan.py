"""
Implementation of `3.1 Appearance-based Gaze Estimation` from
[Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf).

Note: Only Python 3 support currently.
"""
from keras import applications
from keras import layers
from keras import models
from keras import objectives
from keras.preprocessing import image
from keras.utils import np_utils
import numpy as np
import tensorflow as tf


#
# dataset directory paths
#
synthesis_eyes_dir = '/Users/mjdietzx/Downloads/SynthEyes_data'  # https://www.cl.cam.ac.uk/research/rainbow/projects/syntheseyes/
mp_gaze_dir = ''  # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/

#
# image dimensions
#
img_width = 55
img_height = 35
img_channels = 1

#
# training params
#
nb_epochs = 50
batch_size = 32
k_d = 1  # number of discriminator updates per step
k_g = 2  # number of generative network updates per step


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
    x = layers.Convolution2D(1, 1, 1, border_mode='same')(x)

    return x


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

    x = layers.Flatten()(x)
    x = layers.Activation('softmax')(x)

    return x

datagen = image.ImageDataGenerator(
    preprocessing_function=applications.xception.preprocess_input,
    dim_ordering='tf')

flow_from_directory_params = {'target_size': (img_width, img_height),
                              'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                              'class_mode': None,
                              'batch_size': batch_size}

synthetic_generator = datagen.flow_from_directory(
    directory=synthesis_eyes_dir,
    **flow_from_directory_params
)

"""real_generator = datagen.flow_from_directory(
    directory=mp_gaze_dir,
    **flow_from_directory_params
)"""


def adversarial_training():
    """Adversarial training of refiner network Rθ."""
    #
    # define model inputs and outputs
    #
    synthetic_image_tensor = layers.Input(shape=(img_width, img_height, img_channels))
    refined_image_tensor = refiner_network(synthetic_image_tensor)

    refined_or_real_image_tensor = layers.Input(shape=(img_width, img_height, img_channels))
    discriminator_output = discriminator_network(refined_or_real_image_tensor)

    combined_output = discriminator_network(refiner_network(synthetic_image_tensor))

    #
    # define models
    #
    refiner_model = models.Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')
    discriminator_model = models.Model(input=refined_or_real_image_tensor, output=discriminator_output, name='discriminator')
    combined_model = models.Model(input=synthetic_image_tensor, output=combined_output, name='combined')

    #
    # define custom loss function for the refiner
    #
    def self_regularization_loss(y_true, y_pred):
        delta = 0.001  # FIXME: need to find ideal value for this
        loss_reg = tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))

        return loss_reg

    def refiner_loss(y_true, y_pred):
        """
        LR(θ) = −log(1 − Dφ(Rθ(xi))) - λ * ||Rθ(xi) − xi||, where ||.|| is the l1 norm

        :param y_true: (discriminator classifies refined image as real, synthetic image tensor)
        :param y_pred: (discriminator's prediction of refined image, refined image tensor)
        :return: The total loss.
        """
        loss_real = objectives.binary_crossentropy(y_true[0], y_pred[0])
        loss_reg = self_regularization_loss(y_true[1], y_pred[1])

        return loss_real + loss_reg

    #
    # compile models (hold off on compiling the refiner)
    #
    discriminator_model.compile(optimizer='sgd', loss='categorical_crossentropy')
    discriminator_model.trainable = False
    combined_model.compile(optimizer='sgd', loss='categorical_crossentropy')

    # the target labels for the cross-entropy loss layer are 0 for every yj and 1 for every xi
    y_real = np_utils.to_categorical(np.zeros(shape=batch_size), nb_classes=2)
    y_refined = np_utils.to_categorical(np.ones(shape=batch_size), nb_classes=2)

    # we first train the Rθ network with just self-regularization loss for 1,000 steps
    for _ in range(1000):
        refiner_model.compile(optimizer='sgd', loss=self_regularization_loss)

        image_batch = synthetic_generator.next()
        if len(image_batch) != batch_size:
            image_batch = synthetic_generator.next()

        refiner_model.train_on_batch(image_batch, image_batch)

    refiner_model.compile(optimizer='sgd', loss=refiner_loss)

    # and Dφ for 200 steps
    for _ in range(200):
        discriminator_model.train_on_batch()

    # see Algorithm 1 in https://arxiv.org/pdf/1612.07828v1.pdf
    for i in range(nb_epochs):
        print('Epoch: {} of {}.'.format(i, nb_epochs))

        # train the refiner
        for _ in range(k_g):
            # sample a mini-batch of synthetic images
            # update θ by taking an SGD step on mini-batch loss LR(θ)
            pass

        for _ in range(k_d):
            # sample a mini-batch of synthetic and real images
            # refine the synthetic images w/ the current refiner
            # update φ by taking an SGD step on mini-batch loss LD(φ)
            pass


def main():
    adversarial_training()


if __name__ == '__main__':
    main()
