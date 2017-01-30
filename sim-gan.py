from keras import layers
from keras import models


def refiner_network(input_image_tensor):
    """
    The refiner network, Rθ, is a residual network (ResNet).

    :param input_image_tensor: Input tensor corresponding to a synthetic image from a simulator.
    :return: A model that improves the realism of synthetic images from a simulator.
    """
    def resnet_block(input_features, nb_features=64, nb_kernel_rows=3, nb_kernel_cols=3):
        """
        A ResNet block with two `nb_kernel_rows` x `nb_kernel_cols` convolutional layers,
        each with `nb_features` feature maps.

        See Figure 6 in https://arxiv.org/pdf/1612.07828v1.pdf.

        :param input_features: Input tensor to ResNet block.
        :return: Output tensor from ResNet block.
        """
        x = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(input_features)
        x = layers.Activation('relu')(x)
        x = layers.Convolution2D(nb_features, nb_kernel_rows, nb_kernel_cols, border_mode='same')(x)

        x = layers.merge([input_features, x], mode='sum')
        return layers.Activation('relu')(x)

    # an input image of size w × h is convolved with 3 × 3 filters that output 64 feature maps
    x = layers.Convolution2D(64, 3, 3, border_mode='same')(input_image_tensor)

    # the output is passed through 4 ResNet blocks
    for i in range(4):
        x = resnet_block(x)

    # the output of the last ResNet block is passed to a 1 × 1 convolutional layer producing 1 feature map
    # corresponding to the refined synthetic image
    x = layers.Convolution2D(1, 1, 1, border_mode='same')(x)

    return models.Model(input=input_image_tensor, output=x, name='refiner')


def discriminator_network(input_image_tensor):
    """
    The discriminator network, Dφ, contains 5 convolution layers and 2 max-pooling layers.

    :param input_image_tensor: Input tensor corresponding to an image, either real or refined.
    :return: A model that determines whether an image is real or refined.
    """
    x = layers.Convolution2D(96, 3, 3, border_mode='same', subsample=(2, 2))(input_image_tensor)
    x = layers.Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2))(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same')(x)
    x = layers.Convolution2D(32, 3, 3, border_mode='same', subsample=(1, 1))(x)
    x = layers.Convolution2D(32, 1, 1, border_mode='same', subsample=(1, 1))(x)
    x = layers.Convolution2D(2, 1, 1, border_mode='same', subsample=(1, 1))(x)
    x = layers.Activation('softmax')(x)

    return models.Model(input=input_image_tensor, output=x, name='discriminator')


def main():
    pass


if __name__ == '__main__':
    main()
