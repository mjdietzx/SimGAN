from keras import backend as K
from keras import applications
from keras import layers
from keras import models
from keras.preprocessing import image


#
# image dimensions
#
img_width = 55
img_height = 35
img_channels = 1


def refiner_network(input_image_tensor):
    """
    The refiner network, Rθ, is a residual network (ResNet).

    :param input_image_tensor: Input tensor corresponding to a synthetic image from a simulator.
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
    x = layers.Activation('softmax')(x)

    return x


def adversarial_training():
    """Adversarial training of refiner network Rθ."""
    #
    # define model inputs and outputs
    #
    synthetic_image_tensor = layers.Input(shape=(img_width, img_height, img_channels))
    refined_image_tensor = refiner_network(synthetic_image_tensor)

    refined_or_real_image_tensor = layers.Input(shape=(img_width, img_height, img_channels))
    discriminator_output = discriminator_network(refined_or_real_image_tensor)

    #
    # define models
    #
    refiner_model = models.Model(input=synthetic_image_tensor, output=refined_image_tensor, name='refiner')
    discriminator_model = models.Model(input=refined_or_real_image_tensor, output=discriminator_output, name='discriminator')
    combined_model = models.Model(input=synthetic_image_tensor, output=discriminator_output, name='combined')

    #
    # define custom loss function for the refiner
    #
    def refiner_loss(y_true, y_pred):
        """
        LR(θ) = −log(1 − Dφ(Rθ(xi))) - λ * ||Rθ(xi) − xi||, where ||.|| is the l1 norm

        :param y_true: (discriminator classifies refined image as real, synthetic image tensor)
        :param y_pred: (discriminator's prediction of refined image, refined image tensor)
        :return: The total loss.
        """
        delta = 0.001

        loss_real = K.mean(K.binary_crossentropy(y_pred[0], y_true[0]), axis=-1)
        loss_reg = K.multiply(delta, K.reduce_sum(K.abs(y_pred[0] - y_true[1])))
        return loss_real + loss_reg

    #
    # compile models
    #
    refiner_model.compile(optimizer='sgd', loss=refiner_loss)
    discriminator_model.compile(optimizer='sgd', loss='binary_crossentropy')

    discriminator_model.trainable = False
    combined_model.compile(optimizer='sgd', loss='binary_crossentropy')


def main():
    adversarial_training()


if __name__ == '__main__':
    main()
