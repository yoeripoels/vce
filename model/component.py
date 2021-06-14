"""Model components used for representation models.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape, BatchNormalization, \
    Conv2DTranspose
import tensorflow.keras.layers
from model.config import DROPOUT_ALPHA, LEAKYRELU_ALPHA

# layers shared through models
LeakyReLU = tensorflow.keras.layers.LeakyReLU(alpha=DROPOUT_ALPHA)
Dropout = tensorflow.keras.layers.Dropout(LEAKYRELU_ALPHA)


def Encoder(input_shape, dim_z):
    # input
    x_in = Input(shape=input_shape, name='encoder_input')

    # hidden layers
    strides = (2, 2)
    strides_small = (1, 1)
    kernel_size = 4
    l = Conv2D(filters=32, kernel_size=kernel_size, strides=strides, padding='same', name='1')(x_in)
    l = LeakyReLU(l)
    l = BatchNormalization()(l)
    l = Conv2D(filters=64, kernel_size=kernel_size, strides=strides_small, padding='same', name='2')(l)
    l = LeakyReLU(l)
    l = BatchNormalization()(l)
    l = Conv2D(filters=128, kernel_size=kernel_size, strides=strides_small, padding='same', name='3')(l)
    l = LeakyReLU(l)
    l = BatchNormalization()(l)
    l = Flatten(name='4')(l)

    # latent space
    z_mean = Dense(dim_z, name='z_mean')(l)
    z_log_sigma = Dense(dim_z, name='z_log_sigma')(l)

    return Model(x_in, [z_mean, z_log_sigma], name='encoder')


def Decoder(dim_z, output_shape=None):
    if output_shape:  # if output shape passed, compute inner dense dim
        h, w = output_shape[0:2]
        assert h % 2 == 0 and w % 2 == 0  # as we apply (2,2) strides, we assume even resolution
        h_in, w_in = h//2, w//2
    else:
        h_in, w_in = 16, 16

    # input
    z_in = Input(shape=dim_z, name='decoder_input')

    # hidden layers
    strides = (2, 2)
    strides_small = (1, 1)
    kernel_size = 4
    l = Dense(h_in * w_in * 128, name='4')(z_in)
    l = LeakyReLU(l)
    l = Reshape((h_in, w_in, 128), name='4_b')(l)
    l = BatchNormalization()(l)
    l = Conv2DTranspose(filters=64, kernel_size=kernel_size, strides=strides_small, padding='same', name='3')(l)
    l = LeakyReLU(l)
    l = BatchNormalization()(l)
    l = Conv2DTranspose(filters=32, kernel_size=kernel_size, strides=strides_small, padding='same', name='2')(l)
    l = LeakyReLU(l)
    l = BatchNormalization()(l)

    # reconstruction
    x_out = Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=strides, padding='same', activation='sigmoid',
                            name='1')(l)

    return Model(z_in, x_out, name='decoder')


def Classifier(dim_z, num_class, num_dense=1, dense_dim=50, use_dropout=False):
    z_in = Input(shape=dim_z, name='classifier_input')
    for i in range(num_dense):
        l = Dense(dense_dim, name=str(i+1))(z_in if i == 0 else l)
        l = LeakyReLU(l)
        l = BatchNormalization()(l)
        if use_dropout:
            l = Dropout(l)
    pred = Dense(num_class, name='prediction', activation='softmax')(z_in if num_dense == 0 else l)
    return Model(z_in, pred, name='classifier')


def Encoder_CD(input_shape, latent_dim):
    # input
    x_in = Input(shape=input_shape, name='encoder_input')

    # hidden layers
    strides = (2, 2)
    strides_small = (1, 1)
    kernel_size = 4
    l = Conv2D(filters=32, kernel_size=kernel_size, strides=strides, padding='same', name='1')(x_in)
    l = LeakyReLU(l)
    l = BatchNormalization()(l)
    l = Conv2D(filters=128, kernel_size=kernel_size, strides=strides_small, padding='same', name='2')(l)
    l = LeakyReLU(l)
    l = BatchNormalization()(l)
    l = Flatten(name='3')(l)

    # latent space
    z_mean = Dense(latent_dim, name='z_mean')(l)
    z_log_sigma = Dense(latent_dim, name='z_log_sigma')(l)

    return Model(x_in, [z_mean, z_log_sigma], name='encoder')


def Decoder_CD(latent_dim, output_shape=None):
    if output_shape:  # if output shape passed, compute inner dense dim
        h, w = output_shape[0:2]
        assert h % 2 == 0 and w % 2 == 0  # as we apply (2,2) strides, we assume even resolution
        h_in, w_in = h//2, w//2
    else:
        h_in, w_in = 16, 16

    # input
    z_in = Input(shape=latent_dim, name='decoder_input')

    # hidden layers
    strides = (2, 2)
    strides_small = (1, 1)
    kernel_size = 4
    l = Dense(h_in * w_in * 128, name='3')(z_in)
    l = LeakyReLU(l)
    l = Reshape((h_in, w_in, 128), name='3_b')(l)
    l = BatchNormalization()(l)
    l = Conv2DTranspose(filters=32, kernel_size=kernel_size, strides=strides_small, padding='same', name='2')(l)
    l = LeakyReLU(l)
    l = BatchNormalization()(l)

    # reconstruction
    x_out = Conv2DTranspose(filters=1, kernel_size=kernel_size, strides=strides, padding='same', activation='sigmoid',
                            name='1')(l)

    return Model(z_in, x_out, name='decoder')


def Discriminator(input_shape):
    # input
    x_in = Input(shape=input_shape, name='discriminator_input')

    # hidden layers
    strides = (2, 2)
    strides_small = (1, 1)
    kernel_size = 4
    l = Conv2D(filters=32, kernel_size=kernel_size, strides=strides, padding='same', name='1')(x_in)
    l = LeakyReLU(l)
    l = Dropout(l)
    l = BatchNormalization()(l)
    l = Conv2D(filters=64, kernel_size=kernel_size, strides=strides_small, padding='same', name='2')(l)
    l = LeakyReLU(l)
    l = Dropout(l)
    l = BatchNormalization()(l)
    l = Conv2D(filters=128, kernel_size=kernel_size, strides=strides_small, padding='same', name='3')(l)
    l = LeakyReLU(l)
    l = Dropout(l)
    l = BatchNormalization()(l)
    l = Flatten(name='4')(l)

    # realism prediction
    pred = Dense(2, name='z_mean', activation='softmax')(l)

    return Model(x_in, pred, name='discriminator')
