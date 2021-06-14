"""Implementations of representation models, REPR.
"""
from model.base import REPR
import tensorflow as tf
from tensorflow import keras
from util.model import grad_reverse
from model.component import Encoder, Decoder, Classifier
from model.config import ADAM_ARGS


class DVAE(REPR):
    """Disentangled VAE, using class labels.
    Splits latent space into z_y and z_x, where the former contains class-related information, and the latter
    contains the remaining information.

    Disentanglement is encouraged using auxiliary classifiers on both latent spaces.
    """

    _name_loss = ['rec', 'kl_y', 'kl_x', 'class']
    _name_acc = ['z_y', 'z_x']

    def __init__(self, input_shape, dim_y, dim_x, num_class=10,
                 optimizer=None, **kwargs):
        super(DVAE, self).__init__()
        self.encoder_y = Encoder(input_shape, dim_y)
        self.encoder_x = Encoder(input_shape, dim_x)
        self.decoder = Decoder(dim_y + dim_x, output_shape=input_shape)
        self.classifier_y = Classifier(dim_y, num_class, num_dense=0)
        self.classifier_x = Classifier(dim_x, num_class)

        self.set_save_info(args={'input_shape': input_shape, 'dim_y': dim_y, 'dim_x': dim_x, 'num_class': num_class},
                           models={'enc_y': self.encoder_y, 'enc_x': self.encoder_x, 'dec': self.decoder,
                           'class_y': self.classifier_y, 'class_x': self.classifier_x})

        optimizer if optimizer is not None else keras.optimizers.Adam(**ADAM_ARGS)
        self.set_train_params(optimizer=optimizer, **kwargs)

    def set_train_params(self, optimizer=None, **kwargs):
        if optimizer:
            self.optimizer = optimizer
        self.parse_weights(**kwargs)

    def compile(self, *args, **kwargs):
        super(DVAE, self).compile(*args, **kwargs)
        self.set_train_params(*args, **kwargs)

    def forward_loss(self, batch):
        x, y = batch
        # infer / sample latent
        mu_y, log_sigma_y = self.encoder_y(x)
        z_y = self.sample(mu_y, log_sigma_y)
        mu_x, log_sigma_x = self.encoder_x(x)
        z_x = self.sample(mu_x, log_sigma_x)

        # reconstruct
        x_rec = self.decoder(tf.concat([z_y, z_x], axis=1))

        # get class predictions based off latent samples
        class_pred_y = self.classifier_y(z_y)
        class_pred_x = self.classifier_x(grad_reverse(z_x))  # z_x has reversed gradient

        # compute all losses
        # ELBO
        l_rec = self.w['rec'] * self.loss_rec(x, x_rec)
        l_kl_y = self.w['kl_y'] * self.loss_kl(mu_y, log_sigma_y)
        l_kl_x = self.w['kl_x'] * self.loss_kl(mu_x, log_sigma_x)

        # Classification / disentanglement losses
        l_class = self.w['class'] * (self.loss_classify(class_pred_y, y) + self.loss_classify(class_pred_x, y))

        return (l_rec, l_kl_y, l_kl_x, l_class), (x_rec, class_pred_y, class_pred_x)

    def train_step(self, batch):
        l = {}
        pred = {}
        with tf.GradientTape() as tape:
            (l['rec'], l['kl_y'], l['kl_x'], l['class']), (_, pred['z_y'], pred['z_x']) = self.forward_loss(batch)
            loss = sum(l.values())
        train_variables = [*self.encoder_y.trainable_variables, *self.encoder_x.trainable_variables,
                           *self.decoder.trainable_variables, *self.classifier_y.trainable_variables,
                           *self.classifier_x.trainable_variables]
        gradients = tape.gradient(loss, train_variables)
        self.optimizer.apply_gradients(zip(gradients, train_variables))
        return self.get_metric(loss=l, pred=pred, y=batch[1])

    def test_step(self, batch):
        l, pred = {}, {}
        (l['rec'], l['kl_y'], l['kl_x'], l['class']), (_, pred['z_y'], pred['z_x']) = self.forward_loss(batch)
        return self.get_metric(loss=l, pred=pred, y=batch[1])

    def encode_y(self, x):
        mu_y, _ = self.encoder_y(x)
        return mu_y

    def encode_x(self, x):
        mu_x, _ = self.encoder_x(x)
        return mu_x

    def decode(self, z_y, z_x):
        return self.decoder(tf.concat([z_y, z_x], axis=1))


if __name__ == '__main__':
    dvae = DVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8)
    dvae.compile()
    print(dvae)

