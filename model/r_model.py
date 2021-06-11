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
    def __init__(self, input_shape, dim_y, dim_x, num_class=10,
                 optimizer=None, w_rec=1, w_kl_y=1, w_kl_x=1, w_class=1):
        super(DVAE, self).__init__()
        self.encoder_y = Encoder(input_shape, dim_y)
        self.encoder_x = Encoder(input_shape, dim_x)
        self.decoder = Decoder(dim_y + dim_x)
        self.classifier_y = Classifier(dim_y, num_class, num_dense=0)
        self.classifier_x = Classifier(dim_x, num_class)

        optimizer = optimizer if optimizer is not None else keras.optimizers.Adam(**ADAM_ARGS)
        self.set_train_params(optimizer=optimizer, w_rec=w_rec, w_kl_y=w_kl_y, w_kl_x=w_kl_x, w_class=w_class)

    def set_train_params(self, optimizer=None, w_rec=None, w_kl_y=None, w_kl_x=None, w_class=None):
        if optimizer:
            self.optimizer = optimizer
        if w_rec:
            self.w_rec = w_rec
        if w_kl_y:
            self.w_kl_y = w_kl_y
        if w_kl_x:
            self.w_kl_x = w_kl_x
        if w_class:
            self.w_class = w_class

    def compile(self, *args, **kwargs):
        super(DVAE, self).compile(*args, **kwargs)
        self.set_train_params(*args, **kwargs)
        # initialize metrics
        self.metric_loss = {name: keras.metrics.Mean() for name in ['l_rec', 'l_kl_y', 'l_kl_x', 'l_class']}
        self.metric_acc = {name: keras.metrics.CategoricalAccuracy() for name in ['acc_z_y', 'acc_z_x']}

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
        class_pred_x = self.classifier_x(z_x)
        class_pred_x = grad_reverse(class_pred_x)  # reverse gradients for x

        # compute all losses
        # ELBO
        l_rec = self.w_rec * self.loss_rec(x, x_rec)
        l_kl_y = self.w_kl_y * self.loss_kl(mu_y, log_sigma_y)
        l_kl_x = self.w_kl_x * self.loss_kl(mu_x, log_sigma_x)

        # Classification / disentanglement losses
        l_class = self.w_class * (self.loss_classify(class_pred_y, y) + self.loss_classify(class_pred_x, y))

        return (l_rec, l_kl_y, l_kl_x, l_class), (x_rec, class_pred_y, class_pred_x)

    def get_metric(self, batch, l_rec, l_kl_y, l_kl_x, l_class, class_pred_y, class_pred_x):
        x, y = batch
        for metric, loss_value in zip(['l_rec', 'l_kl_y', 'l_kl_x', 'l_class'], [l_rec, l_kl_y, l_kl_x, l_class]):
            self.metric_loss[metric].update_state(loss_value)

        for metric, pred in zip(['acc_z_y', 'acc_z_x'], [class_pred_y, class_pred_x]):
            self.metric_acc[metric].update_state(pred, y)

        return {name: metric.result() for name, metric in list(self.metric_loss.items()) + list(self.metric_acc.items())}

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            (l_rec, l_kl_y, l_kl_x, l_class), (_, class_pred_y, class_pred_x) = self.forward_loss(batch)
            loss = l_rec + l_kl_y + l_kl_x + l_class
        train_variables = [*self.encoder_y.trainable_variables, *self.encoder_x.trainable_variables,
                           *self.decoder.trainable_variables, *self.classifier_y.trainable_variables,
                           *self.classifier_x.trainable_variables]
        gradients = tape.gradient(loss, train_variables)
        self.optimizer.apply_gradients(zip(gradients, train_variables))
        return self.get_metric(batch, l_rec, l_kl_y, l_kl_x, l_class, class_pred_y, class_pred_x)

    def test_step(self, batch):
        (l_rec, l_kl_y, l_kl_x, l_class), (_, class_pred_y, class_pred_x) = self.forward_loss(batch)
        return self.get_metric(batch, l_rec, l_kl_y, l_kl_x, l_class, class_pred_y, class_pred_x)

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

