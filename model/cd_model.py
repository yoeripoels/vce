"""Implementations of change-discriminator models, CD.
"""
from model.base import CD
from model.r_model import DVAE
import tensorflow as tf
from model.component import Classifier, Encoder_CD, Decoder_CD
import tensorflow.keras as keras

class CD_DVAE(DVAE, CD):
    """Change discriminator model, built upon DVAE.
    Uses a disentangled z_y (class) and z_x (remaining).
    We use the absolute (dimension-wise) difference of z_y between a pair of datapoints, to classify a sample
    as a 'good' or 'bad' change.
    """

    _name_loss = ['rec', 'kl_y', 'kl_x', 'class', 'chg_disc']
    _name_weight_extra = ['full']
    _name_acc = ['z_y', 'z_x', 'chg_disc']

    def __init__(self, input_shape, dim_y, dim_x, num_class=10,
                 optimizer=None, cnn_list=None, dense_list=None, **kwargs):
        if cnn_list is None:  # default encoder config is different
            cnn_list = [(32, 4, (2, 2), 'same'),
                        (128, 4, (1, 1), 'same')]
        super(CD_DVAE, self).__init__(input_shape, dim_y, dim_x, num_class, optimizer, cnn_list, dense_list, **kwargs)
        self.classifier_y = Classifier(dim_y, num_class)
        self.classifier_x = Classifier(dim_x, num_class)

        self.change_discriminator = Classifier(dim_y, num_class=2, use_dropout=True)
        self.set_save_info(args={'input_shape': input_shape, 'dim_y': dim_y, 'dim_x': dim_x, 'num_class': num_class,
                                 'cnn_list': cnn_list, 'dense_list': dense_list},
                           models={'enc_y': self.encoder_y, 'enc_x': self.encoder_x, 'dec': self.decoder,
                                   'class_y': self.classifier_y, 'class_x': self.classifier_x,
                                   'chg_disc': self.change_discriminator})

        train_variables = [*self.encoder_y.trainable_variables, *self.encoder_x.trainable_variables,
                           *self.decoder.trainable_variables, *self.classifier_y.trainable_variables,
                           *self.classifier_x.trainable_variables, *self.change_discriminator.trainable_variables]

        self.init_optimizer('main', train_variables=train_variables)

    def forward_loss(self, batch):
        x_a, x_b, disc_true, x_full, y_full = batch

        # regular VAE loss for both pairs
        loss_a, data_a = self.elbo_loss(x_a)
        z_y_a = data_a[2]
        loss_b, data_b = self.elbo_loss(x_b)
        z_y_b = data_b[2]

        l_rec, l_kl_y, l_kl_x = [(l_a + l_b) for l_a, l_b in zip(loss_a, loss_b)]

        # DISCRIMINATE LOSS
        z_dif = tf.math.abs(z_y_a - z_y_b)
        # z_dif = tf.clip_by_value(z_dif, 0, 2)  # clip to 2x std
        disc_pred = self.change_discriminator(z_dif)
        l_disc = self.w['chg_disc'] * self.loss_classify(disc_pred, disc_true)

        (l_rec_full, l_kl_y_full, l_kl_x_full, l_class), (_, _, _, class_pred_y, class_pred_x, _) = \
            super(CD_DVAE, self).forward_loss((x_full, y_full))
        l_rec += l_rec_full * self.w['full']
        l_kl_y += l_kl_y_full * self.w['full']
        l_kl_x += l_kl_x_full * self.w['full']

        # (l_rec, l_kl_y, l_kl_x, l_class), (z_y, z_x, x_rec, class_pred_y, class_pred_x)

        return (l_rec, l_kl_y, l_kl_x, l_class, l_disc), \
               (class_pred_y, class_pred_x, y_full, disc_pred, disc_true)

    def train_step(self, batch):
        l = {}
        pred = {}
        with tf.GradientTape() as tape:
            (l['rec'], l['kl_y'], l['kl_x'], l['class'], l['chg_disc']), \
            data_return = self.forward_loss(batch)
            pred['z_y'], pred['z_x'], y, pred_d, true_d = data_return
            loss = sum(l.values())

        # train model + discriminator
        train_variables = self._optimizers['main']['train_variables']
        gradients = tape.gradient(loss, train_variables)
        self._optimizers['main']['optimizer'].apply_gradients(zip(gradients, train_variables))

        # update metrics
        self.update_metric(l, metric_type='loss')
        self.update_metric(pred, y=y, metric_type='acc')
        self.update_metric_single('chg_disc', pred_d, y=true_d, metric_type='acc')
        self.update_metric_single(value=loss, metric_type='class_loss')
        return self.get_metric()

    def test_step(self, batch):
        l = {}
        pred = {}
        (l['rec'], l['kl_y'], l['kl_x'], l['class'], l['chg_disc']), \
        data_return = self.forward_loss(batch)
        pred['z_y'], pred['z_x'], y, pred_d, true_d = data_return

        # update metrics
        self.update_metric(l, metric_type='loss')
        self.update_metric(pred, y=y, metric_type='acc')
        self.update_metric_single('chg_disc', pred_d, y=true_d, metric_type='acc')
        self.update_metric_single(value=sum(l.values()), metric_type='class_loss')
        return self.get_metric()

    def discriminate(self, x_a, x_b):
        # encode both, discriminate using inferred means
        mu_y_a, _ = self.encoder_y(x_a)
        mu_y_b, _ = self.encoder_y(x_b)
        z_dif = tf.math.abs(mu_y_a - mu_y_b)
        z_dif = tf.clip_by_value(z_dif, 0, 2)  # clip to 2x std
        disc_pred = self.change_discriminator(z_dif)
        return disc_pred
