"""Implementations of representation models, REPR.
"""
from model.base import REPR
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from util.model import grad_reverse
from model.component import Encoder, Decoder, Classifier, Discriminator
from model.config import ADAM_ARGS
import numpy as np


class DVAE(REPR):
    """Disentangled VAE, using class labels.
    Splits latent space into z_y and z_x, where the former contains class-related information, and the latter
    contains the remaining information.
    Disentanglement is encouraged using auxiliary classifiers on both latent spaces.

    This class-based disentanglement method is based on label-based disentanglement methods, e.g.:
    "R. Cai, Z. Li, P. Wei, J. Qiao, K. Zhang, and Z. Hao. Learning disentangled semantic representation for domain
    adaptation." - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6759585/
    "Z. Ding, Y. Xu, W. Xu, G. Parmar, Y. Yang, M. Welling, and Z. Tu. Guided variational autoencoder for
    disentanglement learning." - https://arxiv.org/abs/2004.01255
    "M. Ilse, J. M. Tomczak, C. Louizos, and M. Welling. Diva: Domain invariant variational autoencoders" -
    https://proceedings.mlr.press/v121/ilse20a/ilse20a.pdf
    "Z. Zheng and L. Sun. Disentangling latent space for VAE by label relevant/irrelevant dimensions." -
    https://arxiv.org/abs/1812.09502
    """

    _name_loss = ['rec', 'kl_y', 'kl_x', 'class']
    _name_weight_extra = []
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

        # save init params for easy access
        self._input_shape = input_shape
        self._dim_y = dim_y
        self._dim_x = dim_x
        self._num_class = num_class

        optimizer if optimizer is not None else keras.optimizers.Adam(**ADAM_ARGS)  # default optimizer
        self.set_train_params(optimizer=optimizer, **kwargs)

    def set_train_params(self, optimizer=None, batch_size=None, **kwargs):
        if optimizer:
            self.optimizer = optimizer
        if batch_size:
            self.batch_size = batch_size
        self.parse_weights(**kwargs)

    def compile(self, *args, **kwargs):
        super(DVAE, self).compile(*args, **kwargs)
        self.set_train_params(*args, **kwargs)

    def elbo_loss(self, x):
        # infer / sample latent
        mu_y, log_sigma_y = self.encoder_y(x)
        z_y = self.sample(mu_y, log_sigma_y)
        mu_x, log_sigma_x = self.encoder_x(x)
        z_x = self.sample(mu_x, log_sigma_x)

        # reconstruct
        x_rec = self.decoder(tf.concat([z_y, z_x], axis=1))

        # ELBO loss
        l_rec = self.w['rec'] * self.loss_rec(x, x_rec)
        l_kl_y = self.w['kl_y'] * self.loss_kl(mu_y, log_sigma_y)
        l_kl_x = self.w['kl_x'] * self.loss_kl(mu_x, log_sigma_x)
        return (l_rec, l_kl_y, l_kl_x), (mu_y, log_sigma_y, z_y, mu_x, log_sigma_x, z_x, x_rec)

    def forward_loss(self, batch):
        x, y = batch

        # regular elbo loss
        (l_rec, l_kl_y, l_kl_x), (_, _, z_y, _, _, z_x, x_rec) = self.elbo_loss(x)

        # get class predictions based off latent samples
        class_pred_y = self.classifier_y(z_y)
        class_pred_x = self.classifier_x(grad_reverse(z_x))  # z_x has reversed gradient

        # classification / disentanglement losses
        l_class = self.w['class'] * (self.loss_classify(class_pred_y, y) + self.loss_classify(class_pred_x, y))

        return (l_rec, l_kl_y, l_kl_x, l_class), (z_y, z_x, x_rec, class_pred_y, class_pred_x, y)

    def train_step(self, batch):
        l = {}
        pred = {}
        with tf.GradientTape() as tape:
            dvae_loss, dvae_data = self.forward_loss(batch)
            l['rec'], l['kl_y'], l['kl_x'], l['class'] = dvae_loss
            _, _, _, pred['z_y'], pred['z_x'], y = dvae_data

            loss = sum(l.values())
        train_variables = [*self.encoder_y.trainable_variables, *self.encoder_x.trainable_variables,
                           *self.decoder.trainable_variables, *self.classifier_y.trainable_variables,
                           *self.classifier_x.trainable_variables]
        gradients = tape.gradient(loss, train_variables)
        self.optimizer.apply_gradients(zip(gradients, train_variables))
        return self.get_metric(loss=l, pred=pred, y=y)

    def test_step(self, batch):
        l, pred = {}, {}
        dvae_loss, dvae_data = self.forward_loss(batch)
        l['rec'], l['kl_y'], l['kl_x'], l['class'] = dvae_loss
        _, _, _, pred['z_y'], pred['z_x'], y = dvae_data
        return self.get_metric(loss=l, pred=pred, y=y)

    def encode_y(self, x):
        mu_y, _ = self.encoder_y(x)
        return mu_y

    def encode_x(self, x):
        mu_x, _ = self.encoder_x(x)
        return mu_x

    def decode(self, z_y, z_x):
        return self.decoder(tf.concat([z_y, z_x], axis=1))


class VAECE(DVAE):
    """Model for VAE-based Contrastive Explanation. Built upon DIVA.
    z_y's dimensions are individually disentangled using an auxiliary change-discriminator, which acts upon
    samples generated that different in a single z_y dimension.
    Further regularized using discriminator (as in GANs) on these reconstructions.
    """

    _name_loss = ['rec', 'kl_y', 'kl_x', 'class', 'chg_disc', 'disc_vae', 'disc']
    _name_acc = ['z_y', 'z_x', 'chg_disc', 'disc']

    def __init__(self, input_shape, dim_y, dim_x, num_class=10,
                 model_cd=None, optimizer=None, optimizer_disc=None, **kwargs):
        super(VAECE, self).__init__(input_shape, dim_y, dim_x, num_class, optimizer, **kwargs)
        self.discriminator = Discriminator(input_shape)
        self.change_discriminator = model_cd
        self.set_save_info(args={'input_shape': input_shape, 'dim_y': dim_y, 'dim_x': dim_x, 'num_class': num_class},
                           models={'enc_y': self.encoder_y, 'enc_x': self.encoder_x, 'dec': self.decoder,
                           'class_y': self.classifier_y, 'class_x': self.classifier_x, 'disc': self.discriminator})

        optimizer_disc = optimizer_disc if optimizer_disc is not None else keras.optimizers.Adam(**ADAM_ARGS)
        self.set_train_params(optimizer_disc=optimizer_disc, **kwargs)

    def set_cd(self, model_cd):
        # note that we must manually set the CD after loading a model from disc, as it is not part of the model
        self.change_discriminator = model_cd

    def set_train_params(self, optimizer_disc=None, **kwargs):
        super(VAECE, self).set_train_params(**kwargs)
        if optimizer_disc:
            self.optimizer_disc = optimizer_disc
        self.parse_weights(**kwargs)

    def compile(self, *args, **kwargs):
        super(VAECE, self).compile(*args, **kwargs)
        self.set_train_params(*args, **kwargs)

    def forward_loss(self, batch):
        x_a, x_b, y_a, y_b, x_real = batch
        # regular DVAE loss for both pairs
        dvae_loss_a, dvae_data_a = super(VAECE, self).forward_loss((x_a, y_a))
        dvae_loss_b, dvae_data_b = super(VAECE, self).forward_loss((x_b, y_b))
        l_rec, l_kl_y, l_kl_x, l_class = [(l_a + l_b)/2 for l_a, l_b in zip(dvae_loss_a, dvae_loss_b)]
        z_y, z_x, x_rec, class_pred_y, class_pred_x, y = [tf.concat([a, b], axis=0) for a, b in
                                                       zip(dvae_data_a, dvae_data_b)]

        ### pair-based dimension conditioning ###
        ## swap pair, get change discriminator loss ##
        # latent states we use
        z_y_a = dvae_data_a[0]
        z_x_a = dvae_data_a[1]
        z_y_b = dvae_data_b[0]

        # create interpolation configuration
        random_interp = tf.random.uniform((self.batch_size, self._dim_y), minval=0, maxval=2, dtype='int32')
        random_interp = tf.cast(random_interp, dtype='float32')
        # the change indices, what dim to swap
        change_idx = np.random.randint(0, self._dim_y, size=self.batch_size)
        updates = [1. - random_interp[i][change_idx[i]] for i in range(self.batch_size)]  # whether we set to 0 or 1

        # convert to [i, idx_to_change] for tensor_scatter_nd_update
        change_idx = np.array([[i, idx] for i, idx in enumerate(change_idx)])
        random_interp_after = tf.tensor_scatter_nd_update(random_interp, change_idx, updates)
        z_y_before = z_y_a * random_interp + z_y_b * (1 - random_interp)
        z_y_after = z_y_a * random_interp_after + z_y_b * (1 - random_interp_after)

        # get amplitude of changes
        dif = tf.reduce_sum(tf.math.abs(z_y_before - z_y_after), axis=-1)

        # normalize such that total dif = dim_y
        dif /= tf.reduce_sum(tf.math.abs(z_y_a - z_y_b), axis=-1)
        dif *= self._dim_y

        # decode both with the same style
        x_before = self.decoder(tf.concat([z_y_before, z_x_a], axis=1))
        x_after = self.decoder(tf.concat([z_y_after, z_x_a], axis=1))

        chg_disc_pred = self.change_discriminator.discriminate(x_before, x_after)
        chg_disc_true = tf.repeat([[0., 1.]], repeats=self.batch_size, axis=0)
        l_chg_disc = self.w['chg_disc'] * self.loss_classify(chg_disc_pred, chg_disc_true)

        ## feed synthesized pairs + real data into discriminator, updated VAECE and discriminator ##
        disc_pred_fake = self.discriminator(tf.concat([x_before, x_after], axis=0))
        disc_pred_real = self.discriminator(tf.concat([x_real, x_real], axis=0))  # to match batch size

        # set up the correct/incorrect predictions
        disc_y_false = tf.repeat([[1., 0.]], repeats=self.batch_size * 2, axis=0)
        disc_y_true = tf.repeat([[0., 1.]], repeats=self.batch_size * 2, axis=0)

        # VAE model loss (i.e., aim to predict all correct / [1., 0]
        l_disc_vae = self.w['disc_vae'] * self.loss_classify(disc_pred_fake, disc_y_true)

        # Discriminator model loss (correctly distinguish real data from samples)
        disc_pred = tf.concat([disc_pred_fake, disc_pred_real], axis=0)
        disc_true = tf.concat([disc_y_false, disc_y_true], axis=0)
        l_disc = self.w['disc'] * self.loss_classify(disc_pred, disc_true)
        return (l_rec, l_kl_y, l_kl_x, l_class, l_chg_disc, l_disc_vae, l_disc), \
               (z_y, z_x, x_rec, class_pred_y, class_pred_x, y, chg_disc_pred, chg_disc_true, disc_pred, disc_true)

    def train_step(self, batch):
        l = {}
        l_disc = {}
        pred = {}
        with tf.GradientTape(persistent=True) as tape:
            vaece_loss, vaece_data = self.forward_loss(batch)
            l['rec'], l['kl_y'], l['kl_x'], l['class'], l['chg_disc'], l['disc_vae'], l_disc['disc'] = vaece_loss
            _, _, _, pred['z_y'], pred['z_x'], y, pred_cd, true_cd, pred_d, true_d = vaece_data
            loss = sum(l.values())
            loss_disc = sum(l_disc.values())

        # train main representation model
        train_variables = [*self.encoder_y.trainable_variables, *self.encoder_x.trainable_variables,
                           *self.decoder.trainable_variables, *self.classifier_y.trainable_variables,
                           *self.classifier_x.trainable_variables]
        gradients = tape.gradient(loss, train_variables)
        self.optimizer.apply_gradients(zip(gradients, train_variables))

        # train discriminator
        train_variables = self.discriminator.trainable_variables
        gradients = tape.gradient(loss_disc, train_variables)
        self.optimizer_disc.apply_gradients(zip(gradients, train_variables))

        # update metrics
        self.update_metric({**l, **l_disc}, metric_type='loss')
        self.update_metric(pred, y=y, metric_type='acc')
        self.update_metric_single('chg_disc', pred_cd, y=true_cd, metric_type='acc')
        self.update_metric_single('disc', pred_d, y=true_d, metric_type='acc')

        return self.get_metric()

    def test_step(self, batch):
        l = {}
        l_disc = {}
        pred = {}
        vaece_loss, vaece_data = self.forward_loss(batch)
        l['rec'], l['kl_y'], l['kl_x'], l['class'], l['chg_disc'], l['disc_vae'], l_disc['disc'] = vaece_loss
        _, _, _, pred['z_y'], pred['z_x'], y, pred_cd, true_cd, pred_d, true_d = vaece_data

        # update metrics
        self.update_metric({**l, **l_disc}, metric_type='loss')
        self.update_metric(pred, y=y, metric_type='acc')
        self.update_metric_single('chg_disc', pred_cd, y=true_cd, metric_type='acc')
        self.update_metric_single('disc', pred_d, y=true_d, metric_type='acc')

        return self.get_metric()


class LVAE(DVAE):
    """Disentangle individual dimensions in z_y with auxiliary dimension-value classifiers. Built upon DVAE.
    To train we supply images alongside a label for each feature. z_y is optimized to accurately predict these
    labels from the corresponding z_y dimension, while not being able to predict it from other dimensions.

    This disentanglement concept is based on that of label-based disentanglement methods (as referred to in DVAE),
    being a multi-dimensional extension thereof.
    """

    _name_loss = ['rec', 'kl_y', 'kl_x', 'class', 'label']
    _name_acc = ['z_y', 'z_x', 'l', 'l_adv']

    def __init__(self, input_shape, dim_y, dim_x, num_class=10, num_label=8,
                 optimizer=None, **kwargs):
        super(LVAE, self).__init__(input_shape, dim_y, dim_x, num_class, optimizer, **kwargs)
        self.num_label = num_label
        self.classifier_l = [Classifier(1, 2, num_dense=0) for i in range(num_label)]
        self.classifier_l_adv = [Classifier(dim_y - 1, 2) for i in range(num_label)]

        self.set_save_info(args={'input_shape': input_shape, 'dim_y': dim_y, 'dim_x': dim_x, 'num_class': num_class,
                                 'num_label': num_label},
                           models={**{'enc_y': self.encoder_y, 'enc_x': self.encoder_x, 'dec': self.decoder,
                                      'class_y': self.classifier_y, 'class_x': self.classifier_x},
                                   **{'class_l_' + str(i): self.classifier_l[i] for i in range(num_label)},
                                   **{'class_l_aux_' + str(i): self.classifier_l_adv[i] for i in range(num_label)}})

        # initialize variables to easily get the complement dimensions during training
        self.complement = []
        for i in range(self.num_label):
            opposite = list(range(self.num_label))
            opposite.remove(i)
            opposite = tf.constant(opposite)
            self.complement.append(opposite)

    def compile(self, *args, **kwargs):
        super(LVAE, self).compile(*args, **kwargs)
        self.set_train_params(*args, **kwargs)

    def label_loss(self, i, dim_label_y, z_y, complement=False):
        if complement:
            samples_gr = grad_reverse(z_y)  # reverse gradient
            dim = tf.gather(samples_gr, self.complement[i], axis=-1)  # get complement dims
            dim = tf.reshape(dim, (self.batch_size, self.num_label - 1))
            predictions = self.classifier_l_adv[i](dim)
        else:
            dim = tf.gather(z_y, i, axis=-1)  # get correct dim
            dim = tf.reshape(dim, (self.batch_size, 1))
            predictions = self.classifier_l[i](dim)
        l_label = self.w['label'] * self.loss_classify(predictions, dim_label_y[i])
        return l_label, predictions

    def forward_loss(self, batch):
        x, y = batch[0:2]
        dim_label_y = batch[2:]

        dvae_loss, dvae_data = super(LVAE, self).forward_loss((x, y))
        l_rec, l_kl_y, l_kl_x, l_class = dvae_loss
        z_y, z_x, x_rec, class_pred_y, class_pred_x = dvae_data

        # LABEL LOSS #
        l_label = 0
        pred_label = []
        pred_label_adv = []
        for i in range(self.num_label):
            l_label_i, pred_i = self.label_loss(i, dim_label_y, z_y)
            l_label += l_label_i
            pred_label.append(pred_i)
            l_label_adv_i, pred_adv_i = self.label_loss(i, dim_label_y, z_y, complement=True)
            l_label += l_label_adv_i
            pred_label_adv.append(pred_adv_i)
        true_label = tf.concat(dim_label_y, axis=0)
        pred_label = tf.concat(pred_label, axis=0)
        pred_label_adv = tf.concat(pred_label_adv, axis=0)

        return (l_rec, l_kl_y, l_kl_x, l_class, l_label), \
               (z_y, z_x, x_rec, class_pred_y, class_pred_x, true_label, pred_label, pred_label_adv)

    def train_step(self, batch):
        l = {}
        pred = {}
        pred_l = {}
        with tf.GradientTape(persistent=True) as tape:
            lvae_loss, lvae_data = self.forward_loss(batch)
            l['rec'], l['kl_y'], l['kl_x'], l['class'], l['label'] = lvae_loss
            _, _, _, pred['z_y'], pred['z_x'], y_label, pred_l['l'], pred_l['l_adv'] = lvae_data
            y_class = batch[1]
            loss = sum(l.values())
        # train main representation model
        train_variables = [*self.encoder_y.trainable_variables, *self.encoder_x.trainable_variables,
                           *self.decoder.trainable_variables, *self.classifier_y.trainable_variables,
                           *self.classifier_x.trainable_variables]
        for m in self.classifier_l + self.classifier_l_adv:
            train_variables.extend(m.trainable_variables)
        gradients = tape.gradient(loss, train_variables)
        self.optimizer.apply_gradients(zip(gradients, train_variables))

        # update metrics
        self.update_metric(l, metric_type='loss')
        self.update_metric(pred, y=y_class, metric_type='acc')
        self.update_metric(pred_l, y=y_label, metric_type='acc')
        return self.get_metric()

    def test_step(self, batch):
        l = {}
        pred = {}
        pred_l = {}
        lvae_loss, lvae_data = self.forward_loss(batch)
        l['rec'], l['kl_y'], l['kl_x'], l['class'], l['label'] = lvae_loss
        _, _, _, pred['z_y'], pred['z_x'], y_label, pred_l['l'], pred_l['l_adv'] = lvae_data
        y_class = batch[1]

        # update metrics
        self.update_metric(l, metric_type='loss')
        self.update_metric(pred, y=y_class, metric_type='acc')
        self.update_metric(pred_l, y=y_label, metric_type='acc')
        return self.get_metric()


class GVAE(DVAE):
    """Disentangle individual dimensions in z_y by merging them. Built upon DVAE.
    To train we show image pairs with one prespecified feature matching, and merge these dimensions
    in the reconstruction.

    This disentanglement concept is based on that of GVAE:
    "H. Hosoya. Group-based learning of disentangled representations with generalizability for novel contents." -
    https://www.ijcai.org/Proceedings/2019/0348.pdf

    (note that ADA-GVAE implemention is also in this class, its implementation is a mere wrapper around GVAE)
    """

    _name_loss = ['rec', 'kl_y', 'kl_x', 'class']
    _name_weight_extra = ['full']
    _name_acc = ['z_y', 'z_x']

    def __init__(self, input_shape, dim_y, dim_x, num_class=10,
                 optimizer=None, adaptive=False, **kwargs):
        super(GVAE, self).__init__(input_shape, dim_y, dim_x, num_class, optimizer, **kwargs)
        self.adaptive = adaptive
        self.set_save_info(args={'input_shape': input_shape, 'dim_y': dim_y, 'dim_x': dim_x, 'num_class': num_class,
                                 'adaptive': adaptive})

    def compile(self, *args, **kwargs):
        super(GVAE, self).compile(*args, **kwargs)
        self.set_train_params(*args, **kwargs)

    def set_train_params(self, batch_size=None, *args, **kwargs):
        super(GVAE, self).set_train_params(**kwargs)
        if batch_size:
            self.batch_size = batch_size
            indices = tf.constant([range(self.batch_size)], dtype='int32')
            self.indices = tf.reshape(indices, (self.batch_size, 1))

    def merge_dim(self, mu_a, log_sigma_a, mu_b, log_sigma_b, dim_idx):
        """Average a single dimension of a and b, as specified by dim_idx"""
        # construct [1, 1, 0.5] array so we can easily merge two z-spaces
        merge = tf.cast(dim_idx, dtype='int32')
        update_indices = tf.concat([self.indices, merge], axis=-1)  # format indices for scatter_nd_update

        # construct [1, 1, 0.5] array so we can easily merge two z-spaces
        ones = tf.ones_like(mu_a)
        update = tf.ones(self.batch_size) * 0.5
        dif = tf.tensor_scatter_nd_update(ones, update_indices, update)

        # merge two samples we propagate, i.e, a = a * [1, 1, 0.5, 1] + b * [0, 0, 0.5, 0] and vice versa
        mu_a_new = mu_a * dif + mu_b * (1 - dif)
        log_sigma_a_new = log_sigma_a * dif + log_sigma_b * (1 - dif)
        mu_b_new = mu_b * dif + mu_a * (1 - dif)
        log_sigma_b_new = log_sigma_b * dif + log_sigma_a * (1 - dif)

        return mu_a_new, log_sigma_a_new, mu_b_new, log_sigma_b_new

    def merge_heuristically(self, mu_a, log_sigma_a, mu_b, log_sigma_b):
        """Average all but one dimensions of a and b. This remaining dimension is picked heuristically, as the
        dimension with the largets KL-divergence"""
        # create normal distr and take kl divergences
        distr_a = tfp.distributions.Normal(loc=mu_a, scale=tf.math.exp(log_sigma_a))
        distr_b = tfp.distributions.Normal(loc=mu_b, scale=tf.math.exp(log_sigma_b))
        kl = distr_a.kl_divergence(distr_b)

        # get indices of max kl-div, merge remaining
        merge = tf.math.argmax(kl, axis=-1)
        merge = tf.reshape(merge, (self.batch_size, 1))
        merge = tf.cast(merge, dtype='int32')
        update_indices = tf.concat([self.indices, merge], axis=-1)

        # construct [0.5, 0.5, 1] array so we can easily merge two z-spaces
        ones = tf.ones_like(mu_a) * 0.5
        update = tf.ones((self.batch_size))
        dif = tf.tensor_scatter_nd_update(ones, update_indices, update)

        # merge two samples we propagate, i.e, a = a * [1, 0.5, 0.5, 0.5] + b * [0, 0.5, 0.5, 0.5] and vice versa
        mu_a_new = mu_a * dif + mu_b * (1 - dif)
        log_sigma_a_new = log_sigma_a * dif + log_sigma_b * (1 - dif)
        mu_b_new = mu_b * dif + mu_a * (1 - dif)
        log_sigma_b_new = log_sigma_b * dif + log_sigma_a * (1 - dif)

        return mu_a_new, log_sigma_a_new, mu_b_new, log_sigma_b_new

    def forward_loss(self, batch):
        if self.adaptive:
            x_p_a, x_p_b, x, y = batch
        else:
            x_p_a, x_p_b, y_p, x, y = batch

        # regular DVAE loss
        dvae_loss, dvae_data = super(GVAE, self).forward_loss((x, y))
        l_rec, l_kl_y, l_kl_x = [self.w['full'] * l for l in dvae_loss[0:3]]  # ELBO losses are scaled
        l_class = dvae_loss[3]

        z_y, z_x, x_rec, class_pred_y, class_pred_x, y = dvae_data

        # AVERAGE THE SUPPLIED SHARED LATENT DIM AND COMPUTE ELBO (GROUP VAE) #

        # infer like normal
        mu_y_a, log_sigma_y_a = self.encoder_y(x_p_a)
        mu_y_b, log_sigma_y_b = self.encoder_y(x_p_b)

        # merge the group's values
        if self.adaptive:
            mu_y_a, log_sigma_y_a, mu_y_b, log_sigma_y_b = \
                self.merge_heuristically(mu_y_a, log_sigma_y_a, mu_y_b, log_sigma_y_b)
        else:
            mu_y_a, log_sigma_y_a, mu_y_b, log_sigma_y_b = \
                self.merge_dim(mu_y_a, log_sigma_y_a, mu_y_b, log_sigma_y_b, y_p)

        # infer z_xs
        mu_x_a, log_sigma_x_a = self.encoder_x(x_p_a)
        mu_x_b, log_sigma_x_b = self.encoder_x(x_p_b)

        # sample all
        z_y_a = self.sample(mu_y_a, log_sigma_y_a)
        z_x_a = self.sample(mu_x_a, log_sigma_x_a)

        z_y_b = self.sample(mu_y_b, log_sigma_y_b)
        z_x_b = self.sample(mu_x_b, log_sigma_x_b)

        # reconstruct both
        x_rec_a = self.decoder(tf.concat([z_y_a, z_x_a], axis=1))
        x_rec_b = self.decoder(tf.concat([z_y_b, z_x_b], axis=1))

        # compute losses
        l_rec += self.w['rec'] * (self.loss_rec(x_p_a, x_rec_a) + self.loss_rec(x_p_b, x_rec_b))
        l_kl_y += self.w['kl_y'] * (self.loss_kl(mu_y_a, log_sigma_y_a) + self.loss_kl(mu_y_b, log_sigma_y_b))
        l_kl_x += self.w['kl_x'] * (self.loss_kl(mu_x_a, log_sigma_x_a) + self.loss_kl(mu_x_b, log_sigma_x_b))
        return (l_rec, l_kl_y, l_kl_x, l_class), (z_y, z_x, x_rec, class_pred_y, class_pred_x, y)

    """
    Train and test are identical to DVAE    
    """


class ADA_GVAE(GVAE):
    """Disentangle individual dimensions in z_y by heuristically identifying the differing dimension, and merging them
    in reconstruction. Built upon GVAE.
    To train we supply pairs of images with 1 difference (as in VAECE's change discriminator).

    This disentanglement concept is based on that of ADA-GVAE (being a 1-dimension differing version thereof):
    "F. Locatello, B. Poole, G. Rätsch, B. Schölkopf, O. Bachem, and M. Tschannen. Weakly supervised
    disentanglement without compromises." - https://proceedings.mlr.press/v119/locatello20a/locatello20a.pdf
    """
    def __init__(self, input_shape, dim_y, dim_x, num_class=10,
                 optimizer=None, **kwargs):
        if 'adaptive' in kwargs:  # if supplied, make sure we don't pass it along twice
            assert kwargs['adaptive'] is True
            del kwargs['adaptive']
        super(ADA_GVAE, self).__init__(input_shape, dim_y, dim_x, num_class, optimizer=optimizer, adaptive=True, **kwargs)


if __name__ == '__main__':
    dvae = DVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8)
    dvae.compile()
    print(dvae)

