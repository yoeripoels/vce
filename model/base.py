"""Definition of abstract base model classes.

REPR defines an abstract representation model, along with helper functions.
CD defines an abstract change-discriminator model.
"""

from abc import abstractmethod, ABCMeta
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle


class REPR(keras.Model, metaclass=ABCMeta):
    """Base class for the representation model.

    We assume the same process for both, that is, two encoders (for 'class-related' and 'class-unrelated'
    information, that is, 'encoder_y' and 'encoder_x') and one decoder (for the concatenation of [z_y, z_x]).

    We specify functions to encode & decode data, where we expect a value for each dimension (i.e., a sample, whether
    this is implemented as, for example, mu_x or z_x ~ N(mu_x, sigma_x) is left up to the implementation).

    Training is delegated to the model implementation, using train_on_batch()
    """

    def __init__(self):
        super(REPR, self).__init__()
        self.w = {name: 1 for name in self._name_loss + self._name_weight_extra}  # initialize loss weights

    @property
    @abstractmethod
    def _name_loss(self):
        """A list with the names of each loss (to keep track of metrics)"""
        pass

    @property
    @abstractmethod
    def _name_acc(self):
        """A list with the names of each accuracy metric"""
        pass

    @property
    @abstractmethod
    def _name_weight_extra(self):
        """A list with the names of extra hyperparameters, that are not loss weights"""
        return []

    @abstractmethod
    def set_train_params(self, *args):
        """Sets the training parameters (loss weights, optimizer(s), device)"""
        pass

    @abstractmethod
    def train_step(self, batch):
        """Training on a single batch of data"""
        pass

    @abstractmethod
    def test_step(self, batch):
        """Testing on a single batch of data"""
        pass

    @abstractmethod
    def encode_y(self, batch, mean=True):
        """Encode a batch to the latent representation (sampled) of the class-specific dimensions"""
        pass

    @abstractmethod
    def encode_x(self, batch, mean=True):
        """Encode a batch to the latent representation (sampled) of the class-unrelated dimensions"""
        pass

    @abstractmethod
    def decode(self, z_y, z_x):
        """Decode a batch of latent representations to the original data"""
        pass

    '''
    Metric/settings related methods
    '''
    def compile(self, batch_size=None, *args, **kwargs):
        """On .compile(), set metrics according to loss and accuracy names"""
        super(REPR, self).compile(*args, **kwargs)
        self._metric_loss = {name: keras.metrics.Mean() for name in self._name_loss}
        self._metric_acc = {name: keras.metrics.CategoricalAccuracy() for name in self._name_acc}
        if batch_size:
            self.batch_size = batch_size

    def parse_weights(self, **kwargs):
        """Set weight values according to loss names"""
        for name, value in kwargs.items():
            if name.startswith('w_'):
                w_name = name[2:]
                if w_name in self._name_loss + self._name_weight_extra:
                    self.w[w_name] = value
                else:
                    print('WARNING: Supplied weight ({}: {}) for loss that does not exist!'.format(name, value))

    def update_metric(self, update_dict, metric_type='loss', y=None):
        """Updates the metrics according to the supplied dict. Metric type is either 'loss' or 'acc'"""
        for name, value in update_dict.items():
            if metric_type == 'loss':
                self._metric_loss[name].update_state(value)
            elif metric_type == 'acc':
                self._metric_acc[name].update_state(value, y)

    def update_metric_single(self, name, value, metric_type='loss', y=None):
        """Updates a single metric. Metric type is either 'loss' or 'acc'"""
        if metric_type == 'loss':
            self._metric_loss[name].update_state(value)
        elif metric_type == 'acc':
            self._metric_acc[name].update_state(value, y)

    def get_metric(self, loss=None, pred=None, y=None):
        """Get dict of metric results. If losses (l) or predictions (pred) are supplied, update metrics first"""
        if loss is not None:
            self.update_metric(loss, metric_type='loss')
        if pred is not None and y is not None:
            self.update_metric(pred, metric_type='acc', y=y)
        return {name: metric.result() for name, metric in
                [('loss_' + n, v) for n, v in self._metric_loss.items()] +
                [('acc_' + n, v) for n, v in self._metric_acc.items()]}

    def set_save_info(self, args=None, models=None):
        """Sets the (init) arguments and models that are saved/loaded
        Should be called in init()"""
        if args:
            self._save_args = args
        if models:
            self._save_models = models

    def save(self, save_name):
        """Save model weights, hyperparameters and arguments to disk"""
        for name, model in self._save_models.items():
            model.save_weights(save_name + '-' + name + '.h5')
        with open(save_name + '-settings.pkl', 'wb') as f:
            pickle.dump(self._save_args, f)
        with open(save_name + '-hyperparams.pkl', 'wb') as f:
            pickle.dump(self.w, f)

    def load(self, load_name):
        """Load model weights and hyperparameters from disk"""
        for name, model in self._save_models.items():
            model.load_weights(load_name + '-' + name + '.h5')
        with open(load_name + '-settings.pkl', 'rb') as f:
            args = pickle.load(f)
            assert self._save_args == args  # these should not deviate
        with open(load_name + '-hyperparams.pkl', 'rb') as f:
            w = pickle.load(f)
            self.w = w

    @classmethod
    def from_disk(cls, load_name):
        """Create model from existing model/configuration from disk"""
        with open(load_name + '-settings.pkl', 'rb') as f:
            args = pickle.load(f)
        model = cls(**args)
        model.load(load_name)
        return model

    '''
    Training-related methods
    '''
    @staticmethod
    def loss_kl(mu, log_sigma):
        kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(2 * log_sigma) - 2 * log_sigma - 1, axis=1)
        return tf.reduce_mean(kl_loss)

    @staticmethod
    def loss_rec(x, x_rec):
        re_loss = tf.reduce_sum(tf.square(K.batch_flatten(x) - K.batch_flatten(x_rec)), axis=1)
        return tf.reduce_mean(re_loss)

    @staticmethod
    def loss_classify(y, y_true):
        CCE = K.categorical_crossentropy(y, y_true)
        return tf.reduce_mean(CCE)

    @staticmethod
    def sample(mu, log_sigma):
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + K.exp(log_sigma) * epsilon
    '''
    SHARED EVALUATION METHODS HERE, using specified functions from above
    '''


class CD(keras.Model, metaclass=ABCMeta):
    """Base class for the change discriminator model (as used in VAE-CE).

    Assume a static model, i.e., it is pre-trained and not changed. All we require is a function to evaluate
    the quality of change-pairs between two batches.
    """

    def __init__(self):
        super(CD, self).__init__()

    @abstractmethod
    def discriminate(self, batch_a, batch_b):
        """Discriminate between two batches of datapoints, i.e., the averaged 'quality' of batch_a_i <-> batch_b_i,
         where i covers all samples in the batch"""
        pass
