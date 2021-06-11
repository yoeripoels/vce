"""Definition of abstract base model classes.

REPR defines an abstract representation model, along with helper functions.
CD defines an abstract change-discriminator model.
"""

from abc import abstractmethod, ABCMeta
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K


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
    def encode_y(self, batch):
        """Encode a batch to the latent representation (sampled) of the class-specific dimensions"""
        pass

    @abstractmethod
    def encode_x(self, batch):
        """Encode a batch to the latent representation (sampled) of the class-unrelated dimensions"""
        pass

    @abstractmethod
    def decode(self, z_y, z_x):
        """Decode a batch of latent representations to the original data"""
        pass

    '''
    SHARED TRAINING METHODS / LOSSES HERE, so we don't have to repeat them in implementations.
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
