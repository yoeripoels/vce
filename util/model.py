"""Extra model-related classes/functions.
"""
import tensorflow as tf


# gradient reversal layer
@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)

    def custom_grad(dy):
        return -dy

    return y, custom_grad