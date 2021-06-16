"""Augmentation of MNIST.

We specify a class that augments MNIST such that we can split images into (some approximation of) lines, which we
assume to be the underlying type of feature that determines the digit.

This augmentation class can be used to generate inputs necessary for methods that assume some sort of
'feature-difference' as supervision (VAE-CE and ADA-GVAE), as we can create such image-groups using these split digits.
"""

class MNISTAugmentor:
    pass

