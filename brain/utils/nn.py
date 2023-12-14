"""This module contains all utils for neural networks."""

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K


class Augment(tf.keras.layers.Layer):
    """Augmenter class.

    Attributes:
        augment_inputs: A function for input augmentation.
        augment_labels: A function for label augmentation.
    """

    def __init__(self, seed: int = 42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def normalize(
    input_image: np.ndarray, input_mask: np.ndarray
) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalizes input image and input mask to [0,1] range and float32 dtype.

    Args:
        input_image: A 3-channel image.
        input_mask: A grayscale image.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: int = 100) -> tf.float32:
    """Function that calculates dice coefficient.

    Args:
        y_true: The truth values e.g. true mask.
        y_pred: The values predicted by network e.g. pred mask.
        smooth: TODO

    Returns:
        Dice coefficient of y_true and y_pred
    """
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return (2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth)


def dice_coef_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.float32:
    """Dice coefficient loss function.

    Args:
        y_true: The truth values e.g. true mask.
        y_pred: The values predicted by network e.g. pred mask.

    Returns:
        Dice coefficient of y_true and y_pred
    """
    return -dice_coef(y_true, y_pred)


def iou(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: int = 100) -> tf.float32:
    """This function implements Intersection over Union.

    Args:
        y_true: The truth values e.g. true mask.
        y_pred: The values predicted by network e.g. pred mask.
        smooth: TODO

    Returns:
        The IoU for y_true and y_pred.
    """
    intersection = K.sum(y_true * y_pred)
    _sum = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (_sum - intersection + smooth)
    return jac
