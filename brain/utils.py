"""Utils module.

Module contains functions for displaying images, augmenting the data etc.

@author: Piotr Baryczkowski (Piotr45)
@author: Paweł Strzelczyk (pawelstrzelczyk)
"""

import cv2
import numpy as np
import tensorflow as tf


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def display(image: np.ndarray, mask: np.ndarray, win_name: str = "demo") -> None:
    """Display images side-by-side.

    Args:
        image: A 3-channel image.
        mask: A grayscale image.
        win_name: Name of the window.
    """
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cv2.imshow(win_name, np.hstack((image, mask_3_channel)))
    cv2.waitKey(0)
    return
