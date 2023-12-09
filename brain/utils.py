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


def normalize(input_image: np.ndarray, input_mask: np.ndarray) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalizes input image and input mask to [0,1] range and float32 dtype.

    Args:
        input_image: A 3-channel image.
        input_mask: A grayscale image.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask


def create_prediction_mask(prediction: tf.Tensor) -> np.ndarray:
    """Creates a mask from a prediction.

    Args:
        prediction: A prediction from a model.
    """
    prediction_mask = np.argmax(prediction, axis=-1)
    prediction_mask = prediction_mask[..., tf.newaxis]
    return prediction_mask


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


def display_prediction(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    win_name: str = "prediction_demo",
) -> None:
    """Display images side-by-side.

    Args:
        image: A 3-channel image.
        mask: A grayscale image.
        win_name: Name of the window.
    """
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # prediction_3_channel = cv2.cvtColor(create_prediction_mask(prediction), cv2.COLOR_GRAY2BGR)

    cv2.imshow(win_name, np.hstack((mask, create_prediction_mask(prediction))))
    cv2.waitKey(0)
    return
