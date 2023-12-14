"""Utils module.

Module contains functions for displaying images, augmenting the data etc.

@author: Piotr Baryczkowski (Piotr45)
@author: Paweł Strzelczyk (pawelstrzelczyk)
"""

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K





def create_prediction_mask(prediction: tf.Tensor) -> np.ndarray:
    """Creates a mask from a prediction.

    Args:
        prediction: A prediction from a model.
    """
    # prediction_mask = np.argmax(prediction, axis=-1)
    prediction_mask = prediction[prediction > 0.5]
    prediction_mask = prediction_mask[..., tf.newaxis]
    return prediction_mask






