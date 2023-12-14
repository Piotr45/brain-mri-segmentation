"""Module for input output operations."""

import cv2
import numpy as np


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
    prediction_3_channel = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

    cv2.imshow(win_name, np.hstack((image, mask_3_channel, prediction_3_channel)))
    cv2.waitKey(0)
    return
