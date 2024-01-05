"""Module for input output operations."""

import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def plot_prediction(
    images: np.ndarray | list[np.ndarray],
    masks: np.ndarray | list[np.ndarray],
    predictions: np.ndarray | list[np.ndarray],
    plot_title: str = "Model prediction",
    n_examples: int = 3,
) -> None:
    """Display images side-by-side.

    Args:
        image: A 3-channel image.
        mask: A grayscale image.
        plot_title: Title of the plot.
    """

    fig = plt.figure(constrained_layout=True, figsize=(14, 14))
    fig.suptitle(plot_title, ha="center", fontsize=24, va="top")
    subfigs = fig.subfigures(n_examples, 1, wspace=0.05, hspace=0.05)
    for i in range(n_examples):
        image = images[i].numpy()
        mask_3_channel = cv2.cvtColor(masks[i].numpy(), cv2.COLOR_GRAY2BGR)
        prediction_3_channel = cv2.cvtColor(predictions[i], cv2.COLOR_GRAY2BGR)
        subfig = subfigs[i]
        subfig.suptitle(f"Example {i+1}", ha="center", fontsize=16, va="top")
        axes = subfig.subplots(1, 3)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0].imshow(image)
        axes[0].set_title("Image")

        axes[1].imshow(mask_3_channel)
        axes[1].set_title("Mask")

        axes[2].imshow(prediction_3_channel)
        axes[2].set_title("Prediction")
    return
