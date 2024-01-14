"""Module for input output operations."""

import os

import cv2
import numpy as np
import tensorflow as tf

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


def save_model_info(
    filename: str,
    batch_size: int,
    epochs: int,
    split: tuple,
    filters: int,
    num_blocks: int,
) -> None:
    """This function writes info about model training to txt file.

    Args:
        filename: A path to file where we want to save info about training.
        batch_size: size of the training batch.
        epochs: A number of epochs.
        split: Info about dataset split (train, valid, test) in %.
        filters: Initial number of filters.
        num_blocks: A number of encoder / decoder blocks.
    """
    with open(filename, "w", encoding="utf-8") as info:
        info.write(
            f"""BATCH SIZE: {batch_size}
EPOCHS: {epochs}
SPLIT: {split}
FILTERS: {filters}
NUM_BLOCKS: {num_blocks}"""
        )


def save_model_training_plots(
    path: str,
    history: tf.keras.callbacks.History(),
) -> None:
    def plot_stats(
        train: list,
        val: list,
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> None:
        """Creates a figure that compares train and validation.

        Args:
            train: Train data that will be compared.
            val: Validation data for statistic comparison.
            title: Plot title.
            xlabel: Label of the X axis.
            ylabel: Label of the Y axis.

        Returns:
            Desired figure.
        """
        fig = plt.figure()
        plt.plot(train)
        plt.plot(val)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(["train", "val"], loc="upper left")
        return fig

    """This function creates and saves plots about training process.

    Args:
        path: Path to the location where we want to save figures.
        history: Model training history.
    """
    # Plot model loss
    plot_stats(
        history.history["loss"],
        history.history["val_loss"],
        "Loss over epochs",
        "epochs",
        "loss",
    ).savefig(os.path.join(path, "loss.png"))
    # Plot model dice coef
    plot_stats(
        history.history["dice_coef"],
        history.history["val_dice_coef"],
        "Dice coef over epochs",
        "epochs",
        "dice coef",
    ).savefig(os.path.join(path, "dice_coef.png"))
    # Plot model iou
    plot_stats(
        history.history["iou"],
        history.history["val_iou"],
        "IOU over epochs",
        "epochs",
        "iou",
    ).savefig(os.path.join(path, "iou.png"))
    # Plot model binary accuracy
    plot_stats(
        history.history["binary_accuracy"],
        history.history["val_binary_accuracy"],
        "Binary accuracy over epochs",
        "epochs",
        "binary accuracy",
    ).savefig(os.path.join(path, "binary_accuracy.png"))
    return
