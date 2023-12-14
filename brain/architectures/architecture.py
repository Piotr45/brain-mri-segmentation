"""Abstract architecture module."""

import os

import tensorflow as tf

from tensorflow.keras.utils import plot_model


class Architecture:
    """Abstract architecture class.

    Attributes:
        model: A model of neural network based on the class architecture.
    """

    def __init__(self) -> None:
        """Initializes the instance of Neural Network Architecture."""
        self.model = None

    def build_model(self) -> tf.keras.Model:
        raise NotImplementedError

    def train_model(self) -> None:
        raise NotImplementedError

    def evaluate_model(self) -> None:
        raise NotImplementedError

    def predict_model(self, data: tf.Tensor) -> None:
        raise NotImplementedError

    def save_model(self) -> None:
        raise NotImplementedError

    def plot_model(self, filename: str = "model_plot", path: str = ".") -> None:
        """Plotting model architecture.

        Args:
            filename: A name for output file e.g. model_plot.png
            path: A path where picture should be saved.
        """
        plot_model(
            self.model,
            to_file=os.path.join(
                path, f"{filename}{'.png' if '.png' not in filename else ''}"
            ),
            show_shapes=True,
            show_layer_names=True,
        )
