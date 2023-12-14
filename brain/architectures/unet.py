"""U-Net architecture module.

@author Piotr Baryczkowski (Piotr45)
"""

import tensorflow as tf

from architectures.architecture import Architecture
from utils.nn import dice_coef_loss, dice_coef, iou


class UNetArchitecture(Architecture):
    """U-Net architecture class.

    Attributes:
        model: A model of neural network based on the class architecture.
        input_size: Size of an input image for our Neural Network model.
        n_classes: The number of classes for our classifier.
        history: Training history of our model.
    """

    def __init__(self, input_size: tuple = (256, 256, 3), n_classes: int = 2) -> None:
        """Initializes the instance of U-Net Neural Network Architecture.

        Args:
            input_size: Size of an input image that will be passed to our model.
            n_calsses: Number of classes that our model will predict.
        """
        self.model: tf.keras.Model = None
        self.input_size: tuple = input_size
        self.n_classes: int = n_classes
        self.history: tf.keras.callbacks.History() = None

    def build_model(self, num_blocks: int = 4, n_filters: int = 32) -> tf.keras.Model:
        """Function for building the Neural Network model.

        Args:
            num_blocks: A number of encoder blocks. Number of decoder blocks equals num_blocks - 1.
            n_filters: A quantity of filters inside Conv2D layers.

        Returns:
            A Neural Network model, that will also be stored inside this class object.
        """
        # TODO asserts
        # Create network input
        inputs = tf.keras.layers.Input(shape=self.input_size)
        # Create encoder blocks
        eblocks = self.__create_encoder_blocks(num_blocks, n_filters, inputs)
        # Create decoder blocks
        ublocks = self.__create_decoder_blocks(num_blocks, n_filters, eblocks)
        # Create last conv layer
        last_conv = tf.keras.layers.Conv2D(
            n_filters,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )(ublocks[-1])
        # Create network output layer
        outputs = tf.keras.layers.Conv2D(self.n_classes, 1, activation="sigmoid")(
            last_conv
        )
        # Save model inside architecture object
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=dice_coef_loss,
            metrics=[dice_coef, iou, "binary_accuracy"],
        )
        return self.model

    def train_model(
        self,
        train_batches: tf.data.Dataset,
        valid_batches: tf.data.Dataset,
        validation_steps: int,
        steps_per_epoch: int,
        batch_size: int,
        epochs: int,
    ) -> None:
        """Model training function.

        Args:
            train_batches: Training dataset as Tensorflow batch dataset.
            valid_batches: Validation dataset as Tensorflow batch dataset.
            validation_steps: Int that defines how many batches it wil yield per epoch for validation.
            steps_per_epoch: Integer value that defines how many batches it wil yield per epoch for train data.
            batch_size: Size of batch.
            epochs: The number of epochs for our training session.
        """
        # train model
        self.history = self.model.fit(
            train_batches,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            epochs=epochs,
            validation_steps=validation_steps,
            validation_data=valid_batches,
        )
        return

    def evaluate_model(self, test: tf.data.Dataset) -> dict:
        """Function for evaluating the model.

        Args:
            test: A test dataset.

        Returns:
            Model statistics.
        """
        return self.model.evaluate(test)

    def predict_model(self, data: tf.Tensor) -> tf.Tensor:
        """Function for predicting the model.

        Args:
            data: A data to predict.

        Returns:
            Model predictions.
        """
        return self.model.predict(data)

    def save_model(self, filepath: str) -> None:
        """Function that saves our model.

        Args:
            filepath: Path where to save the model.
        """
        self.model.save(filepath=filepath, save_format="tf")

    @staticmethod
    def create_encoder_block(
        inputs: tf.keras.layers.Input,
        n_filters: int = 32,
        dropout: float = 0.3,
        max_pooling: bool = True,
    ) -> tf.keras.layers.Layer:
        """Function for creating single encoder block.

        Args:
            inputs: Previous layer that will be linked with our encoder block.
            n_filters: Number of filters in single Conv2D layer.
            dropout: A probability of how many layers will be dropped.
            max_pooling: A boolean indicating whether we want to use Max Pooling or not.

        Returns:
            Single mini encoder block.
        """
        conv = tf.keras.layers.Conv2D(n_filters, (3, 3), padding="same")(inputs)
        activation = tf.keras.layers.Activation("relu")(conv)
        conv = tf.keras.layers.Conv2D(n_filters, (3, 3), padding="same")(activation)
        batch_norm = tf.keras.layers.BatchNormalization(axis=3)(conv)
        next_layer = tf.keras.layers.Activation("relu")(batch_norm)

        if dropout > 0:
            next_layer = tf.keras.layers.Dropout(dropout)(batch_norm)
        if max_pooling:
            next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(next_layer)
        else:
            next_layer = conv
        skip_connection = conv
        return next_layer, skip_connection

    @staticmethod
    def create_decoder_block(
        prev_layer_input: tf.keras.layers.Layer,
        skip_layer_input: tf.keras.layers.Layer,
        n_filters: int = 32,
    ) -> tf.keras.layers.Layer:
        """Function for creating single decoder block.

        Args:
            prev_layer_input: Previous layer that will be linked with our decoder block.
            skip_layer_input: Skip layer that will be linked with our mini decoder block.
            n_filters: Number of filters in single Conv2D layer.

        Returns:
            Single mini decoder block.
        """
        upscale = tf.keras.layers.concatenate(
            [
                tf.keras.layers.Conv2DTranspose(
                    n_filters, (2, 2), strides=(2, 2), padding="same"
                )(prev_layer_input),
                skip_layer_input,
            ],
            axis=3,
        )
        conv = tf.keras.layers.Conv2D(n_filters, (3, 3), padding="same")(upscale)
        activation = tf.keras.layers.Activation("relu")(conv)
        conv = tf.keras.layers.Conv2D(n_filters, (3, 3), padding="same")(activation)
        batch_norm = tf.keras.layers.BatchNormalization(axis=3)(conv)
        activation = tf.keras.layers.Activation("relu")(batch_norm)

        return activation

    def __create_encoder_blocks(
        self, num_blocks: int, n_filters: int, inputs: tf.keras.layers.Input
    ) -> list:
        """Helper function for creating multiple encoding blocks.

        Args:
            num_blocks: The number of mini encoding blocks that will be created.
            n_filters: A number of filters in single Conv2D layer.
            inputs: Input layer for the first encoding block.

        Returns:
            List of mini encoding blocks.
        """

        eblocks = [None for _ in range(num_blocks)]
        for block_num in range(num_blocks):
            if block_num == 0:
                eblocks[block_num] = self.create_encoder_block(
                    inputs, n_filters, dropout=0, max_pooling=True
                )
                continue
            eblocks[block_num] = self.create_encoder_block(
                eblocks[block_num - 1][0],
                n_filters * 2**block_num,
                dropout=0,
                max_pooling=False
                if block_num == num_blocks - 1
                else True,  # Disable pooling on the last layer
            )
        return eblocks

    def __create_decoder_blocks(
        self, num_blocks: int, n_filters: int, eblocks: list
    ) -> list:
        """This function creates decoder block structure.

        Args:
            num_blocks: A number of encoder blocks.
            n_filrers: Number of filters in single Conv2D layer.
            eblocks: List of encoding blocks that will be linked with decoders.

        Returns:
            Last layer of the decoding chain.
        """
        ublocks = [None for _ in range(num_blocks - 1)]
        for block_num in range(num_blocks - 1):
            if block_num == 0:
                ublocks[block_num] = self.create_decoder_block(
                    eblocks[-1][0],
                    eblocks[-2][1],
                    n_filters * 2 ** (num_blocks - block_num - 2),
                )
                continue
            ublocks[block_num] = self.create_decoder_block(
                ublocks[block_num - 1],
                eblocks[num_blocks - block_num - 2][1],
                n_filters * 2 ** (num_blocks - block_num - 2),
            )
        return ublocks
