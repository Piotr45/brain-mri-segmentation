"""Dataset utils module."""

import tensorflow as tf

from tensorflow.keras import backend as K

from dataloader import DatasetLoaderGen
from utils.nn import Augment


def prepare_dataset(
    dataset_loader: DatasetLoaderGen,
    batch_size: int = 32,
    split: tuple = (0.70, 0.15, 0.15),
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Function for creating train, valid, test datasets.

    Args:
        batch_size: A int that indicates size of the batch size.
        split: A tuple with percentage size of dataset for train, valid, test.

    Returns:
        Three dataset: train, valid, test.
    """
    dataest_num_samples = dataset_loader.dataset_info["num_samples"]

    dataset = tf.data.Dataset.from_generator(
        dataset_loader,
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 128, 1), dtype=tf.float32),
        ),
    )

    # split dataset info train, valid and test
    train_ds_size, valid_ds_size, test_ds_size = [
        int(percent * dataest_num_samples) for percent in split
    ]

    test_ds = dataset.take(test_ds_size)
    valid_ds = dataset.skip(test_ds_size).take(valid_ds_size)
    train_ds = dataset.skip(test_ds_size + valid_ds_size)

    # transform datasets into batch datasets
    train_batches = (
        train_ds.cache()
        .shuffle(train_ds_size)
        .batch(batch_size)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_batches = test_ds.batch(batch_size)
    valid_batches = valid_ds.batch(batch_size)

    return train_batches, valid_batches, test_batches
