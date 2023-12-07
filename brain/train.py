"""Script for training neural networks.

With this script you can use brain package for training neural networks, 
that specialize in brain tumor segmentation.

@author: Piotr Baryczkowski (Piotr45)
@author: Paweł Strzelczyk (pawelstrzelczyk)
"""

import argparse

import tensorflow as tf

from dataloader import DatasetLoaderGen
from architectures.unet import UNetArchitecture
from utils import Augment

DATASET_LENGTH = (
    3929  # it is a rough value from kaggle, TODO obtain this information from code.
)


def prepare_dataset(batch_size: int = 32) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset_loader = DatasetLoaderGen(download=False)

    dataset = tf.data.Dataset.from_generator(
        dataset_loader,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.uint8),
        ),
    )
    test_ds_size = int(DATASET_LENGTH * 0.3)
    train_ds_size = DATASET_LENGTH - test_ds_size

    test_ds = dataset.take(test_ds_size)
    train_ds = dataset.skip(test_ds_size)

    train_batches = (
        train_ds.cache()
        .shuffle(train_ds_size)
        .batch(batch_size)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_batches = test_ds.batch(batch_size)

    return train_batches, test_batches


def main() -> None:
    batch_size = 32
    steps_per_epoch = (DATASET_LENGTH - int(DATASET_LENGTH * 0.3)) // batch_size
    train, test = prepare_dataset()

    unet = UNetArchitecture()
    unet.build_model()
    unet.plot_model()

    unet.train_model(train, steps_per_epoch, batch_size, 10)
    model_stats = unet.evaluate_model(test)
    print(model_stats, type(model_stats))
    return


if __name__ == "__main__":
    main()
