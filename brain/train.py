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
from utils import Augment, display_prediction

DATASET_LENGTH = (
    3929  # it is a rough value from kaggle, TODO obtain this information from code.
)


def prepare_dataset(batch_size: int = 32) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset_loader = DatasetLoaderGen(download=False)

    dataset = tf.data.Dataset.from_generator(
        dataset_loader,
        output_signature=(
            tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 128, 1), dtype=tf.float32),
        ),
    )
    test_ds_size = int(DATASET_LENGTH * 0.15)
    train_ds_size = DATASET_LENGTH - test_ds_size

    test_ds = dataset.take(test_ds_size)
    valid_ds = dataset.skip(test_ds_size).take(test_ds_size)
    train_ds = dataset.skip(test_ds_size * 2)

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


def main() -> None:
    batch_size = 16
    steps_per_epoch = (DATASET_LENGTH - int(DATASET_LENGTH * 0.3)) // batch_size
    train, valid, test = prepare_dataset()

    unet = UNetArchitecture(input_size=(128, 128, 3), n_classes=1)
    unet.build_model()
    unet.plot_model()

    print(train.element_spec)
    print(valid.element_spec)

    unet.train_model(
        train,
        valid,
        int(DATASET_LENGTH * 0.15) // batch_size,
        steps_per_epoch,
        batch_size,
        10,
    )
    model_stats = unet.evaluate_model(test)
    # for images, masks in test.take(1):
    #     prediction = unet.model.predict_on_batch(images)
    #     for i in range(batch_size):
    #         print(images[i].shape, masks[i].shape)
    #         display_prediction(images[i].numpy(), masks[i].numpy(), prediction[i])
    print(model_stats, type(model_stats))
    return


if __name__ == "__main__":
    main()
