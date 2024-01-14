"""Script for training neural networks.

With this script you can use brain package for training neural networks, 
that specialize in brain tumor segmentation.

@author: Piotr Baryczkowski (Piotr45)
@author: Paweł Strzelczyk (pawelstrzelczyk)
"""

import argparse
import sys
import os

import tensorflow as tf

from dataloader import DatasetLoaderGen
from architectures.unet import UNetArchitecture
from utils.dataset import prepare_dataset
from utils.io import display_prediction, save_model_info, save_model_training_plots


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    """The code will parse arguments from std input."""
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "--dataset-dir",
        type=str,
        action="store",
        required=False,
        default="./dataset",
        help="Directory with our dataset.",
    )

    arg_parser.add_argument(
        "--model-output",
        type=str,
        action="store",
        required=False,
        default="../models/test_model",
        help="Path where to save the model.",
    )

    arg_parser.add_argument(
        "--epochs",
        type=int,
        action="store",
        required=False,
        default=10,
        help="Number of epochs for our trainig session.",
    )

    arg_parser.add_argument(
        "--batch-size",
        type=int,
        action="store",
        required=False,
        default=16,
        help="The batch size that will be applied to dataset.",
    )

    arg_parser.add_argument(
        "--num-blocks",
        type=int,
        action="store",
        required=False,
        default=4,
        help="The number of encoder blocks insied U-Net architecture.",
    )

    arg_parser.add_argument(
        "--filters",
        type=int,
        action="store",
        required=False,
        default=32,
        help="Start value of filters that will be applied for U-Net architecture. Number of filters is doubled in each encoder block.",
    )

    arg_parser.add_argument(
        "--split",
        nargs="+",
        type=float,
        required=False,
        default=(0.7, 0.15, 0.15),
        help="Information about how to split dataset into train, valid and test.",
    )

    arg_parser.add_argument(
        "--resize-shape",
        nargs="+",
        type=int,
        required=False,
        default=(256, 256),
        help="Information about shape to which image data should be resized.",
    )

    arg_parser.add_argument(
        "--download", action="store_true", help="Whether to download dataset or not."
    )

    return arg_parser.parse_args(argv)


def main() -> None:
    # parse arguments
    args = parse_arguments(sys.argv[1:])
    batch_size = args.batch_size

    assert len(args.split) == 3, "You have to pass three arguments for --split"
    assert sum(args.split) == 1.0, "Your arguments should sum to 1"
    assert (
        len(args.resize_shape) == 2
    ), "You should pass two arguments for resize shape e.g. 128 128"

    if not os.path.exists(args.model_output):
        os.makedirs(args.model_output)

    # handle the data
    dataset_loader = DatasetLoaderGen(
        download=args.download,
        resize_shape=(None if args.resize_shape == (256, 256) else args.resize_shape),
    )
    data_shape = (args.resize_shape[0], args.resize_shape[1], 3)
    mask_shape = (args.resize_shape[0], args.resize_shape[1], 1)
    dataest_num_samples = dataset_loader.dataset_info["num_samples"]

    steps_per_epoch = (
        dataest_num_samples - int(dataest_num_samples * args.split[0])
    ) // batch_size
    train, valid, test = prepare_dataset(
        dataset_loader, batch_size, args.split, data_shape, mask_shape
    )

    # create model
    input_size = (args.resize_shape[0], args.resize_shape[1], 3)
    unet = UNetArchitecture(input_size=input_size, n_classes=1)
    unet.build_model(args.num_blocks, args.filters)
    unet.plot_model(filename="model_architecture", path=args.model_output)
    save_model_info(
        os.path.join(args.model_output, "info.txt"),
        args.batch_size,
        args.epochs,
        args.split,
        args.filters,
        args.num_blocks,
    )

    # train model
    unet.train_model(
        train_batches=train,
        valid_batches=valid,
        validation_steps=int(dataest_num_samples * args.split[1]) // batch_size,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        epochs=args.epochs,
    )

    # evaluation process
    dice_loss, dice_coef, iou, binary_acc = unet.evaluate_model(test)
    unet.save_model(args.model_output)
    with open(
        os.path.join(args.model_output, "evaluation.txt"), "w", encoding="utf-8"
    ) as eval_file:
        eval_file.write(
            f"Dice loss: {dice_loss}\nDice coef: {dice_coef}\nIOU: {iou}\nBinary accuracy: {binary_acc}"
        )

    save_model_training_plots(args.model_output, unet.history)

    # display results from 1 batch
    # for images, masks in test.take(1):
    #     prediction = unet.model.predict_on_batch(images)
    #     for i in range(batch_size):
    #         display_prediction(images[i].numpy(), masks[i].numpy(), prediction[i])
    return


if __name__ == "__main__":
    main()
