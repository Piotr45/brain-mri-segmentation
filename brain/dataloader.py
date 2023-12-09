"""Module for loading dataset from kaggle.

@author: Piotr Baryczkowski (Piotr45)
"""

import os
import glob

import cv2
import kaggle
import numpy as np

from PIL import Image

from utils import display, normalize


DATASET_PATH = os.path.join(
    os.path.relpath(os.path.join(os.path.dirname(__file__), os.pardir)), "dataset"
)


class DatasetLoaderGen:
    """DatasetLoaderGenerator class.

    Instance of this class can download and unzip dataset from Kaggle.
    It is also a generator which returns data as tuple of images.

    Attributes:
        path: A string that contains path where dataset will be located.
        downloand: A boolean indicating if we want to download dataset with kaggle API or not.
    """

    def __init__(self, path: str = DATASET_PATH, download: bool = True) -> None:
        """Initializes the instance of DatasetLoader.

        Args:
            path: Defines path to dataset.
            download: Defines if we want to download dataset via kaggle API.
        """
        self.path = path

        if download:
            kaggle.api.authenticate()

            kaggle.api.dataset_download_files(
                "mateuszbuda/lgg-mri-segmentation",
                path=self.path,
                quiet=False,
                unzip=True,
            )

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """Walks through dataset dirs and returns data.

        Returns:
            Tuple of images, original image and mask.
        """
        source_dirs = [
            source_dir
            for source_dir in glob.glob(
                os.path.join(self.path, "kaggle_3m", "*"), recursive=True
            )
            if os.path.isdir(source_dir)
        ]
        for source_dir in source_dirs:
            mask_source_files = glob.glob(os.path.join(source_dir, "*_mask.tif"))
            for mask_source in mask_source_files:
                # yield original image and mask
                # TODO find prettier way to do this
                image = np.array(Image.open(mask_source.replace("_mask", "")))
                mask = np.array(Image.open(mask_source))
                resized_image = cv2.resize(
                    image, (128, 128), interpolation=cv2.INTER_AREA
                )
                resized_mask = cv2.resize(
                    mask, (128, 128), interpolation=cv2.INTER_AREA
                )
                normalized_image, normalized_mask = normalize(
                    resized_image, resized_mask
                )
                yield normalized_image, np.expand_dims(normalized_mask, axis=2)


if __name__ == "__main__":
    dataset_loader = DatasetLoaderGen(download=False)
    image, mask = next(dataset_loader())
    display(image, mask)
