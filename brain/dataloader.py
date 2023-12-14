"""Module for loading dataset from kaggle.

@author: Piotr Baryczkowski (Piotr45)
"""

import os
import glob

import cv2
import kaggle
import numpy as np

from PIL import Image

from utils.io import display
from utils.nn import normalize


DATASET_PATH = os.path.join(
    os.path.relpath(os.path.join(os.path.dirname(__file__), os.pardir)), "dataset"
)


class DatasetLoaderGen:
    """DatasetLoaderGenerator class.

    Instance of this class can download and unzip dataset from Kaggle.
    It is also a generator which returns data as tuple of images.

    Attributes:
        path: A string that contains path where dataset will be located.
        resize_shape: A tuple that defines output image shape, None if you want the original.
        dataset_info: Dict with basic information about dataset e.g. number of samples, and path to them.
    """

    def __init__(
        self,
        path: str = DATASET_PATH,
        resize_shape: tuple | None = None,
        download: bool = True,
    ) -> None:
        """Initializes the instance of DatasetLoader.

        Args:
            path: Defines path to dataset.
            resize_shape: If you want to resize output image, pass shape e.g. (128, 128).
            downloand: A boolean indicating if we want to download dataset with kaggle API or not.
        """
        self.path: str = path
        self.resize_shape: tuple | None = resize_shape

        if download:
            kaggle.api.authenticate()

            kaggle.api.dataset_download_files(
                "mateuszbuda/lgg-mri-segmentation",
                path=self.path,
                quiet=False,
                unzip=True,
            )

        self.dataset_info = self.__get_dataset_info()

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """Walks through dataset dirs and returns data.

        Returns:
            Tuple of images, original image and mask.
        """
        source_dirs = self.dataset_info["source_dirs"]
        for source_dir in source_dirs:
            mask_source_files = glob.glob(os.path.join(source_dir, "*_mask.tif"))
            for mask_source in mask_source_files:
                # yield original image and mask
                # TODO find prettier way to do this
                image = np.array(Image.open(mask_source.replace("_mask", "")))
                mask = np.array(Image.open(mask_source))
                # resize the data
                if self.resize_shape is not None:
                    image, mask = self.__resize_data(image, mask)
                # normalize the data
                normalized_image, normalized_mask = normalize(image, mask)
                yield normalized_image, np.expand_dims(normalized_mask, axis=2)

    def __get_dataset_info(self) -> dict:
        """Get information about dataset.

        Returns:
            Basic information about dataset: source directories, number of samples in dataset.
        """
        source_dirs = [
            source_dir
            for source_dir in glob.glob(
                os.path.join(self.path, "kaggle_3m", "*"), recursive=True
            )
            if os.path.isdir(source_dir)
        ]
        num_samples = sum(
            [
                len(glob.glob(os.path.join(source_dir, "*_mask.tif")))
                for source_dir in source_dirs
            ]
        )
        return {"source_dirs": source_dirs, "num_samples": num_samples}

    def __resize_data(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Function for resizing data.

        Args:
            image: Image that we want to resize.
            mask: Mask associated with the image.

        Returns:
            Resized data with resize_shape attribute.
        """
        return cv2.resize(
            image, self.resize_shape, interpolation=cv2.INTER_AREA
        ), cv2.resize(mask, self.resize_shape, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    dataset_loader = DatasetLoaderGen(download=False)
    image, mask = next(dataset_loader())
    display(image, mask)
