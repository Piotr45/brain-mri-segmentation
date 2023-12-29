# Brain MRI segmentation

Segmentation of brain tumor from MRI images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

All prerequisities are listed in `requirements.txt`.
You should also install [Graphviz](https://graphviz.gitlab.io/download/).

### Installing

A step by step series of examples that tell you how to get a development env running.

Create and install venv with all packages listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train model with `train.py` script or look at `brain.ipynb`.

```bash
usage: train.py [-h] [--dataset-dir DATASET_DIR] [--model-output MODEL_OUTPUT] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--num-blocks NUM_BLOCKS] [--filters FILTERS] [--split SPLIT [SPLIT ...]]
                [--resize-shape RESIZE_SHAPE [RESIZE_SHAPE ...]] [--download]

options:
  -h, --help            show this help message and exit
  --dataset-dir DATASET_DIR
                        Directory with our dataset. (default: ./dataset)
  --model-output MODEL_OUTPUT
                        Path where to save the model. (default: ../models/test_model)
  --epochs EPOCHS       Number of epochs for our trainig session. (default: 10)
  --batch-size BATCH_SIZE
                        The batch size that will be applied to dataset. (default: 16)
  --num-blocks NUM_BLOCKS
                        The number of encoder blocks insied U-Net architecture. (default: 4)
  --filters FILTERS     Start value of filters that will be applied for U-Net architecture. Number of filters is doubled in each encoder block. (default: 32)
  --split SPLIT [SPLIT ...]
                        Information about how to split dataset into train, valid and test. (default: (0.7, 0.15, 0.15))
  --resize-shape RESIZE_SHAPE [RESIZE_SHAPE ...]
                        Information about shape to which image data should be resized. (default: (128, 128))
  --download            Whether to download dataset or not. (default: False)
```

```bash
cd brain
python train.py
```

## Authors

* **Piotr Baryczkowski** - *Initial work, UNet implementation* - [Piotr45](https://github.com/Piotr45)
* **Paweł Strzelczyk** - *Initial work* - [pawelstrzelczyk](https://github.com/pawelstrzelczyk)

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Hat tip to authors of [dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
