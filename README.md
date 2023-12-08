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
