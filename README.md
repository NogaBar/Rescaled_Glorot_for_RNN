# Rescaled Glorot for RNN

This repository includes implementation for linear RNN and linear diagonal RNN and is based on the on the [minimal LRU unofficial github repository](https://github.com/NicolasZucchet/minimal-LRU) of the LRU.



## Requirements & Installation

To run the code on your own machine, run `pip install -r requirements.txt`. The GPU installation of
JAX can be tricky; further instructions are available on how to install it
[here](https://github.com/google/jax#installation). PyTorch and torchtext also needs to be installed separately
because of interference issues with jax: install the CPU version of pytorch from
[this page](https://pytorch.org/get-started/locally/).

### Data Download

Downloading the raw data differs for each dataset. The following datasets require no action:

- Text (IMDb)
- Image (Cifar black & white)
- sMNIST
- psMNIST
- Cifar (Color)

The remaining datasets need to be manually downloaded. To download _everything_,
run `./bin/download_all.sh`. This will download quite a lot of data and will take some time. Below is a summary of the steps for each dataset:

- ListOps: run `./bin/download_lra.sh` to download the full LRA dataset.

## Repository Structure

Directories and files that ship with GitHub repo:

```
lru/                   Source code for models, datasets, etc.
    dataloaders/       Code mainly derived from S4 processing each dataset.
    dataloading.py     Dataloading functions.
    model.py           Defines the LRU, linear RNN and diagonal RNN modules, individual layers and entire models.
    train.py           Training loop code.
    train_helpers.py   Functions for optimization, training and evaluation steps.
    utils/             A range of utility functions.
bin/                   Shell scripts for downloading data.
requirements.txt       Requirements for running the code.
run_train.py           Training loop entrypoint.
```

Directories that may be created on-the-fly:

```
raw_datasets/       Raw data as downloaded.
cache_dir/          Precompiled caches of data. Can be copied to new locations to avoid preprocessing.
wandb/              Local WandB log files.
```

## Run experiments

Running the different experiments requires a Weights and Biases account to log the results.


### Sequential CIFAR

The task is here to look at a 32x32 CIFAR-10 image and predict the class of the image. Chance level
is at an accuracy of 10%.

```
python run_train.py --model diag-rnn --n_layers 6 --dataset cifar-classification --warmup 18 --epochs 180 --batch_size 50 --d_model 512 --d_hidden 384 --lr_factor 0.025 --A_init fixed
```

Note: the LRA benchmark uses grey images (dataset: cifar-lra-classification) but the LRU paper uses
colored images (dataset: cifar-classification).

### ListOps

The ListOps examples are comprised of summary operations on lists of single-digit integers, written
in prefix notation. The full sequence has a corresponding solution which is also a single-digit
integer, thus making it a ten-way balanced classification problem.

```
 python run_train.py --model diag-rnn --n_layers 6 --dataset listops-classification --warmup_end 5 --d_model 256 --d_hidden 192 --epochs 50 --batch_size 32 --A_init fixed --lr_factor 0.05 --p_dropout 0
```

### IMdB

The network is given a sequence of bytes representing a text and has to classify the document into
two categories.

```
python run_train.py --model diag-rnn --n_layers 6 --dataset imdb-classification --warmup_end 7 --d_model 192 --d_hidden 256  --batch_size 32 --A_init fixed --epochs 65 --lr_factor 0.025```

