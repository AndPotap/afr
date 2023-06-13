# Automatic Feature Reweighting (AFR)
This repository contains the code for
[Simple and Fast Group Robustness by Automatic Feature Reweighting](https://openreview.net/pdf?id=s5F1a6s1HS)
by Shikai Qiu, Andres Potapczynski, Pavel Izmailov and Andrew Gordon Wilson (ICML 2023).

## Motivation
Robustness to spurious correlations is of foundational importance for the reliable application of machine learning systems in the real-world,
where it is common that features learned during training fail to generalize or completely disregard minority groups.
There has been an extraordinary progress in the development of approaches and principles that are applicable to tackling these problems
but there has remained a significant open question: can we leverage these advances to develop approaches that do not require (1) knowledge of group labels, (2) manual intervention from experts, and (3) computationally intensive training procedures.

As a solution to the previous three limitations, we propose _Automatic Feature Reweighting_ (AFR), which is a group robustness method that is able to achieve **state-of-the-art results**
across a wide variety of benchmarks **without requiring group labels through training**, and with **fast training times** that are only marginally longer than standard ERM.

<figure>
  <img src="./figs/wga.png" alt="Image">
  <figcaption>(Left) Test worst-group accuracy and (right) training time comparison on Waterbirds for state-of-the-art methods that do not use group information during training.
   AFR outperforms the baselines, while only requiring a small fraction of compute time.</figcaption>
</figure>

## Citation
If you find AFR useful, please cite our work:
```bibtex
@article{qiu2023afr,
    title={{Simple and Fast Group Robustness by Automatic Feature Reweighting}},
    author={Shikai Qiu and Andres Potapczynski and Pavel Izmailov and Andrew Gordon Wilson},
    journal={International Conference on Machine Learning (ICML)},
    year={2023}
}
```

## Installation and Requirements
Simply clone the repo and install the requirements by sourcing `scripts/requirements.sh`.
We opted for having the requirements in a bash script rather than the usual formats
to make it easier to see and install the version of PyTorch that we used in the project.
The most important requirements are:
* `PyTorch`. ML framework used in the project.
* `transformers`. To load language models used for text benchmarks.
* `timm`. To load vision models used for image benchmarks.
* `wandb`. To run hyperparameter sweeps.

It is also required that you have access to the datasets that you want to run as
benchmarks.

## File Structure
```
.
├── data --> groups several utils for loading dataset and augmentations
│   ├── augmix_transforms.py
│   ├── dataloaders.py
│   ├── datasets.py --> contains the base clases for managing the datasets and groups
│   ├── data_transforms.py
│   ├── __init__.py
├── logs --> where experimental results get saved
├── losses --> comprises the loss functions used
│   └── __init__.py
├── models --> incorporates all the model definitions used in the experiments
│   ├── __init__.py  --> most model definitions
│   ├── model_utils.py
│   ├── preresnet.py
│   └── text_models.py
├── optimizers --> has the optimizers used in the experiments
│   └── __init__.py
├── README.md
├── scripts
│   └── requirements.sh
├── setup.cfg --> yapf linter settings used in the project
├── train_embeddings.py --> main script to train from embeddings (second stage)
├── train_supervised.py --> main script used to run checkpoints (first stage)
└── utils --> utils to run experiments, log results and track metrics
    ├── common_utils.py
    ├── general.py
    ├── __init__.py
    ├── logging.py
    ├── logging_utils.py
    └── supervised_utils.py
```

## Running AFR
### First Stage
To replicate the experimental results, run the following commands.
We now show the command line arguments that need to be run for the first and
second stage. On the next section we explain each of the variables in detail.
Using waterbirds as an example, for the first stage run
```shell
ARCH='imagenet_resnet50_pretrained'
DATA_DIR='/datasets/waterbirds_official'
SEED=21
PROP=80
OMP_NUM_THREADS=4 python3 train_supervised.py \
    --output_dir=logs/waterbirds/${PROP}_${SEED} \
    --project=waterbirds \
    --seed=${SEED} \
    --eval_freq=10 \
    --save_freq=10 \
    --data_dir=${DATA_DIR} \
    --data_transform=AugWaterbirdsCelebATransform \
    --model=${ARCH} \
    --train_prop=${PROP} \
    --num_epochs=40 \
    --batch_size=32 \
    --optimizer=sgd_optimizer \
    --scheduler=constant_lr_scheduler \
    --init_lr=0.003 \
    --weight_decay=1e-4 \
```
| Name | Description |
| :------------ |  :-----------: |
| `output_dir` | Specifies where the results are saved. |
| `project` | Name of the wandb project. |
| `seed` | Seed to use. |
| `eval_freq` | How often (in epochs) to evaluate the current model on the validation set. |
| `save_freq` | How often (in epochs) to save a model checkpoint. |
| `data_dir` | File path where the dataset is located. |
| `data_transform` | Type of augmentation being used. |
| `model` | Model architecture. |
| `train_prop` | % of train dataset to use for the 1st stage. |
| `num_epochs` | Number of epochs to run. |
| `batch_size` | Size of the mini-batches. |
| `optimizer` | Type of optimizer to use. |
| `scheduler` | Type of scheduler to use. |
| `init_lr` | Initial learning rate (starting point for the scheduler). |
| `weight_decay` | Weight decay value. |

### Second Stage
To run the second stage in our waterbirds example use the following command
```shell
ARCH='imagenet_resnet50_pretrained'
DATA_DIR='/datasets/waterbirds_official'
SEED=1
PROP=80
TRAIN_PROP=$(($PROP - 100))
BASE_MODEL="./logs/waterbirds/$PROP_$SEED"
OMP_NUM_THREADS=4 python3 train_embeddings.py \
    --output_dir=logs/waterbirds/emb \
    --project=waterbirds \
    --seed=21 \
    --base_model_dir=${BASE_MODEL} \
    --model=${ARCH} \
    --data_dir=${DATA_DIR} \
    --data_transform=AugWaterbirdsCelebATransform\
    --num_epochs=500 \
    --batch_size=128 \
    --optimizer=sgd_optimizer \
    --scheduler=constant_lr_scheduler \
    --init_lr=0.02 \
    --weight_decay=0. \
    --loss=fixed_cwxe \
    --train_prop=${TRAIN_PROP} \
    --focal_loss_gamma=14 \
    --num_augs=10 \
    --grad_norm=0.0 \
    --reg_coeff=0.2 \
```
| Name | Description |
| :------------ |  :-----------: |
| `output_dir` | Specifies where the results are saved. |
| `project` | Name of the wandb project. |
| `seed` | Seed to use. |
| `base_model_dir` | Location of the first stage checkpoint. |
| `model` | Model architecture. |
| `data_dir` | File path where the dataset is located. |
| `data_transform` | Type of augmentation being used. |
| `num_epochs` | Number of epochs to run. |
| `batch_size` | Size of the mini-batches. |
| `optimizer` | Type of optimizer to use. |
| `scheduler` | Type of scheduler to use. |
| `init_lr` | Initial learning rate (starting point for the scheduler). |
| `weight_decay` | Weight decay value. |
| `loss` | Name of loss function to be used. |
| `train_prop` | % of train dataset use for 2nd stage (should be negative). |
| `focal_loss_gamma` | Gamma value. |
| `num_augs` | Number of augmentations to use for embeddings. |
| `grad_norm` | Gradient norm cap. |
| `reg_coeff` | Regularization coefficient. |

## Examples

To run AFR on Waterbirds use the following two commands.
```shell
python3 train_supervised.py --use_wandb="" --output_dir=logs/waterbirds/80_21 --project=waterbirds --seed=21 --eval_freq=10 --save_freq=10 --data_dir='/datasets/waterbirds_official' --dataset=SpuriousDataset --data_transform=AugWaterbirdsCelebATransform --model='imagenet_resnet50_pretrained' --max_prop=1.0 --train_prop=${PROP} --num_epochs=40 --batch_size=32 --optimizer=sgd_optimizer --scheduler=constant_lr_scheduler --init_lr=0.003 --weight_decay=1e-4
python3 train_embeddings.py --use_wandb="" --output_dir=logs/waterbirds/emb --project=waterbirds --seed=21 --base_model_dir="./logs/waterbirds/80_21" --model='imagenet_resnet50_pretrained' --data_dir='/datasets/waterbirds_official' --data_transform=AugWaterbirdsCelebATransform --num_epochs=500 --batch_size=128 --emb_batch_size=-1 --optimizer=sgd_optimizer --scheduler=constant_lr_scheduler --init_lr=0.02 --momentum=0. --weight_decay=0. --loss=fixed_cwxe --tune_on=train --max_prop=1.0 --train_prop=-20 --val_prop=0.05 --finetune_on_val=True --focal_loss_gamma=14 --num_augs=10 --grad_norm=0.0 --reg_coeff=0.2
```
The first command runs the 1st stage of AFR and saves the checkpoint on
`logs/waterbirds/80_1`.
The second command runs the 2nd stage of AFR.
We added the flag `--use_wandb=""` to avoid using wandb but in case you do want to use
wandb simply take that flag out of the command.

To run AFR on CelebA use the following two commands. 
```shell
python3 train_supervised.py --use_wandb="" --output_dir=logs/celeba/80_1 --project=celeba --seed=1 --eval_freq=1 --save_freq=2 --data_dir='/datasets/CelebA' --data_transform=AugWaterbirdsCelebATransform --model='imagenet_resnet50_pretrained' --max_prop=1.0 --train_prop=80 --num_epochs=20 --batch_size=100 --optimizer=sgd_optimizer --scheduler=cosine_lr_scheduler --init_lr=3e-3 --weight_decay=1e-4
python3 train_embeddings.py --use_wandb="" --output_dir=logs/celeba/emb --project=celeba --seed=1 --base_model_dir="./logs/celeba/80_1" --model='imagenet_resnet50_pretrained' --data_dir='/datasets/CelebA' --data_transform=AugWaterbirdsCelebATransform --num_epochs=100 --batch_size=128 --emb_batch_size=-1 --optimizer=sgd_optimizer --scheduler=constant_lr_scheduler --init_lr=0.01 --momentum=0.0 --weight_decay=0. --loss=fixed_cwxe --tune_on=train --train_prop=-0.2 --focal_loss_gamma=1.8 --num_augs=1 --grad_norm=0.0 --reg_coeff=0.00 --checkpoint=final_checkpoint.pt
```


## Data Access

### Waterbirds and CelebA

Please follow the instructions in the [DFR repo](https://github.com/PolinaKirichenko/deep_feature_reweighting#data-access) to prepare the Waterbirds and CelebA datasets.

### CivilComments

ToDo

### MultiNLI

ToDo

### Chest X-Ray

ToDo
