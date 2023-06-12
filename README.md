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
├── capture_dataset_images.py
├── data
│   ├── augmix_transforms.py
│   ├── dataloaders.py
│   ├── datasets.py
│   ├── data_transforms.py
│   ├── __init__.py
├── eval_chkp.py
├── generate_circles_data.py
├── logs
├── losses
│   └── __init__.py
├── models
│   ├── circles_model.py
│   ├── cmnist_model.py
│   ├── custom_head_domino.py
│   ├── gating_model.py
│   ├── __init__.py
│   ├── model_utils.py
│   ├── preresnet.py
│   └── text_models.py
├── multi_layer.py
├── notebooks
│   ├── celeba_tuning.ipynb
│   ├── chest-x-ray.py
│   ├── dfr_comparision.ipynb
│   ├── lbfgs_example.py
│   ├── loss_plot.py
│   ├── mixed-chest-x-ray.py
│   ├── mnli.ipynb
│   ├── plot_fig1.py
│   ├── plot_jtt_early.py
│   ├── plot_jtt_heatmap.py
│   ├── plot_jtt_upweight.py
│   ├── plot_scale.py
│   ├── plot_time_bars.py
│   ├── plot_val_exp.py
│   ├── wb_celeba.ipynb
│   └── wb_tuning.ipynb
├── optimizers
│   └── __init__.py
├── README.md
├── scripts
├── setup.cfg
├── show_images.py
├── train_2_steps.py
├── train_dfr.py
├── train_edfr.py
├── train_embeddings.py
├── train_jtt.py
├── train_phases.py
├── train_supervised.py
└── utils
    ├── common_utils.py
    ├── general.py
    ├── __init__.py
    ├── logging.py
    ├── logging_utils.py
    └── supervised_utils.py
```

## Experiments

## Variables / Arguments

## Examples
