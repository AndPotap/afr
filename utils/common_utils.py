import torch
import os
import data
import argparse
from copy import deepcopy
import numpy as np
from functools import partial
import pandas as pd
from types import SimpleNamespace
from torch.utils.data import DataLoader
from data.datasets import EmbeddingsDataset
from utils.logging_utils import get_config_for_wandb
from utils.logging_utils import prepare_logging
from utils.general import add_timestamp_with_random
import wandb


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def proportion(num):
    num = float(num)
    if abs(num) > 1:
        return num / 100
    else:
        return num


def get_embeddings_loader(emb_batch_size, embeddings, targets, weights):
    if emb_batch_size > 0:
        ds = EmbeddingsDataset(embeddings=embeddings, targets=targets, weights=weights)
        batch_size = emb_batch_size if emb_batch_size > 0 else len(targets)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    else:
        loader = [(embeddings, targets, weights)]
    return loader


def determine_dfr_train_eval_loaders(tune_on, train_loader, holdout_loaders):
    if tune_on == "train":
        loader = train_loader
        eval_on = {"test": holdout_loaders["test"]}
    else:
        loader = holdout_loaders["val"]
        eval_on = holdout_loaders
    return loader, eval_on


def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def initialize_logger(args):
    args.output_dir = add_timestamp_with_random(args.output_dir, ending='')
    if args.use_wandb:
        os.makedirs(args.output_dir, exist_ok=True)
        config = get_config_for_wandb(args)
        wandb.init(dir=args.output_dir, project=args.project, name=args.run_name, config=config)
    defaults = vars(args)
    for key, value in defaults.items():
        print(f"{key}: {value}")
    set_seed(args.seed)
    logger = prepare_logging(args.output_dir, args)
    return logger


def get_y_s(g, n_spurious=1):
    y = g // n_spurious
    s = g % n_spurious
    return y, s


def update_dict(acc_groups, y, g, logits):
    preds = torch.argmax(logits, axis=1)
    correct_batch = (preds == y)
    g = g.cpu()
    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)


def get_results(acc_groups):
    groups = acc_groups.keys()
    results = {}
    for group in groups:
        text = f"accuracy_{get_y_s(group)[0]}_{get_y_s(group)[1]}"
        results.update({text: acc_groups[group].avg})
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"mean_accuracy": all_correct / all_total if all_total > 0 else 0})
    results.update({"worst_accuracy": min(results.values())})
    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_params_names(model):
    for name, _ in model.named_parameters():
        print(name)


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="base model")
    parser.add_argument(
        "--loss", type=str, default="cross_entropy", choices=[
            "cross_entropy",
        ])
    parser.add_argument("--data_dir", type=str, required=True, help="Train dataset directory")
    parser.add_argument("--train_prop", type=proportion, default=1)
    parser.add_argument("--val_prop", type=float, default=1)
    parser.add_argument("--max_prop", type=float, default=1)
    parser.add_argument("--val_size", type=int, default=-1)
    parser.add_argument(
        "--data_transform", type=str, required=False, default="AugWaterbirdsCelebATransform",
        choices=[
            "NoTransform",
            "WildsBase",
            "AugDominoTransform",
            "CropDominoTransform",
            "CropFlipDominoTransform",
            "FlipDominoTransform",
            "CropFlipBlurDominoTransform",
            "NoAugDominoTransform",
            "SimCLRDominoTransform",
            "MaskedDominoTransform",
            "AugWaterbirdsCelebATransform",
            "SimCLRWaterbirdsCelebATransform",
            "NoAugWaterbirdsCelebATransform",
            "NoAugNoNormWaterbirdsCelebATransform",
            "MaskedWaterbirdsCelebATransform",
            "ImageNetAugmixTransform",
            "ImageNetRandomErasingTransform",
            "SimCLRCifarTransform",
            "BertTokenizeTransform",
        ], help="data preprocessing transformation")
    parser.add_argument(
        "--dataset", type=str, required=False, default="SpuriousDataset", choices=[
            "SpuriousDataset",
            "SpuriousCIFAR10",
            "MultiNLIDataset",
            "FakeSpuriousCIFAR10",
            "DummyMNIST",
            "MultiColoredMNIST",
            "ChestXRay",
            "CirclesData",
            "CXR",
            "CXR2",
            "Camelyon17",
            "WildsFMOW",
            "WildsCivilCommentsCoarse",
        ], help="dataset type")
    parser.add_argument("--cmnist_spurious_corr", type=float, default=0.995)
    parser.add_argument("--project", type=str, help="wandb project name")
    parser.add_argument("--run_name", type=str, help="wandb run name")
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument(
        "--optimizer", type=str, required=False, default="sgd_optimizer",
        choices=["sgd_optimizer", "adamw_optimizer", "lbfgs",
                 "bert_adamw_optimizer"], help="optimizer name")
    parser.add_argument(
        "--scheduler", type=str, required=False, default="constant_lr_scheduler",
        choices=["cosine_lr_scheduler", "constant_lr_scheduler",
                 "bert_lr_scheduler"], help="scheduler name")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--gradient_starv_lam", type=float, default=0)
    parser.add_argument("--reweight_groups", action='store_true', help="reweight groups")
    parser.add_argument("--reweight_classes", action='store_true', help="reweight classes")
    parser.add_argument("--reweight_spurious", action='store_true',
                        help="reweight based on spurious attribute")
    parser.add_argument("--pass_n", type=int, default=0)
    parser.add_argument("--warm_iters", type=int, default=0)
    parser.add_argument("--pass_n_train", type=int, default=0)
    parser.add_argument(
        "--data_transform_second", type=str, required=False, default="AugWaterbirdsCelebATransform",
        choices=[
            "NoTransform",
            "AugWaterbirdsCelebATransform",
            "ImageNetAugmixTransform",
        ])
    parser.add_argument("--tune_on", type=str, default="train")
    parser.add_argument("--grad_norm", type=float, default=-1.)
    return parser


def get_minimal_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="base model")
    parser.add_argument("--data_dir", type=str, required=True, help="Train dataset directory")
    parser.add_argument("--train_prop", type=proportion, default=1)
    parser.add_argument("--val_prop", type=float, default=1)
    parser.add_argument("--max_prop", type=float, default=1)
    parser.add_argument(
        "--data_transform", type=str, required=False, default="AugWaterbirdsCelebATransform",
        choices=[
            "NoTransform",
            "WildsBase",
            "AugDominoTransform",
            "CropDominoTransform",
            "CropFlipDominoTransform",
            "FlipDominoTransform",
            "CropFlipBlurDominoTransform",
            "NoAugDominoTransform",
            "SimCLRDominoTransform",
            "MaskedDominoTransform",
            "AugWaterbirdsCelebATransform",
            "SimCLRWaterbirdsCelebATransform",
            "NoAugWaterbirdsCelebATransform",
            "NoAugNoNormWaterbirdsCelebATransform",
            "MaskedWaterbirdsCelebATransform",
            "ImageNetAugmixTransform",
            "ImageNetRandomErasingTransform",
            "SimCLRCifarTransform",
            "BertTokenizeTransform",
        ], help="data preprocessing transformation")
    parser.add_argument(
        "--dataset", type=str, required=False, default="SpuriousDataset", choices=[
            "SpuriousDataset",
            "Camelyon17",
            "MultiNLIDataset",
            "SpuriousCIFAR10",
            "FakeSpuriousCIFAR10",
            "DummyMNIST",
            "MultiColoredMNIST",
            "ChestXRay",
            "CXR",
            "CXR2",
            "WildsFMOW",
            "WildsCivilCommentsCoarse",
        ], help="dataset type")
    parser.add_argument("--project", type=str, help="wandb project name")
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--cmnist_spurious_corr", type=float, default=0.995)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument(
        "--optimizer", type=str, required=False, default="sgd_optimizer",
        choices=["sgd_optimizer", "adamw_optimizer", "lbfgs",
                 "bert_adamw_optimizer"], help="optimizer name")
    parser.add_argument(
        "--scheduler", type=str, required=False, default="constant_lr_scheduler",
        choices=["cosine_lr_scheduler", "constant_lr_scheduler",
                 "bert_lr_scheduler"], help="scheduler name")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--init_lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--tune_on", type=str, default="train")
    parser.add_argument("--grad_norm", type=float, default=-1.)
    return parser


def get_eval_args():
    parser = get_train_dfr_args()
    parser.add_argument("--gamma2", type=float, default=0)
    return parser


def get_traintwo_args():
    parser = get_default_args()
    parser.add_argument("--first_phase_iters", type=int, default=0)
    parser.add_argument("--second_phase_iters", type=int, default=0)
    parser.add_argument("--pass_n_train", type=int, default=0)
    parser.add_argument("--policy", type=str, default="nothing")
    parser.add_argument(
        "--data_transform_second", type=str, required=False, default="AugWaterbirdsCelebATransform",
        choices=[
            "NoTransform",
            "AugWaterbirdsCelebATransform",
            "ImageNetAugmixTransform",
        ])
    return parser


def get_embeddings_args():
    parser = get_minimal_args()
    parser.add_argument(
        "--loss", type=str, default="afr", choices=[
            "cross_entropy",
            "afr",
        ])
    parser.add_argument("--checkpoint", type=str, default="final_checkpoint.pt")
    parser.add_argument("--reuse_embeddings", type=str2bool, default=True)
    parser.add_argument("--save_embeddings", type=str2bool, default=True)
    parser.add_argument("--rebalance_weights", type=str2bool, default=True)
    parser.add_argument("--num_augs", type=int, default=1)
    parser.add_argument("--reg_coeff", type=float, default=0.0)
    parser.add_argument("--run_name", type=str, help="wandb run name")
    parser.add_argument("--base_model_dir", type=str, default=None)
    parser.add_argument("--finetune_on_val", type=str2bool, default=True)
    parser.add_argument("--pass_n", type=int, default=0)
    parser.add_argument("--val_size", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--emb_batch_size", type=int, default=-1)
    return parser


def get_train_dfr_args():
    parser = get_default_args()
    parser.add_argument("--base_model_dir", type=str, default=None)
    parser.add_argument("--balance_val", action='store_true')
    parser.add_argument("--finetune_on_val", type=str2bool, default=True)
    parser.add_argument("--policy", type=str, default="fael")
    return parser


def get_jtt_args():
    parser = get_default_args()
    parser.add_argument("--checkpoint", type=str, default="final_checkpoint.pt")
    parser.add_argument("--base_model_dir", type=str, default=None)
    parser.add_argument("--upweight", type=int, default=-1)
    parser.add_argument("--lr_wd_case", type=int, default=-1)
    parser.add_argument("--finetune_on_val", type=str2bool, default=True)
    return parser


def overwrite_wd_lr_jtt_cases(args):
    aux = vars(args)
    if aux["data_dir"].__contains__("waterbirds"):
        if aux["lr_wd_case"] == 1:
            aux["init_lr"], aux["weight_decay"] = 1e-3, 1e-4
        elif aux["lr_wd_case"] == 2:
            aux["init_lr"], aux["weight_decay"] = 1e-4, 1e-1
        elif aux["lr_wd_case"] == 3:
            aux["init_lr"], aux["weight_decay"] = 1e-5, 1e-0
    elif aux["data_dir"].__contains__("CelebA"):
        if aux["lr_wd_case"] == 1:
            aux["init_lr"], aux["weight_decay"] = 1e-4, 1e-4
        elif aux["lr_wd_case"] == 2:
            aux["init_lr"], aux["weight_decay"] = 1e-4, 1e-2
        elif aux["lr_wd_case"] == 3:
            aux["init_lr"], aux["weight_decay"] = 1e-5, 1e-1
    args = SimpleNamespace(**aux)
    return args


def get_jtt_loader(subset, args):
    mask = subset["preds"] != subset["ys"]
    transform_cls = getattr(data, args.data_transform)
    transform = transform_cls(train=True)
    if args.dataset.__contains__("CIFAR10"):
        subset, name = mask, "JTTSpuriousCIFAR10"
    elif args.dataset.__contains__("CXR"):
        subset = subset[subset["preds"] != subset["ys"]]["imgs"].tolist()
        name = "JTTCXR"
    else:
        subset = subset[subset["preds"] != subset["ys"]]["imgs"].tolist()
        name = "JTTSpuriousDataset"

    dataset_cls = getattr(data, name)
    dataset = dataset_cls(basedir=args.data_dir, subset=subset, split="train", transform=transform,
                          upweight=args.upweight, prop=args.train_prop, max_prop=args.max_prop)
    loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=8,
                        pin_memory=True)
    return loader


def get_shrinked_data(subset, data_dir, split, data_transform):
    transform_cls = getattr(data, data_transform)
    transform = transform_cls(train=True)
    dataset_cls = getattr(data, "ShrinkedSpuriousDataset")
    dataset = dataset_cls(basedir=data_dir, subset=subset, split=split, transform=transform)
    return dataset

def get_data(args, finetune_on_val=False):
    transform_cls = getattr(data, args.data_transform)
    train_transform = transform_cls(train=True)
    test_transform = transform_cls(train=False)

    dataset_cls = getattr(data, args.dataset)
    if args.dataset.__contains__("Colored"):
        dataset_cls = partial(dataset_cls, spurious_correlation=args.cmnist_spurious_corr)
    trainset = dataset_cls(basedir=args.data_dir, split="train", transform=train_transform,
                           prop=args.train_prop, max_prop=args.max_prop)

    holdoutsets = {}
    for split in ["val", "test"]:
        transform = train_transform if (split == "val" and finetune_on_val) else test_transform
        # transform = train_transform if (split == "test" and finetune_on_val) else test_transform
        prop = 1 if split == "test" else args.val_prop
        holdoutsets[split] = dataset_cls(basedir=args.data_dir, split=split, transform=transform,
                                         prop=prop)

    if args.pass_n > 0:
        trainset, holdoutsets = pass_data_from_train_to_val(trainset, holdoutsets,
                                                            pass_n=args.pass_n)

    if args.val_size != -1:
        print(f"Using only {args.val_size} samples of val data")
        if args.balance_val:
            print("Using balanced group ratios for validation set")
            group_ratios = np.array([1 / 4] * 4)
        else:
            print("Using group ratios from train set for validation set")
            group_ratios = trainset.group_counts / trainset.group_counts.sum()
            group_ratios = group_ratios / group_ratios.sum()
        data.subsample_to_size_and_ratio(holdoutsets["val"], args.val_size, group_ratios)
        # data.subsample_to_size_and_ratio(holdoutsets["test"], args.val_size, group_ratios)

    # collate_fn = data.get_collate_fn(mixup=False, num_classes=trainset.n_classes)
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 16, 'pin_memory': True}
    # sampler = data.get_sampler(trainset, args)
    # train_loader = DataLoader(trainset, shuffle=True, sampler=sampler, collate_fn=collate_fn,
    #                           **loader_kwargs)
    train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)
    holdout_loaders = {}
    for name, ds in holdoutsets.items():
        shuffle = True if (name == "val" and finetune_on_val) else False
        # shuffle = True if (name == "test" and finetune_on_val) else False
        holdout_loaders[name] = DataLoader(ds, shuffle=shuffle, **loader_kwargs)
    return train_loader, holdout_loaders
    # new_holdout_loaders = {}
    # new_holdout_loaders["val"] = holdout_loaders["test"]
    # new_holdout_loaders["test"] = holdout_loaders["val"]
    # return train_loader, new_holdout_loaders


def get_data_phases(args, finetune_on_val=True):
    transform_cls = getattr(data, args.data_transform)
    train_transform = transform_cls(train=True)
    train_transform_second = getattr(data, args.data_transform_second)(train=True)
    test_transform = transform_cls(train=False)

    dataset_cls = getattr(data, args.dataset)
    trainset = dataset_cls(basedir=args.data_dir, split="train", transform=train_transform)

    holdoutsets = {}
    for split in ["val", "test"]:
        transform = train_transform_second if (split == "val"
                                               and finetune_on_val) else test_transform
        holdoutsets[split] = dataset_cls(basedir=args.data_dir, split=split, transform=transform)

    if args.val_size != -1:
        print("Using group ratios from train set for validation set")
        group_ratios = trainset.group_counts / trainset.group_counts.sum()
        group_ratios = group_ratios / group_ratios.sum()
        data.subsample_to_size_and_ratio(holdoutsets["val"], args.val_size, group_ratios)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 16, 'pin_memory': True}
    train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)
    holdout_loaders = {}
    for name, ds in holdoutsets.items():
        shuffle = True if (name == "val" and finetune_on_val) else False
        holdout_loaders[name] = DataLoader(ds, shuffle=shuffle, **loader_kwargs)
    return train_loader, holdout_loaders


def get_data_2_steps(args, finetune_on_val=False):
    transform_cls = getattr(data, args.data_transform)
    train_transform = transform_cls(train=True)
    train_transform_second = getattr(data, args.data_transform_second)(train=True)
    test_transform = transform_cls(train=False)

    dataset_cls = getattr(data, args.dataset)
    if args.dataset.__contains__("Colored"):
        dataset_cls = partial(dataset_cls, spurious_correlation=args.cmnist_spurious_corr)
    trainset = dataset_cls(basedir=args.data_dir, split="train", transform=train_transform)

    holdoutsets = {}
    for split in ["val", "test"]:
        transform = train_transform if (split == "val" and finetune_on_val) else test_transform
        holdoutsets[split] = dataset_cls(basedir=args.data_dir, split=split, transform=transform)

    if args.pass_n_train > 0:
        trainset1, trainset2 = make_two_trainsets(trainset, pass_n=args.pass_n_train)
    trainset2.transform = train_transform_second

    if args.val_size != -1:
        print("Using group ratios from train set for validation set")
        group_ratios = trainset.group_counts / trainset.group_counts.sum()
        group_ratios = group_ratios / group_ratios.sum()
        data.subsample_to_size_and_ratio(holdoutsets["val"], args.val_size, group_ratios)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 16, 'pin_memory': True}
    train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)
    train_loader1 = DataLoader(trainset1, shuffle=True, **loader_kwargs)
    train_loader2 = DataLoader(trainset2, shuffle=True, **loader_kwargs)
    holdout_loaders = {}
    for name, ds in holdoutsets.items():
        shuffle = True if (name == "val" and finetune_on_val) else False
        holdout_loaders[name] = DataLoader(ds, shuffle=shuffle, **loader_kwargs)
    train_loaders = (train_loader, train_loader1, train_loader2)
    return train_loaders, holdout_loaders


def make_two_trainsets(trainset, pass_n, seed=21):
    print(trainset.y_array.shape)

    ind = np.arange(len(trainset))
    np.random.seed(seed=seed)
    np.random.shuffle(ind)
    ind2 = ind[:pass_n]
    ind1 = ind[pass_n:]
    trainset1 = deepcopy(trainset)
    trainset2 = deepcopy(trainset)

    trainset1.y_array = trainset.y_array[ind1]
    trainset1.group_array = trainset.group_array[ind1]
    trainset1.spurious_array = trainset.spurious_array[ind1]
    trainset1.filename_array = trainset.filename_array[ind1]
    trainset1.metadata_df = trainset.metadata_df.iloc[ind1]
    trainset1.group_counts = torch.from_numpy(np.bincount(trainset.group_array))

    trainset2.y_array = trainset.y_array[ind2]
    trainset2.group_array = trainset.group_array[ind2]
    trainset2.spurious_array = trainset.spurious_array[ind2]
    trainset2.filename_array = trainset.filename_array[ind2]
    trainset2.metadata_df = trainset.metadata_df.iloc[ind2]
    trainset2.group_counts = torch.from_numpy(np.bincount(trainset.group_array))

    print(trainset1.y_array.shape)
    print(trainset2.y_array.shape)
    return trainset1, trainset2


def pass_data_from_train_to_val(train_data, val_data, pass_n):
    print(train_data.y_array.shape)
    print(val_data["val"].y_array.shape)

    ind = np.arange(len(train_data))
    np.random.shuffle(ind)
    ind1 = ind[:pass_n]
    ind2 = ind[pass_n:]

    aux1 = train_data.y_array[ind1]
    aux2 = train_data.group_array[ind1]
    aux3 = train_data.spurious_array[ind1]
    aux4 = train_data.filename_array[ind1]
    aux5 = train_data.metadata_df.iloc[ind1]

    val_data["val"].y_array = np.concatenate((val_data["val"].y_array, aux1))
    val_data["val"].group_array = np.concatenate((val_data["val"].group_array, aux2))
    val_data["val"].spurious_array = np.concatenate((val_data["val"].spurious_array, aux3))
    val_data["val"].filename_array = np.concatenate((val_data["val"].filename_array, aux4))
    val_data["val"].metadata_df = pd.concat((val_data["val"].metadata_df, aux5), axis=0)
    val_data["val"].group_counts = torch.from_numpy(np.bincount(val_data["val"].group_array))

    train_data.y_array = train_data.y_array[ind2]
    train_data.group_array = train_data.group_array[ind2]
    train_data.spurious_array = train_data.spurious_array[ind2]
    train_data.filename_array = train_data.filename_array[ind2]
    train_data.metadata_df = train_data.metadata_df.iloc[ind2]
    train_data.group_counts = torch.from_numpy(np.bincount(train_data.group_array))
    print(train_data.y_array.shape)
    print(val_data["val"].y_array.shape)
    return train_data, val_data


def get_train_group_ratios(args):
    dataset_cls = getattr(data, args.dataset)
    trainset = dataset_cls(basedir=args.data_dir, split="train", transform=None)
    group_ratios = trainset.group_counts / trainset.group_counts.sum()
    return group_ratios
