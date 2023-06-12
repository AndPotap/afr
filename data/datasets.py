import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from typing import Callable, cast, Tuple
import wilds


def _bincount_array_as_tensor(arr):
    return torch.from_numpy(np.bincount(arr)).long()


def _randomly_split_in_two(samples, prop=0.8, seed=21, max_prop=1.0):
    random.seed(seed)
    random.shuffle(samples)
    samples = samples[:int(len(samples) * max_prop)]
    first_total = int(len(samples) * np.abs(prop))
    if prop >= 0:
        return samples[:first_total]
    else:
        return samples[-first_total:]


def get_group_array(y_array, spurious_array, all_groups):
    group_array = np.zeros(y_array.shape[0], dtype=np.int64)
    for idx, (y, ss) in enumerate(all_groups):
        mask1 = np.array(spurious_array == ss)
        mask2 = np.array(y_array == y)
        mask = np.logical_and(mask1, mask2)
        group_array[mask] = idx
    return group_array


def _get_split(split):
    try:
        return ["train", "val", "test"].index(split)
    except ValueError:
        raise (f"Unknown split {split}")


def _cast_int(arr):
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    elif isinstance(arr, torch.Tensor):
        return arr.int()
    else:
        raise NotImplementedError


class SpuriousDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None, prop=1.0, seed=21, max_prop=1.0):
        self.basedir = basedir
        self.transform = transform

        self.metadata_df = self._get_metadata(split)
        indices = np.arange(len(self.metadata_df))
        ind = _randomly_split_in_two(indices, prop=prop, seed=seed, max_prop=max_prop)
        self.metadata_df = self.metadata_df.iloc[np.sort(ind)]

        self.y_array = self.metadata_df["y"].values
        self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()
        self.filename_array = self.metadata_df["img_filename"].values

    def _get_metadata(self, split):
        split_i = _get_split(split)
        metadata_df = pd.read_csv(os.path.join(self.basedir, "metadata.csv"))
        metadata_df = metadata_df[metadata_df["split"] == split_i]
        return metadata_df

    def _count_attributes(self):
        self.n_classes = np.unique(self.y_array).size
        self.n_spurious = np.unique(self.spurious_array).size
        self.y_counts = _bincount_array_as_tensor(self.y_array)
        self.spurious_counts = _bincount_array_as_tensor(self.spurious_array)

    def _count_groups(self):
        self.group_counts = _bincount_array_as_tensor(self.group_array)
        self.n_groups = len(self.group_counts)
        self.active_groups = np.unique(self.group_array).tolist()

    def _get_class_spurious_groups(self):
        self.group_array = _cast_int(self.y_array * self.n_spurious + self.spurious_array)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        group = self.group_array[idx]
        is_spurious = self.spurious_array[idx]
        x = self._image_getitem(idx)
        file_path = self.filename_array[idx]
        return x, y, group, is_spurious, file_path

    def _image_getitem(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class JTTSpuriousDataset(SpuriousDataset):
    def __init__(self, basedir, subset, upweight, split="train", transform=None, prop=1.,
                 max_prop=1.):
        del prop
        del max_prop
        selected_cols = ["img_filename", "y", "split", "place"]
        self.basedir = basedir
        self.metadata_df = self._get_metadata(split)
        mistakes = pd.DataFrame({"img": subset * int(upweight - 1)})
        aux = pd.merge(left=self.metadata_df, right=mistakes, left_on="img_filename",
                       right_on="img")
        self.metadata_df = pd.concat((self.metadata_df[selected_cols], aux[selected_cols]))
        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()
        self.filename_array = self.metadata_df["img_filename"].values


class ShrinkedSpuriousDataset(SpuriousDataset):
    def __init__(self, basedir, subset, split="train", transform=None):
        self.basedir = basedir
        self.metadata_df = self._get_metadata(split)
        subset = subset.reset_index()
        subset = subset.rename(columns={"index": "img", 0: "group", 1: "focal"})
        aux = pd.merge(left=self.metadata_df, right=subset, left_on="img_filename", right_on="img")
        self.metadata_df = aux[["img_filename", "y", "split", "place"]]
        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()
        self.filename_array = self.metadata_df["img_filename"].values


class PhasesDataset(SpuriousDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform)

    def __getitem__(self, idx):
        file_path = self.filename_array[idx]
        label = self.y_array[idx]
        group = self.group_array[idx]
        is_spurious = self.spurious_array[idx]
        image = self._image_getitem(idx)
        return file_path, image, label, group, is_spurious


class MultiNLIDataset(SpuriousDataset):
    """Adapted from https://github.com/kohpangwei/group_DRO/blob/master/data/multinli_dataset.py
    """
    def __init__(self, basedir, split="train", transform=None):
        assert transform is None, "transfrom should be None"

        # utils_glue module in basedir is needed to load data
        import sys
        sys.path.append(basedir)

        self.basedir = basedir
        self.metadata_df = pd.read_csv(os.path.join(self.basedir, "metadata_random.csv"))
        bert_filenames = [
            "cached_train_bert-base-uncased_128_mnli", "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm"
        ]
        features_array = sum(
            [torch.load(os.path.join(self.basedir, name)) for name in bert_filenames], start=[])
        all_input_ids = torch.tensor([f.input_ids for f in features_array]).long()
        all_input_masks = torch.tensor([f.input_mask for f in features_array]).long()
        all_segment_ids = torch.tensor([f.segment_ids for f in features_array]).long()
        # all_label_ids = torch.tensor([
        #     f.label_id for f in self.features_array]).long()

        split_i = _get_split(split)
        split_mask = (self.metadata_df["split"] == split_i).values

        self.x_array = torch.stack((all_input_ids, all_input_masks, all_segment_ids),
                                   dim=2)[split_mask]
        self.metadata_df = self.metadata_df[split_mask]
        self.y_array = self.metadata_df['gold_label'].values
        self.spurious_array = (self.metadata_df['sentence2_has_negation'].values)
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        x = self.x_array[idx]
        return x, y, g, s


class EmbeddingsDataset:
    def __init__(self, embeddings, targets, weights):
        self.embeddings = embeddings
        self.targets = targets
        self.weights = weights

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        y = self.targets[idx]
        w = self.weights[idx]
        return emb, y, w


class CirclesData(SpuriousDataset):
    def __init__(self, basedir, split="train", transform=None, prop=1.0, seed=21, max_prop=1.0):
        x, y = prepare_circles(basedir, prop=prop, split=split)
        self.transform = transform
        self.y_array = y
        self.x = x

        self.group_array = y
        self.n_spurious = 4
        self.n_classes = 4
        self.n_groups = 4
        self.group_counts = _bincount_array_as_tensor(self.group_array)
        self.active_groups = list(range(self.n_groups))

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        group = y
        is_spurious = y
        x = self.x[idx]
        file_path = ""
        return x, y, group, is_spurious, file_path


class BaseWildsDataset(SpuriousDataset):
    def __init__(self, ds_name, basedir, split, transform, y_name, spurious_name, prop, seed=21):
        self.basedir = basedir
        self.root_dir = "/".join(self.basedir.split("/")[:-1])
        base_dataset = wilds.get_dataset(dataset=ds_name, download=False, root_dir=self.root_dir)
        self.dataset = base_dataset.get_subset(split, transform=transform)
        if np.abs(prop) < 1.0:
            ind = _randomly_split_in_two(np.arange(len(self.dataset)), prop=prop, seed=seed)
            # self.dataset = WILDSSubset(self.dataset, indices=ind, transform=transform)
            self.dataset.indices = self.dataset.indices[ind]
            # self.dataset.metadata_array =  self.dataset.metadata_array[ind]

        column_names = self.dataset.metadata_fields
        if y_name:
            y_idx = column_names.index(y_name)
            self.y_array = self.dataset.metadata_array[:, y_idx]
        if spurious_name:
            s_idx = column_names.index(spurious_name)
            self.spurious_idx = s_idx
            self.spurious_array = self.dataset.metadata_array[:, s_idx]
        if y_name and spurious_name:
            self._count_attributes()

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]
        s = metadata[self.spurious_idx]
        g = self.group_array[idx]
        return x, y, g, s, "img"

    def __len__(self):
        return len(self.dataset)


class Camelyon17(BaseWildsDataset):
    def __init__(self, basedir, split, transform, prop=1.0, seed=21, **kwargs):
        super().__init__(ds_name="camelyon17", basedir=basedir, split=split, transform=transform,
                         y_name="y", spurious_name="slide", prop=prop, seed=seed)
        self.n_spurious = 50
        # all_groups = [(y, ss) for y in [0, 1] for ss in range(self.n_spurious)]
        # self.group_array = _get_group_array(self.y_array, self.spurious_array, all_groups)
        self.group_array = self.spurious_array
        self._count_groups()


class WildsFMOW(BaseWildsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__("fmow", basedir, split, transform, "y", "region")
        self.group_array = self.spurious_array
        self._count_groups()


class WildsPoverty(BaseWildsDataset):
    def __init__(self, basedir, split="train", transform=None):
        # assert transform is None, "transfrom should be None"
        super().__init__("poverty", basedir, split, transform, "y", "urban")
        self.n_classes = None
        self.group_array = self.spurious_array
        self._count_groups()


class WildsCivilCommentsCoarse(BaseWildsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__("civilcomments", basedir, split, transform, "y", None)
        attributes = [
            "male", "female", "LGBTQ", "black", "white", "christian", "muslim", "other_religions"
        ]
        column_names = self.dataset.metadata_fields
        self.spurious_cols = [column_names.index(a) for a in attributes]
        self.spurious_array = self.get_spurious(self.dataset.metadata_array)
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()

    def get_spurious(self, metadata):
        if len(metadata.shape) == 1:
            return metadata[self.spurious_cols].sum(-1).clip(max=1)
        else:
            return metadata[:, self.spurious_cols].sum(-1).clip(max=1)

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]
        s = self.get_spurious(metadata)
        g = y * self.n_spurious + s
        return x, y, g, s


class DummyMNIST(SpuriousDataset):
    def __init__(self, basedir, split="train", transform=None, spurious_correlation=0.90, **_):
        del basedir, split, transform
        self.spurious_correlation = spurious_correlation
        size_n = 50
        self.images = torch.randn(size=(size_n, 3, 28, 28))
        self.targets = torch.randint(low=0, high=1, size=(size_n, ))
        self.group, self.spurious = self.targets.clone(), self.targets.clone()
        self.n_classes, self.n_spurious, self.n_groups = 2, 1, 4
        self.group_array = np.array([0, 0, 0, 0, 1, 3, 3, 3, 2])
        self.group_counts = _bincount_array_as_tensor(self.group_array)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, label = self.images[idx], self.targets[idx]
        group, spurious = self.group[idx], self.spurious[idx]
        return image, label, group, spurious


class MultiColoredMNIST(SpuriousDataset):
    def __init__(self, basedir, spurious_correlation, split="train", transform=None, seed=21,
                 prop=1.0, max_prop=1.0):
        use_train = True if split in ["train", "val"] else False
        transform = ToTensor()
        self.ds = MNIST(root=basedir, train=use_train, download=True, transform=transform)
        self.subset_data_based_on_split(split)
        data = (self.ds.data, self.ds.targets)
        if split in ["train", "val", "test"]:
            out = self.add_spurious_correlation(data, spurious_correlation=spurious_correlation)
        else:
            out = self.add_spurious_correlation(data, spurious_correlation=0.)
        self.images, self.targets, self.group, self.is_spurious = out
        indices = np.arange(len(self.images))
        ind = _randomly_split_in_two(indices, prop=prop, seed=seed, max_prop=max_prop)
        self.images = self.images[ind]
        self.targets = self.targets[ind]
        self.group = self.group[ind]
        self.is_spurious = self.is_spurious[ind]
        # self.flip_label(per=0.25)
        # self.create_minority(trim_pc=0.05, class_to_trim=0)

        self.data_size = self.images.shape[0]
        self.n_spurious = 2
        self.n_classes = len(torch.unique(self.targets))
        # self.n_groups = len(torch.unique(self.group))
        self.n_groups = len(self.group.unique())
        self.active_groups = list(range(self.n_groups))
        self.group_array = self.group.float()
        self.group_counts = _bincount_array_as_tensor(self.group_array)

    def flip_label(self, per):
        n_classes = len(torch.unique(self.targets))
        flip_label = torch.rand(self.targets.shape[0]) < per
        pseudo_labels = torch.randint(low=0, high=n_classes, size=(self.targets.shape[0], ))
        self.targets = torch.where(flip_label, pseudo_labels, self.targets)

    def create_minority(self, trim_pc, class_to_trim):
        mask = self.targets == class_to_trim
        ind_trim = torch.masked_select(torch.arange(len(mask)), mask)
        ind_other = torch.masked_select(torch.arange(len(mask)), torch.logical_not(mask))
        selected_n = int(len(ind_trim) * trim_pc)
        ind = torch.concat([ind_trim[:selected_n], ind_other])
        ind, _ = torch.sort(ind)
        self.images = self.images[ind]
        self.targets = self.targets[ind]
        self.group = self.group[ind]
        self.is_spurious = self.is_spurious[ind]

    def subset_data_based_on_split(self, split):
        train_size = 48_000
        if split == "train":
            self.ds.data = self.ds.data[:train_size, :, :]
            self.ds.targets = self.ds.targets[:train_size]
        elif split == "val":
            self.ds.data = self.ds.data[train_size:, :, :]
            self.ds.targets = self.ds.targets[train_size:]

    def add_spurious_correlation(self, data, spurious_correlation):
        self.cases = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        colors = self.get_colors()
        images, targets = data[0], data[1]
        colored_images, labels, spurious, group = self.prepare_placeholders(images, targets)
        for idx in range(images.shape[0]):
            spurious[idx] = torch.rand(1) < spurious_correlation
            labels[idx] = self.get_label(targets[idx])
            color_num = self.get_color_label(labels[idx], torch.logical_not(spurious[idx]))
            group[idx] = self.determine_group(labels[idx], spurious[idx])
            aux = torch.broadcast_to(images[idx], size=(3, 28, 28)).clone().float()
            colored_images[idx] = aux.clone()
            colored_images[idx] /= 256.
            color = colors[int(color_num)]
            for j in range(len(color)):
                colored_images[idx, j, :, :] *= color[j]
        return colored_images, labels.long(), group.long(), spurious.long()

    @staticmethod
    def prepare_placeholders(images, targets):
        colored_images = torch.zeros(size=(images.shape[0], 3, 28, 28))
        labels = torch.zeros(size=(targets.shape[0], ))
        spurious = torch.zeros(size=(targets.shape[0], )).bool()
        group = torch.zeros(size=(targets.shape[0], ))
        return colored_images, labels, spurious, group

    def get_label(self, target):
        target = int(target.clone().detach())
        for idx, case in enumerate(self.cases):
            lower, upper = case
            if (target >= lower) and (target <= upper):
                label = torch.tensor(idx, dtype=torch.int64)
        return label

    def get_color_label(self, label, randomize_color):
        color_label = torch.tensor(int(label), dtype=torch.int64)
        if randomize_color:
            # options = [idx for idx in range(len(self.cases)) if idx != int(label)]
            options = [idx for idx in range(len(self.cases))]
            selected = int(torch.randint(high=len(options), size=(1, )))
            color_label = torch.tensor(options[selected], dtype=torch.int64)
        return color_label

    @staticmethod
    def determine_group(label, spurious):
        group = label.clone()
        if spurious == 1:
            group += 5
        return torch.tensor(group, dtype=torch.int64)

    @staticmethod
    def get_colors():
        purple = (127 / 256, 0., 255 / 256)
        pink = (255 / 256, 0., 255 / 256)
        blue = (0., 255 / 256, 255 / 256)
        orange = (255 / 256, 128 / 256, 0.)
        green = (128 / 256, 255 / 256, 0.)
        colors = [purple, blue, pink, orange, green]
        return colors

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image, label = self.images[idx], self.targets[idx]
        group, is_spurious = self.group[idx], self.is_spurious[idx]
        return image, label, group, is_spurious


class FakeSpuriousCIFAR10(SpuriousDataset):
    """CIFAR10 with SpuriousCorrelationDataset API.

    Groups are the same as classes.
    """
    def __init__(self, basedir, split, transform=None, val_size=5000):
        split_i = _get_split(split)
        self.ds = CIFAR10(root=basedir, train=(split_i != 2), download=True, transform=transform)
        if split_i == 0:
            self.ds.data = self.ds.data[:-val_size]
            self.ds.targets = self.ds.targets[:-val_size]
        elif split_i == 1:
            self.ds.data = self.ds.data[-val_size:]
            self.ds.targets = self.ds.targets[-val_size:]

        self.y_array = np.array(self.ds.targets)
        self.n_classes = 10
        self.spurious_array = np.zeros_like(self.y_array)
        self.n_spurious = 1
        self.group_array = self.y_array

        self.n_groups = 10
        self.group_counts = _bincount_array_as_tensor(self.group_array)
        self.y_counts = _bincount_array_as_tensor(self.y_array)
        self.spurious_counts = _bincount_array_as_tensor(self.spurious_array)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, y, y, 0


class SpuriousCIFAR10(CIFAR10):
    def __init__(self, basedir, split="train", transform=None, prop=1.0, seed=21, max_prop=1.0):
        train = True if split == "train" else False
        super().__init__(root=basedir, transform=transform, train=train)
        ind = _randomly_split_in_two(np.arange(len(self.data)), prop=prop, seed=seed,
                                     max_prop=max_prop)
        self.data = self.data[ind]
        self.targets = torch.tensor(self.targets)[ind]

        self.add_spurious_attributes()

    def add_spurious_attributes(self):
        self.y_array = np.array(self.targets)
        self.n_classes = 10
        self.spurious_array = np.zeros_like(self.y_array)
        self.n_spurious = 1
        self.group_array = self.y_array

        self.n_groups = 10
        self.active_groups = [idx for idx in range(self.n_groups)]
        self.group_counts = _bincount_array_as_tensor(self.group_array)
        self.y_counts = _bincount_array_as_tensor(self.y_array)
        self.spurious_counts = _bincount_array_as_tensor(self.spurious_array)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        dummy_spurious, dummy_path = 0, "img"
        return img, target, target, dummy_spurious, dummy_path


class JTTSpuriousCIFAR10(SpuriousCIFAR10):
    def __init__(self, basedir, subset, upweight, split="train", transform=None):
        super().__init__(basedir, split, transform, prop=1.0, seed=21)
        data_subset = self.data[subset]
        data_subset = np.concatenate([data_subset for _ in range(upweight)], axis=0)
        self.data = np.concatenate((self.data, data_subset), axis=0)

        targets_subset = np.array(self.targets)[subset]
        targets_subset = np.concatenate([targets_subset for _ in range(upweight)], axis=0)
        self.targets = torch.tensor(np.concatenate((self.targets, targets_subset)))

        self.add_spurious_attributes()


def remove_minority_groups(trainset, num_remove):
    if num_remove == 0:
        return
    print("Removing minority groups")
    print("Initial groups", np.bincount(trainset.group_array))
    group_counts = trainset.group_counts
    minority_groups = np.argsort(group_counts.numpy())[:num_remove]
    minority_groups
    idx = np.where(
        np.logical_and.reduce([trainset.group_array != g for g in minority_groups],
                              initial=True))[0]
    trainset.y_array = trainset.y_array[idx]
    trainset.group_array = trainset.group_array[idx]
    trainset.confounder_array = trainset.confounder_array[idx]
    trainset.filename_array = trainset.filename_array[idx]
    trainset.metadata_df = trainset.metadata_df.iloc[idx]
    print("Final groups", np.bincount(trainset.group_array))


def balance_groups(ds):
    print("Original groups", ds.group_counts)
    group_counts = ds.group_counts.long().numpy()
    min_group = np.min(group_counts)
    group_idx = [np.where(ds.group_array == g)[0] for g in range(ds.n_groups)]
    for idx in group_idx:
        np.random.shuffle(idx)
    group_idx = [idx[:min_group] for idx in group_idx]
    idx = np.concatenate(group_idx, axis=0)
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))
    print("Final groups", ds.group_counts)


def unbalance_groups(ds, group_ratios):
    # keep as much data as possible but with the given group ratios,
    # assuming original groups roughly balanced scale ratios so that largest one is 1
    group_ratios = np.array(group_ratios)
    group_ratios = group_ratios / np.max(group_ratios)
    print('Unbalancing groups with ratios', group_ratios)
    print("Original groups", ds.group_counts)
    group_counts = ds.group_counts.long().numpy()
    min_group = np.min(group_counts)
    group_idx = [np.where(ds.group_array == g)[0] for g in range(ds.n_groups)]
    for idx in group_idx:
        np.random.shuffle(idx)
    group_idx = [idx[:int(min_group * r)] for idx, r in zip(group_idx, group_ratios)]
    idx = np.concatenate(group_idx, axis=0)
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))
    print("Final groups", ds.group_counts)


def subsample(ds, frac, generator=torch.Generator().manual_seed(42)):
    subsample_size = int(len(ds) * frac)
    idx = torch.randperm(len(ds), generator=generator).tolist()[:subsample_size]
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))


def subsample_to_size_and_ratio(ds, final_size, final_ratio):
    # subsample ds so that the final dataset has final_size samples and
    # final_ratio for each group where final_ratio sums to 1
    final_ratio = np.array(final_ratio)
    final_ratio = final_ratio / np.sum(final_ratio)
    print(f'Subsampling to size {final_size} samples with ratios {final_ratio}')
    print("Original groups", ds.group_counts)
    group_counts = ds.group_counts.long().numpy()
    check = (final_size * final_ratio <= group_counts).all()
    assert check, 'Cannot subsample to desired size and ratio'
    group_idx = [np.where(ds.group_array == g)[0] for g in range(ds.n_groups)]
    group_idx = [idx[:int(final_size * r)] for idx, r in zip(group_idx, final_ratio)]
    idx = np.concatenate(group_idx, axis=0)
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))
    print("Final groups", ds.group_counts)


def subset(ds, idx):
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))


def concate(ds1, ds2):
    ds1.y_array = np.concatenate((ds1.y_array, ds2.y_array), axis=0)
    ds1.group_array = np.concatenate((ds1.group_array, ds2.group_array), axis=0)
    ds1.spurious_array = np.concatenate((ds1.spurious_array, ds2.spurious_array), axis=0)
    ds1.filename_array = np.concatenate((ds1.filename_array, ds2.filename_array), axis=0)
    ds1.metadata_df = pd.concat((ds1.metadata_df, ds2.metadata_df), axis=0)
    ds1.group_counts = torch.from_numpy(np.bincount(ds1.group_array))


class ChestXRay(ImageFolder):
    def __init__(self, basedir, downsamples=None, split="train", transform=None, prop=1.0,
                 max_prop=1.0):
        self.prop = prop
        self.max_prop = max_prop
        if downsamples is None:
            downsamples = (0.12, 1.0, 0.5, 0.5) if split == "train" else (1.0, 1.0, 1.0, 1.0)
        self.split = "val" if split in ["test", "val"] else "train"
        self.ds = downsamples
        self.downsample_seed = 13
        super().__init__(root=basedir, transform=transform)
        self.split_dataset(split)
        self.n_classes, self.n_spurious, self.n_groups = 2, 1, 4
        self.group_array = np.array(list(map(lambda x: x[-1][1], self.samples)))
        self.group_counts = _bincount_array_as_tensor(self.group_array)
        self.active_groups = np.unique(self.group_array).tolist()

    def split_dataset(self, split):
        if split in ["test", "val"]:
            # samples[1] = nihcxr/train/00021676_000.png
            # samples[-1] = patient46237/study1/view1_frontal.jpg
            s1 = _randomly_split_in_two(self.samples, prop=0.5)
            s2 = _randomly_split_in_two(self.samples, prop=-0.5)
            # samples[1] = patient48639/study2/view1_frontal.jpg
            # samples[-1] = nihcxr/train/00003249_000.png
            self.samples = s1 if split == "val" else s2

        elif np.abs(self.prop) < 1.:
            indices = np.arange(len(self.samples))
            ind = _randomly_split_in_two(indices, self.prop, max_prop=self.max_prop)
            self.samples = [self.samples[idx] for idx in ind]

    @staticmethod
    def _use_negative_label_policy(df):
        df['Pneumonia'] = np.abs(df['Pneumonia'])
        return df

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        del class_to_idx

        def is_valid_file(x: str) -> bool:
            return x.lower().endswith(cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        directory = os.path.expanduser(directory)
        chexlabels = pd.read_csv(os.path.join(directory, 'chexpert/trainval.csv'))
        # chexlabels.shape = (223,648, 19)
        chexlabels = chexlabels[~np.isnan(chexlabels['Pneumonia'])]
        # chexlabels.shape = (27,842, 19) !!
        # chexlabels['Pneumonia'] = np.abs(chexlabels['Pneumonia'])
        chexlabels = self._use_negative_label_policy(chexlabels)
        # chexlabels['Pneumonia'].unique() = [0., -1., 1.]
        # np.mean(mask = chexlabels['Pneumonia'] >= 0.) = 0.3258
        # TODO: why abs? maybe a negative number signifies something
        chexlabels.index = chexlabels['Path']

        nihlabels = pd.read_csv(os.path.join(directory, 'nihcxr/nih_labels.csv'))
        nihlabels.index = nihlabels['Image Index']

        image_and_labels = []
        np.random.seed(self.downsample_seed)

        for root, _, fnames in os.walk(os.path.join(directory, 'nihcxr/train')):
            for pic_name in fnames:
                label = nihlabels.loc[pic_name, 'Pneumonia']
                has_image_extension = is_valid_file(pic_name)
                is_in_correct_fold = nihlabels.loc[pic_name, 'fold'] == self.split
                if has_image_extension and is_in_correct_fold:
                    cond1 = not label and np.random.rand() <= self.ds[0]
                    cond2 = label and np.random.rand() <= self.ds[1]
                    if (cond1) or (cond2):
                        path = os.path.join(root, pic_name)
                        group = 0 if label < 1 else 1
                        image_and_labels.append((path, (int(label), group, -1)))

        for root, _, fnames in os.walk(os.path.join(directory, f'chexpert/{self.split}')):
            for pic_name in fnames:
                current_name = os.path.join(directory, f'chexpert/{self.split}')
                df_name = f'CheXpert-v1.0-small/{self.split}'
                pic_name = os.path.join(root.replace(current_name, df_name), pic_name)
                if pic_name in chexlabels.index:
                    row = chexlabels.loc[pic_name]
                    label = row['Pneumonia']
                    has_image_extension = is_valid_file(pic_name)
                    is_frontal = row['Frontal/Lateral'] == 'Frontal'
                    if has_image_extension and is_frontal:
                        cond1 = not label and np.random.rand() <= self.ds[2]
                        cond2 = label and np.random.rand() <= self.ds[3]
                        if (cond1) or (cond2):
                            path = os.path.join(root, pic_name.split("/")[-1])
                            group = 2 if label < 1 else 3
                            image_and_labels.append((path, (int(label), group, -1)))
        return image_and_labels

    def __getitem__(self, index):
        path, target = self.samples[index]
        group, spurious = target[1], target[2]
        target = target[0]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, group, spurious, path


class CXR(ChestXRay):
    def __init__(self, basedir, downsamples=None, split="train", transform=None, prop=1.0,
                 max_prop=1.0):
        if downsamples is None:
            downsamples = (0.12, 1.0, 1.0, 0.1) if split == "train" else (1.0, 1.0, 1.0, 1.0)
        super().__init__(basedir, downsamples, split, transform, prop, max_prop)

    @staticmethod
    def _use_negative_label_policy(df):
        mask = df['Pneumonia'] == -1
        return df[np.logical_not(mask)]


class CXR2(ChestXRay):
    def __init__(self, basedir, downsamples=None, split="train", transform=None, prop=1.0,
                 max_prop=1.0):
        if downsamples is None:
            downsamples = (0.12, 1.0, 0.5, 0.5) if split == "train" else (1.0, 1.0, 1.0, 1.0)
        super().__init__(basedir, downsamples, split, transform, prop, max_prop)


class JTTCXR(CXR):
    def __init__(self, basedir, subset, upweight, split="train", transform=None, prop=1.0,
                 max_prop=1.0, downsamples=None):
        super().__init__(basedir, downsamples, split, transform, prop, max_prop)
        subset_samples = []
        for idx in subset:
            for jdx in self.samples:
                if idx == jdx[0]:
                    subset_samples.append(jdx)
        subset_samples *= upweight
        self.samples += subset_samples

        self.n_classes, self.n_spurious, self.n_groups = 2, 1, 4
        self.group_array = np.array(list(map(lambda x: x[-1][1], self.samples)))
        self.group_counts = _bincount_array_as_tensor(self.group_array)
        self.active_groups = np.unique(self.group_array).tolist()


def prepare_circles(file_path, prop, split, seed=21):
    out = np.load(file_path)
    x = torch.tensor(out["x"], dtype=torch.float32)
    y = torch.tensor(out["y"], dtype=torch.float32)
    train_n = int(x.shape[0] * prop)
    ind = np.arange(x.shape[0])
    np.random.seed(seed=seed)
    np.random.shuffle(ind)
    x_train, y_train = x[ind[:train_n]], y[ind[:train_n]]
    x_test, y_test = x[ind[train_n:]], y[ind[train_n:]]
    if split == "train":
        return x_train, y_train
    else:
        return x_test, y_test


class DatasetGroup:
    def __init__(self, data):
        self.x, self.y = data

    def __getitem__(self, index):
        return self.x[index], int(self.y[index]), 0, 0

    def __len__(self):
        return self.x.shape[0]


def generate_multiple_circles_data(dataset_size, dim_n, proportions):
    categories_n = len(proportions)
    sizes = [int(dataset_size * prop) for prop in proportions]
    sizes[-1] = dataset_size - sum(sizes[:-1])
    ys, xs = [], []
    lims = [(3 * i, 3 * (i + 1)) for i in range(categories_n)]
    for idx, (lower, upper) in enumerate(lims):
        x_i = np.zeros(shape=(sizes[idx], dim_n))
        y_i = np.zeros(shape=(sizes[idx], ), dtype=np.int32)
        radius = np.random.uniform(low=lower, high=upper, size=sizes[idx])
        out = fill_in_cirle(x_i, radius, dim_n=dim_n)
        xs.append(out)
        ys.append(y_i + idx)

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def fill_in_cirle(x, radius, dim_n):
    for idx in range(x.shape[0]):
        recurrent_sum = 0
        r = radius[idx]
        r2 = radius[idx]**2.
        for cat in range(dim_n - 1):
            if recurrent_sum > r2:
                break
            x[idx, cat] = np.random.uniform(low=-r, high=np.sqrt(r2 - recurrent_sum))
            recurrent_sum += x[idx, cat]**2.
            sign = np.random.choice(a=[-1., 1.], size=1)[0]
        out = sign * np.sqrt(r2 - recurrent_sum) if r2 >= recurrent_sum else 0.
        x[idx, dim_n - 1] = out
    return x


def fill_spurious(x, y, categories, alpha, spurious_dim, locs=(1., 0.), scales=(0.1, 10.0)):
    spur_mean, noise_mean = locs
    spur_std, noise_std = scales
    mask = np.isin(y, categories)

    spurious_loc = np.random.choice(a=2, size=(x.shape[0], 1), p=[alpha, 1 - alpha])
    x_new = np.random.normal(loc=noise_mean, scale=noise_std, size=(x.shape[0], spurious_dim))
    spur_feat = np.random.normal(loc=spur_mean, scale=spur_std, size=(x.shape[0], spurious_dim))
    x_new[mask] = np.where(spurious_loc[mask] == 0, spur_feat[mask], x_new[mask])

    x_new = np.concatenate((x, x_new), axis=1)
    return x_new
