import os
import tqdm
import pandas as pd
import torch
import numpy as np
from torch.nn import BatchNorm2d
from torch.nn import Identity
from torch.nn import Linear
from torch.nn.utils import clip_grad_norm_
import utils
from utils import AverageMeter
from utils.logging import load_object
from utils.logging import save_object
from losses import get_exp_weights

_bar_format = '{l_bar}{bar:50}{r_bar}{bar:-10b}'


def compute_groups_stats(groups, metric):
    metric = np.array(metric)
    n_groups = len(np.unique(np.array(groups)))
    text = ""
    for i in range(n_groups):
        mask = groups == i
        text += f"\nGroup ({i}): "
        text += f"Q05: {np.quantile(metric[mask], q=0.05):2.5f} | "
        text += f"Q25: {np.quantile(metric[mask], q=0.25):2.5f} | "
        text += f"Q50: {np.quantile(metric[mask], q=0.5):2.5f} | "
        text += f"Q75: {np.quantile(metric[mask], q=0.75):2.5f} | "
        text += f"Q95: {np.quantile(metric[mask], q=0.95):2.5f} | "
    print(text)


def get_dfr_correction(weights, groups):
    active_groups = groups.unique().tolist()
    total = 1 / len(active_groups)
    for g_val in active_groups:
        mask = groups == g_val
        weights[mask] = (weights[mask] * total) / weights[mask].sum()
    return weights


def normalize_weights(weights):
    weights /= weights.sum()
    return weights


def rebalance_weights(weights, targets):
    unique_classes = torch.unique(targets)
    for c in unique_classes:
        class_count = (targets == c).sum()
        weights[targets == c] *= len(targets) / class_count
    return weights.detach()


def get_last_layer_model(init_weights):
    w0, b0 = init_weights
    last_layer = Linear(w0.shape[1], w0.shape[0], bias=True)
    last_layer.weight.data = w0.clone()
    last_layer.bias.data = b0.clone()
    # last_layer.weight.data = classifier.weight.clone()
    # last_layer.bias.data = classifier.bias.clone()
    # last_layer.cuda()
    return last_layer


def determine_freeze_policy(policy):
    if policy == "fael":
        fn = freeze_all_params_except_last_layer
    elif policy == "faebnl":
        fn = freeze_all_params_except_last_layer_or_bn
    elif policy == "fab":
        fn = freeze_as_beginning
    elif policy == "lb":
        fn = freeze_except_last_block
    else:
        fn = lambda _: None
    return fn


def freeze_except_last_block(model):
    freeze_all_params(model)
    for name, param in model.named_parameters():
        if name.__contains__("layer4.2"):
            param.requires_grad = True
    model.fc.weight.requires_grad, model.fc.bias.requires_grad = True, True


def freeze_all_params_except_last_layer_or_bn(model):
    freeze_all_params(model)
    for _, module in model.named_modules():
        if isinstance(module, BatchNorm2d):
            module.weight.requires_grad = True
            module.bias.requires_grad = True
    model.fc.weight.requires_grad, model.fc.bias.requires_grad = True, True


def freeze_as_beginning(model):
    freeze_all_params(model)
    model.fc.weight.requires_grad = True


def freeze_all_params_except_last_layer(model):
    freeze_batchnorm_stats(model)
    freeze_all_params(model)
    model.fc.weight.requires_grad, model.fc.bias.requires_grad = True, True


def freeze_batchnorm_stats(model):
    for _, module in model.named_modules():
        if isinstance(module, BatchNorm2d):
            module.eval()


def freeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = False


def get_classifier_and_feature_extractor(model):
    classifier = get_classifier(model)
    feature_extractor = get_feature_extractor(model)
    return classifier.eval(), feature_extractor.eval()


def get_feature_extractor(model):
    model.fc = Identity()
    return model


def get_classifier(model):
    return model.fc


@torch.no_grad()
def get_all_weights(model, loader, gamma, device):
    weights = []
    for batch in loader:
        x, y, *_ = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        weights.append(get_exp_weights(logits, y, gamma).detach().cpu())
    return torch.cat(weights, axis=0)


class EmbeddingManager:
    def __init__(self, device, args):
        self.device = device
        self.base_dir = args.base_model_dir
        self.num_augs = args.num_augs
        self.reuse_embeddings = args.reuse_embeddings
        self.save_embeddings = args.save_embeddings

    def get_notebook_embeddings(self, emb_path):
        print(f'Loading embeddings from {emb_path}')
        emb_dict = torch.load(emb_path)
        embeddings, predictions, y, groups = emb_dict['e'], emb_dict['pred'], emb_dict[
            'y'], emb_dict['g']
        test_embeddings, test_predictions, test_y, test_groups = emb_dict['test_e'], emb_dict[
            'test_pred'], emb_dict['test_y'], emb_dict['test_g']
        val_embeddings, val_predictions, val_y, val_groups = emb_dict['val_e'], emb_dict[
            'val_pred'], emb_dict['val_y'], emb_dict['val_g']
        out = (embeddings, predictions, groups, y)
        out_test = (test_embeddings, test_predictions, test_groups, test_y)
        out_val = (val_embeddings, val_predictions, val_groups, val_y)
        # w0, b0 = emb_dict["w0"], emb_dict["b0"]
        return out, out_test, out_val

    def get_train_test_embeddings(self, classifier, feature_extractor, loaders):
        filepath = os.path.join(self.base_dir, "embeddings.pkl")
        if (os.path.exists(filepath) and self.reuse_embeddings):
            print("Found embeddings")
            out, out_test, out_val = load_object(filepath)
        else:
            print("Computings embeddings")
            out, out_test, out_val = self._get_train_test_embeddings(classifier, feature_extractor,
                                                                     loaders)
            if self.save_embeddings:
                print("Saving embeddings")
                save_object((out, out_test, out_val), filepath)
        out = self.place_in_device(out)
        out_test = self.place_in_device(out_test)
        out_val = self.place_in_device(out_val)
        return out, out_test, out_val

    def place_in_device(self, out):
        new = tuple([element.to(self.device) for element in out])
        return new

    def _get_train_test_embeddings(self, classifier, feature_extractor, loaders):
        train_loader, holdout_loaders = loaders
        val_loader, test_loader = holdout_loaders["val"], holdout_loaders["test"]
        out = get_embeddings_and_rest(classifier, feature_extractor, train_loader,
                                      num_augs=self.num_augs, device=self.device)
        out_test = get_embeddings_and_rest(classifier, feature_extractor, test_loader,
                                           num_augs=self.num_augs, device=self.device)
        out_val = get_embeddings_and_rest(classifier, feature_extractor, val_loader,
                                          num_augs=self.num_augs, device=self.device)
        return out, out_test, out_val


@torch.no_grad()
def get_embeddings_and_rest(classifier, feature_extractor, loader, num_augs, device):
    embeddings, predictions, targets, groups = [], [], [], []
    for _ in range(num_augs):
        for batch in loader:
            x, y, group, *_ = batch
            x, y = x.to(device), y.to(device)
            z = feature_extractor(x)
            logits = classifier(z)
            predictions.append(torch.argmax(logits.detach().cpu(), axis=1))
            embeddings.append(z.detach().cpu())
            groups.append(group.detach().cpu())
            targets.append(y.detach().cpu())
    embeddings = torch.cat(embeddings, axis=0)
    predictions = torch.cat(predictions, axis=0)
    groups = torch.cat(groups, axis=0)
    targets = torch.cat(targets, axis=0)
    return embeddings, predictions, groups, targets


def take_step_multi(gating_model, model1, model2, data, optimizer, criterion, scheduler,
                    max_norm=-1):
    embeddings, targets, weights = data
    optimizer.zero_grad()
    # rho = gating_model(embeddings)
    rho = gating_model(embeddings)**2.
    logits = rho * (model1(embeddings)) + (1 - rho) * (model2(embeddings))
    loss = criterion(logits, targets, weights)
    loss.backward()
    if max_norm > 0:
        clip_grad_norm_(gating_model.parameters(), max_norm=max_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss, logits


def take_step(model, data, optimizer, criterion, scheduler, init_weights, reg_coeff, max_norm=-1):
    w0, b0 = init_weights
    embeddings, targets, weights = data
    optimizer.zero_grad()
    logits = model(embeddings)
    loss = criterion(logits, targets, weights)
    reg = ((model.weight - w0).pow(2).sum() + (model.bias - b0).pow(2).sum())
    loss += reg_coeff * reg
    loss.backward()
    if max_norm > 0:
        clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss, logits


def train_epoch_multiple(model, loaders, optimizer, criterion, device, counter, policy=None):
    model.train()
    if policy is not None:
        policy(model)
    loss_meter = AverageMeter()
    n_groups = loaders[0].dataset.n_groups
    acc_groups = {g_idx: AverageMeter() for g_idx in range(n_groups)}
    shrinked, counter = iter(loaders[-1]), 0

    for batch in (pbar := tqdm.tqdm(loaders[0], bar_format=_bar_format)):
        x1, y1, group1, *_ = batch
        x1, y1 = x1.to(device), y1.to(device)
        empty = True if counter == len(shrinked) else False
        if empty:
            shrinked, counter = iter(loaders[-1]), 0
        batch2 = next(shrinked)
        counter += 1
        x2, y2, group2, *_ = batch2
        x2, y2 = x2.to(device), y2.to(device)
        X = torch.concat((x1, x2))
        Y = torch.concat((y1, y2))
        group = torch.concat((group1, group2))

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y, counter)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss, X.size(0))

        preds = torch.argmax(logits, dim=1)
        utils.update_dict(acc_groups, Y, group, logits)
        acc = (preds == Y).float().mean()

        text = f"Loss: {loss.item():3f} ({loss_meter.avg:.3f}); Acc: {acc:3f}"
        pbar.set_description(text)

    return loss_meter, acc_groups


def train_epoch(model, loader, optimizer, criterion, device, counter, policy=None,
                max_grad_norm=-1.0):
    model.train()
    if policy is not None:
        policy(model)
    loss_meter = AverageMeter()
    # n_groups = loader.dataset.n_groups
    # acc_groups = {g_idx: AverageMeter() for g_idx in range(n_groups)}
    acc_groups = {g_idx: AverageMeter() for g_idx in loader.dataset.active_groups}

    for batch in (pbar := tqdm.tqdm(loader, bar_format=_bar_format)):
        x, y, group, *_ = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y, counter)
        loss.backward()
        if max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        loss_meter.update(loss, x.size(0))
        preds = torch.argmax(logits, dim=1)

        utils.update_dict(acc_groups, y, group, logits)
        acc = (preds == y).float().mean()
        text = f"Loss: {loss.item():3f} ({loss_meter.avg:.3f}); Acc: {acc:3f}"
        pbar.set_description(text)

    return loss_meter, acc_groups


@torch.no_grad()
def get_img_predictions(model, loader, device):
    model.eval()
    imgs, ys, preds = [], [], []
    for batch in tqdm.tqdm(loader, bar_format=_bar_format):
        x, y, *_, img_path = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, axis=1)
        imgs += img_path
        ys.append(y.cpu())
        preds.append(pred.cpu())
    predictions_df = {
        'imgs': imgs,
        'preds': torch.concat(preds, axis=0),
        'ys': torch.concat(ys, axis=0)
    }
    return pd.DataFrame(predictions_df)


def eval_model(model, test_loader_dict, device):
    model.eval()
    results_dict = {}
    with torch.no_grad():
        for test_name, test_loader in test_loader_dict.items():
            acc_groups = {g_idx: AverageMeter() for g_idx in test_loader.dataset.active_groups}
            for batch in tqdm.tqdm(test_loader, bar_format=_bar_format):
                x, y, group, *_ = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                utils.update_dict(acc_groups, y, group, logits)
            results_dict[test_name] = acc_groups
    return results_dict
