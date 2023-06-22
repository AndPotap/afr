import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from functools import partial
import logging
from datetime import datetime
import inspect
import random
import string
import pickle
import yaml
import wandb
from torch import save as save_checkpoint
from utils.common_utils import AverageMeter
from utils.common_utils import update_dict


class TrainLogger:
    def __init__(self, args):
        self.args = args
        self.use_wandb = args.use_wandb
        self.logger = self.initialize_logging(args)
        log_args(args, logger=self.logger)
        log_inputs(vars(args), log_dir=self.logger.name)
        self.best_val_wga, self.best_test_wga = 0., 0.
        self.best_val_mean, self.best_test_mean = 0., 0.
        self.metrics = {epoch: {} for epoch in range(args.num_epochs + 1)}

    def initialize_logging(self, args):
        # args.output_dir = add_timestamp_with_random(args.output_dir, ending='')
        os.makedirs(args.output_dir, exist_ok=True)
        logger = prepare_logger(args.output_dir)
        if self.use_wandb:
            config = get_config_for_wandb(args)
            wandb.init(dir=args.output_dir, project=args.project, name=args.run_name, config=config)
        return logger

    def finalize_logging(self, model):
        # save_checkpoint(model.state_dict(), join(self.args.base_model_dir, 'final_last_layer.pt'))
        # save_checkpoint(model.state_dict(), join(self.args.output_dir, 'final_last_layer.pt'))
        save_checkpoint(model.state_dict(), join(self.args.output_dir, 'final_checkpoint.pt'))
        save_object(self.metrics, join(self.args.output_dir, 'metrics.pkl'))
        if self.use_wandb:
            wandb.finish()

    def log_data_stats(self, train_data, test_data, val_data=None):
        self.logger.info("\nLogging data stats")
        get_ys_func = partial(get_y_s, n_spurious=train_data.dataset.n_spurious)
        for data, name in [(train_data, "Train"), (test_data, "Test"), (val_data, "Val")]:
            if data:
                self.logger.info(f'\n{name} Data (total {len(data.dataset)})')
                self.logger.info(f"N groups {data.dataset.n_groups}")
                for group_idx in data.dataset.active_groups:
                    y_idx, s_idx = get_ys_func(group_idx)
                    self.logger.info(f'    Group {group_idx} (y={y_idx}, s={s_idx}):'
                                     f' n = {data.dataset.group_counts[group_idx]:.0f}')

    def log_train_results_and_save_chkp(self, epoch, acc_groups, model, optimizer, scheduler):
        if epoch % self.args.save_freq == 0:
            state_dict = create_state_dict(model, optimizer, scheduler, epoch)
            save_checkpoint(state_dict, join(self.args.output_dir, 'resumable_checkpoint.pt'))
        self.log_train_results(epoch=epoch, acc_groups=acc_groups)

    def log_train_results(self, epoch, acc_groups):
        log_results(self.logger, epoch=epoch, acc_groups=acc_groups, tag="train",
                    use_wandb=self.use_wandb, metrics=self.metrics)

    def log_holdout_results(self, epoch, results_dict):
        for partition, acc_groups in results_dict.items():
            log_results(self.logger, self.metrics, epoch=epoch, acc_groups=acc_groups,
                        tag=partition, use_wandb=self.use_wandb)

    def log_results_save_chkp(self, model, epoch, results_dict):
        self.log_holdout_results(epoch, results_dict)

        if "val" in results_dict.keys():
            val_wga = get_results(results_dict["val"])["worst_accuracy"]
            if val_wga > self.best_val_wga:
                self.logger.info(f"\nNew best validation wga: {val_wga:1.3e}")
                self.best_val_wga = val_wga
                test_at_val = get_results(results_dict["test"])["worst_accuracy"]
                self.logger.info(f"\nNew best test @ validation: {test_at_val:1.3e}")
                self.metrics[epoch].update({"best_test_at_val": test_at_val})
                if self.use_wandb:
                    wandb.log({"best_val_wga": self.best_val_wga}, step=epoch)
                    wandb.log({"best_test_wga_at_val": test_at_val}, step=epoch)
                save_checkpoint(model.state_dict(), join(self.args.output_dir, 'best_checkpoint.pt'))

        test_wga = get_results(results_dict["test"])["worst_accuracy"]
        if test_wga > self.best_test_wga:
            self.logger.info(f"\nNew best test wga: {test_wga:1.3e}")
            self.best_test_wga = test_wga
            if self.use_wandb:
                wandb.log({"best_test_wga": self.best_test_wga}, step=epoch)

        self.metrics[epoch].update({
            "best_val_wga": self.best_val_wga,
            "best_test_wga": self.best_test_wga
        })
        save_checkpoint(model.state_dict(), join(self.args.output_dir, f'checkpoint_{epoch}.pt'))


class EmbeddingsLogger(TrainLogger):
    def __init__(self, args):
        super().__init__(args)

    def log_group_weights(self, weights, groups, epoch):
        active_groups = groups.unique().tolist()
        for g_val in active_groups:
            mask = groups == g_val
            text = f"w{g_val}"
            agg_weight =  weights[mask].sum().item()
            self.metrics[epoch][text] = agg_weight
            if self.use_wandb:
                wandb.log({text: agg_weight})

    def save_holdout_data(self, test_data, val_data):
        self.test_embeddings, self.test_y, self.test_groups = test_data
        self.val_embeddings, self.val_y, self.val_groups = val_data

    def log_results_save_chkp(self, epoch, holdout_logits, model=None):
        logits_test, logits_val = holdout_logits
        # logits_test = classifier(self.test_embeddings)
        test_acc_groups = self.generate_acc_groups(logits_test, self.test_y, self.test_groups,
                                                   epoch, "test")
        # logits_val = classifier(self.val_embeddings)
        val_acc_groups = self.generate_acc_groups(logits_val, self.val_y, self.val_groups, epoch,
                                                  "val")
        results_dict = {"val": val_acc_groups, "test": test_acc_groups}

        val_wga = get_results(results_dict["val"])["worst_accuracy"]
        if val_wga > self.best_val_wga:
            self.logger.info(f"\nNew best validation wga: {val_wga:1.3e}")
            self.best_val_wga = val_wga
            test_at_val = get_results(results_dict["test"])["worst_accuracy"]
            self.logger.info(f"\nNew best test @ validation: {test_at_val:1.3e}")
            self.metrics[epoch].update({"best_test_at_val": test_at_val})
            if self.use_wandb:
                wandb.log({"best_val_wga": self.best_val_wga}, step=epoch)
                wandb.log({"best_test_wga_at_val": test_at_val}, step=epoch)

            if model is not None:
                save_checkpoint(model.state_dict(), join(self.args.base_model_dir, 'best_last_layer.pt'))

        val_mean = get_results(results_dict["val"])["mean_accuracy"]
        if val_mean > self.best_val_mean:
            self.logger.info(f"\nNew best validation wga: {val_mean:1.3e}")
            self.best_val_mean = val_mean
            test_mean_val = get_results(results_dict["test"])["mean_accuracy"]
            self.logger.info(f"\nNew best test mean @ val: {test_mean_val:1.3e}")
            self.metrics[epoch].update({"best_test_mean_val": test_mean_val})
            if self.use_wandb:
                wandb.log({"best_val_mean": self.best_val_mean}, step=epoch)

        test_wga = get_results(results_dict["test"])["worst_accuracy"]
        if test_wga > self.best_test_wga:
            self.logger.info(f"\nNew best test wga: {test_wga:1.3e}")
            self.best_test_wga = test_wga
            if self.use_wandb:
                wandb.log({"best_test_wga": self.best_test_wga}, step=epoch)

        test_mean = get_results(results_dict["test"])["mean_accuracy"]
        if test_mean > self.best_test_mean:
            self.logger.info(f"\nNew best test wga: {test_mean:1.3e}")
            self.best_test_mean = test_mean
            if self.use_wandb:
                wandb.log({"best_test_mean": self.best_test_mean}, step=epoch)

        self.metrics[epoch].update({
            "best_val_wga": self.best_val_wga,
            "best_test_wga": self.best_test_wga,
            "best_val_mean": self.best_val_mean,
            "best_test_mean": self.best_test_mean,
        })

    def generate_acc_groups(self, logits, targets, groups, epoch, partition="train"):
        active_groups = groups.unique().tolist()
        acc_groups = {g_idx: AverageMeter() for g_idx in active_groups}
        update_dict(acc_groups, targets, groups, logits)
        log_results(self.logger, self.metrics, epoch, acc_groups, partition, self.use_wandb)
        return acc_groups

    def save_plot(self, path):
        df = pd.DataFrame.from_dict(self.metrics, orient="index")
        self.logger.info(f"Best test at val:     {df['best_test_at_val'].max():2.3f}")
        self.logger.info(f"Best test WGA:        {df['best_test_wga'].max():2.3f}")
        self.logger.info(f"Best val WGA:         {df['best_val_wga'].max():2.3f}")
        self.logger.info(f"Best test mean val:   {df['best_test_mean_val'].max():2.3f}")
        self.logger.info(f"Best test mean:       {df['best_test_mean'].max():2.3f}")
        self.logger.info(f"Best val mean:        {df['best_val_mean'].max():2.3f}")

        xs = np.arange(len(df))
        plt.figure(dpi=150)
        cols = ["test_accuracy_0_0", "test_accuracy_1_0", "test_accuracy_2_0", "test_accuracy_3_0"]
        for col in cols:
            plt.plot(xs, df[col], label=col)
        plt.legend()
        plt.savefig(os.path.join(path, "plot_groups.png"))

        xs = np.arange(len(df))
        plt.figure(dpi=150)
        cols = ["train_worst_accuracy", "test_worst_accuracy", "val_worst_accuracy"]
        for col in cols:
            plt.plot(xs, df[col], label=col)
        plt.axvline(x=df['best_val_wga'].argmax())
        plt.legend()
        plt.savefig(os.path.join(path, "plot_wga.png"))

        xs = np.arange(len(df))
        plt.figure(dpi=150)
        cols = ["train_mean_accuracy", "test_mean_accuracy", "val_mean_accuracy"]
        for col in cols:
            plt.plot(xs, df[col], label=col)
        plt.plot(xs, df[col], label=col)
        plt.axvline(x=df['best_val_mean'].argmax())
        plt.legend()
        plt.savefig(os.path.join(path, "plot_mean.png"))

        xs = np.arange(len(df))
        cols = df.columns.tolist()
        plt.figure(dpi=150)
        for col in cols:
            if col[0] == "w":
                plt.plot(xs, df[col].fillna(value=df[col][0]), label=col)
        plt.legend()
        plt.ylim([0., 1.])
        plt.savefig(os.path.join(path, "plot_weights.png"))


def create_state_dict(model, optimizer, scheduler, epoch):
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    if scheduler is not None:
        scheduler.step()
        state_dict["scheduler"] = scheduler.state_dict()
    return state_dict


def print_results_nicely(results, logger):
    text = ""
    for key, vals in results.items():
        text += f"{key}: {vals:1.3e} | "
    logger.info(text)


def log_results(logger, metrics, epoch, acc_groups, tag, use_wandb):
    results = get_results(acc_groups)
    tagged_results = {f"{tag}_{k}": results[k] for k in results}
    metrics[epoch].update(tagged_results)
    logger.info(f"{tag} results")
    print_results_nicely(results, logger)
    if use_wandb:
        wandb.log(tagged_results, step=epoch)
        wandb.log({"epoch": epoch}, step=epoch)


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


def get_y_s(g, n_spurious=1):
    y = g // n_spurious
    s = g % n_spurious
    return y, s


def prepare_logger(log_dir):
    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, "info.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger


def log_args(args, logger):
    defaults = vars(args)
    for key, value in defaults.items():
        logger.info(f"{key}: {value}")


def get_config_for_wandb(args):
    keys2ignore = [
        'data_dir',
        'test_data_dir',
        'project',
        'run_name',
        'resume',
        'eval_freq',
        'save_freq',
        'use_wandb',
        'reweight_groups',
        'reweight_classes',
        'reweight_spurious',
        'no_shuffle_train',
        'concate_train_val',
        'num_minority_groups_remove',
    ]
    config = {}
    for key, vals in args.__dict__.items():
        if key not in keys2ignore:
            config[key] = vals
    return config


def log_inputs(defaults, log_dir):
    with open(os.path.join(log_dir, 'meta.yaml'), mode='w+') as f:
        yaml.dump(defaults, f, allow_unicode=True)
    save_object(defaults, filepath=os.path.join(log_dir, 'defaults.pkl'))


def get_default_args(func):
    signature = inspect.signature(func)
    output = {
        k: v.default
        for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty
    }
    return output


def generate_log_dir(log_dir='./logs/'):
    log_dir = add_timestamp(log_dir)
    log_dir = add_random_characters(log_dir + '_', size_to_add=5)
    log_dir += '/'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def add_timestamp(beginning='./params_'):
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = beginning + time_stamp
    return output_file


def add_random_characters(beginning, size_to_add=10):
    letters_nums = string.ascii_letters + string.digits
    ind = random.choices(letters_nums, k=size_to_add)
    stamp = ''.join(ind)
    return beginning + stamp


def add_timestamp_with_random(beginning='./params_', ending='.pkl'):
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_stamp += '_' + add_random_characters("", size_to_add=4)
    output_file = join(time_stamp, ending)
    output_file = join(beginning, output_file)
    return output_file


def load_object(filepath):
    with open(file=filepath, mode='rb') as f:
        obj = pickle.load(file=f)
    return obj


def save_object(obj, filepath, use_highest=True):
    protocol = pickle.HIGHEST_PROTOCOL if use_highest else pickle.DEFAULT_PROTOCOL
    with open(file=filepath, mode='wb') as f:
        pickle.dump(obj=obj, file=f, protocol=protocol)
