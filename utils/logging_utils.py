import os
from functools import partial
import sys
import json
import wandb
import utils
from torch import save as save_checkpoint


class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *_):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def get_config_for_wandb(args):
    keys2ignore = [
        'data_dir',
        'test_data_dir',
        'project',
        'run_name',
        'resume',
        'output_dir',
        'eval_freq',
        'save_freq',
        'use_wandb',
        'reweight_groups',
        'reweight_classes',
        'reweight_spurious',
        'no_shuffle_train',
        'concate_train_val',
        'base_model_dir',
        'num_minority_groups_remove',
    ]
    config = {}
    for key, vals in args.__dict__.items():
        if key not in keys2ignore:
            config[key] = vals
    return config


def prepare_logging(output_dir, args):
    print('Preparing directory %s' % output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        args_json = json.dumps(vars(args))
        f.write(args_json)

    logger = Logger(os.path.join(output_dir, 'log.txt'))
    return logger


class ResultsTracker:
    def __init__(self, logger, output_dir, use_wandb):
        self.logger = logger
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.best_val_wga, self.best_test_wga = 0., 0.

    def log_results_save_chkp(self, state_dict, epoch, results_dict):
        for partition, acc_groups in results_dict.items():
            utils.log_results(self.logger, epoch, acc_groups, partition, use_wandb=self.use_wandb)

        if "val" in results_dict.keys():
            val_wga = utils.get_results(results_dict["val"])["worst_accuracy"]
            if val_wga > self.best_val_wga:
                self.logger.write(f"\nNew best validation wga: {val_wga:1.3e}")
                self.best_val_wga = val_wga
                test_at_val = utils.get_results(results_dict["test"])["worst_accuracy"]
                self.logger.write(f"\nNew best test @ validtion: {test_at_val:1.3e}")
                if self.use_wandb:
                    wandb.log({"best_val_wga": self.best_val_wga}, step=epoch)
                    wandb.log({"best_test_wga_at_val": test_at_val}, step=epoch)
                save_checkpoint(state_dict, os.path.join(self.output_dir, 'best_checkpoint.pt'))

        test_wga = utils.get_results(results_dict["test"])["worst_accuracy"]
        if test_wga > self.best_test_wga:
            self.logger.write(f"\nNew best test wga: {test_wga:1.3e}")
            self.best_test_wga = test_wga
            if self.use_wandb:
                wandb.log({"best_test_wga": self.best_test_wga}, step=epoch)

        save_checkpoint(state_dict, os.path.join(self.output_dir, f'checkpoint_{epoch}.pt'))


def print_results_nicely(results, logger):
    text = ""
    for key, vals in results.items():
        text += f"{key}: {vals:1.3e} | "
    logger.write(text + "\n")


def log_results(logger, epoch, acc_groups, tag, use_wandb):
    results = utils.get_results(acc_groups)
    logger.write(f"{tag} results \n")
    print_results_nicely(results, logger)
    if use_wandb:
        wandb.log({f"{tag}_{k}": results[k] for k in results}, step=epoch)
        wandb.log({"epoch": epoch + 1}, step=epoch)


def log_data_stats(logger, train_data, test_data, val_data=None):
    print("Logging data stats")
    get_ys_func = partial(get_y_s, n_spurious=train_data.dataset.n_spurious)
    for data, name in [(train_data, "Train"), (test_data, "Test"), (val_data, "Val")]:
        if data:
            logger.write(f'{name} Data (total {len(data.dataset)})\n')
            print("N groups ", data.dataset.n_groups)
            for group_idx in range(data.dataset.n_groups):
                y_idx, s_idx = get_ys_func(group_idx)
                logger.write(f'    Group {group_idx} (y={y_idx}, s={s_idx}):'
                             f' n = {data.dataset.group_counts[group_idx]:.0f}\n')


def get_y_s(g, n_spurious=1):
    y = g // n_spurious
    s = g % n_spurious
    return y, s
