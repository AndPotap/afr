import time
import torch
import models
import optimizers
import losses
import utils
from utils.common_utils import get_default_args
from utils.common_utils import set_seed
from utils.supervised_utils import train_epoch
from utils.supervised_utils import eval_model
from utils.logging import TrainLogger
from utils.general import print_time_taken


def train_supervised(args):
    tic = time.time()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    Log = TrainLogger(args)

    train_loader, holdout_loaders = utils.get_data(args)
    Log.log_data_stats(train_loader, holdout_loaders['test'], holdout_loaders['val'])

    model = getattr(models, args.model)(train_loader.dataset.n_classes).to(device)
    Log.logger.info(f'Model has {utils.count_parameters(model) / 1e6:.2g}M parameters')

    optimizer = getattr(optimizers, args.optimizer)(model, args)
    scheduler = getattr(optimizers, args.scheduler)(optimizer, args)
    criterion = getattr(losses, args.loss)(args)

    for epoch in range(args.num_epochs):
        info = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        loss_meter, acc_groups = info
        Log.logger.info(f"E: {epoch} | L: {loss_meter.avg:2.5e}\n")
        Log.log_train_results_and_save_chkp(epoch, acc_groups, model, optimizer, scheduler)

        if (epoch % args.eval_freq == 0) or (epoch == args.num_epochs - 1):
            results_dict = eval_model(model, holdout_loaders, device=device)
            Log.log_results_save_chkp(model, epoch, results_dict)

    Log.finalize_logging(model)
    toc = time.time()
    print_time_taken(toc - tic, logger=Log.logger)


if __name__ == '__main__':
    parser = get_default_args()
    args = parser.parse_args()
    assert args.reweight_groups + args.reweight_classes <= 1
    train_supervised(args)
