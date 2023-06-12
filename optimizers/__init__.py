from torch.optim import AdamW
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


def sgd_optimizer_fromparams(params, lr, momentum, weight_decay):
    optimizer = SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def sgd_optimizer(model, args):
    lr, momentum, weight_decay = args.init_lr, args.momentum, args.weight_decay
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def adamw_optimizer(model, args):
    lr, weight_decay = args.init_lr, args.weight_decay
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def cosine_lr_scheduler(optimizer, args):
    lr, num_steps = args.init_lr, args.num_epochs
    return CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr / 100.)


def constant_lr_scheduler(*_):
    return None
