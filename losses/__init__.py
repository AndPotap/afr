import torch


def cross_entropy(*_):
    def criterion(logits, y, *_):
        xe = torch.nn.functional.cross_entropy(logits, y, reduction='none')
        loss = xe.mean()
        return loss

    return criterion


def mixed_loss(args):
    def criterion(logits, y, counter):
        if counter >= args.warm_iters:
            output = focal_fn(logits, y, args.focal_loss_gamma)
            output = output.mean()
        else:
            output = torch.nn.functional.cross_entropy(logits, y)
        return output

    return criterion


def fixed_cwxe(*_):
    def criterion(logits, y, weights):
        return fixed_cwxe_fn(logits, y, weights)

    return criterion


def fixed_cwxe_fn(logits, y, weights):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    out = weights * ce
    return out.sum()


def changing_cwxe(args, *_):
    def criterion(logits, y, *_):
        return changing_cwxe_fn(logits, y, gamma=args.focal_loss_gamma).mean()

    return criterion


def changing_cwxe_fn(logits, y, gamma):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    weights = get_exp_weights(logits, y, gamma)
    weights = weights.detach()
    cwxe = weights * ce
    return cwxe


def get_exp_weights(logits, y, gamma):
    p = logits.softmax(-1)
    y_onehot = torch.zeros_like(logits).scatter_(-1, y.unsqueeze(-1), 1)
    p_true = (p * y_onehot).sum(-1)
    weights = (-gamma * p_true).exp()
    return weights


def confidence_wxe(args, *_):
    def criterion(logits, y, *_):
        return confidence_wxe_fn(logits, y, gamma=args.focal_loss_gamma).mean()

    return criterion


def confidence_wxe_fn(logits, y, gamma):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    weights = get_weights(logits, y, gamma)
    weights = weights.detach()
    cwxe = weights * ce
    return cwxe


def focal_loss(args, *_):
    def criterion(logits, y, *_):
        return focal_fn(logits, y, gamma=args.focal_loss_gamma).mean()

    return criterion


def focal_fn(logits, y, gamma):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    weights = get_weights(logits, y, gamma)
    focal = weights * ce
    return focal


def get_weights(logits, y, gamma):
    p = logits.softmax(-1)
    y_onehot = torch.zeros_like(logits).scatter_(-1, y.unsqueeze(-1), 1)
    p_true = (p * y_onehot).sum(-1)
    eps = 1e-5
    weights = ((1 - p_true + eps)**gamma)
    return weights


def gs_loss(args):
    def criterion(logits, y, *_):
        xe = torch.nn.functional.cross_entropy(logits, y, reduction='none')
        reg = torch.mean((logits)**2.)
        loss = xe.mean() + args.gradient_starv_lam * reg
        return loss

    return criterion
