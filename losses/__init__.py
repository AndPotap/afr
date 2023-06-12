import torch


def cross_entropy(*_):
    def criterion(logits, y, *_):
        xe = torch.nn.functional.cross_entropy(logits, y, reduction='none')
        loss = xe.mean()
        return loss

    return criterion


def fixed_cwxe(*_):
    def criterion(logits, y, weights):
        return fixed_cwxe_fn(logits, y, weights)

    return criterion


def fixed_cwxe_fn(logits, y, weights):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    out = weights * ce
    return out.sum()
