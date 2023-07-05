import torch


def cross_entropy(*_):
    def criterion(logits, y, *_):
        xe = torch.nn.functional.cross_entropy(logits, y, reduction='none')
        loss = xe.mean()
        return loss

    return criterion

def afr(*_):
    def criterion(logits, y, weights):
        return afr_fn(logits, y, weights)
    return criterion

def afr_fn(logits, y, weights):
    ce = torch.nn.functional.cross_entropy(logits, y, reduction='none')
    out = weights * ce
    return out.sum()

def get_exp_weights(logits, y, gamma):
    p = logits.softmax(-1)
    y_onehot = torch.zeros_like(logits).scatter_(-1, y.unsqueeze(-1), 1)
    p_true = (p * y_onehot).sum(-1)
    weights = (-gamma * p_true).exp()
    return weights
