import numpy as np
from .utils import *


def chamfer_loss(x, y):
    d = torch.cdist(x, y)
    return d.min(dim=0).values.mean() + d.min(dim=1).values.mean()


def continuity_loss(x):
    d = (x[1:] - x[:-1]).norm(dim=-1, p=2)
    return d.mean()


def svg_length_loss(p_pred, p_target):
    pred_length, target_length = get_length(p_pred), get_length(p_target)

    return (target_length - pred_length).abs() / target_length


def svg_emd_loss(p_pred, p_target,
                 first_point_weight=False, return_matched_indices=False):
    n, m = len(p_pred), len(p_target)

    if n == 0:
        return 0.

    # Make target point lists clockwise
    p_target = make_clockwise(p_target)

    # Compute length distribution
    distr_pred =  torch.linspace(0., 1., n).to(p_pred.device)
    distr_target = get_length_distribution(p_target, normalize=True)
    d = torch.cdist(distr_pred.unsqueeze(-1), distr_target.unsqueeze(-1))
    matching = d.argmin(dim=-1)
    p_target_sub = p_target[matching]

    # EMD
    i = np.argmin([torch.norm(p_pred - reorder(p_target_sub, i), dim=-1).mean() for i in range(n)])

    losses = torch.norm(p_pred - reorder(p_target_sub, i), dim=-1)

    if first_point_weight:
        weights = torch.ones_like(losses)
        weights[0] = 10.
        losses = losses * weights

    if return_matched_indices:
        return losses.mean(), (p_pred, p_target, reorder(matching, i))

    return losses.mean()
