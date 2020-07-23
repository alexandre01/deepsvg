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


def svg_align_loss(p_pred, p_target, align_start_both=False, align_start=False, align_furthest_point=False, proportional_length=True, interpolate=False, return_matched_indices=False,
                   add_unmatched_length=False, align_ratio=1.0, pred_uniform_distr=False, make_pred_clockwise=True, first_point_weight=False):
    """
    Args:
        p_pred: Predicted list of points. Shape [N, 2]
        p_target: Target list of points. Shape [M, 2]
        align_start: If False, alignment is done using the closest pair of points. If True, starting point in p_pred is used.
        proportional_length: Whether to normalize the length distribution in pred and target or not.
        interpolate: Whether to use weighted loss using the closest 2 target points instead of closest target point.
    """
    n, m = len(p_pred), len(p_target)

    if n == 0:
        return 0.

    # Make points lists clockwise
    if make_pred_clockwise:
        p_pred = make_clockwise(p_pred)
    p_target = make_clockwise(p_target)

    # Align points lists
    if align_start_both:
        i, j = 0, 0
    elif align_start:
        d = torch.cdist(p_pred[0].unsqueeze(0), p_target)
        i, j = 0, torch.argmin(d)
    elif align_furthest_point:
        # Min max
        d = torch.cdist(p_pred, p_target)
        vals, inds = d.max(dim=-1)
        _, i = vals.min(dim=0)
        j = inds[i]
    else:
        d = torch.cdist(p_pred, p_target)
        min_index = torch.argmin(d)
        i, j = min_index // m, min_index % m
    p_pred, p_target = reorder(p_pred, i), reorder(p_target, j)

    # Compute length distribution
    if pred_uniform_distr:
        l = 1. if proportional_length else get_length(p_pred)
        distr_pred = torch.linspace(0., l, n).to(p_pred.device)
    else:
        distr_pred = get_length_distribution(p_pred, normalize=proportional_length)

    distr_target = get_length_distribution(p_target, normalize=proportional_length)

    if interpolate:
        # Match each pred point to a pair of target points
        inds, weights = align.match_distr(distr_pred, distr_target)

        losses = torch.norm(p_pred.unsqueeze(dim=1) - p_target[inds], dim=-1)
        losses = losses * weights

        matched_indices = inds[:, 0]
    else:
        # Simply select the closest matching target index
        distr_d = torch.cdist(distr_pred.unsqueeze(-1), distr_target.unsqueeze(-1))
        matched_indices = distr_d.argmin(dim=-1)

        losses = torch.norm(p_pred - p_target[matched_indices], dim=-1)

    # Align ratio
    n_points = int(align_ratio * len(losses))
    losses = losses[:n_points]

    if first_point_weight:
        weights = torch.ones_like(losses)
        weights[0] = 10.
        losses = losses * weights

    if return_matched_indices:
        return losses.mean(), (p_pred, p_target, matched_indices)

    return losses.mean()


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
