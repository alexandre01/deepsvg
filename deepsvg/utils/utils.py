import torch


def linear(a, b, x, min_x, max_x):
    """
    b             ___________
                /|
               / |
    a  _______/  |
              |  |
           min_x max_x
    """
    return a + min(max((x - min_x) / (max_x - min_x), 0), 1) * (b - a)


def batchify(data, device):
    return (d.unsqueeze(0).to(device) for d in data)


def _make_seq_first(*args):
    # N, G, S, ... -> S, G, N, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(2, 1, 0, *range(3, arg.dim())) if arg is not None else None
    return (*(arg.permute(2, 1, 0, *range(3, arg.dim())) if arg is not None else None for arg in args),)


def _make_batch_first(*args):
    # S, G, N, ... -> N, G, S, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(2, 1, 0, *range(3, arg.dim())) if arg is not None else None
    return (*(arg.permute(2, 1, 0, *range(3, arg.dim())) if arg is not None else None for arg in args),)


def _pack_group_batch(*args):
    # S, G, N, ... -> S, G * N, ...
    if len(args) == 1:
        arg, = args
        return arg.reshape(arg.size(0), arg.size(1) * arg.size(2), *arg.shape[3:]) if arg is not None else None
    return (*(arg.reshape(arg.size(0), arg.size(1) * arg.size(2), *arg.shape[3:]) if arg is not None else None for arg in args),)


def _unpack_group_batch(N, *args):
    # S, G * N, ... -> S, G, N, ...
    if len(args) == 1:
        arg, = args
        return arg.reshape(arg.size(0), -1, N, *arg.shape[2:]) if arg is not None else None
    return (*(arg.reshape(arg.size(0), -1, N, *arg.shape[2:]) if arg is not None else None for arg in args),)
