import shutil
import torch
import torch.nn as nn
import os
import random
import numpy as np
import glob


def save_ckpt(checkpoint_dir, model, cfg=None, optimizer=None, scheduler_lr=None, scheduler_warmup=None,
              stats=None, train_vars=None):
    if is_multi_gpu(model):
        model = model.module

    state = {
        "model": model.state_dict()
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler_lr is not None:
        state["scheduler_lr"] = scheduler_lr.state_dict()
    if scheduler_warmup is not None:
        state["scheduler_warmup"] = scheduler_warmup.state_dict()
    if cfg is not None:
        state["cfg"] = cfg.to_dict()
    if stats is not None:
        state["stats"] = stats.to_dict()
    if train_vars is not None:
        state["train_vars"] = train_vars.to_dict()

    checkpoint_path = os.path.join(checkpoint_dir, "{:06d}.pth.tar".format(stats.step))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, checkpoint_path)

    if stats.is_best():
        best_model_path = os.path.join(checkpoint_dir, "best.pth.tar")
        shutil.copyfile(checkpoint_path, best_model_path)


def save_ckpt_list(checkpoint_dir, model, cfg=None, optimizers=None, scheduler_lrs=None, scheduler_warmups=None,
              stats=None, train_vars=None):
    if is_multi_gpu(model):
        model = model.module

    state = {
        "model": model.state_dict()
    }

    if optimizers is not None:
        state["optimizers"] = [optimizer.state_dict() if optimizer is not None else optimizer for optimizer in optimizers]
    if scheduler_lrs is not None:
        state["scheduler_lrs"] = [scheduler_lr.state_dict() if scheduler_lr is not None else scheduler_lr for scheduler_lr in scheduler_lrs]
    if scheduler_warmups is not None:
        state["scheduler_warmups"] = [scheduler_warmup.state_dict() if scheduler_warmup is not None else None for scheduler_warmup in scheduler_warmups]
    if cfg is not None:
        state["cfg"] = cfg.to_dict()
    if stats is not None:
        state["stats"] = stats.to_dict()
    if train_vars is not None:
        state["train_vars"] = train_vars.to_dict()

    checkpoint_path = os.path.join(checkpoint_dir, "{:06d}.pth.tar".format(stats.step))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, checkpoint_path)

    if stats.is_best():
        best_model_path = os.path.join(checkpoint_dir, "best.pth.tar")
        shutil.copyfile(checkpoint_path, best_model_path)


def load_ckpt(checkpoint_dir, model, cfg=None, optimizer=None, scheduler_lr=None, scheduler_warmup=None,
              stats=None, train_vars=None):
    if not os.path.exists(checkpoint_dir):
        return False

    if os.path.isfile(checkpoint_dir):
        checkpoint_path = checkpoint_dir
    else:
        ckpts_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "./[0-9]*.pth.tar")))
        if not ckpts_paths:
            return False
        checkpoint_path = ckpts_paths[-1]

    state = torch.load(checkpoint_path)

    if is_multi_gpu(model):
        model = model.module
    model.load_state_dict(state["model"], strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler_lr is not None:
        scheduler_lr.load_state_dict(state["scheduler_lr"])
    if scheduler_warmup is not None:
        scheduler_warmup.load_state_dict(state["scheduler_warmup"])
    if cfg is not None:
        cfg.load_dict(state["cfg"])
    if stats is not None:
        stats.load_dict(state["stats"])
    if train_vars is not None:
        train_vars.load_dict(state["train_vars"])

    return True


def load_ckpt_list(checkpoint_dir, model, cfg=None, optimizers=None, scheduler_lrs=None, scheduler_warmups=None,
              stats=None, train_vars=None):
    if not os.path.exists(checkpoint_dir):
        return False

    if os.path.isfile(checkpoint_dir):
        checkpoint_path = checkpoint_dir
    else:
        ckpts_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "./[0-9]*.pth.tar")))
        if not ckpts_paths:
            return False
        checkpoint_path = ckpts_paths[-1]

    state = torch.load(checkpoint_path)

    if is_multi_gpu(model):
        model = model.module
    model.load_state_dict(state["model"], strict=False)

    for optimizer, scheduler_lr, scheduler_warmup, optimizer_sd, scheduler_lr_sd, scheduler_warmups_sd in zip(optimizers, scheduler_lrs, scheduler_warmups, state["optimizers"], state["scheduler_lrs"], state["scheduler_warmups"]):
        if optimizer is not None and optimizer_sd is not None:
            optimizer.load_state_dict(optimizer_sd)
        if scheduler_lr is not None and scheduler_lr_sd is not None:
            scheduler_lr.load_state_dict(scheduler_lr_sd)
        if scheduler_warmup is not None and scheduler_warmups_sd is not None:
            scheduler_warmup.load_state_dict(scheduler_warmups_sd)
    if cfg is not None and state["cfg"] is not None:
        cfg.load_dict(state["cfg"])
    if stats is not None and state["stats"] is not None:
        stats.load_dict(state["stats"])
    if train_vars is not None and state["train_vars"] is not None:
        train_vars.load_dict(state["train_vars"])

    return True


def load_model(checkpoint_path, model):
    state = torch.load(checkpoint_path)

    if is_multi_gpu(model):
        model = model.module
    model.load_state_dict(state["model"], strict=False)


def is_multi_gpu(model):
    return isinstance(model, nn.DataParallel)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_sequence(sequences, batch_first=False, padding_value=0, max_len=None):
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]

    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def set_seed(_seed=42):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    os.environ['PYTHONHASHSEED'] = str(_seed)


def infinite_range(start_idx=0):
    while True:
        yield start_idx
        start_idx += 1
