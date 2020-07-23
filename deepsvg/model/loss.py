import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsvg.difflib.tensor import SVGTensor
from .utils import _get_padding_mask, _get_visibility_mask
from .config import _DefaultConfig


class SVGLoss(nn.Module):
    def __init__(self, cfg: _DefaultConfig):
        super().__init__()

        self.cfg = cfg

        self.args_dim = 2 * cfg.args_dim if cfg.rel_targets else cfg.args_dim + 1

        self.register_buffer("cmd_args_mask", SVGTensor.CMD_ARGS_MASK)

    def forward(self, output, labels, weights):
        loss = 0.
        res = {}

        # VAE
        if self.cfg.use_vae:
            mu, logsigma = output["mu"], output["logsigma"]
            loss_kl = -0.5 * torch.mean(1 + logsigma - mu.pow(2) - torch.exp(logsigma))
            loss_kl = loss_kl.clamp(min=weights["kl_tolerance"])

            loss += weights["loss_kl_weight"] * loss_kl
            res["loss_kl"] = loss_kl

        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        # 2-stage visibility
        if self.cfg.decode_stages == 2:
            visibility_logits = output["visibility_logits"]
            loss_visibility = F.cross_entropy(visibility_logits.reshape(-1, 2), visibility_mask.reshape(-1).long())

            loss += weights["loss_visibility_weight"] * loss_visibility
            res["loss_visibility"] = loss_visibility

        # Commands & args
        tgt_commands, tgt_args, padding_mask = tgt_commands[..., 1:], tgt_args[..., 1:, :], padding_mask[..., 1:]

        mask = self.cmd_args_mask[tgt_commands.long()]

        loss_cmd = F.cross_entropy(command_logits[padding_mask.bool()].reshape(-1, self.cfg.n_commands), tgt_commands[padding_mask.bool()].reshape(-1).long())
        loss_args = F.cross_entropy(args_logits[mask.bool()].reshape(-1, self.args_dim), tgt_args[mask.bool()].reshape(-1).long() + 1)  # shift due to -1 PAD_VAL

        loss += weights["loss_cmd_weight"] * loss_cmd \
                + weights["loss_args_weight"] * loss_args

        res.update({
            "loss": loss,
            "loss_cmd": loss_cmd,
            "loss_args": loss_args
        })

        return res
