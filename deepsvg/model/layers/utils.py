import torch


def to_negative_mask(mask):
    if mask is None:
        return

    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_adj_subsequent_mask(sz):
    mask = torch.diag(torch.ones(sz), diagonal=0) + torch.diag(torch.ones(sz-1), diagonal=-1)

    if sz >= 2:
        mask = mask + torch.diag(torch.ones(sz-2), diagonal=-2)

    return to_negative_mask(mask)


def generate_adj_mask(sz):
    mask = torch.diag(torch.ones(sz), diagonal=0) +\
           torch.diag(torch.ones(sz - 1), diagonal=+1) +\
           torch.diag(torch.ones(sz - 1), diagonal=-1)

    if sz >= 2:
        mask = mask + torch.diag(torch.ones(sz - 2), diagonal=-2) +\
               torch.diag(torch.ones(sz - 2), diagonal=+2)

    return to_negative_mask(mask)
