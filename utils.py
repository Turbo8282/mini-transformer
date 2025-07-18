import torch

def create_pad_mask(seq, pad_idx):
    # (B, S) → (B, 1, 1, S)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_subsequent_mask(size):
    # (S, S)
    return torch.tril(torch.ones(size, size)).bool()

def create_mask(src, tgt, src_pad_idx, tgt_pad_idx):
    B, src_len = src.size()
    B, tgt_len = tgt.size()

    # (B, 1, 1, src_len)
    src_mask = create_pad_mask(src, src_pad_idx)

    # (B, 1, tgt_len, 1)
    tgt_pad_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(3)

    # (1, 1, tgt_len, tgt_len)
    tgt_sub_mask = create_subsequent_mask(tgt_len).to(tgt.device).unsqueeze(0).unsqueeze(0)

    # Combine: (B, 1, tgt_len, tgt_len)
    tgt_mask = tgt_pad_mask & tgt_sub_mask

    return src_mask, tgt_mask
