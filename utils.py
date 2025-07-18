import torch

def create_pad_mask(seq, pad_idx):
    # seq: (batch_size, seq_len)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # shape: (B, 1, 1, S)

def create_subsequent_mask(size):
    # Prevent decoder from seeing future tokens
    return torch.tril(torch.ones(size, size)).bool()  # (S, S)

def create_mask(src, tgt, src_pad_idx, tgt_pad_idx):
    src_mask = create_pad_mask(src, src_pad_idx)         # (B, 1, 1, src_len)
    tgt_pad_mask = create_pad_mask(tgt, tgt_pad_idx)     # (B, 1, 1, tgt_len)
    tgt_sub_mask = create_subsequent_mask(tgt.size(1)).to(tgt.device)  # (tgt_len, tgt_len)

    tgt_mask = tgt_pad_mask & tgt_sub_mask.unsqueeze(0).unsqueeze(0)   # (B, 1, tgt_len, tgt_len)

    return src_mask, tgt_mask
