import torch
import torch.nn as nn
from layers import Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=512, num_heads=8, ff_dim=2048,
                 num_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, dropout)

        self.output_projection = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [batch_size, src_seq_len]
        # tgt: [batch_size, tgt_seq_len]

        enc_output = self.encoder(src, mask=src_mask)             # (B, src_len, D)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)  # (B, tgt_len, D)

        logits = self.output_projection(dec_output)  # (B, tgt_len, vocab_size)
        return logits
