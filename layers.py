import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    # Initialise
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Final linear layer to combine heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention()

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()

        # Linear projection
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Reshape for multi-heads: (batch, heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention to each head
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Concatenate heads back: (batch, seq_len, embed_dim)
        out = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Final projection
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        # Create a matrix of shape (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)     # even indices
        pe[:, 1::2] = torch.cos(position * div_term)     # odd indices

        pe = pe.unsqueeze(0)  # shape: (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual + normalization
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual + normalization
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Token + position embeddings
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through stacked encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        # Masked self-attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (queries from decoder, keys/values from encoder)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Masked self-attention (decoder to itself, but can’t see future)
        x2 = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(x2))

        # Cross-attention (decoder attends to encoder output)
        x2 = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        x = self.norm2(x + self.dropout2(x2))

        # Feed-forward
        x2 = self.ffn(x)
        x = self.norm3(x + self.dropout3(x2))

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Token + positional embeddings
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through stacked decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        return x
