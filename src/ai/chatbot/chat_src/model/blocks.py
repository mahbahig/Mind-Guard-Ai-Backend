"""
model/blocks.py — PositionalEncoding, Encoder, Decoder
========================================================
Positional encoding and full encoder/decoder stacks.
"""

import math
import torch
import torch.nn as nn

from src.ai.chatbot.chat_src.model.layers import EncoderLayer, DecoderLayer


# ── 1. Positional Encoding ─────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Injects position information into token embeddings using fixed sine/cosine waves.

    For position pos and dimension i:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This allows the model to learn relative positions through attention.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the fixed positional encoding matrix [max_len, d_model]
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()          # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                   # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)   # even dims → sine
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  dims → cosine

        pe = pe.unsqueeze(0)                            # [1, max_len, d_model]
        self.register_buffer("pe", pe)                  # not a parameter — fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── 6. Full Encoder ────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        n_heads:    int,
        n_layers:   int,
        d_ff:       int,
        max_len:    int,
        dropout:    float,
        pad_idx:    int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.scale = math.sqrt(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(src) * self.scale
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ── 7. Full Decoder ────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        n_heads:    int,
        n_layers:   int,
        d_ff:       int,
        max_len:    int,
        dropout:    float,
        pad_idx:    int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.scale = math.sqrt(d_model)

    def forward(
        self,
        tgt:      torch.Tensor,
        enc_out:  torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt) * self.scale
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x
