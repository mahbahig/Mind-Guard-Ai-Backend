"""
model/layers.py — FeedForward, EncoderLayer, DecoderLayer
===========================================================
Position-wise feed-forward network and single encoder/decoder layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ai.chatbot.chat_src.model.attention import MultiHeadAttention


# ── 3. Feed-Forward Network ────────────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applied independently to each position:
      FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2

    d_ff is typically 4x d_model. We use 2x for our small model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ── 4. Encoder Layer ───────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    """
    Single encoder layer:
      1. Multi-Head Self-Attention   (every token attends to every token)
      2. Add & Norm
      3. Feed-Forward
      4. Add & Norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff        = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention + residual
        attn_out = self.self_attn(x, x, x, src_mask)
        x        = self.norm1(x + self.dropout(attn_out))

        # Feed-forward + residual
        ff_out = self.ff(x)
        x      = self.norm2(x + self.dropout(ff_out))
        return x


# ── 5. Decoder Layer ───────────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    """
    Single decoder layer:
      1. Masked Multi-Head Self-Attention  (title tokens can't see future words)
      2. Add & Norm
      3. Multi-Head Cross-Attention        (attend to encoder output = source message)
      4. Add & Norm
      5. Feed-Forward
      6. Add & Norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x:         torch.Tensor,
        enc_out:   torch.Tensor,
        src_mask:  torch.Tensor = None,
        tgt_mask:  torch.Tensor = None,
    ) -> torch.Tensor:
        # 1. Masked self-attention (decoder attends to itself, no future)
        self_out = self.self_attn(x, x, x, tgt_mask)
        x        = self.norm1(x + self.dropout(self_out))

        # 2. Cross-attention (decoder queries, encoder keys/values)
        cross_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x         = self.norm2(x + self.dropout(cross_out))

        # 3. Feed-forward
        ff_out = self.ff(x)
        x      = self.norm3(x + self.dropout(ff_out))
        return x
