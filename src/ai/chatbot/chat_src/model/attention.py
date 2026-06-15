"""
model/attention.py — Multi-Head Attention
==========================================
Scaled Dot-Product Multi-Head Attention module.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 2. Multi-Head Attention ────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Scaled Dot-Product Multi-Head Attention.

    Splits d_model into n_heads parallel attention heads.
    Each head learns different aspects of token relationships.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Then concatenate all heads and project back to d_model.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads   # dimension per head

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, d_model] → [batch, n_heads, seq, d_k]"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)   # [batch, n_heads, seq, d_k]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, n_heads, seq, d_k] → [batch, seq, d_model]"""
        batch, _, seq, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query:  torch.Tensor,
        key:    torch.Tensor,
        value:  torch.Tensor,
        mask:   torch.Tensor = None,
    ) -> torch.Tensor:
        """
        query, key, value: [batch, seq, d_model]
        mask: optional [batch, 1, 1, seq] or [batch, 1, seq, seq]
        """
        Q = self._split_heads(self.W_q(query))   # [batch, heads, seq_q, d_k]
        K = self._split_heads(self.W_k(key))     # [batch, heads, seq_k, d_k]
        V = self._split_heads(self.W_v(value))   # [batch, heads, seq_k, d_k]

        # Scaled dot-product attention
        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # [batch, heads, seq_q, seq_k]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn    = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)          # [batch, heads, seq_q, d_k]

        # Merge heads and project
        out = self._merge_heads(context)         # [batch, seq_q, d_model]
        return self.W_o(out)
