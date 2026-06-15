"""
model/transformer.py — TitleTransformer
=========================================
Full Encoder-Decoder Transformer for title generation.
"""

import torch
import torch.nn as nn

from src.ai.chatbot.chat_src.model.blocks import Encoder, Decoder


# ── 8. Full Seq2Seq Transformer ────────────────────────────────────────────────
class TitleTransformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for title generation.

    Input:  tokenized user message   [batch, src_len]
    Output: logits over vocabulary   [batch, tgt_len, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 128,
        n_heads:    int   = 4,
        n_layers:   int   = 3,
        d_ff:       int   = 256,
        max_len:    int   = 128,
        dropout:    float = 0.1,
        pad_idx:    int   = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(vocab_size, d_model, n_heads, n_layers,
                               d_ff, max_len, dropout, pad_idx)
        self.decoder = Decoder(vocab_size, d_model, n_heads, n_layers,
                               d_ff, max_len, dropout, pad_idx)

        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights (Xavier uniform — standard for transformers)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Mask padding tokens in the source. [batch, 1, 1, src_len]"""
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Mask future tokens in the target (causal mask).
        Combines padding mask + look-ahead mask.
        [batch, 1, tgt_len, tgt_len]
        """
        tgt_len  = tgt.size(1)
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)          # [batch,1,1,tgt_len]
        causal   = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        return pad_mask & causal                                              # [batch,1,tgt_len,tgt_len]

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        src: [batch, src_len]   — tokenized input message
        tgt: [batch, tgt_len]   — tokenized title (teacher forcing during training)

        Returns logits: [batch, tgt_len, vocab_size]
        """
        src_mask = self._make_src_mask(src)
        tgt_mask = self._make_tgt_mask(tgt)

        enc_out  = self.encoder(src, src_mask)
        dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        logits   = self.output_proj(dec_out)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src:       torch.Tensor,
        bos_idx:   int,
        eos_idx:   int,
        max_steps: int = 12,
    ) -> list[int]:
        """
        Greedy decoding — generates one token at a time.

        At each step:
          1. Encoder runs once on the source (cached)
          2. Decoder takes all generated tokens so far as input
          3. Pick the highest-probability next token
          4. Stop at [EOS] or max_steps

        Args:
            src:      [1, src_len] tokenized input message
            bos_idx:  beginning-of-sequence token ID
            eos_idx:  end-of-sequence token ID
            max_steps: maximum title tokens to generate

        Returns:
            List of token IDs (the generated title, without BOS/EOS)
        """
        self.eval()
        src_mask = self._make_src_mask(src)
        enc_out  = self.encoder(src, src_mask)

        # Start with [BOS]
        generated = [bos_idx]
        seen_tokens: set[int] = set()   # no-repeat penalty

        for _ in range(max_steps):
            tgt      = torch.tensor([generated], dtype=torch.long, device=src.device)
            tgt_mask = self._make_tgt_mask(tgt)
            dec_out  = self.decoder(tgt, enc_out, src_mask, tgt_mask)
            logits   = self.output_proj(dec_out)[0, -1]    # [vocab]

            # Suppress already-generated tokens (no-repeat)
            for tok in seen_tokens:
                logits[tok] = float("-inf")
            # Suppress special tokens (PAD=0, UNK=1, BOS=2)
            logits[0] = float("-inf")
            logits[1] = float("-inf")
            logits[2] = float("-inf")

            next_tok = logits.argmax().item()              # greedy pick

            if next_tok == eos_idx:
                break
            generated.append(next_tok)
            seen_tokens.add(next_tok)

        return generated[1:]   # strip BOS
