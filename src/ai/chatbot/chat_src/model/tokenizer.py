"""
model/tokenizer.py — WordTokenizer
=====================================
Simple word-level tokenizer built from training data vocabulary.
"""


# ── 9. Tokenizer ───────────────────────────────────────────────────────────────
class WordTokenizer:
    """
    Simple word-level tokenizer built from training data vocabulary.

    Special tokens:
      [PAD] = 0   padding to equal sequence lengths
      [UNK] = 1   unknown words not seen during training
      [BOS] = 2   begin of sequence
      [EOS] = 3   end of sequence
    """

    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._built = False

    def build(self, texts: list[str], min_freq: int = 1):
        """
        Build vocabulary from a list of texts.
        Words appearing < min_freq times are mapped to [UNK].
        """
        from collections import Counter
        counts = Counter()
        for t in texts:
            counts.update(self._tokenize(t))

        # Start with special tokens
        vocab = self.SPECIAL + [
            w for w, c in counts.most_common() if c >= min_freq
        ]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._built   = True

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase + split on whitespace/punctuation."""
        import re
        text = text.lower().strip()
        text = re.sub(r"([.!?,;:\"'()])", r" \1 ", text)
        return text.split()

    def encode(self, text: str, max_len: int = None) -> list[int]:
        tokens = self._tokenize(text)
        ids    = [self.word2idx.get(t, self.UNK) for t in tokens]
        if max_len:
            ids = ids[:max_len]
        return ids

    def encode_with_bos_eos(self, text: str, max_len: int = None) -> list[int]:
        ids = [self.BOS] + self.encode(text, max_len) + [self.EOS]
        return ids

    def decode(self, ids: list[int]) -> str:
        words = [
            self.idx2word.get(i, "[UNK]")
            for i in ids
            if i not in (self.PAD, self.BOS, self.EOS)
        ]
        return " ".join(words)

    def pad(self, ids: list[int], max_len: int) -> list[int]:
        return ids[:max_len] + [self.PAD] * max(0, max_len - len(ids))

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        import json
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.word2idx = {w: int(i) for w, i in data["word2idx"].items()}
        tok.idx2word = {int(i): w for w, i in tok.word2idx.items()}
        tok._built   = True
        return tok
