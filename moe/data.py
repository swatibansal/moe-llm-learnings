"""Character-level dataset: last 5% of the text is held out for validation."""
import numpy as np


class CharData:
    def __init__(self, path: str, block: int, val_frac: float = 0.05):
        text = open(path, encoding="utf-8").read()
        self.chars = sorted(set(text))
        self.vocab = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        ids = np.array([self.stoi[c] for c in text], dtype=np.int32)
        n_val = int(len(ids) * val_frac)
        self.train_ids, self.val_ids = ids[:-n_val], ids[-n_val:]
        self.block = block
        print(f"data: {len(ids):,} chars, vocab {self.vocab}, train {len(self.train_ids):,} / val {len(self.val_ids):,}")

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def _batch(self, ids, starts):
        x = np.stack([ids[s: s + self.block] for s in starts])
        y = np.stack([ids[s + 1: s + self.block + 1] for s in starts])
        return x, y

    def train_batch(self, rng: np.random.Generator, batch: int):
        starts = rng.integers(0, len(self.train_ids) - self.block - 1, size=batch)
        return self._batch(self.train_ids, starts)

    def val_batches(self, batch: int, n_batches: int):
        """Deterministic, evenly spaced windows so every eval is comparable."""
        n = len(self.val_ids) - self.block - 1
        starts = np.linspace(0, n, batch * n_batches, dtype=np.int64)
        for i in range(n_batches):
            yield self._batch(self.val_ids, starts[i * batch:(i + 1) * batch])
