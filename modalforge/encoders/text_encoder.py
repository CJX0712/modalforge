"""Text encoder: TF-IDF over a fixed vocabulary, pure numpy.

Tokeniser handles the simple templated captions (lowercase English words). The
vocabulary is capped at ``tfidf_max_features`` by descending document frequency,
making the output dimension stable and the encoder fully offline.
"""

from __future__ import annotations

import re

import numpy as np

from ..core.config import Config
from ..core.errors import E300EncoderError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class TfidfEncoder:
    """Deterministic TF-IDF encoder (numpy only)."""

    name = "tfidf"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "TfidfEncoder":
        if not texts:
            raise E300EncoderError("cannot fit on empty text list")
        docs = [_tokenize(t) for t in texts]
        df: dict[str, int] = {}
        for d in docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        # keep top-k by document frequency
        ordered = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))
        top = ordered[: self.cfg.tfidf_max_features]
        self.vocab_ = {w: i for i, (w, _) in enumerate(top)}
        n = len(docs)
        idf = np.zeros(len(self.vocab_), dtype=np.float64)
        for w, i in self.vocab_.items():
            idf[i] = np.log((1.0 + n) / (1.0 + df[w])) + 1.0  # smoothed idf
        self.idf_ = idf
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf_ is None:
            raise E300EncoderError("TfidfEncoder.transform called before fit")
        vocab = self.vocab_
        idf = self.idf_
        out = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for r, t in enumerate(texts):
            counts: dict[int, float] = {}
            toks = _tokenize(t)
            for w in toks:
                if w in vocab:
                    counts[vocab[w]] = counts.get(vocab[w], 0.0) + 1.0
            if counts:
                vec = np.zeros(len(vocab), dtype=np.float64)
                for i, c in counts.items():
                    vec[i] = 1.0 + np.log(c)  # sublinear tf
                vec *= idf
                nrm = np.linalg.norm(vec)
                if nrm > 1e-8:
                    vec = vec / nrm
                out[r] = vec.astype(np.float32)
        return out

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
