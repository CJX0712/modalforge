"""Shared data structures for ModalForge.

Contracts (keep cross-module fairness):
* An *embedding* is always a 2D float32 array ``(n_items, dim)``.
* An *aligner* always returns two embeddings ``(Z_img, Z_txt)`` in a **shared**
  space where the i-th row of ``Z_img`` and the i-th row of ``Z_txt`` describe
  the same item. Retrieval is therefore symmetric and comparable across methods.
* Retrieval scores are cosine similarities in ``[-1, 1]``; larger == more similar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Embedding:
    """A modality-specific embedding matrix plus provenance metadata."""

    matrix: np.ndarray  # (n_items, dim) float32
    modality: str  # "image" | "text"
    dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.matrix = np.ascontiguousarray(self.matrix, dtype=np.float32)
        if self.matrix.ndim != 2:
            raise ValueError("Embedding.matrix must be 2D (n_items, dim)")
        self.dim = int(self.matrix.shape[1])

    @property
    def n_items(self) -> int:
        return int(self.matrix.shape[0])


@dataclass
class AlignedPair:
    """One image + its matching caption, sharing ``item_id``."""

    item_id: int
    image: np.ndarray  # (H, W) or (H, W, 3) uint8
    caption: str


@dataclass
class AlignedCorpus:
    """A corpus of aligned image/text pairs."""

    item_ids: List[int]
    images: np.ndarray  # (n, H, W) or (n, H, W, 3) uint8
    captions: List[str]

    def __len__(self) -> int:
        return len(self.item_ids)

    def to_pairs(self) -> List[AlignedPair]:
        out: List[AlignedPair] = []
        for i, iid in enumerate(self.item_ids):
            out.append(AlignedPair(iid, self.images[i], self.captions[i]))
        return out


@dataclass
class RetrievalResult:
    """Ranked retrieval results for a batch of queries."""

    # ranks[k][q] = item_id ranked at position k for query q  (int array (k, n_q))
    ranks: np.ndarray
    # scores[k][q] = cosine similarity of that ranked item      (float array (k, n_q))
    scores: np.ndarray

    @property
    def k(self) -> int:
        return int(self.ranks.shape[0])


@dataclass
class BenchmarkRow:
    """One row of the cross-aligner benchmark table."""

    aligner: str
    available: bool
    recall_at_1_img2txt: float
    recall_at_5_img2txt: float
    map_img2txt: float
    mrr_img2txt: float
    recall_at_1_txt2img: float
    recall_at_5_txt2img: float
    map_txt2img: float
    mrr_txt2img: float
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "aligner": self.aligner,
            "available": self.available,
            "recall@1_img2txt": round(self.recall_at_1_img2txt, 4),
            "recall@5_img2txt": round(self.recall_at_5_img2txt, 4),
            "map_img2txt": round(self.map_img2txt, 4),
            "mrr_img2txt": round(self.mrr_img2txt, 4),
            "recall@1_txt2img": round(self.recall_at_1_txt2img, 4),
            "recall@5_txt2img": round(self.recall_at_5_txt2img, 4),
            "map_txt2img": round(self.map_txt2img, 4),
            "mrr_txt2img": round(self.mrr_txt2img, 4),
            "notes": self.notes,
        }
