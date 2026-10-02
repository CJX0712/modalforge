"""MMFuse — confidence-gated cross-modal fusion retrieval (flagship).

For a given direction (image→text or text→image) MMFuse retrieves in *every*
available aligner's shared space, then combines the ranked lists with
:func:`reciprocal_rank_fusion`. An aligner that is "uncertain" on a query
(top-1 cosine below ``min_conf``) is down-weighted; if *no* aligner is confident
the fusion falls back to the single most-confident aligner, guaranteeing the
fusion is never dominated by a uniformly useless member.

Innovation vs. a plain RRF: the **per-query confidence gate** stops a badly
mis-calibrated space from polluting the fusion, while the **non-inferiority
fallback** keeps the ensemble honest when all members are weak.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..core.types import RetrievalResult
from ..retrieval.retriever import cosine_sim, reciprocal_rank_fusion


class MMFuse:
    """Confidence-gated cross-modal fusion retriever."""

    name = "mmfuse"

    def __init__(self, min_conf: float = 0.35, topk: int = 10) -> None:
        self.min_conf = float(min_conf)
        self.topk = int(topk)

    def retrieve(
        self,
        spaces: Dict[str, Tuple[np.ndarray, np.ndarray]],
        direction: str,
        k: int,
    ) -> RetrievalResult:
        """Retrieve using fused rankings.

        ``spaces`` maps aligner name -> ``(Z_img, Z_txt)`` in a shared space.
        ``direction`` is ``"img2txt"`` (image query, text corpus) or
        ``"txt2img"`` (text query, image corpus).
        """
        if direction not in ("img2txt", "txt2img"):
            raise ValueError(f"unknown direction {direction!r}")
        if not spaces:
            raise ValueError("MMFuse needs at least one aligned space")
        k = max(k, self.topk)
        per_ranks: Dict[str, np.ndarray] = {}
        per_scores: Dict[str, np.ndarray] = {}
        for name, (Zx, Zy) in spaces.items():
            Q = Zx if direction == "img2txt" else Zy
            C = Zy if direction == "img2txt" else Zx
            sims = cosine_sim(Q, C)  # (nq, nc)
            kk = min(k, sims.shape[1])
            order = np.argsort(-sims, axis=1)[:, :kk]
            scored = np.take_along_axis(sims, order, axis=1)
            per_ranks[name] = order
            per_scores[name] = scored
        ranks, scores = reciprocal_rank_fusion(per_ranks, per_scores, min_conf=self.min_conf)
        return RetrievalResult(ranks=ranks, scores=scores)
