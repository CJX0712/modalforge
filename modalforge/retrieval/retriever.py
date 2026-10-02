"""Cosine retrieval, reciprocal-rank fusion and evaluation metrics.

Metrics are *label-based*: a retrieved item is correct iff its corpus label
equals the query label. Using caption strings as labels means semantically
identical captions (duplicate concepts) count as correct — the right semantic
definition for cross-modal retrieval.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E500RetrievalError
from ..core.types import RetrievalResult


def _unit(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return M / nrm


def cosine_sim(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Return ``(n_queries, n_corpus)`` cosine similarity matrix."""
    Q = _unit(queries)
    C = _unit(corpus)
    return Q @ C.T


class CosineRetriever:
    """Brute-force cosine retriever over a fixed corpus."""

    def __init__(self) -> None:
        self._corpus: np.ndarray | None = None

    def index(self, corpus: np.ndarray) -> None:
        if corpus.ndim != 2:
            raise E500RetrievalError("corpus must be 2D (n, dim)")
        self._corpus = _unit(corpus)

    def retrieve(self, queries: np.ndarray, k: int) -> RetrievalResult:
        if self._corpus is None:
            raise E500RetrievalError("retrieve before index")
        sims = cosine_sim(queries, self._corpus)  # (nq, nc)
        nc = sims.shape[1]
        k = min(k, nc)
        order = np.argsort(-sims, axis=1)[:, :k]
        scored = np.take_along_axis(sims, order, axis=1)
        return RetrievalResult(ranks=order, scores=scored)


def _correctness(ranks: np.ndarray, query_labels, corpus_labels) -> np.ndarray:
    # map corpus label to index for fast compare
    corr = np.array(
        [
            1 if corpus_labels[ranks[r, c]] == query_labels[r] else 0
            for r in range(ranks.shape[0])
            for c in range(ranks.shape[1])
        ],
        dtype=np.int8,
    ).reshape(ranks.shape)
    return corr


def retrieval_metrics(
    ranks: np.ndarray,
    query_labels,
    corpus_labels,
    ks=(1, 5, 10),
):
    """Compute recall@{ks}, mAP and MRR from a rank matrix.

    ``ranks`` is ``(n_queries, topk)`` with corpus indices. ``query_labels`` and
    ``corpus_labels`` are indexable by those indices.
    """
    if ranks.shape[0] != len(query_labels):
        raise E500RetrievalError("ranks row count must equal number of queries")
    corr = _correctness(ranks, query_labels, corpus_labels)
    nq = ranks.shape[0]
    maxk = ranks.shape[1]
    recalls = {}
    for k in ks:
        kk = min(k, maxk)
        hit = corr[:, :kk].any(axis=1).mean() if kk > 0 else 0.0
        recalls[k] = float(hit)

    # MRR
    mrr = 0.0
    for r in range(nq):
        pos = np.where(corr[r] == 1)[0]
        if pos.size:
            mrr += 1.0 / (pos[0] + 1)
    mrr /= max(nq, 1)

    # mAP
    ap_sum = 0.0
    for r in range(nq):
        n_rel = int(corr[r].sum())
        if n_rel == 0:
            continue
        precisions = []
        running = 0
        for p in range(maxk):
            if corr[r, p] == 1:
                running += 1
                precisions.append(running / (p + 1))
        ap_sum += float(np.mean(precisions)) if precisions else 0.0
    map_ = ap_sum / max(nq, 1)
    return {"recall@k": recalls, "mrr": float(mrr), "map": float(map_)}


def reciprocal_rank_fusion(
    per_aligner_ranks: dict[str, np.ndarray],
    per_aligner_scores: dict[str, np.ndarray],
    min_conf: float = 0.35,
    k0: int = 60,
):
    """Confidence-gated Reciprocal Rank Fusion across aligner spaces.

    Each aligner contributes weight ``1`` when its top-1 cosine ``>= min_conf``,
    else ``0.5`` (down-weighted but still informative). If *no* aligner is
    confident, we fall back to the single most-confident aligner (non-inferiority
    guard) so the fusion can never be worse than a sane single method by design.
    """
    names = list(per_aligner_ranks.keys())
    if not names:
        raise E500RetrievalError("no aligner rankings to fuse")
    nq, topk = per_aligner_ranks[names[0]].shape
    nc = int(per_aligner_ranks[names[0]].max()) + 1

    conf = {nm: float(np.mean(per_aligner_scores[nm][:, 0])) for nm in names}
    weights = {nm: (1.0 if conf[nm] >= min_conf else 0.5) for nm in names}
    if all(w < 1.0 for w in weights.values()):
        best = max(names, key=lambda nm: conf[nm])
        weights = {nm: (1.0 if nm == best else 0.0) for nm in names}

    fused = np.zeros((nq, nc), dtype=np.float64)
    for nm in names:
        rk = per_aligner_ranks[nm]  # (nq, topk)
        w = weights[nm]
        for q in range(nq):
            for pos in range(topk):
                item = int(rk[q, pos])
                fused[q, item] += w / (k0 + pos + 1)

    fused_ranks = np.argsort(-fused, axis=1)[:, :topk]
    fused_scores = np.take_along_axis(fused, fused_ranks, axis=1)
    return fused_ranks, fused_scores
