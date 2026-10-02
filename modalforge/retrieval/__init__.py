"""Retrieval package: cosine retrieval + cross-modal evaluation metrics."""

from .retriever import (
    CosineRetriever,
    cosine_sim,
    reciprocal_rank_fusion,
    retrieval_metrics,
)

__all__ = [
    "CosineRetriever",
    "cosine_sim",
    "reciprocal_rank_fusion",
    "retrieval_metrics",
]
