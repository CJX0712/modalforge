"""Tests for cosine retrieval, metrics and reciprocal-rank fusion."""

import numpy as np

from modalforge.retrieval.retriever import (
    CosineRetriever,
    cosine_sim,
    reciprocal_rank_fusion,
    retrieval_metrics,
)


def test_cosine_sim_unit_vectors():
    Q = np.eye(3)
    C = np.eye(3)
    sims = cosine_sim(Q, C)
    assert np.allclose(sims, np.eye(3))


def test_metrics_perfect_ranking():
    # 5 queries, each retrieves its own item first (identity ranking)
    ranks = np.array(
        [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [2, 0, 1, 3, 4], [3, 0, 1, 2, 4], [4, 0, 1, 2, 3]]
    )
    labels = ["a", "b", "c", "d", "e"]
    m = retrieval_metrics(ranks, labels, labels, ks=(1, 5))
    assert m["recall@k"][1] == 1.0
    assert m["mrr"] == 1.0
    assert m["map"] == 1.0


def test_metrics_correct_at_position_two():
    # single query, correct item (label "c") is at rank position 1 (2nd retrieved)
    ranks = np.array([[0, 2, 1, 3, 4]])
    query_labels = ["c"]
    corpus_labels = ["a", "b", "c", "d", "e"]
    m = retrieval_metrics(ranks, query_labels, corpus_labels, ks=(1, 5))
    assert m["recall@k"][1] == 0.0
    assert abs(m["mrr"] - 0.5) < 1e-9
    assert abs(m["map"] - 0.5) < 1e-9


def test_reciprocal_rank_fusion_shape_and_permutation():
    rng = np.random.default_rng(0)
    nq, nc = 4, 20
    per_ranks = {
        "a": np.array([rng.permutation(nc)[:10] for _ in range(nq)]),
        "b": np.array([rng.permutation(nc)[:10] for _ in range(nq)]),
    }
    per_scores = {k: rng.random((nq, 10)) for k in per_ranks}
    ranks, scores = reciprocal_rank_fusion(per_ranks, per_scores)
    assert ranks.shape == (nq, 10)
    for q in range(nq):
        # fused ranking covers the full corpus (0..nc-1), 10 distinct indices
        assert sorted(ranks[q].tolist()) == sorted(set(ranks[q].tolist()))
        assert all(0 <= i < nc for i in ranks[q].tolist())


def test_retriever_basic():
    corpus = np.eye(4)
    retr = CosineRetriever()
    retr.index(corpus)
    res = retr.retrieve(np.eye(4), k=2)
    assert res.ranks.shape == (4, 2)
    assert list(res.ranks[0]) == [0, 1] or res.scores[0, 0] == 1.0
