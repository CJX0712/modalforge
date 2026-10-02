"""Tests for core types, config and errors."""

from modalforge.core.config import Config, config_from_env
from modalforge.core.errors import (
    E100ConfigError,
    ModalForgeError,
)
from modalforge.core.types import AlignedCorpus, Embedding


def test_embedding_shape_and_dtype():
    e = Embedding(matrix=__import__("numpy").zeros((5, 3)), modality="image")
    assert e.n_items == 5
    assert e.dim == 3
    assert e.matrix.dtype == "float32"


def test_embedding_requires_2d():
    import numpy as np

    try:
        Embedding(matrix=np.zeros((5,)), modality="image")
        assert False, "should reject 1D"
    except ValueError:
        pass


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("MODALFORGE_SEED", "7")
    monkeypatch.setenv("MODALFORGE_N_TRAIN", "400")
    monkeypatch.setenv("MODALFORGE_SHAPE_VOCAB", '["circle","square"]')
    cfg = config_from_env(Config())
    assert cfg.seed == 7
    assert cfg.n_train == 400
    assert tuple(cfg.shape_vocab) == ("circle", "square")


def test_config_env_bad_value_raises(monkeypatch):
    monkeypatch.setenv("MODALFORGE_SEED", "not-an-int")
    try:
        config_from_env(Config())
        assert False, "should raise"
    except E100ConfigError:
        pass


def test_error_hierarchy():
    assert issubclass(E100ConfigError, ModalForgeError)


def test_aligned_corpus_roundtrip(tmp_path):
    import numpy as np

    from modalforge.data.loaders import load_corpus, save_corpus

    corpus = AlignedCorpus(
        item_ids=[0, 1],
        images=np.zeros((2, 8, 8, 3), dtype="uint8"),
        captions=["a red circle", "a blue square"],
    )
    p = tmp_path / "corpus"
    save_corpus(corpus, str(p))
    loaded = load_corpus(str(p))
    assert loaded.captions == corpus.captions
    assert loaded.images.shape == corpus.images.shape
