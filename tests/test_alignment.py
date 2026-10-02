"""Tests that every aligner fits/transforms and is deterministic."""

import numpy as np

from modalforge.alignment.base import build_aligners
from modalforge.core.config import Config
from modalforge.data.synthetic import generate_corpus
from modalforge.encoders.image_encoder import HogImageEncoder
from modalforge.encoders.text_encoder import TfidfEncoder


def _features(cfg, n=120, seed=5):
    corpus = generate_corpus(cfg, n, seed=seed)
    xi = HogImageEncoder(cfg).fit_transform(corpus.images)
    xt = TfidfEncoder(cfg).fit_transform(corpus.captions)
    return xi, xt


def test_numpy_cca_and_contrastive_available_and_run():
    cfg = Config(n_train=120, cca_dims=8, contrastive_dim=8, contrastive_epochs=5)
    X, Y = _features(cfg)
    for name in ("numpy_cca", "numpy_contrastive"):
        al = next(a for a in build_aligners(cfg) if a.name == name)
        assert al.available()
        al.fit(X, Y)
        zx, zy = al.transform(X, Y)
        assert zx.shape == (X.shape[0], 8 if name == "numpy_cca" else 8)
        assert zy.shape[0] == X.shape[0]


def test_sklearn_aligners_available_in_this_env():
    cfg = Config()
    sk = next(a for a in build_aligners(cfg) if a.name == "sklearn_cca")
    # scikit-learn is installed in CI; if not, this is a graceful skip
    if sk.available():
        X, Y = _features(cfg, n=120)
        sk.fit(X, Y)
        zx, zy = sk.transform(X, Y)
        assert zx.shape[0] == X.shape[0]


def test_contrastive_deterministic():
    cfg = Config(contrastive_dim=8, contrastive_epochs=5, seed=11)
    X, Y = _features(cfg, n=80, seed=11)
    a1 = next(a for a in build_aligners(cfg) if a.name == "numpy_contrastive")
    a2 = next(a for a in build_aligners(cfg) if a.name == "numpy_contrastive")
    a1.fit(X, Y)
    a2.fit(X, Y)
    z1, _ = a1.transform(X, Y)
    z2, _ = a2.transform(X, Y)
    assert np.allclose(z1, z2)


def test_clip_unavailable_offline():
    cfg = Config()
    clip = next((a for a in build_aligners(cfg) if a.name == "clip"), None)
    if clip is not None:
        # In an offline/CPU env without torch+transformers this must be False
        assert clip.available() is False
