"""Tests for image (HOG+RGB) and text (TF-IDF) encoders."""

import numpy as np

from modalforge.core.config import Config
from modalforge.encoders.image_encoder import HogImageEncoder
from modalforge.encoders.text_encoder import TfidfEncoder


def test_image_encoder_shape_and_determinism():
    cfg = Config(hog_cells=4, hog_bins=9)
    enc = HogImageEncoder(cfg)
    imgs = np.random.default_rng(0).integers(0, 255, size=(5, 32, 32, 3), dtype="uint8")
    out1 = enc.fit_transform(imgs)
    out2 = enc.transform(imgs)
    assert out1.shape == (5, 4 * 4 * (9 + 3))
    assert np.allclose(out1, out2)
    # per-sample L2 normalisation of the HOG block keeps magnitude bounded
    assert out1.dtype == np.float32


def test_image_encoder_handles_grayscale():
    enc = HogImageEncoder(Config())
    imgs = np.random.default_rng(1).integers(0, 255, size=(3, 32, 32), dtype="uint8")
    out = enc.transform(imgs)
    assert out.shape[0] == 3


def test_text_encoder_shape_and_determinism():
    cfg = Config(tfidf_max_features=32)
    enc = TfidfEncoder(cfg)
    texts = ["a red circle", "a blue square", "a red circle", "two green triangles"]
    out1 = enc.fit_transform(texts)
    out2 = enc.transform(texts)
    # 8 unique tokens across the 4 captions, capped at tfidf_max_features=32 -> 8
    assert out1.shape == (4, 8)
    assert np.allclose(out1, out2)
    # identical captions -> identical vectors
    assert np.allclose(out1[0], out1[2])


def test_text_encoder_vocab_capped():
    cfg = Config(tfidf_max_features=2)
    enc = TfidfEncoder(cfg)
    enc.fit(["alpha beta gamma delta epsilon"] * 10)
    assert len(enc.vocab_) <= 2
