"""Synthetic but *aligned* image<->text corpus.

Each item is a small RGB scene (32x32) of ``count`` shapes of a given ``color``
and ``shape`` type, paired with a templated caption such as
``"a photo of 2 red circles"``. Because the image genuinely encodes shape
(edges) and color (RGB) and the caption encodes the same concepts in language,
a good cross-modal aligner can recover the pairing far above the random floor.

Deterministic: same ``seed`` -> identical corpus (no downloads, fully offline).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ..core.config import Config
from ..core.types import AlignedCorpus

_COLOR_RGB = {
    "red": (220, 60, 60),
    "green": (60, 200, 90),
    "blue": (70, 120, 235),
    "yellow": (235, 205, 70),
}
_BG = (244, 244, 244)


def list_concepts(cfg: Config) -> list[tuple]:
    out = []
    for s in cfg.shape_vocab:
        for c in cfg.color_vocab:
            for n in cfg.count_vocab:
                out.append((s, c, n))
    return out


def _caption(shape: str, color: str, count: int) -> str:
    noun = shape if count == 1 else f"{shape}s"
    return f"a photo of {count} {color} {noun}"


def _draw_scene(
    size: int, shape: str, color: str, count: int, rng: np.random.Generator
) -> np.ndarray:
    img = Image.new("RGB", (size, size), _BG)
    draw = ImageDraw.Draw(img)
    rgb = _COLOR_RGB[color]
    positions = _place(count, size, rng)
    for cx, cy in positions:
        r = int(size * 0.18)
        if shape == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
        elif shape == "square":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
        else:  # triangle
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb)
    return np.asarray(img, dtype=np.uint8)


def _place(count: int, size: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    # spread anchors on a loose grid with deterministic jitter
    if count == 1:
        return [(size // 2, size // 2)]
    pts: list[tuple[int, int]] = []
    # simple non-overlapping-ish layout via grid cells
    cells = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.5, 0.5)]
    for i in range(count):
        fx, fy = cells[i % len(cells)]
        jx = rng.integers(-3, 4)
        jy = rng.integers(-3, 4)
        pts.append((int(size * fx) + jx, int(size * fy) + jy))
    return pts


def _sample_concepts(cfg: Config, n: int, rng: np.random.Generator) -> list[tuple]:
    concepts = list_concepts(cfg)
    idx = rng.integers(0, len(concepts), size=n)
    return [concepts[i] for i in idx]


def generate_corpus(cfg: Config, n: int, seed: int | None = None) -> AlignedCorpus:
    """Generate ``n`` aligned (image, caption) pairs."""
    if seed is None:
        seed = cfg.seed
    rng = np.random.default_rng(seed)
    size = cfg.image_size
    concepts = _sample_concepts(cfg, n, rng)
    images = np.empty((n, size, size, 3), dtype=np.uint8)
    captions: list[str] = []
    item_ids: list[int] = []
    for i, (shape, color, count) in enumerate(concepts):
        # per-item sub-seed keeps rendering deterministic but varied
        irng = np.random.default_rng(seed * 1000003 + i)
        images[i] = _draw_scene(size, shape, color, count, irng)
        captions.append(_caption(shape, color, count))
        item_ids.append(i)
    return AlignedCorpus(item_ids=item_ids, images=images, captions=captions)


def generate_train_test(
    cfg: Config, train_seed: int | None = None, test_seed: int | None = None
) -> tuple[AlignedCorpus, AlignedCorpus]:
    """Generate disjoint train and test corpora."""
    tr = generate_corpus(cfg, cfg.n_train, seed=train_seed if train_seed else cfg.seed)
    te = generate_corpus(
        cfg,
        cfg.n_test,
        seed=(test_seed if test_seed else cfg.seed) + 7777,
    )
    # re-index test ids to avoid collision with train ids
    te = AlignedCorpus(
        item_ids=[i + 100000 for i in te.item_ids],
        images=te.images,
        captions=te.captions,
    )
    return tr, te
