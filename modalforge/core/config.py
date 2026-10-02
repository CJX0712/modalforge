"""Configuration with environment overrides.

Every field can be overridden by ``MODALFORGE_<UPPER_FIELD>``. Example:
``MODALFORGE_SEED=7 MODALFORGE_N_TRAIN=400 python -m modalforge.examples.run_demo``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class Config:
    # reproducibility
    seed: int = 20261002

    # corpus
    n_train: int = 480
    n_test: int = 120
    image_size: int = 32  # square, grayscale used internally
    shape_vocab: tuple = ("circle", "square", "triangle")
    color_vocab: tuple = ("red", "green", "blue", "yellow")
    count_vocab: tuple = (1, 2, 3)

    # image encoder (HOG-like cell histogram)
    hog_cells: int = 4  # grid cells per axis -> hog_cells^2 blocks
    hog_bins: int = 9  # orientation bins
    hog_block_norm: bool = True

    # text encoder
    tfidf_max_features: int = 64

    # aligners
    cca_dims: int = 16  # shared-space dimensionality for CCA / PLS
    contrastive_dim: int = 24
    contrastive_lr: float = 0.05
    contrastive_epochs: int = 60
    contrastive_temperature: float = 0.2
    clip_name: str = "openai/clip-vit-base-patch32"

    # retrieval / fusion
    recall_ks: tuple = (1, 5, 10)
    fusion_min_conf: float = 0.35  # cosine threshold below which an aligner is "uncertain"
    fusion_topk: int = 10

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def config_from_env(base: Config | None = None) -> Config:
    cfg = base or Config()
    for f in fields(cfg):
        env_key = f"MODALFORGE_{f.name.upper()}"
        if env_key in os.environ:
            raw = os.environ[env_key]
            cfg = _apply_override(cfg, f.name, raw, f.type)
    return cfg


def _apply_override(cfg: Config, name: str, raw: str, ftype: Any) -> Config:  # noqa: ANN401
    import json

    t = str(ftype)
    try:
        if "int" in t:
            val: Any = int(raw)
        elif "float" in t:
            val = float(raw)
        elif "bool" in t:
            val = raw.lower() in ("1", "true", "yes", "on")
        elif "tuple" in t:
            val = (
                tuple(json.loads(raw)) if raw.startswith("[") else tuple(json.loads(f"[{raw}]"))
            )
        else:
            val = raw
    except (ValueError, json.JSONDecodeError) as exc:  # pragma: no cover
        from .errors import E100ConfigError

        raise E100ConfigError(f"bad env override {name}={raw!r}: {exc}") from exc
    setattr(cfg, name, val)
    return cfg
