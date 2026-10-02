"""Aligner construction + registry.

The pipeline consumes the list returned by :func:`build_aligners`. Each aligner
exposes ``name``, ``available()``, ``fit(X_img, X_txt)``, ``transform(...)`` and
is tried independently — if an aligner is unavailable or raises during fit, the
pipeline marks it skipped rather than crashing (offline resilience).
"""

from __future__ import annotations

from typing import List

from ..core.config import Config
from .clip_backend import ClipAligner
from .contrastive import NumpyContrastive
from .numpy_cca import NumpyCCA
from .sklearn_cca import SklearnCCA, SklearnPLS

# priority order: pure-numpy first (guaranteed offline), then sklearn, then CLIP
_ALIGNER_FACTORIES = [
    ("numpy_cca", lambda c: NumpyCCA(n_components=c.cca_dims)),
    ("sklearn_cca", lambda c: SklearnCCA(n_components=c.cca_dims)),
    ("sklearn_pls", lambda c: SklearnPLS(n_components=c.cca_dims)),
    (
        "numpy_contrastive",
        lambda c: NumpyContrastive(
            dim=c.contrastive_dim,
            lr=c.contrastive_lr,
            epochs=c.contrastive_epochs,
            temperature=c.contrastive_temperature,
            seed=c.seed,
        ),
    ),
    ("clip", lambda c: ClipAligner(model_name=c.clip_name)),
]


def build_aligners(cfg: Config) -> List[object]:
    """Return aligner instances in priority order."""
    out = []
    for _, factory in _ALIGNER_FACTORIES:
        try:
            out.append(factory(cfg))
        except Exception:  # pragma: no cover - defensive
            continue
    return out


def aligner_names(cfg: Config) -> List[str]:
    return [n for n, _ in _ALIGNER_FACTORIES]
