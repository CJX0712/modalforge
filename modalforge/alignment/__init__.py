"""ModalForge alignment: cross-modal embedding alignment backends.

Each aligner projects image and text embeddings into a shared semantic space
so that paired samples are close under cosine similarity. Backends range from a
dependency-free numpy CCA (offline fallback) to scikit-learn CCA/PLS and an
optional CLIP dual-encoder.
"""

from .base import build_aligners
from .clip_backend import ClipAligner
from .contrastive import NumpyContrastive
from .numpy_cca import NumpyCCA
from .sklearn_cca import SklearnCCA, SklearnPLS

__all__ = [
    "build_aligners",
    "NumpyCCA",
    "SklearnCCA",
    "SklearnPLS",
    "NumpyContrastive",
    "ClipAligner",
]
