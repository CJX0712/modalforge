"""Core package: shared types, errors, configuration and interfaces.

All cross-module contracts live here so that data -> encoders -> alignment ->
retrieval -> fusion -> pipeline form a single, acyclic dependency chain that
only ever reaches *down* toward ``modalforge.core``.
"""

from .config import Config, config_from_env
from .errors import (
    E100ConfigError,
    E200DataError,
    E300EncoderError,
    E400AlignmentError,
    E500RetrievalError,
    ModalForgeError,
)
from .interfaces import (
    Aligner,
    ImageEncoder,
    Retriever,
    TextEncoder,
)
from .types import (
    AlignedCorpus,
    AlignedPair,
    BenchmarkRow,
    Embedding,
    RetrievalResult,
)

__all__ = [
    "AlignedCorpus",
    "AlignedPair",
    "BenchmarkRow",
    "Embedding",
    "RetrievalResult",
    "E100ConfigError",
    "E200DataError",
    "E300EncoderError",
    "E400AlignmentError",
    "E500RetrievalError",
    "ModalForgeError",
    "Config",
    "config_from_env",
    "Aligner",
    "ImageEncoder",
    "Retriever",
    "TextEncoder",
]
