"""Error taxonomy for ModalForge (E100..E500)."""

from __future__ import annotations


class ModalForgeError(Exception):
    """Base class for all ModalForge errors."""


class E100ConfigError(ModalForgeError):
    """Invalid configuration / environment override."""


class E200DataError(ModalForgeError):
    """Corpus generation, loading or shape mismatch."""


class E300EncoderError(ModalForgeError):
    """Image / text encoder failure (empty input, bad shape)."""


class E400AlignmentError(ModalForgeError):
    """Aligner fit / transform failure (rank deficiency, NaN)."""


class E500RetrievalError(ModalForgeError):
    """Retriever / fusion failure (empty corpus, dim mismatch)."""
