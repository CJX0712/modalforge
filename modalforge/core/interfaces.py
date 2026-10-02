"""Protocol (structural) interfaces — the only contracts modules depend on."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ImageEncoder(Protocol):
    """Maps raw images to a 2D embedding matrix ``(n, d_img)``."""

    name: str

    def fit(self, images: np.ndarray) -> "ImageEncoder": ...

    def transform(self, images: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, images: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class TextEncoder(Protocol):
    """Maps captions to a 2D embedding matrix ``(n, d_txt)``."""

    name: str

    def fit(self, texts: list[str]) -> "TextEncoder": ...

    def transform(self, texts: list[str]) -> np.ndarray: ...

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class Aligner(Protocol):
    """Projects image & text embeddings into a *shared* space.

    ``transform`` returns ``(Z_img, Z_txt)`` where row i of each describes the
    same item, enabling symmetric cross-modal retrieval.
    """

    name: str

    def fit(self, X_img: np.ndarray, X_txt: np.ndarray) -> "Aligner": ...

    def transform(
        self, X_img: np.ndarray, X_txt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def available(self) -> bool: ...


@runtime_checkable
class Retriever(Protocol):
    """Cosine retrieval over a fixed corpus."""

    def index(self, corpus: np.ndarray) -> None: ...

    def retrieve(self, queries: np.ndarray, k: int) -> "RetrievalResult":  # noqa: F821
        ...
