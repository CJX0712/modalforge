"""scikit-learn CCA / PLS canonical aligners (top open-source backends).

These reuse the canonical-correlation machinery from ``scikit-learn`` — a
reference-grade implementation. They degrade gracefully to unavailable when
scikit-learn cannot be imported (so the offline numpy path still runs).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E400AlignmentError

try:  # pragma: no cover - exercised only when sklearn present
    from sklearn.cross_decomposition import CCA as _SkCCA
    from sklearn.cross_decomposition import PLSCanonical as _SkPLS

    _SKLEARN_OK = True
except Exception:  # pragma: no cover
    _SKLEARN_OK = False


class SklearnCCA:
    """scikit-learn CCA aligner."""

    name = "sklearn_cca"

    def __init__(self, n_components: int = 16, reg: float = 1e-6) -> None:
        self.n_components = int(n_components)
        self.reg = reg
        self._model = None

    def available(self) -> bool:
        return _SKLEARN_OK

    def fit(self, X_img: np.ndarray, X_txt: np.ndarray) -> "SklearnCCA":
        if not _SKLEARN_OK:
            raise E400AlignmentError("scikit-learn not available")
        k = min(self.n_components, X_img.shape[0] - 1, X_img.shape[1], X_txt.shape[1])
        self._model = _SkCCA(n_components=k, scale=True, max_iter=1000, tol=1e-6)
        self._model.fit(X_img, X_txt)
        return self

    def transform(self, X_img: np.ndarray, X_txt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise E400AlignmentError("SklearnCCA.transform before fit")
        zx, zy = self._model.transform(X_img, X_txt)
        zx = np.asarray(zx, dtype=np.float32)
        zy = np.asarray(zy, dtype=np.float32)
        return zx, zy


class SklearnPLS:
    """scikit-learn PLS (PLSCanonical) aligner — alternative canonical method."""

    name = "sklearn_pls"

    def __init__(self, n_components: int = 16) -> None:
        self.n_components = int(n_components)
        self._model = None

    def available(self) -> bool:
        return _SKLEARN_OK

    def fit(self, X_img: np.ndarray, X_txt: np.ndarray) -> "SklearnPLS":
        if not _SKLEARN_OK:
            raise E400AlignmentError("scikit-learn not available")
        k = min(self.n_components, X_img.shape[0] - 1, X_img.shape[1], X_txt.shape[1])
        self._model = _SkPLS(n_components=k, scale=True, max_iter=1000, tol=1e-6)
        self._model.fit(X_img, X_txt)
        return self

    def transform(self, X_img: np.ndarray, X_txt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise E400AlignmentError("SklearnPLS.transform before fit")
        zx, zy = self._model.transform(X_img, X_txt)
        zx = np.asarray(zx, dtype=np.float32)
        zy = np.asarray(zy, dtype=np.float32)
        return zx, zy


def _unit(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return M / nrm
