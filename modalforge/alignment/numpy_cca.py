"""From-scratch Canonical Correlation Analysis (CCA), pure numpy.

Key invariant (cross-validated against scikit-learn in the test-suite):
``canonical_correlations`` returned here equal ``sklearn.cross_decomposition.CCA``
``.corrs_`` to 1e-5, and the column correlations of the projected ``Zx, Zy``
reproduce those same values.

Math: find ``A, B`` maximising ``corr(AᵀX, BᵀY)``. The image vectors ``A`` are the
top eigenvectors of ``M = Sxx⁻¹ Sxy Syy⁻¹ Sxyᵀ``; canonical correlations are the
square roots of ``M``'s eigenvalues; ``B = Syy⁻¹ Sxyᵀ A / λ``.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E400AlignmentError


def _center(M: np.ndarray) -> np.ndarray:
    return M - M.mean(axis=0, keepdims=True)


def _whitening_sqrt_inv(M: np.ndarray) -> np.ndarray:
    """Return ``M^{-1/2}`` via symmetric eigendecomposition (M symmetric PSD)."""
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals, 0.0, None)
    inv_sqrt = np.zeros_like(vals)
    nz = vals > 1e-12
    inv_sqrt[nz] = 1.0 / np.sqrt(vals[nz])
    return (vecs * inv_sqrt) @ vecs.T


def cca_solve(
    X: np.ndarray, Y: np.ndarray, n_components: int, reg: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve CCA via the whitened-SVD formulation.

    Whitening: ``Xw = Xc @ Sxx^{-1/2}``, ``Yw = Yc @ Syy^{-1/2}`` so both have
    identity covariance. Then ``R = Sxx^{-1/2} Sxy Syy^{-1/2}`` and its singular
    values **are** the canonical correlations; the projection matrices are
    ``A = Sxx^{-1/2} U[:, :k]``, ``B = Syy^{-1/2} V[:, :k]``. This is numerically
    clean and its projected column correlations equal the singular values exactly
    (cross-validated against scikit-learn).

    Returns ``(A, B, correlations)`` where A:(dx,k) B:(dy,k).
    """
    if X.shape[0] != Y.shape[0]:
        raise E400AlignmentError("X and Y must share the first dimension")
    Xc = _center(X).astype(np.float64)
    Yc = _center(Y).astype(np.float64)
    n = Xc.shape[0]
    Sxx = (Xc.T @ Xc) / max(n - 1, 1) + reg * np.eye(Xc.shape[1])
    Syy = (Yc.T @ Yc) / max(n - 1, 1) + reg * np.eye(Yc.shape[1])
    Sxy = (Xc.T @ Yc) / max(n - 1, 1)
    dx, dy = Sxx.shape[0], Syy.shape[0]
    k = min(n_components, dx, dy, n - 1)
    if k < 1:
        raise E400AlignmentError(
            f"cannot compute {n_components} components (n={n}, dx={dx}, dy={dy})"
        )
    Xw_sqrt_inv = _whitening_sqrt_inv(Sxx)
    Yw_sqrt_inv = _whitening_sqrt_inv(Syy)
    R = Xw_sqrt_inv @ Sxy @ Yw_sqrt_inv
    # full_matrices=False -> U:(dx,k) s:(k,) Vt:(k,dy); s sorted descending
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    s = np.clip(s, 0.0, 1.0)  # canonical correlations live in [0, 1]
    A = Xw_sqrt_inv @ U[:, :k]  # (dx, k)
    B = Yw_sqrt_inv @ Vt[:k, :].T  # (dy, k)
    return A, B, s[:k]


def cca_project(
    X: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray, mx: np.ndarray, my: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project into the shared CCA space.

    Returns the raw canonical variates ``Zx = (X-mx)@A``, ``Zy = (Y-my)@B``.
    Do **not** per-sample L2-normalise: that would destroy the canonical
    correlations (Pearson). Cosine retrieval (scale-invariant per sample) is
    unaffected, and the retriever normalises internally.
    """
    Zx = (X.astype(np.float64) - mx) @ A
    Zy = (Y.astype(np.float64) - my) @ B
    return Zx.astype(np.float32), Zy.astype(np.float32)


class NumpyCCA:
    """Pure-numpy CCA aligner (always available, offline)."""

    name = "numpy_cca"

    def __init__(self, n_components: int = 16, reg: float = 1e-6) -> None:
        self.n_components = int(n_components)
        self.reg = reg
        self._A: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._mx: np.ndarray | None = None
        self._my: np.ndarray | None = None
        self.correlations_: np.ndarray | None = None

    def available(self) -> bool:
        return True

    def fit(self, X_img: np.ndarray, X_txt: np.ndarray) -> "NumpyCCA":
        A, B, lam = cca_solve(X_img, X_txt, self.n_components, self.reg)
        self._A, self._B, self.correlations_ = A, B, lam
        self._mx = X_img.astype(np.float64).mean(axis=0)
        self._my = X_txt.astype(np.float64).mean(axis=0)
        return self

    def transform(self, X_img: np.ndarray, X_txt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._A is None:
            raise E400AlignmentError("NumpyCCA.transform before fit")
        return cca_project(X_img, X_txt, self._A, self._B, self._mx, self._my)
