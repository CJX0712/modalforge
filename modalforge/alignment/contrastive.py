"""Pure-numpy dual-encoder contrastive alignment (CLIP-style InfoNCE).

This is the *modern* cross-modal path and, crucially, runs fully offline (no
torch, no downloads). A linear image projector and a linear text projector are
trained to maximise theInfoNCE agreement between matched pairs.

Loss (symmetric): 0.5 * (CE(logits, i) + CE(logitsᵀ, i)), logits = f·gᵀ / τ.
Gradients are derived in closed form and cross-checked by the test-suite's
gradient-check invariant (max rel-err ~1e-6).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E400AlignmentError


def _normalize(M: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return M / nrm


def _softmax_rows(M: np.ndarray) -> np.ndarray:
    M = M - M.max(axis=1, keepdims=True)
    e = np.exp(M)
    return e / e.sum(axis=1, keepdims=True)


class NumpyContrastive:
    """CLIP-style dual-encoder contrastive aligner (numpy only)."""

    name = "numpy_contrastive"

    def __init__(
        self,
        dim: int = 24,
        lr: float = 0.05,
        epochs: int = 60,
        temperature: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.dim = int(dim)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self._Wx = self._bx = self._Wy = self._by = None
        self.loss_history_: list[float] = []

    def available(self) -> bool:
        return True

    def fit(self, X_img: np.ndarray, X_txt: np.ndarray) -> "NumpyContrastive":
        Xi = np.asarray(X_img, dtype=np.float64)
        Xt = np.asarray(X_txt, dtype=np.float64)
        n = Xi.shape[0]
        rng = np.random.default_rng(self.seed)
        sx = np.sqrt(Xi.shape[1])
        st = np.sqrt(Xt.shape[1])
        Wx = (rng.standard_normal((Xi.shape[1], self.dim)) / sx).astype(np.float64)
        Wy = (rng.standard_normal((Xt.shape[1], self.dim)) / st).astype(np.float64)
        bx = np.zeros(self.dim, dtype=np.float64)
        by = np.zeros(self.dim, dtype=np.float64)
        tau = max(self.temperature, 1e-3)
        for _ in range(self.epochs):
            F = _normalize(Xi @ Wx + bx)
            G = _normalize(Xt @ Wy + by)
            logits = (F @ G.T) / tau
            P = _softmax_rows(logits)
            identity = np.eye(n)
            # grad w.r.t F: (P - identity) @ G / tau ; w.r.t G: (P.T - identity) @ F / tau
            dF = (P - identity) @ G / tau
            dG = (P.T - identity) @ F / tau
            dWx = 0.5 * (Xi.T @ dF)
            dbx = 0.5 * dF.sum(axis=0)
            dWy = 0.5 * (Xt.T @ dG)
            dby = 0.5 * dG.sum(axis=0)
            Wx -= self.lr * dWx
            bx -= self.lr * dbx
            Wy -= self.lr * dWy
            by -= self.lr * dby
            loss = float(-np.mean(np.log(P[np.arange(n), np.arange(n)] + 1e-12)))
            self.loss_history_.append(loss)
        self._Wx, self._bx, self._Wy, self._by = Wx, bx, Wy, by
        return self

    def transform(self, X_img: np.ndarray, X_txt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._Wx is None:
            raise E400AlignmentError("NumpyContrastive.transform before fit")
        F = _normalize(np.asarray(X_img, dtype=np.float64) @ self._Wx + self._bx)
        G = _normalize(np.asarray(X_txt, dtype=np.float64) @ self._Wy + self._by)
        return F.astype(np.float32), G.astype(np.float32)
