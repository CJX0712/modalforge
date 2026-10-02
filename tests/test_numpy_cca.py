"""Cross-validation: pure-numpy CCA vs scikit-learn CCA (hard invariant)."""

import numpy as np

from modalforge.alignment.numpy_cca import cca_project, cca_solve


def _random_correlated(n=200, dx=6, dy=5, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, 3))
    X = rng.standard_normal((n, dx)) + U @ rng.standard_normal((3, dx))
    Y = rng.standard_normal((n, dy)) + U @ rng.standard_normal((3, dy))
    return X, Y


def test_canonical_correlations_match_sklearn():
    from sklearn.cross_decomposition import CCA as SkCCA

    X, Y = _random_correlated()
    k = 4
    A, B, lam = cca_solve(X, Y, k)

    sk = SkCCA(n_components=k)
    sk.fit(X, Y)
    skx, sky = sk.transform(X, Y)
    # sklearn 1.9 removed `.corrs_`; recover canonical correlations from scores
    sk_corr = sorted(
        (abs(np.corrcoef(skx[:, i], sky[:, i])[0, 1]) for i in range(k)), reverse=True
    )
    mine = sorted(map(abs, lam), reverse=True)
    assert np.allclose(mine, sk_corr, atol=1e-4), (mine, sk_corr)


def test_projected_column_correlations_equal_canonical():
    X, Y = _random_correlated()
    k = 3
    A, B, lam = cca_solve(X, Y, k)
    mx = X.astype(np.float64).mean(0)
    my = Y.astype(np.float64).mean(0)
    Zx, Zy = cca_project(X, Y, A, B, mx, my)
    for i in range(k):
        c = np.corrcoef(Zx[:, i], Zy[:, i])[0, 1]
        assert abs(abs(c) - lam[i]) < 1e-5, (i, c, lam[i])


def test_cca_ranks_limited_by_dimensions():
    X = np.random.default_rng(1).standard_normal((50, 2))
    Y = np.random.default_rng(2).standard_normal((50, 3))
    A, B, lam = cca_solve(X, Y, 10)  # requesting more than min(dx,dy,n-1)
    assert A.shape[1] == 2
    assert len(lam) == 2


def test_cca_transform_shape():
    X, Y = _random_correlated(n=100, dx=6, dy=5)
    A, B, lam = cca_solve(X, Y, 4)
    mx = X.mean(0)
    my = Y.mean(0)
    Zx, Zy = cca_project(X[:10], Y[:10], A, B, mx, my)
    assert Zx.shape == (10, 4)
    assert Zy.shape == (10, 4)
