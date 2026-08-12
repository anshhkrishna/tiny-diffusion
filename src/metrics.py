"""Distributional distance metric shared by baselines, the DDPM experiment, and rigor.

Energy distance (Szekely & Rizzo, 2013, "Energy statistics: A class of statistics based
on distances") rather than per-dimension Wasserstein: it is a proper metric on
multivariate distributions (zero iff the distributions are equal) and, unlike computing
`scipy.stats.wasserstein_distance` independently per dimension, it does not throw away
correlation structure between the two coordinates -- which matters for two_moons and
swiss_roll_2d, both of which are defined by their (non-independent) shape. scipy has no
built-in multivariate energy/Wasserstein distance, so this is a small hand-written
O(n*m) pairwise-distance implementation.
"""
import numpy as np


def _mean_pairwise_distance(a, b, exclude_diagonal):
    diffs = a[:, None, :] - b[None, :, :]
    d = np.sqrt((diffs ** 2).sum(axis=-1))
    if exclude_diagonal:
        n = len(a)
        assert a is b or a.shape == b.shape  # only called with a is b in that case
        mask = ~np.eye(n, dtype=bool)
        return d[mask].mean()
    return d.mean()


def energy_distance(x, y):
    """Unbiased two-sample energy distance between point sets x (n, d) and y (m, d).

    E(X, Y) = 2 * E|X - Y| - E|X - X'| - E|Y - Y'|, with the within-sample terms using
    the U-statistic (i != j) form so E(X, X) == 0 exactly. Zero iff x and y are drawn
    from the same distribution as n, m -> infinity; smaller is better (closer to target).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    e_xy = _mean_pairwise_distance(x, y, exclude_diagonal=False)
    e_xx = _mean_pairwise_distance(x, x, exclude_diagonal=True)
    e_yy = _mean_pairwise_distance(y, y, exclude_diagonal=True)
    return float(2 * e_xy - e_xx - e_yy)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    same = rng.standard_normal((300, 2))
    other = rng.standard_normal((300, 2)) + 3.0
    print("energy_distance(N(0,I), N(0,I) different draw):", energy_distance(same, rng.standard_normal((300, 2))))
    print("energy_distance(N(0,I), N(3,I)):", energy_distance(same, other))
