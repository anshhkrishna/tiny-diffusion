"""Synthetic 2D toy distributions + a tiny bundled image set.

Everything here is generated or loaded in-process: `make_moons` and `make_swiss_roll`
are sampled from closed-form generators bundled with scikit-learn, and `load_digits`
ships its data as a package resource. No network call, no download, no auth, no cost.
"""
import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll, load_digits

TOY_NAMES = ("two_moons", "swiss_roll_2d")


def _normalize(x):
    """Zero mean, unit variance per dimension — matches the N(0, I) noise prior."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def load(name, n=1000, seed=0):
    """Return an (n, 2) normalized array for one of TOY_NAMES."""
    if name == "two_moons":
        x, _ = make_moons(n_samples=n, noise=0.05, random_state=seed)
    elif name == "swiss_roll_2d":
        x3, _ = make_swiss_roll(n_samples=n, noise=0.05, random_state=seed)
        x = x3[:, [0, 2]]  # drop the roll's height axis, keep the 2D spiral
    else:
        raise ValueError(f"unknown toy distribution: {name!r} (expected one of {TOY_NAMES})")
    return _normalize(x).astype(np.float64)


def load_digits_data(seed=0):
    """Return (X, y): bundled 8x8 grayscale digit images flattened to (n, 64), scaled to [0, 1]."""
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0  # pixel values are integers in [0, 16]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    return x[order], digits.target[order]


if __name__ == "__main__":
    for name in TOY_NAMES:
        pts = load(name, n=5, seed=0)
        print(f"{name}: shape={pts.shape} mean={pts.mean(axis=0)} std={pts.std(axis=0)}")
    x, y = load_digits_data()
    print(f"digits: shape={x.shape} labels={sorted(set(y.tolist()))}")
