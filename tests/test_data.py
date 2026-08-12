import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load, load_digits_data, TOY_NAMES


def test_toy_shapes():
    for name in TOY_NAMES:
        pts = load(name, n=256, seed=0)
        assert pts.shape == (256, 2)
        assert np.all(np.isfinite(pts))


def test_toy_normalized():
    for name in TOY_NAMES:
        pts = load(name, n=2000, seed=0)
        assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-8)
        assert np.allclose(pts.std(axis=0), 1.0, atol=1e-8)


def test_toy_seed_determinism():
    for name in TOY_NAMES:
        a = load(name, n=128, seed=42)
        b = load(name, n=128, seed=42)
        assert np.array_equal(a, b)


def test_toy_seed_variation():
    for name in TOY_NAMES:
        a = load(name, n=128, seed=1)
        b = load(name, n=128, seed=2)
        assert not np.array_equal(a, b)


def test_toy_distributions_are_distinguishable():
    # two_moons and swiss_roll_2d should not be the same shape in disguise: their
    # covariance structure (elongation ratio between principal axes) should differ.
    def elongation(pts):
        cov = np.cov(pts.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        return eigvals[0] / eigvals[1]

    moons = load("two_moons", n=4000, seed=0)
    swiss = load("swiss_roll_2d", n=4000, seed=0)
    assert abs(elongation(moons) - elongation(swiss)) > 0.05


def test_digits_shape_and_range():
    x, y = load_digits_data(seed=0)
    assert x.shape[1] == 64
    assert x.shape[0] > 1000  # sklearn's bundled digits has 1797 samples
    assert x.shape[0] == y.shape[0]
    assert np.all(np.isfinite(x))
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert set(np.unique(y).tolist()) == set(range(10))


def test_digits_seed_determinism():
    x1, y1 = load_digits_data(seed=0)
    x2, y2 = load_digits_data(seed=0)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_unknown_name_raises():
    try:
        load("not_a_real_distribution")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown distribution name")
