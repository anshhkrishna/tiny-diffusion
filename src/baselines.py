"""Two baselines scored on the same held-out-eval / energy-distance protocol the DDPM
experiment will reuse.

Evaluation protocol, fixed here so later comparisons stay honest:
  - seed=0 generates the "reference" draw a method is allowed to use (e.g. to fit the
    Gaussian baseline's mean/covariance, or later to train DDPM).
  - seed=1 generates a held-out evaluation set from the *same* target distribution,
    independent of seed=0, used only to score energy distance. Nothing is fit or tuned
    against the eval set.
Both baselines and any later method are scored against the identical eval set per toy
distribution, so numbers are directly comparable.
"""
import numpy as np

from src.data import TOY_NAMES, load
from src.metrics import energy_distance

REFERENCE_SEED = 0
EVAL_SEED = 1
N = 500


def noise_prior_baseline(n, seed):
    """Samples straight from the N(0, I) prior -- the "did nothing" floor.

    Valid without further scaling because src/data.py normalizes every toy target to
    zero mean, unit variance per dimension.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2))


def gaussian_fit_baseline(reference, n, seed):
    """Samples from a single Gaussian fit (mean, full covariance) to `reference`."""
    mean = reference.mean(axis=0)
    cov = np.cov(reference, rowvar=False)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=n)


def run():
    results = {}
    for name in TOY_NAMES:
        reference = load(name, n=N, seed=REFERENCE_SEED)
        eval_set = load(name, n=N, seed=EVAL_SEED)

        noise_samples = noise_prior_baseline(N, seed=REFERENCE_SEED)
        gaussian_samples = gaussian_fit_baseline(reference, N, seed=REFERENCE_SEED)

        d_noise = energy_distance(noise_samples, eval_set)
        d_gaussian = energy_distance(gaussian_samples, eval_set)

        results[name] = {"noise_prior": d_noise, "gaussian_fit": d_gaussian}
        print(f"{name}: n={N} eval_seed={EVAL_SEED}")
        print(f"  noise_prior  energy_distance = {d_noise:.4f}")
        print(f"  gaussian_fit energy_distance = {d_gaussian:.4f}")
    return results


if __name__ == "__main__":
    run()
