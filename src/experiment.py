"""Train DDPM for real on both 2D toy targets, then sample
at a range of step counts using `DDPM.sample_respaced` and score each against the same
held-out eval set / energy-distance protocol `src/baselines.py` already fixed, so the
numbers here are directly comparable to `results/baseline.log`.

Reuses REFERENCE_SEED=0 / EVAL_SEED=1 / N=500 from `src.baselines` -- not reinvented.
Run with `python -m src.experiment`, real stdout saved verbatim to `results/run.log`.
"""
import time

import numpy as np

from src.baselines import (
    EVAL_SEED,
    N,
    REFERENCE_SEED,
    gaussian_fit_baseline,
    noise_prior_baseline,
)
from src.data import TOY_NAMES, load
from src.ddpm import DDPM
from src.metrics import energy_distance

T = 200  # diffusion steps used for training (chosen for the CPU time budget)
N_EPOCHS = 2000
BATCH_SIZE = 64
LR = 2e-3
MODEL_SEED = 0  # MLP weight init
TRAIN_RNG_SEED = 123  # minibatch order + forward-noise draws during training
SAMPLE_SEED = 2  # distinct from REFERENCE_SEED/EVAL_SEED, used for every sampler call

# Step counts to sweep at sampling time. A longer list (1000, 500, ...)
# assumes a larger T than the T=200 chosen here for the CPU time budget, so the sweep is
# capped at T and otherwise kept log-ish spaced, always including the full-T reference.
STEP_COUNTS = (200, 100, 50, 20, 10, 5, 2, 1)


def train(x0, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
          model_seed=MODEL_SEED, train_rng_seed=TRAIN_RNG_SEED):
    ddpm = DDPM(T=T, seed=model_seed)
    rng = np.random.default_rng(train_rng_seed)
    losses = []
    for epoch in range(n_epochs):
        perm = rng.permutation(len(x0))
        epoch_losses = []
        for start in range(0, len(x0), batch_size):
            batch = x0[perm[start:start + batch_size]]
            epoch_losses.append(ddpm.train_step(batch, rng, lr=lr))
        losses.append(float(np.mean(epoch_losses)))
    return ddpm, losses


def run():
    overall_start = time.time()
    results = {}
    for name in TOY_NAMES:
        print(f"=== {name} ===")
        reference = load(name, n=N, seed=REFERENCE_SEED)
        eval_set = load(name, n=N, seed=EVAL_SEED)

        t0 = time.time()
        ddpm, losses = train(reference)
        train_time = time.time() - t0
        first5, last5 = np.mean(losses[:5]), np.mean(losses[-5:])
        print(f"train: {N_EPOCHS} epochs, T={T}, batch={BATCH_SIZE}, lr={LR}, "
              f"time={train_time:.2f}s")
        print(f"  loss: first 5 epochs mean={first5:.4f}, last 5 epochs mean={last5:.4f}")
        assert np.isfinite(losses).all(), "training loss went non-finite"
        assert last5 < first5, "training loss did not decrease"

        noise_samples = noise_prior_baseline(N, seed=REFERENCE_SEED)
        gaussian_samples = gaussian_fit_baseline(reference, N, seed=REFERENCE_SEED)
        d_noise = energy_distance(noise_samples, eval_set)
        d_gaussian = energy_distance(gaussian_samples, eval_set)
        print(f"  baseline noise_prior  energy_distance = {d_noise:.4f}")
        print(f"  baseline gaussian_fit energy_distance = {d_gaussian:.4f}")

        step_results = {}
        print(f"  {'steps':>6}  {'energy_distance':>16}  {'sample_time_s':>13}")
        for num_steps in STEP_COUNTS:
            t1 = time.time()
            samples = ddpm.sample_respaced(n=N, seed=SAMPLE_SEED, num_steps=num_steps)
            sample_time = time.time() - t1
            assert np.all(np.isfinite(samples)), f"non-finite samples at num_steps={num_steps}"
            d = energy_distance(samples, eval_set)
            step_results[num_steps] = d
            print(f"  {num_steps:>6}  {d:>16.4f}  {sample_time:>13.4f}")

        results[name] = {
            "train_time_s": train_time,
            "loss_first5": first5,
            "loss_last5": last5,
            "baseline_noise_prior": d_noise,
            "baseline_gaussian_fit": d_gaussian,
            "steps": step_results,
        }

        full_step_d = step_results[T]
        beats_gaussian = full_step_d < d_gaussian
        print(f"  sanity check: full-step ({T}) energy_distance={full_step_d:.4f} "
              f"{'<' if beats_gaussian else '>='} gaussian_fit baseline={d_gaussian:.4f} "
              f"-> {'beats' if beats_gaussian else 'DOES NOT beat'} the baseline")
        print()

    total_time = time.time() - overall_start
    print(f"total wall-clock time: {total_time:.2f}s")
    return results


if __name__ == "__main__":
    run()
