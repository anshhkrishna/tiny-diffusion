"""Rigor: repeat src.experiment's train + step-count sweep across
>=3 seeds, report mean +/- std of energy_distance at every step count in
src.experiment.STEP_COUNTS, and confirm the "DDPM beats gaussian_fit at full steps"
sanity check holds seed-by-seed, not just for the single seed src.experiment used.

Only MODEL_SEED and TRAIN_RNG_SEED vary per seed (one model init + one training-batch
stream per seed); REFERENCE_SEED, EVAL_SEED, and SAMPLE_SEED stay fixed at the values
src.baselines / src.experiment already fixed, so the data, the held-out eval set, and
the sampling noise are identical across seeds -- only the model's training stochasticity
differs, isolating what an uncertainty estimate on this metric is supposed to measure.

Run with `python -m src.rigor`, real stdout saved verbatim to `results/rigor.log`.
"""
import time

import numpy as np

from src.baselines import EVAL_SEED, N, REFERENCE_SEED, gaussian_fit_baseline
from src.data import TOY_NAMES, load
from src.experiment import SAMPLE_SEED, STEP_COUNTS, T, train
from src.metrics import energy_distance

SEEDS = (0, 1, 2)


def run():
    overall_start = time.time()
    for name in TOY_NAMES:
        print(f"=== {name} ===")
        reference = load(name, n=N, seed=REFERENCE_SEED)
        eval_set = load(name, n=N, seed=EVAL_SEED)

        gaussian_samples = gaussian_fit_baseline(reference, N, seed=REFERENCE_SEED)
        d_gaussian = energy_distance(gaussian_samples, eval_set)
        print(f"gaussian_fit baseline energy_distance = {d_gaussian:.4f}")

        per_step = {s: [] for s in STEP_COUNTS}
        beats_count = 0
        for seed in SEEDS:
            t0 = time.time()
            ddpm, losses = train(reference, model_seed=seed, train_rng_seed=1000 + seed)
            train_time = time.time() - t0
            assert np.isfinite(losses).all(), f"seed={seed} training loss went non-finite"

            full_step_samples = ddpm.sample_respaced(n=N, seed=SAMPLE_SEED, num_steps=T)
            d_full = energy_distance(full_step_samples, eval_set)
            beats_gaussian = d_full < d_gaussian
            beats_count += int(beats_gaussian)
            print(f"seed {seed}: train_time={train_time:.2f}s "
                  f"full_step_energy_distance={d_full:.4f} beats_gaussian={beats_gaussian}")

            for num_steps in STEP_COUNTS:
                if num_steps == T:
                    d = d_full
                else:
                    samples = ddpm.sample_respaced(n=N, seed=SAMPLE_SEED, num_steps=num_steps)
                    assert np.all(np.isfinite(samples)), \
                        f"seed={seed} non-finite samples at num_steps={num_steps}"
                    d = energy_distance(samples, eval_set)
                per_step[num_steps].append(d)

        print(f"sanity: beats gaussian_fit in {beats_count}/{len(SEEDS)} seeds")
        print()
        for num_steps in STEP_COUNTS:
            vals = per_step[num_steps]
            print(f"steps={num_steps} mean={np.mean(vals):.4f} std={np.std(vals):.4f} "
                  f"n_seeds={len(vals)}")
        print()

        mean_full = np.mean(per_step[T])
        mean_worst = np.mean(per_step[min(STEP_COUNTS)])
        holds = mean_full < mean_worst
        print(f"core_claim: mean_energy_distance(steps={T})={mean_full:.4f} < "
              f"mean_energy_distance(steps={min(STEP_COUNTS)})={mean_worst:.4f} "
              f"-> {'holds' if holds else 'DOES NOT HOLD'}")
        print()

    total_time = time.time() - overall_start
    print(f"total wall-clock time: {total_time:.2f}s")


if __name__ == "__main__":
    run()
