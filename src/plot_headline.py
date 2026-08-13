"""Regenerates results/headline.png from the committed results/rigor.log -- parses the
real mean/std numbers the rigor step already printed (3-seed uncertainty), rather than
re-running training or re-deriving numbers, so the plot always matches exactly what's
checked into the log.

Run with `python -m src.plot_headline`.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LOG_PATH = RESULTS_DIR / "rigor.log"
BASELINE_LOG_PATH = RESULTS_DIR / "baseline.log"
OUT_PATH = RESULTS_DIR / "headline.png"

SECTION_RE = re.compile(r"^=== (\S+) ===$")
GAUSSIAN_RE = re.compile(r"gaussian_fit baseline energy_distance = (-?\d+\.\d+)")
STEP_RE = re.compile(r"^steps=(\d+) mean=(-?\d+\.\d+) std=(-?\d+\.\d+) n_seeds=(\d+)$")

BASELINE_SECTION_RE = re.compile(r"^(\S+): n=\d+ eval_seed=\d+$")
NOISE_RE = re.compile(r"noise_prior\s+energy_distance = (-?\d+\.\d+)")


def parse_log(path=LOG_PATH):
    """Returns {dataset: {"gaussian": float, "steps": [(n, mean, std), ...]}}."""
    datasets = {}
    current = None
    for line in path.read_text().splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1)
            datasets[current] = {"gaussian": None, "steps": []}
            continue
        if current is None:
            continue
        m = GAUSSIAN_RE.search(line)
        if m:
            datasets[current]["gaussian"] = float(m.group(1))
            continue
        m = STEP_RE.match(line)
        if m:
            datasets[current]["steps"].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    for name, d in datasets.items():
        assert d["steps"], f"no step rows parsed for {name} -- log format changed?"
        assert d["gaussian"] is not None, f"missing gaussian baseline for {name}"
    return datasets


def parse_noise_prior(path=BASELINE_LOG_PATH):
    """Returns {dataset: noise_prior energy_distance}, read from the committed baseline log.

    The headline finding is that very-few-step sampling lands on the noise prior, so the
    plot has to show where that floor actually is.
    """
    floors = {}
    current = None
    for line in path.read_text().splitlines():
        m = BASELINE_SECTION_RE.match(line)
        if m:
            current = m.group(1)
            continue
        if current is None:
            continue
        m = NOISE_RE.search(line)
        if m:
            floors[current] = float(m.group(1))
    assert floors, "no noise_prior rows parsed -- log format changed?"
    return floors


def plot(datasets, noise_floors=None, out_path=OUT_PATH):
    fig, ax = plt.subplots(figsize=(1600 / 150, 900 / 150), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = {"two_moons": "#1f77b4", "swiss_roll_2d": "#d62728"}

    for name, d in datasets.items():
        steps_sorted = sorted(d["steps"], key=lambda p: p[0])
        xs = [s for s, _, _ in steps_sorted]
        ys = [m for _, m, _ in steps_sorted]
        errs = [s for _, _, s in steps_sorted]
        color = colors.get(name, None)
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=f"{name} (DDPM, mean +/- std, 3 seeds)",
                    color=color)
        ax.axhline(d["gaussian"], linestyle="--", linewidth=1.5, color=color, alpha=0.7,
                    label=f"{name} gaussian-fit baseline")
        if noise_floors and name in noise_floors:
            ax.axhline(noise_floors[name], linestyle=":", linewidth=1.5, color=color, alpha=0.7,
                        label=f"{name} noise-prior baseline")

    ax.set_xscale("log")
    ax.set_xlabel("sampling steps (log scale)", fontsize=13)
    ax.set_ylabel("energy distance to held-out eval set\n(lower is better)", fontsize=13)
    ax.set_title("sample quality holds down to about 10-20 steps, then collapses", fontsize=15)
    ax.axhline(0, linewidth=0.8, color="black", alpha=0.3)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot(parse_log(), parse_noise_prior())
