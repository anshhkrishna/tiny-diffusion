"""Regenerates results/quality_vs_steps.png from the committed results/run.log --
parses the real numbers the experiment already printed, rather than re-running training or
re-deriving numbers, so the plot always matches exactly what's checked into the log.

Run with `python -m src.plot_quality_vs_steps`.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LOG_PATH = RESULTS_DIR / "run.log"
OUT_PATH = RESULTS_DIR / "quality_vs_steps.png"

SECTION_RE = re.compile(r"^=== (\S+) ===$")
GAUSSIAN_RE = re.compile(r"baseline gaussian_fit energy_distance = (-?\d+\.\d+)")
NOISE_RE = re.compile(r"baseline noise_prior +energy_distance = (-?\d+\.\d+)")
STEP_ROW_RE = re.compile(r"^\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def parse_log(path=LOG_PATH):
    """Returns {dataset: {"gaussian": float, "noise": float, "steps": [(n, d), ...]}}."""
    datasets = {}
    current = None
    for line in path.read_text().splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1)
            datasets[current] = {"gaussian": None, "noise": None, "steps": []}
            continue
        if current is None:
            continue
        m = GAUSSIAN_RE.search(line)
        if m:
            datasets[current]["gaussian"] = float(m.group(1))
            continue
        m = NOISE_RE.search(line)
        if m:
            datasets[current]["noise"] = float(m.group(1))
            continue
        m = STEP_ROW_RE.match(line)
        if m:
            datasets[current]["steps"].append((int(m.group(1)), float(m.group(2))))
    for name, d in datasets.items():
        assert d["steps"], f"no step rows parsed for {name} -- log format changed?"
        assert d["gaussian"] is not None and d["noise"] is not None, f"missing baselines for {name}"
    return datasets


def plot(datasets, out_path=OUT_PATH):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    colors = {"two_moons": "#1f77b4", "swiss_roll_2d": "#d62728"}

    for name, d in datasets.items():
        steps_sorted = sorted(d["steps"], key=lambda p: p[0])
        xs = [s for s, _ in steps_sorted]
        ys = [v for _, v in steps_sorted]
        color = colors.get(name, None)
        ax.plot(xs, ys, marker="o", label=f"{name} (DDPM)", color=color)
        ax.axhline(d["gaussian"], linestyle="--", linewidth=1, color=color, alpha=0.6,
                   label=f"{name} gaussian_fit baseline")

    ax.set_xscale("log")
    ax.set_xlabel("sampling steps (log scale)")
    ax.set_ylabel("energy distance to held-out eval set (lower is better)")
    ax.set_title("Sample quality vs. number of reverse-diffusion steps")
    ax.axhline(0, linewidth=0.8, color="black", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    plot(parse_log())
