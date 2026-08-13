"""Covers the core claim: across seeds, DDPM ancestral sampling at full
sampling steps reliably beats both very-few-step sampling and the gaussian_fit
baseline, on the real held-out energy_distance metric.

Parses the committed `results/rigor.log` (produced by `python -m src.rigor`, a
mean/std sweep over 3 training seeds) rather than retraining inside the test suite --
retraining takes tens of seconds per seed and would make pytest slow to the point
nobody runs it. This still checks real, committed numbers, not invented ones: if
`src/rigor.py` is ever rerun and the log changes, this test is checking the new log,
not a value frozen at authoring time.
"""
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RIGOR_LOG = RESULTS_DIR / "rigor.log"

SECTION_RE = re.compile(r"^=== (\S+) ===$")
GAUSSIAN_RE = re.compile(r"^gaussian_fit baseline energy_distance = (-?\d+\.\d+)$")
SANITY_RE = re.compile(r"^sanity: beats gaussian_fit in (\d+)/(\d+) seeds$")
STEPS_RE = re.compile(r"^steps=(\d+) mean=(-?\d+\.\d+) std=(-?\d+\.\d+) n_seeds=(\d+)$")
CORE_CLAIM_RE = re.compile(
    r"^core_claim: mean_energy_distance\(steps=(\d+)\)=(-?\d+\.\d+) < "
    r"mean_energy_distance\(steps=(\d+)\)=(-?\d+\.\d+) -> (holds|DOES NOT HOLD)$"
)


def parse_rigor_log(path=RIGOR_LOG):
    """Returns {dataset: {"gaussian": float, "beats": (k, n), "steps": {n: (mean, std,
    n_seeds)}, "core_claim": (full_steps, full_mean, few_steps, few_mean, verdict)}}."""
    datasets = {}
    current = None
    for line in path.read_text().splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1)
            datasets[current] = {"gaussian": None, "beats": None, "steps": {},
                                  "core_claim": None}
            continue
        if current is None:
            continue
        m = GAUSSIAN_RE.match(line)
        if m:
            datasets[current]["gaussian"] = float(m.group(1))
            continue
        m = SANITY_RE.match(line)
        if m:
            datasets[current]["beats"] = (int(m.group(1)), int(m.group(2)))
            continue
        m = STEPS_RE.match(line)
        if m:
            n_steps = int(m.group(1))
            datasets[current]["steps"][n_steps] = (
                float(m.group(2)), float(m.group(3)), int(m.group(4)))
            continue
        m = CORE_CLAIM_RE.match(line)
        if m:
            datasets[current]["core_claim"] = (
                int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4)),
                m.group(5))
    return datasets


def test_rigor_log_covers_both_toy_datasets():
    datasets = parse_rigor_log()
    assert set(datasets) == {"two_moons", "swiss_roll_2d"}


def test_at_least_three_seeds_per_step_count():
    datasets = parse_rigor_log()
    for name, d in datasets.items():
        assert d["steps"], f"no step-count rows parsed for {name}"
        for n_steps, (_, _, n_seeds) in d["steps"].items():
            assert n_seeds >= 3, f"{name} steps={n_steps} used only {n_seeds} seeds"


def test_full_step_ddpm_beats_gaussian_fit_baseline_every_seed():
    datasets = parse_rigor_log()
    for name, d in datasets.items():
        k, n = d["beats"]
        assert n >= 3, f"{name}: sanity check ran fewer than 3 seeds"
        assert k == n, (
            f"{name}: full-step DDPM only beat the gaussian_fit baseline in {k}/{n} "
            f"seeds -- the method claim does not hold across seeds"
        )


def test_full_step_energy_distance_below_gaussian_fit_mean():
    datasets = parse_rigor_log()
    for name, d in datasets.items():
        full_steps = max(d["steps"])
        mean_full, _, _ = d["steps"][full_steps]
        assert mean_full < d["gaussian"], (
            f"{name}: mean full-step ({full_steps}) energy_distance {mean_full} is not "
            f"below the gaussian_fit baseline {d['gaussian']}"
        )


def test_core_claim_full_steps_beats_few_steps_across_seeds():
    """The claim under test: full-step sampling reliably beats very-few-step
    sampling. Checked two ways -- src.rigor's own logged core_claim verdict, and an
    independent recomputation from the parsed per-step-count means, so a bug in how
    src.rigor prints its verdict can't silently pass this test.
    """
    datasets = parse_rigor_log()
    for name, d in datasets.items():
        assert d["core_claim"] is not None, f"{name}: no core_claim line found in log"
        full_steps, full_mean, few_steps, few_mean, verdict = d["core_claim"]
        assert verdict == "holds", (
            f"{name}: src.rigor logged core_claim as '{verdict}', not 'holds'"
        )
        assert full_steps == max(d["steps"])
        assert few_steps == min(d["steps"])

        recomputed_full = d["steps"][full_steps][0]
        recomputed_few = d["steps"][few_steps][0]
        assert recomputed_full < recomputed_few, (
            f"{name}: recomputed mean energy_distance at {full_steps} steps "
            f"({recomputed_full}) is not below {few_steps} steps ({recomputed_few})"
        )
