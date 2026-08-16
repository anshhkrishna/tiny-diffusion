# tiny-diffusion

> how few reverse steps can a ddpm get away with before sample quality falls apart?

a ddpm written from scratch in numpy — forward process, denoiser, loss and backward pass,
no pytorch, no autodiff — plus a respaced ancestral sampler that can draw using any
subset of the trained timesteps. that sampler is what makes the question askable.

## the knee is at 10-20 steps

quality is energy distance between generated samples and a held-out eval set, lower is
better, averaged over 3 training seeds (`results/rigor.log`):

| steps | two_moons | swiss_roll_2d |
|---|---|---|
| 200 | -0.0029 +/- 0.0014 | 0.0009 +/- 0.0010 |
| 20 | -0.0011 +/- 0.0033 | 0.0000 +/- 0.0009 |
| 10 | 0.0023 +/- 0.0033 | 0.0012 +/- 0.0011 |
| 5 | 0.0099 +/- 0.0037 | 0.0037 +/- 0.0010 |
| 2 | 0.0477 +/- 0.0151 | 0.0281 +/- 0.0036 |
| 1 | 0.0411 +/- 0.0000 | 0.0180 +/- 0.0001 |

flat, within noise of the full 200-step number, all the way down to about 10-20 steps.
then it falls off a cliff. the full 8-step-count table is in `results/rigor.log`, the
single-seed version in `results/run.log`.

![sample quality vs. sampling steps, with error bars and both baselines](results/headline.png)

**at 2 steps the model is worth roughly nothing.** 0.0477 against a noise-prior floor of
0.0483 on two_moons, 0.0281 against 0.0265 on swiss_roll_2d. two reverse steps buys
almost nothing over sampling straight from n(0, i).

## does it work at all?

before the sweep means anything the model has to beat two references, both computed on
this project's own data draw:

| reference | two_moons | swiss_roll_2d |
|---|---|---|
| trained ddpm, 200 steps | -0.0029 +/- 0.0014 | 0.0009 +/- 0.0010 |
| gaussian fit (mean/cov of target) | 0.0282 | 0.0203 |
| noise prior (straight from n(0, i)) | 0.0483 | 0.0265 |

the gaussian fit is the cheapest distribution-shaped guess that needs no diffusion
machinery at all, so beating it is the real bar. the trained model clears it at full
steps in 3/3 seeds on both datasets (`results/rigor.log`, `sanity: beats gaussian_fit in
3/3 seeds`).

## the one-step oddity

one step scores *better* than two on both targets: 0.0411 vs 0.0477 on two_moons, 0.0180
vs 0.0281 on swiss_roll_2d. both sit at or above the noise-prior floor, so this is a
comparison between two ways of failing, not evidence that fewer steps help. the
single-hop sampler adds no intermediate noise, which plausibly explains why it lands
marginally closer — but this project did not test that, and it is left as a loose end
rather than dressed up as a finding.

short version in `results/FINDING.md`.

## data and budget

two 2d synthetic targets from `sklearn.datasets` (`make_moons`, and `make_swiss_roll`
projected to 2d), generated locally: no download, no network, no auth. a tiny image set
(`sklearn.datasets.load_digits`, bundled with scikit-learn) is loaded and shape-tested in
`tests/test_data.py` as a stretch target, but never used in the headline experiment —
training a diffusion model on 8x8 images did not fit the 10-minute cpu budget this
project set itself.

it came in well under that. full training plus a complete step-count sweep for one
dataset runs in under 5 seconds (`results/run.log`); `src.rigor`, which is 3 seeds x 2
datasets with a full 8-step sweep each, takes 51.03s (`results/rigor.log`).

## reproduce

```bash
pip install -r requirements.txt

python -m src.baselines           # baselines, writes results/baseline.log
python -m src.experiment          # single-seed train + step-count sweep, writes results/run.log
python -m src.rigor               # 3-seed train + sweep, writes results/rigor.log
python -m src.plot_headline       # regenerates results/headline.png from results/rigor.log
pytest tests/
```

13 tests: data loader shapes, ranges and determinism, plus the core claim itself
(full-step energy distance reliably beats very-few-step energy distance across seeds),
parsed from the committed `results/rigor.log`.
