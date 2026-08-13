A from-scratch DDPM (NumPy only, no PyTorch) was trained on two 2D toy distributions
and sampled at step counts from 200 down to 1, scored by energy distance to a held-out
eval set, averaged over 3 training seeds. At full sampling steps (200) it beats a
Gaussian-fit baseline on both datasets: two moons scores -0.0029 versus 0.0282 for the
baseline, and swiss roll 2D scores 0.0009 versus 0.0203 (`results/rigor.log`). Quality
barely moves between 200 and about 10-20 steps, then drops off sharply below that. The
surprising part: at 2 steps the model is not just worse, it is statistically
indistinguishable from pure noise. Its energy distance (0.0477 +/- 0.0151 two moons,
0.0281 +/- 0.0036 swiss roll) lands right on top of the noise-prior baseline (0.0483 and
0.0265, `results/baseline.log`), meaning two reverse steps buys almost nothing over
sampling straight from N(0, I).
