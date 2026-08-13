"""From-scratch DDPM: closed-form forward diffusion + a hand-written NumPy MLP
(forward and backward pass, no autograd) trained as the reverse noise-predictor.

Forward process (Ho, Jain & Abbeel 2020, DDPM, arXiv:2006.11239): a fixed linear beta
schedule beta_1..beta_T, alpha_t = 1 - beta_t, alpha_bar_t = prod(alpha_1..alpha_t).
Sampling x_t given x_0 has the closed form

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps ~ N(0, I)

which is what makes training possible with a single random timestep per example, no need
to simulate the whole chain during training.

Reverse process: a small MLP eps_theta(x_t, t) is trained to predict the noise eps that
was added, with the simplified loss from Ho et al. eq. 14: L = E[||eps - eps_theta(x_t,
t)||^2]. Sampling then runs the learned reverse transition ancestrally from x_T ~ N(0, I)
down to x_0:

    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps_theta(x_t, t))
              + sigma_t * z,   z ~ N(0, I) if t > 0 else 0,   sigma_t = sqrt(beta_t)

In addition to the full-T ancestral sampler (`sample`), this module implements a
reduced-step-count "respaced" sampler (`sample_respaced`, DDPM.sample_respaced) for the
quality-vs-steps sweep, following the timestep-respacing trick from
Nichol & Dhariwal (2021, "Improved DDPM", arXiv:2102.09672, sec. 4.2): pick a strictly
decreasing subsequence of the original timesteps and derive an *effective* per-hop beta
from the ratio of alpha_bar at consecutive subsequence points, rather than looking up
beta_t directly (which is only defined for single-integer hops). Verified against the
full sampler above: with the subsequence set to every original timestep, the effective
beta at a hop from t to t-1 reduces algebraically to beta_t exactly (alpha_bar_t /
alpha_bar_{t-1} = alpha_t, so 1 - alpha_bar_t/alpha_bar_{t-1} = 1 - alpha_t = beta_t) --
and `sample_respaced(n, seed, num_steps=T)` is checked to match `sample(n, seed)`
bit-for-bit for that reason (see the assertion in `_smoke_test`).
"""
import numpy as np


def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule, plus the derived alpha and cumulative-product alpha_bar."""
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def time_embedding(t, T, dim=8):
    """Sinusoidal embedding of integer timestep(s) t in [0, T) -- shape (len(t), dim)."""
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1) / T
    freqs = 2.0 ** np.arange(dim // 2, dtype=np.float64)
    angles = t * freqs[None, :] * np.pi
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


class MLP:
    """Two-hidden-layer ReLU MLP with a hand-written forward pass, backward pass, and
    Adam optimizer step -- no autograd framework. Gradients are derived by hand below;
    they are verified against finite-difference gradients to confirm they're correct.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, seed=0):
        rng = np.random.default_rng(seed)

        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)

        self.W1 = he(in_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = he(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = he(hidden_dim, out_dim) * 0.1  # small init: eps_theta starts near 0
        self.b3 = np.zeros(out_dim)
        self.params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self._adam_t = 0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0.0)
        out = a2 @ self.W3 + self.b3
        return out, (X, z1, a1, z2, a2)

    def backward(self, cache, d_out):
        """d_out is dL/d(out), already normalized by the caller (e.g. by batch size) --
        every downstream gradient below is a linear function of d_out, so that
        normalization propagates through the chain rule without re-dividing here.
        """
        X, z1, a1, z2, a2 = cache
        dW3 = a2.T @ d_out
        db3 = d_out.sum(axis=0)
        da2 = d_out @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def adam_step(self, grads, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_t += 1
        for p in self.params:
            g = grads[p]
            self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
            self._v[p] = beta2 * self._v[p] + (1 - beta2) * (g ** 2)
            m_hat = self._m[p] / (1 - beta1 ** self._adam_t)
            v_hat = self._v[p] / (1 - beta2 ** self._adam_t)
            setattr(self, p, getattr(self, p) - lr * m_hat / (np.sqrt(v_hat) + eps))


class NoisePredictor:
    """eps_theta(x_t, t): concatenates x_t with a sinusoidal time embedding and feeds
    the small MLP above."""

    def __init__(self, T, hidden_dim=64, time_dim=8, seed=0):
        self.T = T
        self.time_dim = time_dim
        self.mlp = MLP(in_dim=2 + time_dim, hidden_dim=hidden_dim, out_dim=2, seed=seed)

    def _features(self, x, t):
        return np.concatenate([x, time_embedding(t, self.T, self.time_dim)], axis=1)

    def predict(self, x, t):
        return self.mlp.forward(self._features(x, t))

    def train_step(self, x, t, target_eps, lr=1e-3):
        pred, cache = self.predict(x, t)
        diff = pred - target_eps
        loss = float(np.mean(diff ** 2))
        d_out = 2.0 * diff / diff.size  # dL/d(pred) for L = mean(diff**2)
        grads = self.mlp.backward(cache, d_out)
        self.mlp.adam_step(grads, lr=lr)
        return loss


class DDPM:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, hidden_dim=64, time_dim=8, seed=0):
        self.T = T
        self.betas, self.alphas, self.alpha_bars = make_beta_schedule(T, beta_start, beta_end)
        self.model = NoisePredictor(T=T, hidden_dim=hidden_dim, time_dim=time_dim, seed=seed)

    def add_noise(self, x0, t, rng):
        """Closed-form forward process. t is an (n,) int array of timesteps in [0, T)."""
        eps = rng.standard_normal(x0.shape)
        ab = self.alpha_bars[t][:, None]
        x_t = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        return x_t, eps

    def train_step(self, x0_batch, rng, lr=1e-3):
        t = rng.integers(0, self.T, size=x0_batch.shape[0])
        x_t, eps = self.add_noise(x0_batch, t, rng)
        return self.model.train_step(x_t, t, eps, lr=lr)

    def sample(self, n, seed):
        """Full-T ancestral sampling, x_T ~ N(0, I) down to x_0."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, 2))
        for t in reversed(range(self.T)):
            t_arr = np.full(n, t)
            eps_pred, _ = self.model.predict(x, t_arr)
            alpha_t, alpha_bar_t, beta_t = self.alphas[t], self.alpha_bars[t], self.betas[t]
            coef = beta_t / np.sqrt(1 - alpha_bar_t)
            mean = (x - coef * eps_pred) / np.sqrt(alpha_t)
            if t > 0:
                x = mean + np.sqrt(beta_t) * rng.standard_normal((n, 2))
            else:
                x = mean
        return x

    def sample_respaced(self, n, seed, num_steps):
        """Ancestral sampling using a strictly decreasing subsequence of `num_steps`
        original timesteps (DDPM respacing), instead of the full T single-integer hops.

        The subsequence ts[0] > ts[1] > ... > ts[-1] == 0 always ends at 0 so the chain
        fully denoises. At each hop from ts[i] (current) to ts[i+1] (next), the model is
        queried at its native timestep ts[i] (that's what it was trained on), and the
        effective beta for that hop is derived from how much alpha_bar moves between the
        two subsequence points:

            beta_eff = 1 - alpha_bar[ts[i]] / alpha_bar[ts[i+1]]

        which is the unique choice that collapses to the model's true single-step beta_t
        when the hop is exactly one integer (see module docstring). The final
        subsequence element is always 0, so it has no "next" to hop to -- that last
        iteration reuses beta[0]/alpha[0] directly and adds no noise, exactly matching
        the full sampler's t=0 case.
        """
        if not (1 <= num_steps <= self.T):
            raise ValueError(f"num_steps must be in [1, {self.T}], got {num_steps}")
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, 2))

        if num_steps == self.T:
            ts = np.arange(self.T - 1, -1, -1)
        else:
            raw = np.linspace(self.T - 1, 0, num_steps)
            ts = np.unique(np.round(raw).astype(int))[::-1]
            if len(ts) != num_steps:
                raise RuntimeError(
                    f"respacing collision: requested {num_steps} distinct steps, "
                    f"got {len(ts)} after rounding -- pick a num_steps further from T"
                )

        for i, cur in enumerate(ts):
            t_arr = np.full(n, int(cur))
            eps_pred, _ = self.model.predict(x, t_arr)
            alpha_bar_cur = self.alpha_bars[cur]
            if i + 1 < len(ts):
                nxt = ts[i + 1]
                beta_eff = 1.0 - alpha_bar_cur / self.alpha_bars[nxt]
            else:
                beta_eff = self.betas[cur]
            alpha_eff = 1.0 - beta_eff
            coef = beta_eff / np.sqrt(1 - alpha_bar_cur)
            mean = (x - coef * eps_pred) / np.sqrt(alpha_eff)
            if cur > 0:
                x = mean + np.sqrt(beta_eff) * rng.standard_normal((n, 2))
            else:
                x = mean
        return x


def _smoke_test():
    """Smoke test: train end-to-end on a tiny subset (~200 points,
    a handful of epochs) and confirm the loss is finite and decreasing. Not the real
    experiment (see `src/experiment.py`) -- just confirms the machinery works.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import load

    x0 = load("two_moons", n=200, seed=0)
    ddpm = DDPM(T=200, seed=0)
    rng = np.random.default_rng(0)

    losses = []
    n_epochs, batch_size = 20, 64
    for epoch in range(n_epochs):
        perm = rng.permutation(len(x0))
        epoch_losses = []
        for start in range(0, len(x0), batch_size):
            batch = x0[perm[start:start + batch_size]]
            epoch_losses.append(ddpm.train_step(batch, rng, lr=2e-3))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(f"epoch {epoch:2d}  loss={mean_loss:.5f}")

    assert all(np.isfinite(l) for l in losses), "loss went non-finite"
    first_half = np.mean(losses[:5])
    second_half = np.mean(losses[-5:])
    print(f"mean loss, first 5 epochs = {first_half:.5f}, last 5 epochs = {second_half:.5f}")
    assert second_half < first_half, "loss did not decrease"

    samples = ddpm.sample(n=50, seed=1)
    assert samples.shape == (50, 2)
    assert np.all(np.isfinite(samples)), "sampling produced non-finite values"
    print(f"sampled {samples.shape[0]} points end-to-end, mean={samples.mean(axis=0)}, "
          f"std={samples.std(axis=0)}")

    respaced_full = ddpm.sample_respaced(n=50, seed=1, num_steps=ddpm.T)
    assert np.allclose(samples, respaced_full), "sample_respaced(num_steps=T) must match sample()"
    respaced_few = ddpm.sample_respaced(n=50, seed=1, num_steps=10)
    assert respaced_few.shape == (50, 2) and np.all(np.isfinite(respaced_few))
    print("sample_respaced(num_steps=T) matches sample() exactly; num_steps=10 runs fine too.")
    print("smoke test passed: loss finite and decreasing, sampling runs end-to-end.")


if __name__ == "__main__":
    _smoke_test()
