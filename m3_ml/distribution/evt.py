"""EVT methods for tail risk (PLAN.md addendum).

GARCH-EVT (McNeil & Frey 2000): filter returns with GARCH / GJR-GARCH
(Student-t likelihood, zero mean), take the in-sample standardized residuals,
model them as an empirical body with GPD tails, and simulate h-day paths
through the GARCH recursion with innovations drawn from that distribution.
"""

import numpy as np
from .benchmarks import first_variance, fit_garch_state, public_simulation

from .quantiles import TAUS
from .tails import fit_gpd_tails, semiparametric_ppf

SIMULATIONS = 10_000
SEED = 0


def fit_garch(inp, refit, last, vol):
    """Fit once on the public contiguous prefix and retain later filtering data."""
    return fit_garch_state(inp, refit, last, vol)


def first_step_variance(res, inp, origins):
    """First unobserved-quote variance for each publication-aware origin."""
    return first_variance(res, inp, origins)[0]


def simulate_sums(params, var1, h, draw, sims=SIMULATIONS, seed=SEED):
    """Sum of h simulated daily returns per origin (origins x sims), GJR recursion:
    s2' = omega + (alpha + gamma * 1[e < 0]) e^2 + beta s2; draw(u) maps uniforms to innovations."""
    omega, alpha = params['omega'], params['alpha[1]']
    gamma, beta = params.get('gamma[1]', 0.0), params['beta[1]']
    rng = np.random.default_rng(seed)
    s2 = np.repeat(np.asarray(var1, dtype=float)[:, None], sims, axis=1)
    total = np.zeros_like(s2)
    for _ in range(h):
        eps = np.sqrt(s2) * draw(rng.random(s2.shape))
        total += eps
        s2 = omega + (alpha + gamma * (eps < 0)) * eps ** 2 + beta * s2
    return total


def predict_garch_evt(inp, h, refit, train, origins, vol, tail, taus=TAUS):
    last = int(origins[-1])
    res = fit_garch(inp, refit, last, vol)
    z = res.std_resid.dropna().to_numpy()
    model = fit_gpd_tails(z, tail)
    return public_simulation(inp, h, origins, res, lambda u: semiparametric_ppf(model, u),
                             SIMULATIONS, taus)


EVT_MODELS = {
    'garch_evt': (predict_garch_evt, [{'vol': v, 'tail': f} for v in ('garch', 'gjr') for f in (0.05, 0.10)]),
}
