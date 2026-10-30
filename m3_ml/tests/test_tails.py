import numpy as np
from scipy import stats

from m3_ml.distribution import tails as tl
from m3_ml.distribution.quantiles import TAUS


def normal_all(n, scale=1.0):
    return np.tile(stats.norm.ppf(tl.ALL_TAUS) * scale, (n, 1))


def test_grid():
    assert len(tl.ALL_TAUS) == 63 and len(np.unique(tl.ALL_TAUS)) == 63
    assert tl.MAIN.sum() == 23 and np.allclose(tl.ALL_TAUS[tl.MAIN], TAUS)


def test_var_es_match_the_normal_distribution():
    Q_main, Q_full = tl.split(normal_all(1))
    for a in tl.ALPHAS:
        exact_es = -stats.norm.pdf(stats.norm.ppf(a)) / a          # E[Z | Z < q_a]
        for side in tl.SIDES:
            v, e = tl.var_es(Q_main, Q_full, a, side)
            assert np.isclose(v[0], stats.norm.ppf(a))             # symmetric: same in both tails
            assert abs(e[0] / exact_es - 1) < 0.03                 # midpoint rule, 4-20 points


def test_fz0_prefers_the_true_var_es():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(400_000)
    a = 0.025
    v, e = stats.norm.ppf(a), -stats.norm.pdf(stats.norm.ppf(a)) / a
    true = np.nanmean(tl.fz0(y, np.full_like(y, v), np.full_like(y, e), a))
    for dv, de in ((0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)):
        wrong = np.nanmean(tl.fz0(y, np.full_like(y, v + dv), np.full_like(y, e + de), a))
        assert wrong > true


def test_z2_accepts_truth_and_rejects_too_thin_tails():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(1000)
    _, p_true = tl.z2_test(normal_all(1000), y, 0.025, 'lower', draws=2000)
    _, p_thin = tl.z2_test(normal_all(1000, 0.6), y, 0.025, 'lower', draws=2000)
    assert p_true > 0.05 and p_thin < 0.01


def test_gpd_tail_recovers_a_student_t_tail():
    rng = np.random.default_rng(2)
    z = stats.t.rvs(4, size=200_000, random_state=rng)
    model = tl.fit_gpd_tails(z, 0.10)
    for u in (0.001, 0.005, 0.01, 0.99, 0.995):
        assert abs(tl.semiparametric_ppf(model, u) / stats.t.ppf(u, 4) - 1) < 0.05


def test_direction_metrics():
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1.0, -1.0, -1.0, 1.0])
    m = tl.direction_metrics(p, y)
    assert m['accuracy'] == 0.5 and m['precision_up'] == 0.5 and m['recall_up'] == 0.5
    assert np.isclose(m['brier'], np.mean((p - (y > 0)) ** 2))
