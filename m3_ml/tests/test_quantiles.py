import numpy as np
from scipy import stats

from m3_ml.distribution import quantiles as qs


def normal_quantiles(n, scale=1.0, loc=0.0):
    return np.tile(stats.norm.ppf(qs.TAUS, loc=loc, scale=scale), (n, 1))


def test_grid():
    assert len(qs.TAUS) == 23 and np.all(np.diff(qs.TAUS) > 0)
    assert qs.TAUS[0] == 0.01 and qs.TAUS[-1] == 0.99 and 0.5 in qs.TAUS


def test_pinball_by_hand():
    Q = np.zeros((1, 23))
    loss = qs.pinball(Q, [1.0])[0]
    assert np.allclose(loss, qs.TAUS)          # y above every quantile: tau * (y - q)
    loss = qs.pinball(Q, [-1.0])[0]
    assert np.allclose(loss, 1 - qs.TAUS)      # y below: (1 - tau) * (q - y)


def test_crps_is_proper_for_the_true_distribution():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000)
    true = qs.crps(normal_quantiles(len(y)), y).mean()
    for wrong in (normal_quantiles(len(y), 0.7), normal_quantiles(len(y), 1.4), normal_quantiles(len(y), 1, 0.3)):
        assert qs.crps(wrong, y).mean() > true


def test_crps_close_to_exact_normal_crps():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(100_000)
    exact = (y * (2 * stats.norm.cdf(y) - 1) + 2 * stats.norm.pdf(y) - 1 / np.sqrt(np.pi)).mean()
    approx = qs.crps(normal_quantiles(len(y)), y).mean()
    assert abs(approx / exact - 1) < 0.15    # 23 unequal levels: close, not identical


def test_coverage_and_pit_for_the_true_distribution():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(100_000)
    Q = normal_quantiles(len(y))
    for level in qs.LEVELS:
        assert abs(qs.covered(Q, y, level).mean() - level) < 0.01
    hist, _ = np.histogram(qs.pit(Q, y), bins=10, range=(0, 1))
    assert np.allclose(hist / len(y), 0.1, atol=0.01)


def test_rearrange_never_increases_loss():
    rng = np.random.default_rng(3)
    Q = normal_quantiles(500) + rng.normal(0, 0.5, (500, 23))
    y = rng.standard_normal(500)
    assert qs.crps(qs.rearrange(Q), y).mean() <= qs.crps(Q, y).mean() + 1e-12


def test_coverage_tests_accept_truth_and_reject_bad():
    rng = np.random.default_rng(4)
    good = rng.random(2000) < 0.9
    bad = rng.random(2000) < 0.8
    assert qs.kupiec_pvalue(good, 0.9) > 0.05 and qs.kupiec_pvalue(bad, 0.9) < 0.01
    assert qs.hac_mean_pvalue(good - 0.9, 1) > 0.05 and qs.hac_mean_pvalue(bad - 0.9, 1) < 0.01
    clustered = np.repeat(rng.random(400) < 0.9, 5)  # violations come in runs
    assert qs.christoffersen_pvalue(good) > 0.05 and qs.christoffersen_pvalue(clustered) < 0.01
