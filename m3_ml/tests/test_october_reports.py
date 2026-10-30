"""The short October evaluation must not emit inferential winner claims."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from m3_ml.distribution import analyze
from m3_ml.distribution.tails import ALL_COLUMNS, ALL_TAUS
from m3_ml.tabular import analyze_direct


def sample_forecasts(h=1):
    dates = pd.bdate_range('2026-10-01', '2026-10-23')
    y = np.linspace(-.025, .04, len(dates))
    blocks = []
    for method, scale in [('hs', .08), ('ewma', .06), ('garch', .05),
                          ('a_ridge', .04), ('c_mixture', .03)]:
        f = pd.DataFrame(np.tile(norm.ppf(ALL_TAUS) * scale, (len(dates), 1)), columns=ALL_COLUMNS)
        f['origin_date'] = dates - pd.offsets.BDay(1)
        f['target_date'] = dates
        f['origin_price'], f['actual_log_return'] = 100., y
        f['horizon'], f['method'] = h, method
        blocks.append(f)
    return pd.concat(blocks, ignore_index=True)


def test_distribution_final_tables_keep_losses_without_significance():
    f = sample_forecasts()
    missing = f.iloc[:1].assign(horizon=60, actual_log_return=np.nan)
    emitted = pd.concat([f, missing], ignore_index=True)
    for result in (analyze.metrics(analyze.scored(f)), analyze.risk_metrics(f)):
        final = analyze.complete_metrics(result, emitted, 'daily_update')
        pvalues = [c for c in final if c.startswith('dm_p_') or c.endswith('_p')]
        assert final[pvalues].isna().all().all()
        for flag in ('skill', 'risk_skill'):
            if flag in final:
                assert not final[flag].any()
        assert final.loc[final.horizon.eq(1), 'n'].eq(17).all()
        assert final.loc[final.horizon.eq(60), 'n'].eq(0).all()
        assert final.loc[final.horizon.eq(60), 'status'].eq('unscored_no_matured_test_targets').all()
        loss = 'crps' if 'crps' in final else 'fz0'
        assert np.isfinite(final.loc[final.horizon.eq(1), loss]).all()


def test_retrospective_comparison_also_withholds_pvalues():
    f = sample_forecasts(h=5)
    out = analyze.non_blind(analyze.scored(f), f)
    assert out[['dm_p_crps', 'dm_p_fz0']].isna().all().all()
    assert np.isfinite(out[['crps_ratio', 'fz0_diff']]).all().all()


def test_tabular_regime_summary_is_descriptive():
    f = pd.DataFrame({'horizon': [1] * 17, 'model': ['example'] * 17,
                      'se': np.linspace(.2, .4, 17), 'se_rw': np.linspace(1., 2., 17),
                      'moved': [True] * 17, 'hit': [True] * 17, 'regime': ['low vol'] * 17})
    out = analyze_direct.summary(f, ['horizon', 'regime'])
    assert out.dm_pvalue.isna().all()
    assert out.inference.eq('descriptive_short_window').all()
    assert out.rmse_vs_rw.lt(1).all()
