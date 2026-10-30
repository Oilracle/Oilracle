"""Verify the daily rolling delivery against the unchanged sealed source."""
import json
import numpy as np
import pandas as pd
from m3_ml.fusion_v3.data import OUT,BASE,ROOT,RAW,KEY,LAG,sha,write_json
from m3_ml.fusion_v3.run import seal_inputs
from m3_ml.distribution.tails import ALL_COLUMNS
from m3_ml.fusion.model import MEDIAN,metrics
from m3_ml.common.protocol import HORIZONS

DELIVERY=OUT/'rolling'


def main():
    manifest=json.loads((ROOT/'m4_reports/final_strategy_manifest.json').read_text())
    if manifest['active_version']=='short_bias_v6':
        from m4_reports.short_strategy import audit
        return audit()
    if manifest['active_version']=='state_reliability_v5':
        from m4_reports.audit_state import main as audit_state
        return audit_state()
    if manifest['active_version']=='rolling_long_history_v4':
        from m4_reports.audit_upgrade import main as audit_upgrade
        return audit_upgrade()
    payload=json.loads((DELIVERY/'presentation_data.json').read_text())
    model=json.loads((OUT/'selected.json').read_text())
    audit=json.loads((OUT/'test_audit.json').read_text())
    digest=sha(OUT/'selected.json')
    assert model['input_sha256']==seal_inputs()
    assert manifest['selected_sha256']==payload['selected_sha256']==audit['selected_sha256']==digest
    assert manifest['active_version']==payload['version']==model['version']=='market_residual_v3'
    assert ROOT/manifest['output']==DELIVERY
    assert ROOT/manifest['model_output']==OUT
    assert manifest['forecast_mode']==payload['forecast_mode']=='daily_update'
    assert manifest['horizons']==payload['horizons']==list(HORIZONS)
    assert 'fixed_path_metrics' not in payload
    assert len(manifest['superseded'])==2
    dates=['origin_date','futures_quote_date','futures_available_date']
    original=pd.read_csv(OUT/'test_forecasts.csv',parse_dates=dates)
    forecasts=pd.read_csv(DELIVERY/'forecasts.csv',parse_dates=dates)
    assert len(original)==87 and len(forecasts)==len(payload['forecasts'])==68
    expected=original[original['mode'].eq('daily_update')].reset_index(drop=True)
    pd.testing.assert_frame_equal(forecasts,expected,rtol=1e-12,atol=1e-12)
    assert forecasts['mode'].eq('daily_update').all()
    assert all(row['mode']=='daily_update' for row in payload['forecasts']+payload['test_metrics'])
    assert set(forecasts.horizon)==set(HORIZONS)
    assert forecasts.groupby('horizon').size().to_dict()=={h:17 for h in HORIZONS}
    assert not forecasts.duplicated(['mode',*KEY]).any()
    assert forecasts.futures_available_date.le(forecasts.origin_date).all()
    assert all(d+LAG==a for d,a in zip(forecasts.futures_quote_date,forecasts.futures_available_date))
    raw=pd.read_csv(RAW,parse_dates=['quote_date']).set_index('quote_date')
    np.testing.assert_allclose(raw.close.reindex(forecasts.futures_quote_date),forecasts.futures_price,rtol=0,atol=1e-12)
    q=forecasts[ALL_COLUMNS].to_numpy()
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-12).all()
    np.testing.assert_allclose(q[:,MEDIAN],forecasts.center,atol=1e-14)
    np.testing.assert_allclose(forecasts.origin_price*np.exp(forecasts.center),forecasts.forecast,atol=1e-10)
    saved=pd.read_csv(DELIVERY/'metrics.csv')
    original_metrics=pd.read_csv(OUT/'test_metrics.csv')
    pd.testing.assert_frame_equal(saved,original_metrics[original_metrics['mode'].eq('daily_update')].reset_index(drop=True),rtol=1e-12,atol=1e-12)
    recalculated=metrics(forecasts,'integrated_v3')
    keys=['mode','horizon'];cols=['n','rmse_usd','rw_rmse_usd','rmse_vs_rw','crps_log','coverage90','width90_usd']
    a=saved.set_index(keys).sort_index();b=recalculated.set_index(keys).sort_index()
    np.testing.assert_allclose(a[cols],b[cols],rtol=1e-10,atol=1e-10,equal_nan=True)
    assert a.loc['daily_update','n'].reindex([1,5,20,60]).tolist()==[17,13,0,0]
    history=pd.read_csv(OUT/'point_crossfit.csv',parse_dates=['origin_date','label_available_date'])
    assert history.origin_date.max()<=pd.Timestamp('2026-09-30') and history.label_available_date.max()<=pd.Timestamp('2026-09-30')
    entries=payload['historical_inference']['comparisons']
    source_inference=json.loads((OUT/'historical_inference.json').read_text())
    expected_entries=[e for e in source_inference['comparisons'] if e['scope'] in {f'h{h}' for h in HORIZONS}]
    assert entries==expected_entries
    for entry in entries:
        if entry['reference']!='rw':continue
        g=history[history.horizon.eq(int(entry['scope'][1:]))&history.origin_date.ge('2025-09-01')]
        ratio=np.sqrt(np.mean((g.forecast-g.actual)**2)/np.mean((g.origin_price-g.actual)**2))
        np.testing.assert_allclose(ratio,entry['rmse_ratio'])
    sensitivity=pd.read_csv(DELIVERY/'module_input_sensitivity.csv')
    assert len(sensitivity)==272 and set(sensitivity.horizon)==set(HORIZONS)
    assert set(sensitivity.member)=={'m1','m2','lstm','ml'}
    assert all(sensitivity.groupby('member').local_price_sensitivity.apply(lambda x:np.max(np.abs(x)))>0)
    scope=json.loads((DELIVERY/'delivery_scope.json').read_text())
    assert scope['forecast_mode']=='daily_update' and scope==payload['delivery_scope']
    for figure in ('october_forecasts.png','strategy_summary.png','historical_validation.png'):
        assert (DELIVERY/'figures'/figure).stat().st_size>10000
    assert sha(BASE/'selected.json')=='fb5dede1373a6c63e488ddab6c7e1704dd41f0d0f6eaa725f8917207343b6dff'
    assert sha(BASE/'test_forecasts.csv')=='38f40fc56f1bd4c1c3f07160edfd1e93d6144482b6b3851355f118a7fbf6c454'
    assert digest=='12d803737e1804b2e69312aede92113535b38f9f7fee743c9b903ecc409c209a'
    assert sha(OUT/'test_forecasts.csv')=='9706ddb057d11a345bb6c554fa535ed7cbdcc3da768ca5f9554e4b173d41940b'
    write_json(DELIVERY/'delivery_audit.json',dict(status='passed',selected_sha256=digest,
        rolling_forecast_sha256=sha(DELIVERY/'forecasts.csv'),source_forecast_sha256=sha(OUT/'test_forecasts.csv'),
        canonical_manifest=True,forecast_mode='daily_update',rows=68,archived_rows=87,
        original_predictions_unchanged=True,metrics_recomputed=True,all_consumed_futures_quotes_verified=True,
        point_training_cutoff='2026-09-30',v2_preserved=True,no_october_parameter_changes=True))
    print('Daily rolling delivery audit passed:',digest,flush=True)


if __name__=='__main__':main()
