"""Audit the v5 canonical delivery without changing any fitted model."""
import numpy as np
import pandas as pd
from m3_ml.fusion.model import MEDIAN,metrics,mature_before
from m3_ml.distribution.tails import ALL_COLUMNS,MAIN
from m3_ml.distribution.quantiles import crps
from m3_ml.fusion_v3.data import RAW,LAG
from m3_ml.fusion_v5.data import OUT,OLD,ROOT,KEY,HORIZONS,EXPERTS,read,write_json,sha,protected
from m3_ml.fusion_v5.run import input_hashes
from m3_ml.fusion_v5.risk_safety import envelope
from m4_reports.state_strategy import DELIVERY


def main():
    protected();model=read(OUT/'selected.json');manifest=read(ROOT/'m4_reports/final_strategy_manifest.json')
    payload=read(DELIVERY/'presentation_data.json');digest=sha(OUT/'selected.json');audit=read(OUT/'test_audit.json')
    assert model['input_sha256']==input_hashes()
    assert digest==manifest['selected_sha256']==payload['selected_sha256']==audit['selected_sha256']
    assert model['version']==manifest['active_version']==payload['version']=='state_reliability_v5'
    assert manifest['forecast_mode']==payload['forecast_mode']=='daily_update'
    assert manifest['horizons']==payload['horizons']==list(HORIZONS) and len(manifest['superseded'])==4
    assert ROOT/manifest['output']==DELIVERY and ROOT/manifest['model_output']==OUT
    assert ROOT/manifest['forecasts']==DELIVERY/'forecasts.csv' and ROOT/manifest['metrics']==DELIVERY/'metrics.csv'
    assert sha(DELIVERY/'forecasts.csv')==sha(OUT/'test_forecasts.csv') and sha(DELIVERY/'metrics.csv')==sha(OUT/'test_metrics.csv')
    f=pd.read_csv(DELIVERY/'forecasts.csv',parse_dates=['origin_date','target_date','futures_quote_date','futures_available_date'])
    assert len(f)==len(payload['forecasts'])==68 and f['mode'].eq('daily_update').all() and not f.duplicated(KEY).any()
    assert f.groupby('horizon').size().to_dict()=={h:17 for h in HORIZONS}
    raw=pd.read_csv(RAW,parse_dates=['quote_date']).set_index('quote_date')
    assert f.futures_available_date.le(f.origin_date).all() and all(d+LAG==a for d,a in zip(f.futures_quote_date,f.futures_available_date))
    np.testing.assert_allclose(raw.close.reindex(f.futures_quote_date),f.futures_price,rtol=0,atol=1e-12)
    q=f[ALL_COLUMNS].to_numpy();assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-12).all()
    np.testing.assert_allclose(q[:,MEDIAN],f.center,atol=1e-14)
    np.testing.assert_allclose(f.origin_price*np.exp(f.center),f.forecast,atol=1e-10)
    weights=f[[e+'_weight' for e in EXPERTS]].to_numpy()
    assert (weights>=-1e-12).all() and (f.incumbent_weight>=.5-1e-12).all()
    np.testing.assert_allclose(weights.sum(axis=1),1.,atol=1e-10)
    np.testing.assert_allclose((weights*f[[e+'_expert' for e in EXPERTS]].to_numpy()).sum(axis=1),f.forecast,atol=1e-10)
    previous=pd.read_csv(OLD/'test_forecasts.csv',parse_dates=['origin_date','target_date'])
    previous=previous.set_index(KEY).reindex(pd.MultiIndex.from_frame(f[KEY])).reset_index()
    np.testing.assert_allclose(previous.forecast,f.v4_forecast,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(previous.actual,f.actual,equal_nan=True,atol=1e-12)
    np.testing.assert_array_equal(previous.target_date,f.target_date)
    base_low=previous['q_0.050']-previous.center+f.center;base_high=previous['q_0.950']-previous.center+f.center
    assert (f['q_0.050']<=base_low+1e-12).all() and (f['q_0.950']>=base_high-1e-12).all()
    capped=1.5*(np.exp(base_high)-np.exp(base_low))
    assert (np.exp(f['q_0.950'])-np.exp(f['q_0.050'])<=capped+1e-10).all()
    saved=pd.read_csv(DELIVERY/'metrics.csv').set_index('horizon').sort_index()
    recomputed=metrics(f,'integrated_v5').set_index('horizon').sort_index()
    cols=['n','rmse_usd','mae_usd','rw_rmse_usd','rmse_vs_rw','crps_log','coverage90','width90_usd']
    np.testing.assert_allclose(saved[cols],recomputed[cols],atol=1e-10,rtol=1e-10,equal_nan=True)
    assert saved.n.tolist()==[17,13,0,0]
    for h,g in f[f.actual.notna()].groupby('horizon'):
        for name,column in [('v4','v4_forecast'),('futures','futures_price')]:
            value=np.sqrt(np.mean((g[column]-g.actual)**2))
            np.testing.assert_allclose(saved.loc[h,name+'_rmse_usd'],value)
            np.testing.assert_allclose(saved.loc[h,'rmse_vs_'+name],saved.loc[h,'rmse_usd']/value)
    historical=pd.read_csv(OUT/'point_crossfit.csv',parse_dates=['origin_date','label_available_date'])
    proposal=pd.read_csv(OUT/'point_proposal.csv',parse_dates=['origin_date','label_available_date'])
    actual=historical.set_index(KEY).forecast.reindex(pd.MultiIndex.from_frame(proposal[KEY]))
    np.testing.assert_allclose(actual,proposal.forecast)
    assert historical.label_available_date.max()<=pd.Timestamp('2026-09-30')
    summary=pd.read_csv(DELIVERY/'historical_point_metrics.csv').set_index('horizon')
    for name,column in [('v5','forecast'),('v4','v4_forecast'),('rw','origin_price'),('futures','futures_price')]:
        proposal[name+'_loss']=(proposal[column]-proposal.actual)**2
        for h,g in proposal.groupby('horizon'):
            np.testing.assert_allclose(summary.loc[h,name+'_rmse_usd'],np.sqrt(g[name+'_loss'].mean()))
    early=mature_before(historical,'2025-09-01');early['loss']=(early.origin_price-early.actual)**2
    scales=early.groupby('horizon').loss.mean();complete=proposal.groupby('origin_date').filter(lambda g:len(g)==4).copy()
    for name in ('v5','v4','rw','futures'):complete[name+'_normalized']=complete[name+'_loss']/complete.horizon.map(scales)
    origin=complete.groupby('origin_date')[[name+'_normalized' for name in ('v5','v4','rw','futures')]].mean()
    inference=read(OUT/'historical_inference.json');assert inference==payload['historical_inference'] and len(origin)==202
    for row in inference['comparisons']:
        if row['scope']=='primary_four_horizons':ratio=np.sqrt(origin.v5_normalized.mean()/origin[row['reference']+'_normalized'].mean())
        else:
            g=proposal[proposal.horizon.eq(int(row['scope'][1:]))];ratio=np.sqrt(g.v5_loss.mean()/g[row['reference']+'_loss'].mean())
        np.testing.assert_allclose(row['rmse_ratio'],ratio)
    risk=pd.read_csv(OUT/'risk_forward_forecasts.csv',parse_dates=['origin_date'])
    raw_risk=pd.read_csv(OUT/'risk_forward_proposal.csv',parse_dates=['origin_date'])
    old_risk=pd.read_csv(OLD/'risk_forward_forecasts.csv',parse_dates=['origin_date'])
    old_risk=old_risk.set_index(KEY).reindex(pd.MultiIndex.from_frame(risk[KEY])).reset_index()
    rq,fraction=envelope(raw_risk[ALL_COLUMNS],old_risk[ALL_COLUMNS],raw_risk.center,1.5)
    np.testing.assert_allclose(rq,risk[ALL_COLUMNS],atol=1e-12)
    np.testing.assert_allclose(fraction,risk.risk_safety_fraction,atol=1e-12)
    rs=pd.read_csv(DELIVERY/'historical_risk_metrics.csv').set_index('horizon')
    new_loss=crps(rq[:,MAIN],risk.y);old_loss=crps(old_risk[ALL_COLUMNS].to_numpy()[:,MAIN],old_risk.y)
    for h,indices in risk.groupby('horizon').groups.items():
        g=risk.loc[indices];previous_g=old_risk.loc[indices]
        np.testing.assert_allclose(rs.loc[h,'crps_vs_v4'],new_loss[indices].mean()/old_loss[indices].mean())
        np.testing.assert_allclose(rs.loc[h,'crps_vs_hs'],new_loss[indices].mean()/g.hs_crps.mean())
        coverage=g.y.between(g['q_0.050'],g['q_0.950']).mean()
        np.testing.assert_allclose(rs.loc[h,'coverage90'],coverage)
        width=(g.origin_price*(np.exp(g['q_0.950'])-np.exp(g['q_0.050']))).mean()
        old_width=(previous_g.origin_price*(np.exp(previous_g['q_0.950'])-np.exp(previous_g['q_0.050']))).mean()
        np.testing.assert_allclose(rs.loc[h,'width90_usd'],width);np.testing.assert_allclose(rs.loc[h,'width_vs_v4'],width/old_width)
    assert read(OUT/'promotion_gate.json')==model['promotion_gate']==payload['promotion_gate'] and model['promotion_gate']['passed']
    for name in ('point_fit_audit.json','point_selection_audit.json','support_audit.json'):
        for row in read(OUT/name):
            if row.get('latest_label_release') is not None:assert row['latest_label_release']<=row['cutoff']<='2026-09-30'
    audit=read(OUT/'risk_audit.json')
    for row in audit['fits']+audit['selections']:assert row['latest_label_release']<=row['cutoff']<='2026-09-30'
    for row in read(OUT/'final_fit_audit.json'):assert row['latest_label_release']<='2026-09-30'
    sensitivity=pd.read_csv(DELIVERY/'module_input_sensitivity.csv')
    assert len(sensitivity)==272 and set(sensitivity.member)=={'m1','m2','lstm','ml'}
    assert all(sensitivity.groupby('member').local_price_sensitivity.apply(lambda s:np.max(np.abs(s)))>0)
    for name in ('october_forecasts.png','strategy_summary.png','historical_validation.png'):assert (DELIVERY/'figures'/name).stat().st_size>10000
    assert read(DELIVERY/'delivery_scope.json')==payload['delivery_scope']
    for h,n in ((1,17),(5,13)):
        path=ROOT/f'data/video/fan_integrated_h{h}.csv';meta=read(path.with_suffix('.json'))
        assert meta['selected_sha256']==digest and meta['version']==model['version'] and meta['n']==n
        video=pd.read_csv(path,parse_dates=['date']);g=f[f.horizon.eq(h)&f.actual.notna()].sort_values('target_date')
        np.testing.assert_array_equal(video.date,g.target_date)
        np.testing.assert_allclose(video.actual,g.actual,rtol=0,atol=.000051)
        for column in (c for c in ALL_COLUMNS if c.startswith('q_')):
            np.testing.assert_allclose(video[column.replace('q_','p_')],g.origin_price*np.exp(g[column]),rtol=0,atol=.000051)
    write_json(DELIVERY/'delivery_audit.json',dict(status='passed',selected_sha256=digest,
        forecast_sha256=sha(DELIVERY/'forecasts.csv'),forecast_mode='daily_update',rows=68,
        historical_and_october_metrics_recomputed=True,source_timing_verified=True,conditional_weights_verified=True,
        risk_envelope_and_width_cap_verified=True,member_inputs_verified=True,video_data_verified=True,
        previous_v4_preserved=True,no_october_parameter_changes=True,conditional_historical_evidence=True))
    print('V5 delivery audit passed:',digest,flush=True)


if __name__=='__main__':main()
