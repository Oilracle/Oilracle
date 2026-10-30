"""Check the v4 delivery against frozen models, causal fits and saved evidence."""
import json
import numpy as np
import pandas as pd
from m3_ml.common.protocol import HORIZONS
from m3_ml.fusion.model import MEDIAN,metrics,mature_before
from m3_ml.distribution.tails import ALL_COLUMNS,MAIN
from m3_ml.distribution.quantiles import crps
from m3_ml.fusion_v3.data import RAW,LAG
from m3_ml.fusion_v4.data import OUT,OLD,ROOT,KEY,sha,write_json
from m3_ml.fusion_v4.run import inputs,protected
from m4_reports.rolling_upgrade import DELIVERY,read


def main():
    protected();model=read(OUT/'selected.json');manifest=read(ROOT/'m4_reports/final_strategy_manifest.json')
    payload=read(DELIVERY/'presentation_data.json');audit=read(OUT/'test_audit.json');digest=sha(OUT/'selected.json')
    assert model['input_sha256']==inputs()
    assert digest==manifest['selected_sha256']==payload['selected_sha256']==audit['selected_sha256']
    assert model['version']==manifest['active_version']==payload['version']=='rolling_long_history_v4'
    assert manifest['forecast_mode']==payload['forecast_mode']==model['forecast_mode']=='daily_update'
    assert manifest['horizons']==payload['horizons']==list(HORIZONS)
    assert ROOT/manifest['output']==DELIVERY and ROOT/manifest['model_output']==OUT
    assert ROOT/manifest['forecasts']==DELIVERY/'forecasts.csv'
    assert ROOT/manifest['metrics']==DELIVERY/'metrics.csv'
    assert len(manifest['superseded'])==3 and 'fixed_path_metrics' not in payload
    assert sha(DELIVERY/'forecasts.csv')==sha(OUT/'test_forecasts.csv')
    assert sha(DELIVERY/'metrics.csv')==sha(OUT/'test_metrics.csv')
    dates=['origin_date','target_date','futures_quote_date','futures_available_date']
    f=pd.read_csv(DELIVERY/'forecasts.csv',parse_dates=dates)
    assert len(f)==len(payload['forecasts'])==68 and f['mode'].eq('daily_update').all()
    assert f.groupby('horizon').size().to_dict()=={h:17 for h in HORIZONS} and not f.duplicated(KEY).any()
    assert f.futures_available_date.le(f.origin_date).all()
    assert all(d+LAG==a for d,a in zip(f.futures_quote_date,f.futures_available_date))
    raw=pd.read_csv(RAW,parse_dates=['quote_date']).set_index('quote_date')
    np.testing.assert_allclose(raw.close.reindex(f.futures_quote_date),f.futures_price,rtol=0,atol=1e-12)
    q=f[ALL_COLUMNS].to_numpy()
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-12).all()
    np.testing.assert_allclose(q[:,MEDIAN],f.center,atol=1e-14)
    np.testing.assert_allclose(f.origin_price*np.exp(f.center),f.forecast,atol=1e-10)
    prior=pd.read_csv(OLD/'test_forecasts.csv',parse_dates=['origin_date','target_date'])
    prior=prior[prior['mode'].eq('daily_update')].set_index(KEY).reindex(pd.MultiIndex.from_frame(f[KEY]))
    np.testing.assert_allclose(prior.forecast,f.incumbent_forecast,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(prior.actual,f.actual,rtol=0,atol=1e-12,equal_nan=True)
    np.testing.assert_array_equal(prior.target_date,f.target_date)
    beta=f.horizon.map({int(h):d['upgrade_weight'] for h,d in model['guard'].items()})
    np.testing.assert_allclose(f.forecast,f.incumbent_forecast+beta*(f.proposal_forecast-f.incumbent_forecast))
    saved=pd.read_csv(DELIVERY/'metrics.csv').set_index('horizon').sort_index()
    fresh=metrics(f,'integrated_v4').set_index('horizon').sort_index()
    columns=['n','rmse_usd','rw_rmse_usd','rmse_vs_rw','crps_log','coverage90','width90_usd']
    np.testing.assert_allclose(saved[columns],fresh[columns],rtol=1e-10,atol=1e-10,equal_nan=True)
    assert saved.n.tolist()==[17,13,0,0]
    for h,g in f[f.actual.notna()].groupby('horizon'):
        for reference,column in [('v3','incumbent_forecast'),('futures','futures_price')]:
            value=np.sqrt(np.mean((g[column]-g.actual)**2))
            np.testing.assert_allclose(saved.loc[h,reference+'_rmse_usd'],value)
            np.testing.assert_allclose(saved.loc[h,'rmse_vs_'+reference],saved.loc[h,'rmse_usd']/value)
    history=pd.read_csv(OUT/'point_crossfit.csv',parse_dates=['origin_date','label_available_date'])
    old_history=pd.read_csv(OLD/'point_crossfit.csv',parse_dates=['origin_date'])
    history=history.merge(old_history[KEY+['forecast']].rename(columns={'forecast':'v3_forecast'}),on=KEY,validate='one_to_one')
    assert history.origin_date.max()<=pd.Timestamp('2026-09-30')
    assert history.label_available_date.max()<=pd.Timestamp('2026-09-30')
    for name,column in [('v4','forecast'),('v3','v3_forecast'),('rw','origin_price'),('futures','futures_price')]:
        history[name+'_loss']=(history[column]-history.actual)**2
    hist_metrics=pd.read_csv(OUT/'historical_point_metrics.csv')
    late=history[history.origin_date.ge('2025-09-01')]
    for row in hist_metrics.itertuples():
        g=(late if row.period=='later_history' else history).loc[lambda x:x.horizon.eq(row.horizon)]
        assert len(g)==row.n
        for name in ('v4','v3','rw','futures'):
            np.testing.assert_allclose(getattr(row,name+'_rmse_usd'),np.sqrt(g[name+'_loss'].mean()))
    scales=mature_before(history,'2025-09-01').groupby('horizon').rw_loss.mean()
    complete=late.groupby('origin_date').filter(lambda g:len(g)==len(HORIZONS)).copy()
    complete['scale']=complete.horizon.map(scales)
    for name in ('v4','v3','rw','futures'):complete[name+'_normalized']=complete[name+'_loss']/complete.scale
    origin=complete.groupby('origin_date')[[n+'_normalized' for n in ('v4','v3','rw','futures')]].mean()
    inference=read(OUT/'historical_inference.json');assert payload['historical_inference']==inference
    assert len(origin)==inference['primary']['n_origins']==202
    for row in inference['comparisons']:
        if row['scope']=='primary_four_horizons':
            value=np.sqrt(origin.v4_normalized.mean()/origin[row['reference']+'_normalized'].mean())
        else:
            g=late[late.horizon.eq(int(row['scope'][1:]))]
            value=np.sqrt(g.v4_loss.mean()/g[row['reference']+'_loss'].mean())
        np.testing.assert_allclose(row['rmse_ratio'],value)
    gate=read(OUT/'point_promotion_gate.json')
    assert gate==model['historical_promotion_gate']==payload['promotion_gate']
    np.testing.assert_allclose(gate['aggregate_rmse_vs_v3'],np.sqrt(origin.v4_normalized.mean()/origin.v3_normalized.mean()))
    assert gate['passed'] and gate['aggregate_rmse_vs_v3']<1 and gate['worst_horizon_rmse_vs_v3']<=1.02 and not gate['october_used']
    risk_metrics=pd.read_csv(OUT/'historical_risk_metrics.csv').set_index(['period','horizon'])
    for period in ('forward','development'):
        r=pd.read_csv(OUT/f'risk_{period}_forecasts.csv')
        for h,g in r.groupby('horizon'):
            rq=g[ALL_COLUMNS].to_numpy();s=risk_metrics.loc[(period,h)]
            assert len(g)==s.n and (np.diff(rq,axis=1)>=-1e-12).all()
            np.testing.assert_allclose(rq[:,MEDIAN],g.center,atol=1e-14)
            np.testing.assert_allclose(s.crps_vs_hs,crps(rq[:,MAIN],g.y).mean()/g.hs_crps.mean())
            coverage=((g.y>=g['q_0.050'])&(g.y<=g['q_0.950'])).mean()
            np.testing.assert_allclose(s.coverage90,coverage)
    for name in ('long_fit_audit.json','family_selection_audit.json','stack_fit_audit.json'):
        for row in read(OUT/name):
            if row.get('latest_label_release') is not None:assert row['latest_label_release']<=row['cutoff']<='2026-09-30'
    for name in ('guard_audit.json','risk_audit.json'):
        for row in read(OUT/name)['fits']:
            if row.get('latest_label_release') is not None:assert row['latest_label_release']<=row['cutoff']<='2026-09-30'
    for row in read(OUT/'risk_audit.json')['selections']:assert row['latest_label_release']<=row['cutoff']
    for row in read(OUT/'risk_audit.json')['calibration']:assert row['calibration']['latest_label_release']<=row['cutoff']
    for row in read(OUT/'final_fit_audit.json'):assert row['latest_label_release']<='2026-09-30'
    for d in model['guard'].values():assert 0<=d['upgrade_weight']<=1 and d['latest_label_release']<='2026-09-30'
    for d in model['risk'].values():assert d['calibration']['latest_label_release']<='2026-09-30'
    sensitivity=pd.read_csv(DELIVERY/'module_input_sensitivity.csv')
    assert len(sensitivity)==272 and set(sensitivity.member)=={'m1','m2','lstm','ml'}
    assert all(sensitivity.groupby('member').local_price_sensitivity.apply(lambda x:np.max(np.abs(x)))>0)
    weights=pd.read_csv(DELIVERY/'component_weights.csv')
    np.testing.assert_allclose(weights.groupby('horizon').effective_weight.sum(),1.,atol=1e-8)
    for filename in ('october_forecasts.png','strategy_summary.png','historical_validation.png'):
        assert (DELIVERY/'figures'/filename).stat().st_size>10000
    assert read(DELIVERY/'delivery_scope.json')==payload['delivery_scope']
    for h,n in ((1,17),(5,13)):
        path=ROOT/f'data/video/fan_integrated_h{h}.csv';meta=read(path.with_suffix('.json'))
        assert meta['selected_sha256']==digest and meta['version']==model['version'] and meta['n']==n
        video=pd.read_csv(path,parse_dates=['date']);g=f[f.horizon.eq(h)&f.actual.notna()].sort_values('target_date')
        np.testing.assert_array_equal(video.date,g.target_date)
        np.testing.assert_allclose(video.actual,g.actual,atol=.000051,rtol=0)
        for column in (c for c in ALL_COLUMNS if c.startswith('q_')):
            np.testing.assert_allclose(video[column.replace('q_','p_')],g.origin_price*np.exp(g[column]),atol=.000051,rtol=0)
    write_json(DELIVERY/'delivery_audit.json',dict(status='passed',selected_sha256=digest,
        rolling_forecast_sha256=sha(DELIVERY/'forecasts.csv'),canonical_manifest=True,forecast_mode='daily_update',rows=68,
        metrics_recomputed=True,historical_metrics_recomputed=True,all_consumed_futures_quotes_verified=True,
        member_inputs_verified=True,video_data_verified=True,point_training_cutoff='2026-09-30',
        previous_v3_preserved=True,no_october_parameter_changes=True,conditional_historical_evidence=True))
    print('V4 rolling delivery audit passed:',digest,flush=True)


if __name__=='__main__':main()
