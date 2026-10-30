import json
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table
from m3_ml.fusion_v3.data import ROOT,RAW,KEY,sha,write_json,market_features
from m3_ml.fusion_v4 import run as prior_run
from m3_ml.common.protocol import HORIZONS

OLD=ROOT/'m3_ml/fusion_v4/outputs/frozen_october_2026'
V3=ROOT/'m3_ml/fusion_v3/outputs/frozen_october_2026'
OUT=ROOT/'m3_ml/fusion_v5/outputs/frozen_october_2026'
FIT_END='2026-09-30'
EXPERTS=['incumbent','futures','basis','m1','m2','lstm','ml','long_ridge','long_en','long_tree','long_huber','long_mean','v3']
STATE=['state_vol','state_basis','state_age','state_trend','state_dispersion','state_season']


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def protected():
    prior_run.protected();model=read(OLD/'selected.json')
    assert model['input_sha256']==prior_run.inputs()
    if (OUT/'protected_v4.json').exists():
        expected=read(OUT/'protected_v4.json')
        assert sha(OLD/'selected.json')==expected['selected_sha256']
        assert sha(OLD/'test_forecasts.csv')==expected['forecasts_sha256']


def specification():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'specification.json').exists():raise ValueError('Do not overwrite a registered v5 specification')
    protected();defs=[]
    def add(**d):defs.append(dict(id=f'p{len(defs):03d}',**d))
    add(kind='incumbent',window=0)
    for window in (126,252,0):
        for penalty in (.03,.3,3.):
            add(kind='static',window=window,penalty=penalty,n_state=0)
            for n_state in (3,6):add(kind='regime',window=window,penalty=penalty,n_state=n_state)
    for window in (252,0):
        for learner,alpha in [('ridge',100.),('ridge',1000.),('tree',0.)]:
            for temperature in (.25,1.,4.):
                for anchor in (4.,16.):
                    add(kind='reliability',window=window,learner=learner,alpha=alpha,temperature=temperature,anchor_prior=anchor)
    spec=dict(version='state_reliability_v5',name='Integrated WTI Forecast',fit_end=FIT_END,forecast_mode='daily_update',
        horizons=list(HORIZONS),experts=EXPERTS,state_features=STATE,point_candidates=defs,
        candidate_start='2025-03-01',outer_start='2025-09-01',min_mature_train=40,
        selector_window=252,selector_min_relative_gain=.01,
        point_promotion=dict(max_aggregate_ratio=.995,max_horizon_ratio=1.02),
        risk_start='2025-09-01',risk_outer_start='2026-03-01',
        risk_sources=['hs','a_ridge','a_lightgbm','b_linear','b_lightgbm','b_forest','futures_vol','implied_vol','hybrid_vol'],
        risk_windows=[63,126,252,0],risk_targets=[.90,.95,.975],risk_sides=['symmetric','asymmetric'],
        risk_constraint=dict(min_coverage=.90,max_tail_miss=.075,max_width_vs_hs=2.,selection_window=252),
        risk_promotion=dict(min_mean_coverage_gap_improvement=.02,max_horizon_gap_degradation=.02,
            max_mean_crps_ratio=1.05,max_horizon_crps_ratio=1.10,max_horizon_width_ratio=1.75),
        risk_policy='Fit standardized tail scores only on mature previously issued errors; select minimum CRPS among candidates satisfying empirical coverage and width limits. Report infeasibility explicitly.',
        promotion='If the point gate fails, retain v4 points before fitting risk. Promote only if the risk gate passes, or point gate passes and risk does not materially regress. Otherwise keep v4 canonical.',
        timing='Quarterly refits within history; selection at each outer cutoff uses only earlier candidate forecasts with released labels. Freeze all functions and calibration at September 30; daily inputs may change.',
        limitations='This adds nested time selection for the new layer. Inherited base specifications and repeated historical research remain conditional; source vintages and publication assumptions are unchanged.',
        october='Previously inspected October outcomes are retrospective evaluation only; no candidate, parameter or promotion selection from October.',
        references=['https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html',
                    'https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html'],
        coverage_caveat='Serial dependence, changing distributions and frozen October calibration mean no distribution-free 90% coverage guarantee is claimed.')
    write_json(OUT/'specification.json',spec)
    write_json(OUT/'protected_v4.json',dict(selected_sha256=sha(OLD/'selected.json'),forecasts_sha256=sha(OLD/'test_forecasts.csv')))
    print('Registered',len(defs),'point specifications x 4 horizons and 216 risk specifications before fitting',flush=True)


def add_state(panel,table,market=None):
    frame=panel.copy().reset_index(drop=True)
    if market is None:market=market_features(table)
    positions=pd.Series(table.index,index=table.date).reindex(frame.origin_date).to_numpy().astype(int)
    m=market.iloc[positions].reset_index(drop=True)
    frame['implied_vol']=table.oil_implied_volatility.iloc[positions].to_numpy()/100
    frame['scale']=np.maximum(1.,frame.futures_price*frame.futures_vol*np.sqrt(frame.horizon+2))
    frame['state_vol']=frame.implied_vol
    frame['state_basis']=m.futures_minus_anchor/np.maximum(1.,frame.futures_price*frame.futures_vol*np.sqrt(5.))
    frame['state_age']=np.log1p(m.spot_age_days)
    frame['state_trend']=m.f_ret_20/np.maximum(.005,m.f_vol_20*np.sqrt(20.))
    values=frame[[e+'_expert' for e in EXPERTS]].to_numpy()
    frame['state_dispersion']=np.std(values,axis=1)/frame.scale
    frame['state_season']=np.sin(2*np.pi*(frame.origin_date.dt.month-1)/12)
    frame['state_season_cos']=np.cos(2*np.pi*(frame.origin_date.dt.month-1)/12)
    frame['state_vol_ratio']=m.f_vol_20/np.maximum(.001,m.f_vol_60)
    frame['state_basis_slope']=(m.basis20-m.basis250)/frame.scale
    assert np.isfinite(frame[STATE+['state_season_cos','state_vol_ratio','state_basis_slope','scale']]).all().all()
    return frame


def historical_inputs():
    protected()
    table=load_table().loc[lambda x:x.date.le(FIT_END)].reset_index(drop=True)
    raw=pd.read_csv(RAW,parse_dates=['quote_date']).loc[lambda x:x.quote_date.le(FIT_END)]
    frame=pd.read_csv(OLD/'point_crossfit.csv',parse_dates=['origin_date','target_date','label_available_date'])
    assert frame.origin_date.max()<=pd.Timestamp(FIT_END) and frame.label_available_date.max()<=pd.Timestamp(FIT_END)
    frame['incumbent_expert']=frame.forecast
    for e in EXPERTS[1:]:frame[e+'_expert']=frame[e+'_family']
    frame['v4_forecast']=frame.forecast
    return add_state(frame,table,market_features(table,raw))


if __name__=='__main__':specification()
