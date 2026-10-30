import json
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table,observable_price,label_available_dates
from m3_ml.common.features import build_features
from m3_ml.common.walkforward import final_training_positions
from m3_ml.common.protocol import HORIZONS,FIT_END
from m3_ml.fusion_v3.data import ROOT,RAW,market_features,write_json,sha

OUT=ROOT/'m3_ml/fusion_v4/outputs/frozen_october_2026'
OLD=ROOT/'m3_ml/fusion_v3/outputs/frozen_october_2026'
KEY=['origin_date','horizon']


def specification():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'specification.json').exists():raise ValueError('Do not overwrite the registered research specification')
    candidates=[]
    def add(**values):candidates.append(dict(id=f'l{len(candidates):02d}',**values))
    for basis in ('basis20','basis60','basis250'):add(kind='basis',basis=basis)
    for window in (252,1260,0):add(kind='mean',window=window)
    for window in (1260,0):
        for feature in ('small','full'):
            for alpha in (100.,1000.):add(kind='ridge',window=window,feature=feature,alpha=alpha,target='dollar')
    for target in ('dollar','scaled'):
        for alpha in (.3,.1,.03):add(kind='elastic_net',window=0,feature='full',alpha_ratio=alpha,target=target)
        for leaves in (4,8):add(kind='lightgbm',window=0,feature='full',leaves=leaves,target=target)
    for window in (1260,0):
        add(kind='huber',window=window,feature='small',alpha=10.,target='dollar')
    spec=dict(version='rolling_long_history_v4',name='Integrated WTI Forecast',fit_end=FIT_END,horizons=list(HORIZONS),
        forecast_mode='daily_update',new_training_start='2012-10-01',candidate_start='2023-09-01',
        meta_start='2024-09-01',rule_lock_date='2025-09-01',new_candidates=candidates,
        family_pool=['futures','basis','m1','m2','lstm','ml','v3','long_ridge','long_en','long_tree','long_huber','long_mean'],
        blend_windows=[126,252,0],blend_penalties=[.01,.1,1.,10.],blend_options=['futures','best','convex'],
        update_schedule='Quarterly historical fits; coefficients and meta weights frozen after final pre-October fitting.',
        references=['published_spot_rw','lagged_futures_carry','v3'],
        primary='Equal-horizon average normalized USD MSE over h1,5,20,60 on common late historical origins; report every horizon and stronger futures benchmark.',
        promotion='Later-history aggregate must improve over v3, and no horizon may degrade by more than 2%; no October ranking.',
        risk_sources=['hs','a_ridge','a_lightgbm','b_linear','b_lightgbm','b_forest','futures_vol','implied_vol','hybrid_vol'],
        risk_windows=[126,252,0],risk_mixtures=[0.,.5,1.],risk_widths=[.75,1.,1.25,1.5,2.],
        risk_start='2025-09-01',risk_outer_start='2026-03-01',
        inference='Paired origin-level block bootstrap, blocks 20 and 60; conditional exploratory historical evidence.',
        october='Previously inspected outcomes are not a fresh blind test; they cannot select candidates, parameters, upgrades or fallbacks.',
        source_limits='Same two-federal-business-day futures lag and current-vintage/publication assumptions as v3; base specifications condition the historical evaluation.')
    write_json(OUT/'specification.json',spec)
    write_json(OUT/'protected_v3.json',dict(selected_sha256=sha(OLD/'selected.json'),forecasts_sha256=sha(OLD/'test_forecasts.csv')))
    print('Registered',len(candidates),'long-history candidates at four rolling horizons before evaluation',flush=True)


def features(table,raw=None):
    # Explicit truncation here is a research-stage boundary, not a statistical
    # preprocessing operation. Forecasting later uses the same causal builder.
    market=market_features(table,raw)
    shared=build_features(table)
    small=market[['basis20','basis60','basis250','spot_age_days','futures_minus_anchor',
                  'f_ret_1','f_ret_5','f_ret_20','f_ret_60','f_vol_20','f_vol_60']].copy()
    small['futures_level']=market.futures
    small['implied_vol']=shared.oil_implied_volatility/100
    small['vol_ratio']=market.f_vol_20/(market.f_vol_60+.001)
    small['basis_slope']=market.basis20-market.basis250
    small['futures_age']=(market.date-market.quote_date).dt.days
    angle=2*np.pi*(table.date.dt.month-1)/12
    for k in (1,2):
        small[f'month_sin{k}']=np.sin(k*angle);small[f'month_cos{k}']=np.cos(k*angle)
    full=pd.concat([small,shared],axis=1)
    for season in ('month_sin1','month_cos1'):
        for condition in ('implied_vol','commercial_crude_stocks_yoy_chg','futures_minus_anchor'):
            full[season+'_x_'+condition]=small[season]*full[condition]
    return market,{'small':small,'full':full}


def historical_inputs():
    table=load_table().loc[lambda x:x.date.le(FIT_END)].reset_index(drop=True)
    raw=pd.read_csv(RAW,parse_dates=['quote_date']).loc[lambda x:x.quote_date.le(FIT_END)]
    market,X=features(table,raw);valid=np.isfinite(X['full']).all(axis=1)&table.date.ge('2012-10-01')&market.futures.gt(0)
    available=label_available_dates(table);anchor=observable_price(table);parts=[]
    for h in HORIZONS:
        target=table.target_wti.shift(-h)
        pos=final_training_positions(table.date,valid,target.gt(0),h,FIT_END)
        release=pd.concat([available.shift(-k) for k in range(h+1)],axis=1).max(axis=1)
        f=pd.DataFrame(dict(origin_date=table.date.iloc[pos].to_numpy(),horizon=h,feature_row=pos,
            target_date=table.date.iloc[pos+h].to_numpy(),actual=target.iloc[pos].to_numpy(),
            label_available_date=release.iloc[pos].to_numpy(),origin_price=anchor.iloc[pos].to_numpy(),
            futures_price=market.futures.iloc[pos].to_numpy(),futures_vol=market.f_vol_20.iloc[pos].to_numpy(),
            implied_vol=table.oil_implied_volatility.iloc[pos].to_numpy()/100))
        for c in ('basis20','basis60','basis250'):f[c]=market[c].iloc[pos].to_numpy()
        f['scale']=np.maximum(1.,f.futures_price*f.futures_vol*np.sqrt(h+2))
        f['y']=np.log(f.actual/f.origin_price);parts.append(f)
    return table,pd.concat(parts,ignore_index=True),X


if __name__=='__main__':specification()
