import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from m3_ml.common.data import load_table,observable_price,label_available_dates
from m3_ml.common.features import build_features
from m3_ml.fusion.prepare import ROOT,PATH_H,write_json

OUT=ROOT/'m3_ml/fusion_v3/outputs/frozen_october_2026'
BASE=ROOT/'m3_ml/fusion_v2/outputs/frozen_october_2026'
RAW=ROOT/'data/raw/futures/wti_nearby_futures_corpus_snapshot.csv'
LAG=2*CustomBusinessDay(calendar=USFederalHolidayCalendar())
MEMBERS=('m1','m2','lstm','ml')
KEY=['origin_date','horizon']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def specification():
    from m3_ml.kronos.corpus import path
    OUT.mkdir(parents=True,exist_ok=True)
    source=path('CL=F')
    if not RAW.exists():
        RAW.parent.mkdir(parents=True,exist_ok=True)
        frame=pd.read_parquet(source).reset_index().rename(columns={'Date':'quote_date','index':'quote_date'})
        frame.to_csv(RAW,index=False)
    raw=pd.read_csv(RAW,parse_dates=['quote_date']).set_index('quote_date')
    eia=pd.read_csv(ROOT/'data/long/eia_api/wti_futures_contract1_daily.csv',parse_dates=['observation_date']).set_index('observation_date').value
    match=raw[['close']].join(eia.rename('eia')).dropna().loc['2011-08-04':'2026-09-30']
    difference=(match.close-match.eia).abs()
    if len(match)<3000 or (difference>.011).any():raise ValueError('Raw futures/EIA overlap verification failed')
    prior_audit=json.loads((OUT/'market_source_audit.json').read_text()) if (OUT/'market_source_audit.json').exists() else {}
    source_hash=sha(source) if source.exists() else prior_audit.get('original_cache_sha256')
    write_json(OUT/'market_source_audit.json',dict(source_snapshot=RAW.relative_to(ROOT).as_posix(),source_sha256=sha(RAW),
        original_cache=source.relative_to(ROOT).as_posix(),original_cache_sha256=source_hash,
        eia_overlap_rows=len(match),maximum_difference=float(difference.max()),median_difference=float(difference.median()),
        availability='Session quote + 2 US federal business days; never same-day OHLCV.',
        limitation='Current-vintage unadjusted Close; verified overlap through EIA historical endpoint, not a complete point-in-time vendor archive.',
        exchange_source='https://www.cmegroup.com/trading/energy/files/NYMEX_Energy_Futures_Daily_Settlement_Procedure.pdf'))
    candidates=[]
    def add(**args):
        candidates.append(dict(id=f'c{len(candidates):02d}',**args))
    for name in ('rw','futures','basis',*MEMBERS):add(kind='carry',base=name,window=0)
    for penalty in (0.,.1,1.):add(kind='convex',penalty=penalty,window=0)
    for window in (0,252):
        for feature,alphas in [('members',(10.,100.)),('full',(100.,1000.))]:
            for alpha in alphas:add(kind='ridge',base='futures',feature=feature,alpha=alpha,window=window)
    for alpha in (.3,.1):add(kind='elastic_net',base='futures',feature='full',alpha_ratio=alpha,window=0)
    for leaves in (4,8):add(kind='lightgbm',base='futures',feature='full',num_leaves=leaves,window=0)
    for alpha in (.3,.1):add(kind='elastic_net',base='m1',feature='full',alpha_ratio=alpha,window=0)
    add(kind='elastic_net',base='m2',feature='full',alpha_ratio=.3,window=0)
    base=json.loads((BASE/'specification.json').read_text())
    spec={k:base[k] for k in ('protocol_id','fit_end','m1_method','m2_method','ml','lstm','base_selection_limitation','october_status')}
    spec.update(name='Integrated WTI Forecast',version='market_residual_v3',horizons=list(PATH_H),candidates=candidates,
                rules=[dict(mode=mode,window=window) for window in (0,252) for mode in ('best','top3','soft8','soft32')],
                candidate_start='2024-03-01',point_start='2024-09-01',rule_lock_date='2025-09-01',
                primary_evaluation='2025-09-01 onward; equal-horizon normalized USD MSE across h=1..17, common complete origins.',
                reference_benchmarks=['published_spot_RW','lagged_futures_carry','v2'],
                risk_start='2025-09-01',risk_outer_start='2026-03-01',risk_windows=[0,126,252],
                risk_timing_note='Risk candidate evaluation starts only after the point rule lock; forward risk selection starts after enough h=60 labels mature.',
                risk_methods=['hs','a_ridge','a_lightgbm','b_linear','b_lightgbm','b_forest','futures_vol'],
                risk_mixtures=[0.,.5,1.],risk_widths=[.75,1.,1.25,1.5],
                point_meaning='Median of the constructed log-return distribution; risk calibration preserves the selected price center.',
                no_october_selection=True,market_availability='2 US federal business days after the futures quote; same convention as existing market inputs.',
                inference='Paired moving-block bootstrap of origin-level losses, blocks 20 and 60; exploratory conditional historical evidence, not a fresh prospective test.')
    write_json(OUT/'specification.json',spec)
    print('Specified',len(candidates),'point candidates and',len(spec['rules']),'selector rules before v3 fitting',flush=True)


def market_features(table,raw=None):
    raw=pd.read_csv(RAW,parse_dates=['quote_date']) if raw is None else raw.copy()
    raw=raw.sort_values('quote_date').reset_index(drop=True)
    if raw.quote_date.duplicated().any():raise ValueError('Duplicate futures dates')
    raw['quote_date']=raw.quote_date.astype('datetime64[ns]')
    raw['available_date']=pd.Series([d+LAG for d in raw.quote_date]).astype('datetime64[ns]')
    # Only Close is used; no adjusted-close reconstruction or future roll fixes.
    price=raw.close.astype(float)
    lp=np.log(price.where(price>0));r=lp.diff()
    for k in (1,5,20,60):raw[f'f_ret_{k}']=lp-lp.shift(k)
    for k in (20,60):raw[f'f_vol_{k}']=r.rolling(k).std()
    cols=['quote_date','available_date','close',*[c for c in raw if c.startswith('f_')]]
    dates=pd.DataFrame({'date':pd.to_datetime(table.date).astype('datetime64[ns]')})
    aligned=pd.merge_asof(dates,raw[cols].sort_values(['available_date','quote_date']),left_on='date',right_on='available_date',direction='backward')
    assert aligned.available_date.le(aligned.date).all() and aligned.quote_date.lt(aligned.date).all()
    if (aligned.date-aligned.quote_date).dt.days.gt(10).any():raise ValueError('Stale futures quote')
    # Spot/futures basis uses only already-published spot observations. Match
    # the future on the spot observation date, not on its later release date.
    contemporaneous=pd.merge_asof(dates,raw[['quote_date','available_date','close']],left_on='date',right_on='quote_date',direction='backward')
    available=label_available_dates(table).reset_index(drop=True)
    basis=table.target_wti.reset_index(drop=True)-contemporaneous.close
    anchors=observable_price(table).reset_index(drop=True)
    stats=[]
    for i,date in enumerate(dates.date):
        known=available.le(date)&contemporaneous.available_date.le(date)&dates.date.lt(date)
        values=basis[known]
        known_dates=dates.loc[known,'date']
        stats.append([values.iloc[-k:].median() if len(values)>=k else np.nan for k in (20,60,250)]
                     +[(date-known_dates.iloc[-1]).days if len(known_dates) else np.nan])
    stats=pd.DataFrame(stats,columns=['basis20','basis60','basis250','spot_age_days'])
    result=pd.concat([aligned,stats],axis=1)
    result['futures']=result.close;result['basis']=result.close+result.basis60
    result['futures_minus_anchor']=result.close-anchors
    return result


def feature_frame(panel,table,market=None):
    market=market_features(table) if market is None else market
    dates=pd.Series(table.index,index=pd.to_datetime(table.date)).reindex(panel.origin_date).to_numpy()
    if pd.isna(dates).any():raise ValueError('Missing feature origin')
    dates=dates.astype(int)
    shared=build_features(table).iloc[dates].reset_index(drop=True)
    state=market.iloc[dates].reset_index(drop=True)
    frame=panel.reset_index(drop=True).copy()
    frame['actual']=frame.origin_price*np.exp(frame.y)
    frame['rw_price']=frame.origin_price
    for member in MEMBERS:frame[member+'_price']=frame.origin_price*np.exp(frame[member])
    frame['futures_price']=state.futures;frame['basis_price']=state.basis
    frame['futures_quote_date']=state.quote_date;frame['futures_available_date']=state.available_date
    frame['futures_vol']=state.f_vol_20
    small=pd.DataFrame({m+'_minus_futures':frame[m+'_price']-frame.futures_price for m in ('rw',*MEMBERS)})
    for c in ('basis20','basis60','basis250','spot_age_days'):small[c]=state[c]
    angle=2*np.pi*(frame.origin_date.dt.month-1)/12
    full=pd.concat([small,shared,state[[c for c in state if c.startswith('f_')]]],axis=1)
    for k in (1,2):
        full[f'month_sin{k}']=np.sin(k*angle);full[f'month_cos{k}']=np.cos(k*angle)
    for w in range(1,5):full[f'weekday{w}']=frame.origin_date.dt.dayofweek.eq(w).astype(float)
    for season in ('month_sin1','month_cos1'):
        for condition in ('oil_implied_volatility','commercial_crude_stocks_yoy_chg','futures_minus_anchor'):
            value=state[condition] if condition in state else shared[condition]
            full[season+'_x_'+condition]=full[season]*value
    if not np.isfinite(small).all().all() or not np.isfinite(full).all().all():raise ValueError('Incomplete historical features')
    return frame,{'members':small,'full':full}


def historical_inputs():
    table=load_table().loc[lambda x:x.date.le('2026-09-30')].reset_index(drop=True)
    panel=pd.read_csv(BASE/'validation_member_panel.csv',parse_dates=['origin_date','target_date','label_available_date'])
    if panel.label_available_date.max()>pd.Timestamp('2026-09-30'):raise ValueError('Future labels')
    # Market basis requires a warm-up; main-table early rows are retained for
    # causal shared-feature construction but only origin rows need a basis.
    market=market_features(table)
    return table,*feature_frame(panel,table,market)
