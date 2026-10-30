"""Select on historical data, seal parameters, then evaluate one October system.

python -m m3_ml.fusion.run select
python -m m3_ml.fusion.run test
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table, observable_price, label_available_dates
from m3_ml.common.protocol import FIT_END, HORIZONS, TEST_START, TEST_END, PROTOCOL_ID, VALIDATION_START
from m3_ml.common.walkforward import final_training_positions, forecast_origins
from m3_ml.distribution.tails import ALL_COLUMNS, MAIN
from m3_ml.distribution.quantiles import crps
from .prepare import ROOT, OUT, PATH_H, write_json
from .model import (MEMBERS, MEDIAN, crossfit_points, point_score, fit_weights, crossfit_risk,
                    risk_score, fit_calibrator, calibrate, metrics, risk_readout)

KEY=['origin_date','horizon']

def read_csv(path):
    f=pd.read_csv(path,parse_dates=['origin_date'])
    if 'protocol_id' in f and not f.protocol_id.eq(PROTOCOL_ID).all(): raise ValueError(f'Wrong protocol: {path}')
    return f

def join_member(frame, source, member, keys=KEY):
    if source.duplicated(keys).any(): raise ValueError(f'Duplicate member keys: {member}')
    f=frame.merge(source[keys+['forecast']].rename(columns={'forecast':member}),on=keys,how='left',validate='one_to_one')
    if f[member].isna().any() or not np.isfinite(f[member]).all() or f[member].le(0).any():
        raise ValueError(f'Incomplete/nonpositive member forecasts: {member}')
    f[member]=np.log(f[member]/f.origin_price)
    return f

def validation_frame():
    table=load_table().loc[lambda x:x.date.le(FIT_END)].reset_index(drop=True)
    anchor=observable_price(table); release=label_available_dates(table)
    parts=[]
    for h in PATH_H:
        pos=final_training_positions(table.date,np.ones(len(table),bool),table.target_wti.gt(0),h,FIT_END)
        pos=pos[table.date.iloc[pos].ge(VALIDATION_START)]
        target=pos+h
        # Full target-window maturity is conservative and matches member ML training.
        label_release=np.array([release.iloc[t:t+h+1].max() for t in pos])
        parts.append(pd.DataFrame(dict(origin_date=table.date.iloc[pos].to_numpy(),horizon=h,
                     target_row=target,target_date=table.date.iloc[target].to_numpy(),origin_price=anchor.iloc[pos].to_numpy(),
                     y=np.log(table.target_wti.iloc[target].to_numpy()/anchor.iloc[pos].to_numpy()),
                     label_available_date=label_release)))
    f=pd.concat(parts,ignore_index=True)
    for member in MEMBERS:
        f=join_member(f,read_csv(OUT/f'{member}_validation.csv'),member)
    if f.target_date.max()>pd.Timestamp(FIT_END) or f.label_available_date.max()>pd.Timestamp(FIT_END):
        raise ValueError('Future selection label')
    return f.sort_values(KEY).reset_index(drop=True)

def historical_distributions(frame,spec):
    source=ROOT/'m3_ml/distribution/outputs/frozen_october_2026/validation_selected_quantiles.csv'
    f=read_csv(source)
    result={}
    for method in spec['risk_methods']+['hs']:
        g=f[f.method.eq(method)]
        if g.duplicated(KEY).any(): raise ValueError('Duplicate distribution validation keys')
        g=frame[KEY+['y']].merge(g[KEY+ALL_COLUMNS],on=KEY,how='left',validate='one_to_one')
        if g[ALL_COLUMNS].isna().any().any(): raise ValueError('Missing distribution: '+method)
        result[method]=g
    frame['hs_crps']=crps(result['hs'][ALL_COLUMNS].to_numpy()[:,MAIN],frame.y)
    return result

def seal_inputs():
    paths=[OUT/'specification.json',*[OUT/f'{m}_validation.csv' for m in MEMBERS],
           ROOT/'m3_ml/distribution/outputs/frozen_october_2026/validation_selected_quantiles.csv',
           *sorted((ROOT/'m3_ml/fusion').glob('*.py')),
           ROOT/'m3_ml/fusion/replay_members.R']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def select():
    spec=json.loads((OUT/'specification.json').read_text())
    frame=validation_frame(); distributions=historical_distributions(frame,spec)
    frame.to_csv(OUT/'validation_member_panel.csv',index=False)
    point_candidates=[]; point_frames={}; point_audit={}
    for penalty in spec['ridge_candidates']:
        pred,audit=crossfit_points(frame,penalty,spec['point_validation_start'],FIT_END)
        point_frames[penalty]=pred; point_audit[penalty]=audit
        point_candidates.append(dict(penalty=penalty,score=point_score(pred),n_scored=int(pred.horizon.isin(HORIZONS).sum()),n_forecasts=len(pred)))
    selected_point=min(point_candidates,key=lambda x:(x['score'],-x['penalty']))
    penalty=selected_point['penalty']; points=point_frames[penalty]
    pd.DataFrame(point_candidates).to_csv(OUT/'point_selection.csv',index=False)
    risk_candidates=[]
    for method in spec['risk_methods']:
        for mix in spec['risk_mixture_candidates']:
            pred,_=crossfit_risk(points,distributions[method],mix,spec['risk_validation_start'],FIT_END)
            score=risk_score(pred)
            risk_candidates.append(dict(method=method,mixture=mix,score=score,n_scored=int(pred.horizon.isin(HORIZONS).sum()),n_forecasts=len(pred)))
            print('Risk candidate',method,mix,score,flush=True)
    winner=min(risk_candidates,key=lambda x:(x['score'],x['method'],-x['mixture']))
    pd.DataFrame(risk_candidates).to_csv(OUT/'risk_selection.csv',index=False)
    historical,risk_audit=crossfit_risk(points,distributions[winner['method']],winner['mixture'],spec['risk_validation_start'],FIT_END)
    historical['mode']='historical_crossfit'
    metrics(historical,'integrated').to_csv(OUT/'validation_metrics.csv',index=False)
    historical.to_csv(OUT/'validation_forecasts.csv',index=False)
    points.to_csv(OUT/'point_crossfit.csv',index=False)
    # Re-estimate weights on every mature development label, and calibrate only
    # with cross-fitted centers, never fitted-in-sample fusion residuals.
    weights=fit_weights(frame,penalty)
    calibration_data=points.merge(distributions[winner['method']][KEY+ALL_COLUMNS],on=KEY,validate='one_to_one')
    calibration=fit_calibrator(calibration_data)
    model=dict(protocol_id=PROTOCOL_ID,fit_end=FIT_END,name='Integrated WTI Forecast',
               members=list(MEMBERS),weights=dict(zip(MEMBERS,map(float,weights))),
               ridge_penalty=penalty,risk_method=winner['method'],risk_mixture=winner['mixture'],
               calibration=calibration,point_validation_score=selected_point['score'],risk_validation_score=winner['score'],
               final_weight_rows=int(frame.horizon.isin(HORIZONS).sum()),historical_panel_rows=len(frame),
               calibration_rows=len(calibration_data),
               max_label_release=str(frame.label_available_date.max().date()),
               base_selection_limitation=spec['base_selection_limitation'],october_status=spec['october_status'],
               point_meaning='median of the calibrated log-return distribution transformed to price',
               input_sha256=seal_inputs())
    write_json(OUT/'selected.json',model)
    write_json(OUT/'crossfit_audit.json',dict(point=point_audit[penalty],risk=risk_audit))
    # Fixed-specification ablations describe member contributions without using
    # October to retain/drop members or to change the selected system.
    rows=[dict(removed='none',point_score=point_score(points),risk_score=winner['score'])]
    for removed in MEMBERS:
        members=tuple(m for m in MEMBERS if m!=removed)
        ab,_=crossfit_points(frame,penalty,spec['point_validation_start'],FIT_END,members)
        ar,_=crossfit_risk(ab,distributions[winner['method']],winner['mixture'],spec['risk_validation_start'],FIT_END)
        rows.append(dict(removed=removed,point_score=point_score(ab),risk_score=risk_score(ar)))
    pd.DataFrame(rows).to_csv(OUT/'ablation.csv',index=False)
    print('SEALED',model['weights'],winner,flush=True)

def october_frame(spec):
    table=load_table(); anchor=observable_price(table); parts=[]
    for mode in ('fixed_origin','daily_update'):
        for origin in forecast_origins(table.date,mode):
            hs=PATH_H if mode=='fixed_origin' else HORIZONS
            for h in hs:
                target=origin+h
                date=table.date.iloc[target] if target<len(table) else pd.NaT
                actual=table.target_wti.iloc[target] if pd.notna(date) and pd.Timestamp(TEST_START)<=date<=pd.Timestamp(TEST_END) else np.nan
                parts.append(dict(mode=mode,origin_date=table.date.iloc[origin],horizon=h,target_row=target,
                                  target_date=date,origin_price=anchor.iloc[origin],actual=actual,
                                  y=np.log(actual/anchor.iloc[origin]) if np.isfinite(actual) else np.nan))
    f=pd.DataFrame(parts); keys=['mode',*KEY]
    base='outputs/frozen_october_2026'
    sources={
        'm1':read_csv(ROOT/f'm1_time_series/{base}/forecasts_2026-10-01_2026-10-23_asof_2026-10-23.csv'),
        'm2':read_csv(ROOT/f'm2_seasonality/{base}/final_predictions.csv'),
        'lstm':read_csv(ROOT/f'm3_ml/lstm/{base}/test_predictions.csv')}
    sources['m1']=sources['m1'][sources['m1'].method.eq(spec['m1_method'])]
    sources['m2']=sources['m2'][sources['m2'].method.eq(spec['m2_method'])&sources['m2'].target_type.eq('price')]
    sources['lstm']=sources['lstm'][sources['lstm'].method.eq('LSTM')]
    ml=[]
    for variant in sorted({x['variant'] for x in spec['ml'].values()}):
        src=read_csv(ROOT/f'm3_ml/tabular/{base}/{variant}_forecasts.csv')
        for h,entry in spec['ml'].items():
            if entry['variant']==variant: ml.append(src[src.horizon.eq(int(h))&src.method.eq(entry['method'])])
    sources['ml']=pd.concat(ml)
    for member,source in sources.items():
        if not source.fit_end.eq(FIT_END).all(): raise ValueError('Member final fit date mismatch')
        f=join_member(f,source,member,keys)
    return f

def final_forecasts(spec,model):
    f=october_frame(spec)
    f['center']=f[list(MEMBERS)].to_numpy()@np.array([model['weights'][m] for m in MEMBERS])
    d=read_csv(ROOT/'m3_ml/distribution/outputs/frozen_october_2026/test_quantiles.csv')
    d=d[d.method.eq(model['risk_method'])]
    f=f.merge(d[['mode',*KEY,*ALL_COLUMNS]],on=['mode',*KEY],how='left',validate='one_to_one')
    Q=calibrate(f,model['calibration'],model['risk_mixture'])
    f.loc[:,ALL_COLUMNS]=Q
    f['forecast']=f.origin_price*np.exp(Q[:,MEDIAN])
    f['lower90']=f.origin_price*np.exp(f['q_0.050']);f['upper90']=f.origin_price*np.exp(f['q_0.950'])
    for column,value in risk_readout(Q).items(): f[column]=value
    f['method']=model['name'];f['protocol_id']=PROTOCOL_ID;f['fit_end']=FIT_END
    f['status']=np.where(f.y.notna(),'realized','unreached')
    return f

def test():
    model=json.loads((OUT/'selected.json').read_text());spec=json.loads((OUT/'specification.json').read_text())
    if seal_inputs()!=model['input_sha256']: raise ValueError('Sealed selection inputs/source changed; do not silently reuse parameters')
    selected_hash=hashlib.sha256((OUT/'selected.json').read_bytes()).hexdigest()
    f=final_forecasts(spec,model)
    if len(f)!=87 or f.duplicated(['mode',*KEY]).any(): raise ValueError('Incomplete final grid')
    for h in HORIZONS:
        a=f[f['mode'].eq('fixed_origin')&f.horizon.eq(h)]
        b=f[f['mode'].eq('daily_update')&f.origin_date.eq(FIT_END)&f.horizon.eq(h)]
        np.testing.assert_allclose(a[ALL_COLUMNS],b[ALL_COLUMNS],atol=1e-12)
    f.to_csv(OUT/'test_forecasts.csv',index=False)
    scores=metrics(f,'integrated');scores.to_csv(OUT/'test_metrics.csv',index=False)
    assert hashlib.sha256((OUT/'selected.json').read_bytes()).hexdigest()==selected_hash
    write_json(OUT/'test_audit.json',dict(status='passed',rows=len(f),selected_sha256=selected_hash,
                                        finite_predictions=True,quantiles_monotone=True,first_origins_equal=True,
                                        actual_counts=scores[['mode','horizon','n']].to_dict('records')))
    print(scores[scores.horizon.isin(HORIZONS)].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','test']);a=p.parse_args()
    (select if a.stage=='select' else test)()
