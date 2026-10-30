"""Replay selected member models before October; never read test scores."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table, targets, observable_price
from m3_ml.common.features import build_features
from m3_ml.common.execution import ValidationCache, run_jobs
from m3_ml.common.protocol import FIT_END, HORIZONS, PROTOCOL_ID, TRAIN_END, VALIDATION_START
from m3_ml.tabular.run_direct import forecast, REFIT_VALIDATION
from m3_ml.tabular.models import MODELS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'm3_ml/fusion/outputs/frozen_october_2026'
PATH_H = tuple(range(1, 18)) + (20, 60)

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')

def specification():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(ROOT/'m1_time_series/outputs/frozen_october_2026/validation_selection_scores.csv')
    m1 = s[s.selected & s.status.eq('ok')].sort_values(['score','complexity','method']).iloc[0]
    s = pd.read_csv(ROOT/'m2_seasonality/outputs/frozen_october_2026/validation_metrics.csv')
    s = s[s['mode'].eq('daily_update') & s.target_type.eq('price') & s.horizon.isin(HORIZONS)]
    rw = s[s.method.eq('random_walk')].set_index('horizon').RMSE
    s = s.assign(relative=s.RMSE/s.horizon.map(rw))
    seasonal = s[~s.method.eq('random_walk')].groupby('method').relative.mean().sort_values()
    variants = ('direct','direct_denoised','direct_news','direct_denoised_news')
    params = {v:json.loads((ROOT/f'm3_ml/tabular/outputs/frozen_october_2026/{v}_params.json').read_text()) for v in variants}
    chosen = {}
    for h in PATH_H:
        loss,v,m = min((params[v]['selected'][f'{m}/h{h}']['validation_rmse'],v,m) for v in variants for m in MODELS)
        chosen[str(h)] = dict(variant=v,method=m,score=loss,params=params[v]['selected'][f'{m}/h{h}']['params'])
    lstm = json.loads((ROOT/'m3_ml/lstm/outputs/frozen_october_2026/models/metadata.json').read_text())
    spec = dict(protocol_id=PROTOCOL_ID,fit_end=FIT_END,m1_method=str(m1.method),m1_score=float(m1.score),
                m2_method=str(seasonal.index[0]),m2_score=float(seasonal.iloc[0]),ml=chosen,
                lstm={k:lstm[k] for k in ('config','epochs','seeds')},channels=['m1','m2','lstm','ml'],
                minimum_member_weight=.05,ridge_candidates=[0.,.1,1.,10.],risk_mixture_candidates=[0.,.5,1.],
                risk_methods=['a_ridge','a_lightgbm','b_linear','b_lightgbm','b_forest'],
                point_validation_start='2024-09-01',risk_validation_start='2025-09-01',fold_months=3,
                criterion='equal-horizon normalized RMSE for point; normalized CRPS for calibrated distribution',
                base_selection_limitation='Base specifications use the full historical development window. Meta forecasts are coefficient-cross-fitted conditional on these specifications, not a fully nested unbiased historical test.',
                october_status='Retrospective held-out evaluation; October was inspected in prior module work. No October observations or scores select fusion parameters.',
                interval_policy='ML conditional scale/shape calibrated on earlier mature cross-fitted fusion errors; forecast is distribution median',
                member_constraint='At least 5% per point member, fixed before fusion evaluation to satisfy all-member integration; optimum is conditional on this constraint.')
    write_json(OUT/'specification.json',spec)
    print('Pre-October specification:',{k:spec[k] for k in ('m1_method','m2_method','lstm')},flush=True)

def ml_job(context,task):
    table,features=context
    h,variant,method,params=task
    y=targets(table,[h])[h]
    p=forecast(MODELS[method][0],params,features[variant],y,table.date,(VALIDATION_START,FIT_END),h,REFIT_VALIDATION)
    return np.column_stack([p.index,observable_price(table).reindex(p.index)*np.exp(p)])

def prepare_ml(spec):
    table=load_table().loc[lambda x:x.date.le(FIT_END)].reset_index(drop=True)
    variants=sorted({v['variant'] for v in spec['ml'].values()})
    features={v:build_features(table,denoise='denoised' in v,news='news' in v) for v in variants}
    tasks=[(h,spec['ml'][str(h)]['variant'],spec['ml'][str(h)]['method'],spec['ml'][str(h)]['params']) for h in PATH_H]
    cache=ValidationCache('fusion_ml_validation',{'table':table,**features,'tasks':tasks,'source':Path(__file__).read_text()})
    preds=run_jobs(ml_job,tasks,(table,features),'Fusion ML replay',cache)
    frames=[]
    for task,values in zip(tasks,preds):
        pos=values[:,0].astype(int)
        frames.append(pd.DataFrame({'origin_date':table.date.iloc[pos].to_numpy(),'horizon':task[0],
                                    'forecast':values[:,1],'method':task[2],'variant':task[1]}))
    pd.concat(frames).to_csv(OUT/'ml_validation.csv',index=False)

def lstm_job(context,seed):
    from m3_ml.lstm.src import oil_lstm as l
    spec=context
    df,cols=l.read_data(ROOT/'data/processed/wti_daily.csv',ROOT/'data/processed/features.csv')
    df=df[df.date.le(FIT_END)].reset_index(drop=True)
    cfg=spec['lstm']['config']
    idx=l.published_origins(df,l.origins_between(len(df),60,0,l.boundary(df,TRAIN_END,0)),TRAIN_END)
    raw=df[cols].to_numpy(float)
    xm,xs=l.fit_scaler(raw[:idx[-1]+1]); yt=l.targets(df,idx); ym,ys=l.fit_scaler(yt)
    x=(raw-xm)/xs; origins=np.flatnonzero(df.date.ge(VALIDATION_START))
    model,_,_,_=l.train_one(cfg,seed,l.windows(x,idx,cfg['lookback']),(yt-ym)/ys,epochs=spec['lstm']['epochs'])
    pred=l.predict(model,l.windows(x,origins,cfg['lookback']))*ys+ym
    pred+=df[l.ANCHOR].to_numpy()[origins,None]
    return pred[:,np.array(PATH_H)-1]

def prepare_lstm(spec):
    os.environ['M3_OUTER_JOBS']='3'
    pred=np.mean(run_jobs(lstm_job,spec['lstm']['seeds'],spec,'Fusion LSTM replay'),axis=0)
    table=load_table().loc[lambda x:x.date.between(VALIDATION_START,FIT_END)].reset_index(drop=True)
    f=pd.DataFrame(pred,columns=PATH_H).assign(origin_date=table.date)
    f.melt(id_vars='origin_date',var_name='horizon',value_name='forecast').to_csv(OUT/'lstm_validation.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['spec','ml','lstm']);a=p.parse_args()
    if a.stage=='spec': specification()
    else:
        spec=json.loads((OUT/'specification.json').read_text())
        (prepare_ml if a.stage=='ml' else prepare_lstm)(spec)
