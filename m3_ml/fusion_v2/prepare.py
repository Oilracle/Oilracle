"""Refresh historical base fits every 63 origins; reuse sealed final base fits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table, label_available_dates
from m3_ml.common.execution import ValidationCache, run_jobs
from m3_ml.common.protocol import FIT_END, VALIDATION_START
from m3_ml.fusion.prepare import ROOT, PATH_H, write_json, OUT as OLD

OUT=ROOT/'m3_ml/fusion_v2/outputs/frozen_october_2026'


def specification():
    OUT.mkdir(parents=True,exist_ok=True)
    base=json.loads((OLD/'specification.json').read_text())
    spec={k:base[k] for k in ('protocol_id','fit_end','m1_method','m2_method','ml','lstm','base_selection_limitation','october_status')}
    spec.update(version='horizon_rw_v2',name='Integrated WTI Forecast',
                channels=['rw','m1','m2','lstm','ml'],minimum_member_weight=0.,
                horizons=list(PATH_H),ridge_candidates=[0.,.01,.1,1.,10.],
                base_refit_origins=63,point_candidate_start='2024-03-01',
                point_validation_start='2024-09-01',risk_validation_start='2025-09-01',
                risk_outer_start='2026-03-01',fold_months=3,
                risk_methods=['hs','a_ridge','a_lightgbm','b_linear','b_lightgbm','b_forest'],
                risk_mixture_candidates=[0.,.5,1.],risk_width_candidates=[.75,1.,1.25],
                point_criterion='Per-horizon USD price RMSE, convex arithmetic-price blend; L2 shrinkage to RW; penalty chosen from earlier mature cross-fitted predictions.',
                risk_criterion='Per-horizon CRPS on earlier mature cross-fitted predictions; preserve the selected price median; calibrate shape and width only.',
                tie_break='Point: lower loss, then stronger RW shrinkage. Risk: lower loss, then identity width, smaller empirical mixture, method name.',
                development_policy='First-stage train/test are development data. All model/strategy fitting and selection labels must be published by 2026-09-30. October is evaluated only after selected.json is sealed.',
                base_fit_policy='All members refit every 63 historical origins, with fixed historically selected specifications; final members refit once on 2026-09-30.',
                membership_policy='All four point modules enter the unrestricted candidate blend; zero weight is permitted. ML distribution supplies risk forecasts. No artificial minimum contribution.',
                point_meaning='Price median; arithmetic convex combination of member price forecasts including RW, preserved by risk calibration.')
    write_json(OUT/'specification.json',spec)
    table=load_table().loc[lambda x:x.date.le(FIT_END)].reset_index(drop=True)
    origins=np.flatnonzero(table.date.ge(VALIDATION_START))
    folds=[]
    for i in range(0,len(origins),63):
        pos=origins[i:i+63]
        folds.append(dict(fold=len(folds),cutoff=str(table.date.iloc[pos[0]].date()),
                          stop=str(table.date.iloc[pos[-1]].date()),start_row=int(pos[0]),end_row=int(pos[-1])))
    write_json(OUT/'base_folds.json',folds)
    # ML already uses exactly this 63-origin refit schedule. This is a replayed
    # historical artifact, not a final-model in-sample prediction.
    pd.read_csv(OLD/'ml_validation.csv').to_csv(OUT/'ml_validation.csv',index=False)
    print('SPECIFIED before revised selection/October evaluation:',len(folds),'base folds',flush=True)


def lstm_job(context,task):
    import torch
    from m3_ml.lstm.src import oil_lstm as l
    df,cols,spec,folds=context
    fold_number,seed=task; fold=folds[fold_number]; cutoff=fold['cutoff']
    original_seed=l.seed_everything
    def seed_one(value):
        original_seed(value)
        torch.set_num_threads(1)
    l.seed_everything=seed_one
    try:
        cfg=spec['lstm']['config']
        idx=l.published_origins(df,l.origins_between(len(df),cfg['lookback'],0,l.boundary(df,cutoff,0)),cutoff)
        raw=df[cols].to_numpy(float)
        xm,xs=l.fit_scaler(raw[:idx[-1]+1]);yt=l.targets(df,idx);ym,ys=l.fit_scaler(yt)
        x=(raw-xm)/xs; origins=np.arange(fold['start_row'],fold['end_row']+1)
        model,_,_,_=l.train_one(cfg,seed,l.windows(x,idx,cfg['lookback']),(yt-ym)/ys,epochs=spec['lstm']['epochs'])
        pred=l.predict(model,l.windows(x,origins,cfg['lookback']))*ys+ym
        pred+=df[l.ANCHOR].to_numpy()[origins,None]
        return pred[:,np.array(PATH_H)-1]
    finally:
        l.seed_everything=original_seed


def prepare_lstm():
    from m3_ml.lstm.src import oil_lstm as l
    spec=json.loads((OUT/'specification.json').read_text());folds=json.loads((OUT/'base_folds.json').read_text())
    df,cols=l.read_data(ROOT/'data/processed/wti_daily.csv',ROOT/'data/processed/features.csv')
    df=df[df.date.le(FIT_END)].reset_index(drop=True)
    tasks=[(f['fold'],s) for f in folds for s in spec['lstm']['seeds']]
    cache=ValidationCache('fusion_v2_lstm_63origins',{'data':df,'spec':spec,'folds':folds,
                          'source':Path(__file__).read_text(),'lstm_source':Path(l.__file__).read_text()})
    predictions=run_jobs(lstm_job,tasks,(df,cols,spec,folds),'Rolling LSTM replay',cache)
    frames=[];audit=[]
    for f in folds:
        values=[p for task,p in zip(tasks,predictions) if task[0]==f['fold']]
        frame=pd.DataFrame(np.mean(values,axis=0),columns=PATH_H)
        frame['origin_date']=df.date.iloc[f['start_row']:f['end_row']+1].to_numpy()
        frame=frame.melt(id_vars='origin_date',var_name='horizon',value_name='forecast')
        frame['base_fit_cutoff']=f['cutoff'];frames.append(frame)
        idx=l.published_origins(df,l.origins_between(len(df),60,0,l.boundary(df,f['cutoff'],0)),f['cutoff'])
        releases=label_available_dates(df)
        latest=max(releases.iloc[i+1:i+61].max() for i in idx)
        assert latest<=pd.Timestamp(f['cutoff'])
        audit.append(dict(**f,n_train=len(idx),latest_label_release=str(latest.date()),seeds=spec['lstm']['seeds']))
    result=pd.concat(frames,ignore_index=True)
    if not np.isfinite(result.forecast).all() or result.forecast.le(0).any():raise ValueError('Invalid historical LSTM price')
    result.to_csv(OUT/'lstm_validation.csv',index=False)
    write_json(OUT/'lstm_refit_audit.json',audit)
    print('Rolling LSTM complete:',len(result),'forecasts',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['spec','lstm']);args=parser.parse_args()
    (specification if args.stage=='spec' else prepare_lstm)()
