import inspect
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge,HuberRegressor
from sklearn.pipeline import make_pipeline
from m3_ml.common.preprocessing import make_preprocessor
from m3_ml.common.execution import ValidationCache,run_jobs
from m3_ml.tabular.models import MODELS
from m3_ml.fusion.model import blocks,mature_before
from .data import OUT,HORIZONS,KEY,historical_inputs,write_json


def fit_predict(train,test,features,d):
    if d['kind']=='basis':return test.futures_price.to_numpy()+test[d['basis']].to_numpy(),None
    if d['window']:train=train.tail(d['window'])
    y=train.actual-train.futures_price
    if d['kind']=='mean':return test.futures_price.to_numpy()+y.mean(),float(y.mean())
    if d['target']=='scaled':y=y/train.scale
    X=features[d['feature']]
    if d['kind']=='ridge':model=make_pipeline(make_preprocessor('scaled'),Ridge(alpha=d['alpha']))
    elif d['kind']=='huber':model=make_pipeline(make_preprocessor('scaled'),HuberRegressor(alpha=d['alpha'],max_iter=2000))
    elif d['kind']=='elastic_net':model=MODELS['elastic_net'][0](alpha_ratio=d['alpha_ratio'],l1_ratio=.5)
    else:model=MODELS['lightgbm'][0](num_leaves=d['leaves'],n_estimators=150)
    model.fit(X.iloc[train.feature_row],y)
    correction=model.predict(X.iloc[test.feature_row])
    if d['target']=='scaled':correction*=test.scale.to_numpy()
    return np.maximum(.01,test.futures_price.to_numpy()+correction),model


def job(context,task):
    panel,features,spec=context;h,d=task;f=panel[panel.horizon.eq(h)];parts=[]
    for cutoff,stop in blocks(spec['candidate_start'],spec['fit_end']):
        train=mature_before(f,cutoff);test=f[f.origin_date.ge(cutoff)&f.origin_date.lt(stop)]
        if test.empty:continue
        assert len(train)>=500 and train.label_available_date.max()<=cutoff
        price,_=fit_predict(train,test,features,d)
        if not np.isfinite(price).all():raise ValueError('Nonfinite long-history prediction')
        parts.append(np.column_stack([test.index,price]))
    return np.concatenate(parts)


def main():
    spec=json.loads((OUT/'specification.json').read_text());_,panel,features=historical_inputs()
    panel.to_parquet(OUT/'long_training_panel.parquet',index=False)
    tasks=[(h,d) for h in HORIZONS for d in spec['new_candidates']]
    cache=ValidationCache('rolling_long_history_v4',dict(panel=panel,features=features['full'],spec=spec,
        fit=inspect.getsource(fit_predict),job=inspect.getsource(job)))
    results=run_jobs(job,tasks,(panel,features,spec),'V4 long-history point fits',cache)
    parts=[]
    for (h,d),array in zip(tasks,results):
        f=panel.loc[array[:,0].astype(int)].copy();f['forecast']=array[:,1];f['candidate']=d['id'];parts.append(f)
    pd.concat(parts,ignore_index=True).to_parquet(OUT/'long_candidate_forecasts.parquet',index=False)
    audit=[]
    for cutoff,_ in blocks(spec['candidate_start'],spec['fit_end']):
        for h in HORIZONS:
            g=mature_before(panel[panel.horizon.eq(h)],cutoff)
            audit.append(dict(cutoff=str(cutoff.date()),horizon=h,n=len(g),latest_label_release=str(g.label_available_date.max().date())))
    write_json(OUT/'long_fit_audit.json',audit)
    print('Long-history OOF forecasts complete; no October data consumed',flush=True)


if __name__=='__main__':main()
