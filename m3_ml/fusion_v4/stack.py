"""Past-only family selection and covariance-aware stacking at four horizons."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from m3_ml.fusion.model import blocks,mature_before
from .data import OUT,OLD,HORIZONS,KEY,write_json


def pick_candidate(candidates,cutoff,window=252):
    past=mature_before(candidates,cutoff)
    if past.empty:return None
    last=past.origin_date.drop_duplicates().sort_values().tail(window)
    past=past[past.origin_date.isin(last)]
    scores=past.assign(loss=(past.forecast-past.actual)**2).groupby('candidate').loss.agg(['mean','size'])
    scores=scores[scores['size'].ge(40)]
    return None if scores.empty else scores.sort_values(['mean'],kind='stable').index[0]


def family_choices(long,old,spec,cutoff):
    kinds={d['id']:d['kind'] for d in spec['new_candidates']};choices={};audit=[]
    mapping={'basis':'basis','long_ridge':'ridge','long_en':'elastic_net','long_tree':'lightgbm','long_huber':'huber','long_mean':'mean'}
    for h in HORIZONS:
        chosen={}
        for family,kind in mapping.items():
            subset=long[long.horizon.eq(h)&long.candidate.map(kinds).eq(kind)]
            selected=pick_candidate(subset,cutoff)
            if selected is None:raise ValueError('Insufficient long-model family selection history')
            chosen[family]=selected
        selected=pick_candidate(old[old.horizon.eq(h)],cutoff)
        chosen['v3']=selected if selected is not None else 'c01'
        choices[str(h)]=chosen
        for source,frame in [('long',long),('v3_candidates',old)]:
            past=mature_before(frame[frame.horizon.eq(h)],cutoff)
            audit.append(dict(cutoff=str(pd.Timestamp(cutoff).date()),horizon=h,source=source,
                latest_label_release=None if past.empty else str(past.label_available_date.max().date()),choices=chosen))
    return choices,audit


def inputs(spec):
    long=pd.read_parquet(OUT/'long_candidate_forecasts.parquet')
    old=pd.read_csv(OLD/'point_candidate_forecasts.csv',usecols=KEY+['label_available_date','actual','forecast','candidate'],
                    parse_dates=['origin_date','label_available_date']).loc[lambda x:x.horizon.isin(HORIZONS)]
    native=pd.read_csv(OLD/'validation_member_panel.csv',parse_dates=['origin_date','target_date','label_available_date'])
    native=native[native.horizon.isin(HORIZONS)]
    metadata=long.drop_duplicates(KEY).set_index(KEY)[['implied_vol']]
    native=native.merge(metadata,on=KEY,validate='one_to_one')
    return long,old,native


def family_panel(long,old,native,spec):
    parts=[];audit=[]
    for cutoff,stop in blocks('2024-03-01',spec['fit_end']):
        selected,evidence=family_choices(long,old,spec,cutoff);audit.extend(evidence)
        test=native[native.origin_date.ge(cutoff)&native.origin_date.lt(stop)].copy()
        for h,g in test.groupby('horizon'):
            frame=g.copy().set_index(KEY)
            for name in ('futures','basis','m1','m2','lstm','ml'):frame[name+'_family']=frame[name+'_price']
            for family,cid in selected[str(int(h))].items():
                src=old if family=='v3' else long
                chosen=src[src.horizon.eq(h)&src.candidate.eq(cid)].set_index(KEY).forecast
                frame[family+'_family']=chosen.reindex(frame.index)
            frame['family_fit_cutoff']=cutoff;parts.append(frame.reset_index())
    result=pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    if not np.isfinite(result[[c+'_family' for c in spec['family_pool']]]).all().all():raise ValueError('Incomplete family forecasts')
    return result,audit


def rules(spec):
    result=[dict(id='s00',kind='futures',window=0,penalty=0.)]
    for window in spec['blend_windows']:
        result.append(dict(id=f's{len(result):02d}',kind='best',window=window,penalty=0.))
        for penalty in spec['blend_penalties']:
            result.append(dict(id=f's{len(result):02d}',kind='convex',window=window,penalty=penalty))
    return result


def fit_weights(train,rule,spec):
    if rule['window']:train=train.sort_values('origin_date').tail(rule['window'])
    names=spec['family_pool'];prior=np.zeros(len(names));prior[names.index('futures')]=1.
    if rule['kind']=='futures':return prior
    x=train[[c+'_family' for c in names]].to_numpy()-train.futures_price.to_numpy()[:,None]
    y=train.actual.to_numpy()-train.futures_price.to_numpy()
    if rule['kind']=='best':
        weight=np.zeros(len(names));weight[np.argmin(np.mean((x-y[:,None])**2,axis=0))]=1.;return weight
    scale=max(float(np.mean(y*y)),1e-10);penalty=rule['penalty']
    def fun(w):return np.mean((x@w-y)**2)/scale+penalty*np.sum((w-prior)**2)
    def jac(w):return 2*x.T@(x@w-y)/(len(y)*scale)+2*penalty*(w-prior)
    fit=minimize(fun,prior,jac=jac,method='SLSQP',bounds=[(0.,1.)]*len(names),
        constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(len(w))},options={'ftol':1e-11,'maxiter':500})
    if not fit.success:raise ValueError('Covariance stacking failed: '+fit.message)
    w=np.maximum(fit.x,0.);return w/w.sum()


def select():
    spec=json.loads((OUT/'specification.json').read_text());long,old,native=inputs(spec)
    panel,family_audit=family_panel(long,old,native,spec)
    panel.to_parquet(OUT/'family_crossfit.parquet',index=False)
    write_json(OUT/'family_selection_audit.json',family_audit)
    candidates=[];meta_audit=[];names=spec['family_pool']
    for cutoff,stop in blocks(spec['meta_start'],spec['fit_end']):
        for h,g in panel[panel.origin_date.ge(cutoff)&panel.origin_date.lt(stop)].groupby('horizon'):
            train=mature_before(panel[panel.horizon.eq(h)],cutoff)
            if len(train)<40:raise ValueError('Insufficient mature family errors')
            for rule in rules(spec):
                w=fit_weights(train,rule,spec);f=g.copy()
                f['forecast']=f[[c+'_family' for c in names]].to_numpy()@w
                f['rule']=rule['id'];f['meta_fit_cutoff']=cutoff;candidates.append(f)
                meta_audit.append(dict(cutoff=str(cutoff.date()),horizon=int(h),rule=rule['id'],weights=dict(zip(names,w.tolist())),
                    latest_label_release=str(train.label_available_date.max().date())))
    candidates=pd.concat(candidates,ignore_index=True)
    candidates.to_parquet(OUT/'stack_candidate_forecasts.parquet',index=False)
    past=mature_before(candidates,spec['rule_lock_date'])
    scores=past.assign(loss=(past.forecast-past.actual)**2).groupby(['horizon','rule']).agg(loss=('loss','mean'),n=('loss','size')).reset_index()
    scores.to_csv(OUT/'point_rule_selection.csv',index=False)
    choices=scores.sort_values(['horizon','loss','rule']).groupby('horizon').head(1)
    parts=[];selected={};byid={r['id']:r for r in rules(spec)}
    for row in choices.itertuples():
        selected[str(int(row.horizon))]=dict(**byid[row.rule],early_loss=row.loss,n=row.n)
        parts.append(candidates[candidates.horizon.eq(row.horizon)&candidates.rule.eq(row.rule)])
    points=pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    points['center']=np.log(points.forecast/points.origin_price);points['mode']='historical_crossfit'
    points.to_csv(OUT/'point_crossfit.csv',index=False)
    write_json(OUT/'locked_rules.json',selected);write_json(OUT/'stack_fit_audit.json',meta_audit)
    weights={}
    for h in HORIZONS:
        tr=mature_before(panel[panel.horizon.eq(h)],spec['fit_end'])
        weights[str(h)]=dict(zip(names,fit_weights(tr,selected[str(h)],spec).tolist()))
    write_json(OUT/'point_weights.json',weights)
    final,evidence=family_choices(long,old,spec,spec['fit_end'])
    write_json(OUT/'final_families.json',final);write_json(OUT/'final_family_audit.json',evidence)
    print('V4 locked historical rules:',json.dumps(selected),flush=True)
    print('V4 final weights:',json.dumps(weights),flush=True)


if __name__=='__main__':select()
