import json
import inspect
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from m3_ml.common.preprocessing import make_preprocessor
from m3_ml.common.execution import ValidationCache,run_jobs
from m3_ml.tabular.models import MODELS
from m3_ml.fusion.model import blocks,mature_before
from .data import OUT,BASE,ROOT,KEY,MEMBERS,PATH_H,write_json,historical_inputs

BLEND=('rw','futures','basis',*MEMBERS)


def fit_predict(train,test,features,definition,return_model=False):
    kind=definition['kind'];window=definition['window']
    if window:train=train.iloc[-window:]
    if kind=='carry':
        return test[definition['base']+'_price'].to_numpy(),None
    if kind=='convex':
        x=train[[m+'_price' for m in BLEND]].to_numpy()
        future=test[[m+'_price' for m in BLEND]].to_numpy()
        y=train.actual.to_numpy();prior=np.zeros(len(BLEND));prior[1]=1.
        scale=max(np.mean((train.futures_price-y)**2),1e-8);penalty=definition['penalty']
        def fun(w):return float(np.mean((x@w-y)**2)/scale+penalty*np.sum((w-prior)**2))
        def jac(w):return 2*x.T@(x@w-y)/(len(y)*scale)+2*penalty*(w-prior)
        result=minimize(fun,prior,jac=jac,method='SLSQP',bounds=[(0.,1.)]*len(prior),
                        constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(len(w))},
                        options={'ftol':1e-10,'maxiter':500})
        if not result.success:raise ValueError('Convex fitting failed: '+result.message)
        w=np.maximum(result.x,0.);w/=w.sum()
        return future@w,dict(kind='convex',weights=w)
    X=features[definition['feature']]
    target=train.actual-train[definition['base']+'_price']
    if kind=='ridge':model=make_pipeline(make_preprocessor('scaled'),Ridge(alpha=definition['alpha']))
    elif kind=='elastic_net':model=MODELS[kind][0](alpha_ratio=definition['alpha_ratio'],l1_ratio=.5)
    else:model=MODELS[kind][0](num_leaves=definition['num_leaves'],n_estimators=100)
    model.fit(X.loc[train.index],target)
    price=test[definition['base']+'_price'].to_numpy()+model.predict(X.loc[test.index])
    return np.maximum(price,.01),model


def job(context,task):
    panel,features,spec=context
    h,definition=task
    frame=panel[panel.horizon.eq(h)];parts=[]
    for cutoff,stop in blocks(spec['candidate_start'],spec['fit_end']):
        train=mature_before(frame,cutoff);test=frame[frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)]
        if test.empty:continue
        if len(train)<40:raise ValueError('Insufficient mature point labels')
        price,_=fit_predict(train,test,features,definition)
        if not np.isfinite(price).all():raise ValueError('Nonfinite forecast')
        parts.append(np.column_stack([test.index,price]))
    return np.concatenate(parts)


def blend_weights(history,rule):
    if rule['window']:
        last=history.origin_date.drop_duplicates().sort_values().iloc[-rule['window']:]
        history=history[history.origin_date.isin(last)]
    losses=history.assign(loss=(history.forecast-history.actual)**2).groupby('candidate').loss.mean().sort_index()
    mode=rule['mode'];weights=pd.Series(0.,index=losses.index)
    if mode=='best':weights.loc[losses.sort_values(kind='stable').index[0]]=1.
    elif mode=='top3':weights.loc[losses.sort_values(kind='stable').index[:3]]=1/3
    else:
        eta=float(mode[4:]);weights=np.exp(-eta*(losses/max(losses.min(),1e-10)-1))
        weights/=weights.sum()
    return weights


def apply_rule(candidates,rule,start,end):
    rows=[];audit=[]
    for cutoff,stop in blocks(start,end):
        history=mature_before(candidates,cutoff)
        test=candidates[candidates.origin_date.ge(cutoff)&candidates.origin_date.lt(stop)]
        if test.empty:continue
        if history.groupby('candidate').size().min()<20:raise ValueError('Insufficient past candidate scores')
        weights=blend_weights(history,rule)
        wide=test.pivot(index=KEY,columns='candidate',values='forecast')
        if wide.isna().any().any():raise ValueError('Unequal candidate forecast coverage')
        meta=test.drop_duplicates(KEY).set_index(KEY).loc[wide.index].copy()
        meta['forecast']=wide[weights.index].to_numpy()@weights.to_numpy()
        meta['meta_fit_cutoff']=cutoff
        rows.append(meta.reset_index())
        audit.append(dict(cutoff=str(cutoff.date()),horizon=int(test.horizon.iloc[0]),weights=weights.to_dict(),
                          latest_selection_release=str(history.label_available_date.max().date()),n_origins=history.origin_date.nunique()))
    return pd.concat(rows,ignore_index=True),audit


def select():
    spec=json.loads((OUT/'specification.json').read_text())
    table,panel,features=historical_inputs()
    panel.to_csv(OUT/'validation_member_panel.csv',index=False)
    tasks=[(h,d) for h in PATH_H for d in spec['candidates']]
    cache=ValidationCache('fusion_v3_points',{'panel':panel,'features':features['full'],'spec':spec,
                         'fit_source':inspect.getsource(fit_predict),'job_source':inspect.getsource(job)})
    results=run_jobs(job,tasks,(panel,features,spec),'V3 historical point fits',cache)
    parts=[]
    for (h,d),values in zip(tasks,results):
        f=panel.loc[values[:,0].astype(int)].copy()
        f['forecast']=values[:,1];f['candidate']=d['id'];parts.append(f)
    candidates=pd.concat(parts,ignore_index=True)
    candidates.to_csv(OUT/'point_candidate_forecasts.csv',index=False)
    chosen_rules={};rule_scores=[];point_parts=[];point_audit=[]
    for h in PATH_H:
        g=candidates[candidates.horizon.eq(h)];by_rule={};audits={}
        for i,rule in enumerate(spec['rules']):
            pred,audit=apply_rule(g,rule,spec['point_start'],spec['fit_end'])
            early=mature_before(pred,spec['rule_lock_date'])
            loss=float(np.mean((early.forecast-early.actual)**2))
            rule_scores.append(dict(horizon=h,rule=i,loss=loss,n=len(early),**rule))
            by_rule[i]=pred;audits[i]=audit
        best=min([r for r in rule_scores if r['horizon']==h],key=lambda r:(r['loss'],r['rule']))
        chosen_rules[str(h)]=best
        point_parts.append(by_rule[best['rule']]);point_audit.extend(audits[best['rule']])
    points=pd.concat(point_parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    points['center']=np.log(points.forecast/points.origin_price)
    points['mode']='historical_crossfit'
    points.to_csv(OUT/'point_crossfit.csv',index=False)
    write_json(OUT/'locked_point_rules.json',chosen_rules)
    pd.DataFrame(rule_scores).to_csv(OUT/'point_rule_selection.csv',index=False)
    write_json(OUT/'point_crossfit_audit.json',point_audit)
    weights={}
    for h in PATH_H:
        past=mature_before(candidates[candidates.horizon.eq(h)],spec['fit_end'])
        weights[str(h)]=blend_weights(past,chosen_rules[str(h)]).to_dict()
    write_json(OUT/'point_weights.json',weights)
    # Persist final regressors using only mature pre-October outcomes. No October
    # feature or outcome file is needed for this fitting phase.
    model_dir=OUT/'models';model_dir.mkdir(exist_ok=True)
    fitted={};audit=[]
    for h in PATH_H:
        tr=mature_before(panel[panel.horizon.eq(h)],spec['fit_end'])
        for d in spec['candidates']:
            if weights[str(h)].get(d['id'],0.)<1e-12:continue
            _,model=fit_predict(tr,tr.iloc[:1],features,d,True)
            fitted[(h,d['id'])]=model
        audit.append(dict(horizon=h,n_train=len(tr),latest_label_release=str(tr.label_available_date.max().date())))
    joblib.dump(fitted,model_dir/'point_models.joblib')
    write_json(OUT/'point_final_fit_audit.json',audit)
    print('V3 point models fitted and selector rules locked from data public before',spec['rule_lock_date'],flush=True)


if __name__=='__main__':select()
