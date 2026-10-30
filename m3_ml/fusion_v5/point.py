"""Nested time selection of state gates and learned member reliability."""
import inspect
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,softmax
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from m3_ml.common.execution import ValidationCache,run_jobs
from m3_ml.fusion.model import blocks,mature_before
from .data import OUT,KEY,HORIZONS,EXPERTS,STATE,read,write_json,historical_inputs


def reliability_features(frame):
    state=frame[STATE+['state_season_cos','state_vol_ratio','state_basis_slope']].to_numpy()
    experts=frame[[e+'_expert' for e in EXPERTS]].to_numpy()
    delta=(experts[:,1:]-experts[:,[0]])/frame.scale.to_numpy()[:,None]
    return np.column_stack([state,np.clip(delta,-20.,20.)])


def memberships(frame,model):
    if model['definition']['kind']=='static':return np.ones((len(frame),1))
    columns=STATE[:model['definition']['n_state']]
    z=(frame[columns].to_numpy()-model['location'])/model['spread']
    upper=expit(np.clip(z,-8.,8.));return np.column_stack([upper,1-upper])/len(columns)


def fit(train,d):
    if d['window']:train=train.sort_values('origin_date').tail(d['window'])
    model=dict(definition=d,n_train=len(train),latest_label_release=str(train.label_available_date.max().date()))
    if d['kind']=='incumbent':return model
    if 'support_decay' in d:
        from .support import bounds
        model['support_bounds']=bounds(train)
    p=train[[e+'_expert' for e in EXPERTS]].to_numpy();actual=train.actual.to_numpy()
    if d['kind'] in ('static','regime'):
        if d['kind']=='regime':
            z=train[STATE[:d['n_state']]].to_numpy()
            model['location']=np.median(z,axis=0)
            model['spread']=np.maximum(np.quantile(z,.75,axis=0)-np.quantile(z,.25,axis=0),.1)
        gamma=memberships(train,model);k=gamma.shape[1];m=len(EXPERTS)
        prior=np.zeros((k,m));prior[:,0]=1.
        x=(gamma[:,:,None]*(p-p[:,[0]])[:,None,:]).reshape(len(p),-1)
        y=actual-p[:,0];scale=max(float(np.mean(y*y)),1e-8)
        penalty=d['penalty']/k
        def objective(w):return np.mean((x@w-y)**2)/scale+penalty*np.sum((w-prior.ravel())**2)
        def gradient(w):return 2*x.T@(x@w-y)/(len(y)*scale)+2*penalty*(w-prior.ravel())
        constraint=np.kron(np.eye(k),np.ones((1,m)))
        result=minimize(objective,prior.ravel(),jac=gradient,method='SLSQP',bounds=[(0.,1.)]*(k*m),
            constraints={'type':'eq','fun':lambda w:constraint@w-1.,'jac':lambda w:constraint},options={'ftol':1e-9,'maxiter':400})
        if not result.success:raise ValueError('Conditional stacking failed: '+result.message)
        weights=np.maximum(result.x.reshape(k,m),0.);model['weights']=weights/weights.sum(axis=1)[:,None]
    else:
        x=reliability_features(train);scaler=StandardScaler().fit(x);x=scaler.transform(x)
        target=np.log1p(((p-actual[:,None])/train.scale.to_numpy()[:,None])**2)
        if d['learner']=='ridge':
            estimator=Ridge(alpha=d['alpha']).fit(x,target)
        else:
            estimator=[LGBMRegressor(n_estimators=100,num_leaves=4,max_depth=3,learning_rate=.03,
                min_child_samples=25,reg_lambda=10.,n_jobs=1,verbosity=-1,random_state=5210,deterministic=True,
                force_col_wise=True).fit(x,target[:,j]) for j in range(len(EXPERTS))]
        model['scaler']=scaler;model['estimator']=estimator
    return model


def predict(frame,model,return_weights=False):
    d=model['definition'];p=frame[[e+'_expert' for e in EXPERTS]].to_numpy()
    if d['kind']=='incumbent':weights=np.tile(np.eye(len(EXPERTS))[0],(len(frame),1))
    elif d['kind'] in ('static','regime'):weights=memberships(frame,model)@model['weights']
    else:
        x=model['scaler'].transform(reliability_features(frame));estimator=model['estimator']
        loss=estimator.predict(x) if d['learner']=='ridge' else np.column_stack([e.predict(x) for e in estimator])
        logits=-np.maximum(loss,0.)/d['temperature'];logits[:,0]+=np.log(d['anchor_prior'])
        weights=softmax(logits,axis=1)
    if 'support_decay' in d:
        from .support import trust
        confidence=trust(frame,model['support_bounds'],d['support_decay'])
        weights*=confidence[:,None];weights[:,0]+=1-confidence
    if 'update_fraction' in d:
        fraction=d['update_fraction'];weights*=fraction;weights[:,0]+=1-fraction
    price=np.sum(weights*p,axis=1)
    if not np.isfinite(price).all() or (price<=0).any():raise ValueError('Invalid gated forecast')
    return (price,weights) if return_weights else price


def job(context,task):
    panel,spec=context;h,d=task;f=panel[panel.horizon.eq(h)];parts=[]
    for cutoff,stop in blocks(spec['candidate_start'],spec['fit_end']):
        train=mature_before(f,cutoff);test=f[f.origin_date.ge(cutoff)&f.origin_date.lt(stop)]
        if test.empty:continue
        if len(train)<spec['min_mature_train']:raise ValueError('Not enough released training labels')
        model=fit(train,d);price=predict(test,model)
        parts.append(np.column_stack([test.index,price,np.full(len(test),cutoff.value),
            np.full(len(test),pd.Timestamp(model['latest_label_release']).value),np.full(len(test),model['n_train'])]))
    return np.concatenate(parts)


def choose(candidates,cutoff,h,spec):
    history=mature_before(candidates[candidates.horizon.eq(h)],cutoff)
    dates=history.origin_date.drop_duplicates().sort_values().tail(spec['selector_window'])
    history=history[history.origin_date.isin(dates)]
    scores=history.assign(loss=(history.forecast-history.actual)**2).groupby('candidate').agg(loss=('loss','mean'),n=('loss','size'))
    scores=scores[scores.n.ge(spec['min_mature_train'])].sort_values(['loss','candidate'])
    if scores.empty:return 'p000',dict(n=0,latest_label_release=None)
    best=scores.index[0];baseline=scores.loc['p000','loss']
    selected=best if scores.loc[best,'loss']<=(1-spec['selector_min_relative_gain'])*baseline else 'p000'
    return selected,dict(n=int(scores.loc[selected,'n']),latest_label_release=str(history.label_available_date.max().date()),
        best=best,best_mse_vs_incumbent=float(scores.loc[best,'loss']/baseline),selected_mse_vs_incumbent=float(scores.loc[selected,'loss']/baseline))


def main():
    spec=read(OUT/'specification.json');panel=historical_inputs();panel.to_parquet(OUT/'expert_panel.parquet',index=False)
    tasks=[(h,d) for h in HORIZONS for d in spec['point_candidates']]
    cache=ValidationCache('state_reliability_v5',dict(panel=panel,spec=spec,code=inspect.getsource(inspect.getmodule(fit))))
    values=run_jobs(job,tasks,(panel,spec),'V5 state and reliability fits',cache)
    pieces=[];audits=[]
    for (h,d),array in zip(tasks,values):
        f=panel.loc[array[:,0].astype(int)].copy();f['forecast']=array[:,1];f['candidate']=d['id']
        f['point_fit_cutoff']=pd.to_datetime(array[:,2].astype(np.int64));pieces.append(f)
        evidence=pd.DataFrame(dict(cutoff=pd.to_datetime(array[:,2].astype(np.int64)),
            latest_label_release=pd.to_datetime(array[:,3].astype(np.int64)),n_train=array[:,4].astype(int))).drop_duplicates()
        for row in evidence.itertuples():audits.append(dict(horizon=h,candidate=d['id'],cutoff=str(row.cutoff.date()),
            latest_label_release=str(row.latest_label_release.date()),n_train=int(row.n_train)))
    candidates=pd.concat(pieces,ignore_index=True);candidates.to_parquet(OUT/'point_candidate_forecasts.parquet',index=False)
    write_json(OUT/'point_fit_audit.json',audits)
    selected=[];selections=[]
    for cutoff,stop in blocks(spec['outer_start'],spec['fit_end']):
        for h in HORIZONS:
            cid,evidence=choose(candidates,cutoff,h,spec)
            g=candidates[candidates.horizon.eq(h)&candidates.candidate.eq(cid)&candidates.origin_date.ge(cutoff)&candidates.origin_date.lt(stop)].copy()
            g['selection_cutoff']=cutoff;selected.append(g)
            selections.append(dict(cutoff=str(cutoff.date()),horizon=h,candidate=cid,**evidence))
    proposal=pd.concat(selected,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    proposal['center']=np.log(proposal.forecast/proposal.origin_price)
    proposal.to_csv(OUT/'point_proposal.csv',index=False);write_json(OUT/'point_selection_audit.json',selections)
    final={str(h):dict(candidate=cid,**evidence) for h in HORIZONS for cid,evidence in [choose(candidates,spec['fit_end'],h,spec)]}
    write_json(OUT/'point_final.json',final)
    print('Nested choices:',selections,flush=True);print('Final candidate IDs:',final,flush=True)


if __name__=='__main__':main()
