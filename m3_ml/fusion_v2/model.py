"""Horizon-specific price stacking and median-preserving risk calibration."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from m3_ml.fusion.model import MEMBERS, MEDIAN, mature_before, blocks, shape_scale, risk_readout, metrics
from m3_ml.distribution.tails import ALL_COLUMNS, ALL_TAUS, MAIN
from m3_ml.distribution.quantiles import crps

CHANNELS=('rw',*MEMBERS)
KEY=['origin_date','horizon']


def price_design(frame,members=MEMBERS):
    anchor=frame.origin_price.to_numpy()
    x=np.column_stack([np.zeros(len(frame)),anchor[:,None]*np.expm1(frame[list(members)].to_numpy())])
    target=anchor*np.expm1(frame.y.to_numpy())
    return x,target


def fit_weights(frame,penalty,members=MEMBERS):
    if frame.horizon.nunique()!=1 or len(frame)<40:raise ValueError('Need >=40 mature labels at one horizon')
    x,y=price_design(frame,members)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite weight inputs')
    prior=np.zeros(x.shape[1]);prior[0]=1.
    if penalty=='rw':return prior
    scale=max(float(np.mean(y*y)),1e-10)
    def fun(w):return float(np.mean((x@w-y)**2)/scale+float(penalty)*np.sum((w-prior)**2))
    def jac(w):return 2*x.T@(x@w-y)/(len(y)*scale)+2*float(penalty)*(w-prior)
    result=minimize(fun,prior,jac=jac,method='SLSQP',bounds=[(0.,1.)]*len(prior),
                    constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(len(w))},
                    options={'ftol':1e-12,'maxiter':500})
    if not result.success:raise ValueError('Weight fit failed: '+result.message)
    w=np.maximum(result.x,0.);w/=w.sum()
    return w


def center_from_weights(frame,weights,members=MEMBERS):
    x,_=price_design(frame,members)
    price=frame.origin_price.to_numpy()+x@np.asarray(weights)
    if np.any(price<=0) or not np.isfinite(price).all():raise ValueError('Invalid convex price center')
    return np.log(price/frame.origin_price.to_numpy())


def point_score(frame):
    actual=frame.origin_price.to_numpy()*np.exp(frame.y.to_numpy())
    price=frame.origin_price.to_numpy()*np.exp(frame.center.to_numpy())
    return float(np.sqrt(np.mean((price-actual)**2)/np.mean((frame.origin_price-actual)**2)))


def point_candidates(frame,spec,members=MEMBERS):
    predictions=[];audit=[]
    for cutoff,stop in blocks(spec['point_candidate_start'],spec['fit_end']):
        train=mature_before(frame,cutoff)
        test=frame[frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)]
        for h,g in test.groupby('horizon'):
            tr=train[train.horizon.eq(h)]
            for penalty in ['rw',*spec['ridge_candidates']]:
                w=fit_weights(tr,penalty,members)
                pred=g.copy();pred['center']=center_from_weights(pred,w,members)
                pred['penalty']=str(penalty);pred['meta_fit_cutoff']=cutoff;predictions.append(pred)
                audit.append(dict(cutoff=str(cutoff.date()),horizon=int(h),penalty=str(penalty),n_train=len(tr),
                                  latest_label_release=str(tr.label_available_date.max().date()),
                                  weights=dict(zip(('rw',*members),w.tolist()))))
    return pd.concat(predictions,ignore_index=True),audit


def choose_penalty(candidates,cutoff):
    past=mature_before(candidates,cutoff);chosen={};rows=[]
    for h,g in past.groupby('horizon'):
        scores=[]
        for penalty,p in g.groupby('penalty'):
            if len(p)<20:raise ValueError('Insufficient past candidate predictions')
            scores.append(dict(horizon=int(h),penalty=str(penalty),score=point_score(p),n=len(p)))
        winner=min(scores,key=lambda r:(round(r['score'],12),-float('inf') if r['penalty']=='rw' else -float(r['penalty'])))
        chosen[str(int(h))]=winner;rows.extend(scores)
    return chosen,rows


def adaptive_points(frame,candidates,spec,members=MEMBERS):
    parts=[];audit=[]
    for cutoff,stop in blocks(spec['point_validation_start'],spec['fit_end']):
        choices,_=choose_penalty(candidates,cutoff)
        train=mature_before(frame,cutoff)
        test=frame[frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)]
        for h,g in test.groupby('horizon'):
            entry=choices[str(int(h))];tr=train[train.horizon.eq(h)]
            w=fit_weights(tr,entry['penalty'],members)
            pred=g.copy();pred['center']=center_from_weights(pred,w,members)
            pred['meta_fit_cutoff']=cutoff;pred['selected_penalty']=entry['penalty'];parts.append(pred)
            audit.append(dict(cutoff=str(cutoff.date()),horizon=int(h),selection=entry,n_train=len(tr),
                              latest_label_release=str(tr.label_available_date.max().date()),
                              latest_selection_release=str(mature_before(candidates[candidates.horizon.eq(h)],cutoff).label_available_date.max().date()),
                              weights=dict(zip(('rw',*members),w.tolist()))))
    return pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True),audit


def fit_calibrator(frame):
    _,scale=shape_scale(frame[ALL_COLUMNS])
    residual=(frame.y.to_numpy()-frame.center.to_numpy())/scale
    result={}
    for h,g in frame.assign(z=residual).groupby('horizon'):
        if len(g)<80:raise ValueError(f'h={h}: need >=80 mature cross-fitted errors')
        q=np.quantile(g.z,ALL_TAUS)
        result[str(int(h))]=dict(n=len(g),quantiles=(q-q[MEDIAN]).tolist(),
                                residual_median_diagnostic=float(q[MEDIAN]),
                                latest_label_release=str(g.label_available_date.max().date()))
    return result


def calibrate(frame,calibration,mixture,width):
    shape,scale=shape_scale(frame[ALL_COLUMNS])
    empirical=np.array([calibration[str(int(h))]['quantiles'] for h in frame.horizon])
    deviation=float(width)*scale[:,None]*((1-float(mixture))*shape+float(mixture)*empirical)
    out=frame.center.to_numpy()[:,None]+deviation
    out[:,MEDIAN]=frame.center.to_numpy()
    if not np.isfinite(out).all() or np.any(np.diff(out,axis=1)<-1e-10):raise ValueError('Invalid risk quantiles')
    return out


def risk_grid(points,distributions,spec):
    losses=[];frames={};audit={}
    for method in spec['risk_methods']:
        f=points.merge(distributions[method][KEY+ALL_COLUMNS],on=KEY,validate='one_to_one')
        pieces={(mix,width):[] for mix in spec['risk_mixture_candidates'] for width in spec['risk_width_candidates']}
        evidence=[]
        for cutoff,stop in blocks(spec['risk_validation_start'],spec['fit_end']):
            train=mature_before(f,cutoff)
            test=f[f.origin_date.ge(cutoff)&f.origin_date.lt(stop)].copy()
            if test.empty:continue
            calibration=fit_calibrator(train)
            evidence.append(dict(cutoff=str(cutoff.date()),calibration=calibration))
            for (mix,width),parts in pieces.items():
                pred=test.copy();Q=calibrate(test,calibration,mix,width)
                pred.loc[:,ALL_COLUMNS]=Q;pred['risk_fit_cutoff']=cutoff
                pred['risk_method']=method;pred['risk_mixture']=mix;pred['risk_width']=width
                parts.append(pred)
        audit[method]=evidence
        for (mix,width),parts in pieces.items():
            pred=pd.concat(parts,ignore_index=True);frames[(method,mix,width)]=pred
            scores=pred[KEY+['label_available_date','hs_crps']].copy()
            scores['method']=method;scores['mixture']=mix;scores['width']=width
            scores['loss']=crps(pred[ALL_COLUMNS].to_numpy()[:,MAIN],pred.y)
            losses.append(scores)
        print('Historical risk candidates complete:',method,flush=True)
    return pd.concat(losses,ignore_index=True),frames,audit


def choose_risk(losses,cutoff):
    past=mature_before(losses,cutoff);chosen={};rows=[]
    for (h,method,mix,width),g in past.groupby(['horizon','method','mixture','width']):
        if len(g)<20:raise ValueError('Need >=20 mature risk validation scores')
        rows.append(dict(horizon=int(h),method=method,mixture=float(mix),width=float(width),
                         score=float(g.loss.mean()/g.hs_crps.mean()),n=len(g)))
    for h in sorted(past.horizon.unique()):
        options=[r for r in rows if r['horizon']==h]
        chosen[str(int(h))]=min(options,key=lambda r:(round(r['score'],12),abs(r['width']-1.),r['mixture'],r['method']))
    return chosen,rows


def selected_risk_frames(frames,choices,start=None,stop=None):
    parts=[]
    for h,entry in choices.items():
        pred=frames[(entry['method'],entry['mixture'],entry['width'])]
        pred=pred[pred.horizon.eq(int(h))].copy()
        if start is not None:pred=pred[pred.origin_date.ge(start)]
        if stop is not None:pred=pred[pred.origin_date.lt(stop)]
        parts.append(pred)
    return pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)


def adaptive_risk(losses,frames,spec):
    parts=[];audit=[]
    for cutoff,stop in blocks(spec['risk_outer_start'],spec['fit_end']):
        choices,_=choose_risk(losses,cutoff)
        pred=selected_risk_frames(frames,choices,cutoff,stop)
        pred['risk_selection_cutoff']=cutoff;parts.append(pred)
        audit.append(dict(cutoff=str(cutoff.date()),choices=choices,
                          latest_selection_release=str(mature_before(losses,cutoff).label_available_date.max().date())))
    return pd.concat(parts,ignore_index=True),audit
