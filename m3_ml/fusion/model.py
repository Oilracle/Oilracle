"""Causal meta-fitting and a single coherent, residual-calibrated quantile curve."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from m3_ml.common.protocol import HORIZONS
from m3_ml.distribution.tails import ALL_TAUS, ALL_COLUMNS, MAIN, var_es
from m3_ml.distribution.quantiles import crps

MEMBERS=('m1','m2','lstm','ml')
MEDIAN=int(np.flatnonzero(ALL_TAUS==.5)[0])

def mature_before(frame, cutoff):
    cutoff=pd.Timestamp(cutoff)
    return frame[frame.origin_date.lt(cutoff) & frame.label_available_date.le(cutoff)].copy()

def fit_weights(frame, penalty, members=MEMBERS, floor=.05):
    f=frame[frame.horizon.isin(HORIZONS)]
    if f.empty or set(f.horizon)!=set(HORIZONS):
        raise ValueError('Each key horizon needs mature training labels')
    x=f[list(members)].to_numpy(); y=f.y.to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite weight training data')
    factors=np.zeros(len(f))
    for h in HORIZONS:
        mask=f.horizon.eq(h).to_numpy()
        factors[mask]=1/(len(HORIZONS)*mask.sum()*max(np.mean(y[mask]**2),1e-10))
    prior=np.full(len(members),1/len(members))
    def fun(w):
        e=x@w-y
        return np.sum(factors*e**2)+penalty*np.sum((w-prior)**2)
    def jac(w): return 2*x.T@(factors*(x@w-y))+2*penalty*(w-prior)
    result=minimize(fun,prior,jac=jac,method='SLSQP',bounds=[(floor,1)]*len(members),
                    constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(w))},
                    options={'ftol':1e-12,'maxiter':300})
    if not result.success: raise ValueError('Weight optimization failed: '+result.message)
    if min(result.x)<floor-1e-9 or abs(sum(result.x)-1)>1e-9: raise ValueError('Invalid weights')
    return result.x

def blocks(start, end):
    starts=pd.date_range(start,end,freq='3MS')
    for i,s in enumerate(starts):
        yield s, starts[i+1] if i+1<len(starts) else pd.Timestamp(end)+pd.Timedelta(days=1)

def crossfit_points(frame, penalty, start, end, members=MEMBERS):
    parts=[]; evidence=[]
    for cutoff,stop in blocks(start,end):
        train=mature_before(frame,cutoff)
        test=frame[frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)].copy()
        if test.empty: continue
        w=fit_weights(train,penalty,members)
        test['center']=test[list(members)].to_numpy()@w
        test['meta_fit_cutoff']=cutoff
        parts.append(test)
        evidence.append(dict(cutoff=str(cutoff.date()),n_train=int(train.horizon.isin(HORIZONS).sum()),
                             n_panel=len(train),n_pred=len(test),
                             latest_training_release=str(train.label_available_date.max().date()),
                             weights=dict(zip(members,map(float,w)))))
    return pd.concat(parts,ignore_index=True), evidence

def point_score(frame, column='center'):
    return float(np.mean([np.sqrt(np.mean((g[column]-g.y)**2)/np.mean(g.y**2))
                          for h,g in frame[frame.horizon.isin(HORIZONS)].groupby('horizon')]))

def shape_scale(Q):
    Q=np.sort(np.asarray(Q,float),axis=1)
    lo=int(np.flatnonzero(ALL_TAUS==.05)[0]); hi=int(np.flatnonzero(ALL_TAUS==.95)[0])
    scale=(Q[:,hi]-Q[:,lo])/(norm.ppf(.95)-norm.ppf(.05))
    if not np.isfinite(Q).all() or np.any(scale<=1e-8): raise ValueError('Degenerate ML uncertainty scale')
    return (Q-Q[:,[MEDIAN]])/scale[:,None],scale

def fit_calibrator(frame):
    _,scale=shape_scale(frame[ALL_COLUMNS])
    residual=(frame.y.to_numpy()-frame.center.to_numpy())/scale
    result={}
    for h,g in frame.assign(z=residual).groupby('horizon'):
        if len(g)<80: raise ValueError(f'h={h}: only {len(g)} mature cross-fitted residuals')
        result[str(int(h))]=dict(n=len(g),quantiles=np.quantile(g.z,ALL_TAUS).tolist(),
                                 latest_label_release=str(g.label_available_date.max().date()))
    return result

def calibrate(frame, calibration, mixture):
    shape,scale=shape_scale(frame[ALL_COLUMNS])
    empirical=np.array([calibration[str(int(h))]['quantiles'] for h in frame.horizon])
    bias=empirical[:,[MEDIAN]]
    z=bias+(1-mixture)*shape+mixture*(empirical-bias)
    Q=frame.center.to_numpy()[:,None]+scale[:,None]*z
    if not np.isfinite(Q).all() or np.any(np.diff(Q,axis=1)<-1e-10): raise ValueError('Invalid calibrated quantile curve')
    return Q

def crossfit_risk(points, distributions, mixture, start, end):
    f=points.merge(distributions[['origin_date','horizon',*ALL_COLUMNS]],on=['origin_date','horizon'],validate='one_to_one',how='left')
    if f[ALL_COLUMNS].isna().any().any(): raise ValueError('Missing historical ML distribution on fusion grid')
    result=[]; evidence=[]
    for cutoff,stop in blocks(start,end):
        train=mature_before(f,cutoff)
        test=f[f.origin_date.ge(cutoff)&f.origin_date.lt(stop)].copy()
        if test.empty: continue
        calibration=fit_calibrator(train)
        Q=calibrate(test,calibration,mixture)
        test.loc[:,ALL_COLUMNS]=Q
        test['risk_fit_cutoff']=cutoff
        result.append(test)
        evidence.append(dict(cutoff=str(cutoff.date()),calibration=calibration))
    return pd.concat(result,ignore_index=True),evidence

def losses(frame):
    Q=frame[ALL_COLUMNS].to_numpy()
    y=frame.y.to_numpy()
    return crps(Q[:,MAIN],y)

def risk_score(frame):
    return float(np.mean([np.mean(losses(g))/np.mean(g.hs_crps)
                          for _,g in frame[frame.horizon.isin(HORIZONS)].groupby('horizon')]))

def risk_readout(Q):
    Q=np.asarray(Q)
    def probability(threshold):
        return np.array([np.interp(threshold,row,ALL_TAUS) for row in Q])
    out={'prob_up':1-probability(0.),'prob_drop_5pct':probability(np.log(.95)),
         'prob_rise_5pct':1-probability(np.log(1.05))}
    for a in (.01,.025,.05):
        v,e=var_es(Q[:,MAIN],Q,a,'lower')
        out[f'var_loss_log_{a:g}']=-v
        out[f'es_loss_log_{a:g}']=-e
    return out

def metrics(frame, label):
    rows=[]
    for (mode,h),g in frame.groupby(['mode','horizon']):
        g=g.dropna(subset=['y']); n=len(g)
        row=dict(method=label,mode=mode,horizon=int(h),n=n)
        if n:
            Q=g[ALL_COLUMNS].to_numpy(); pred=g.origin_price.to_numpy()*np.exp(Q[:,MEDIAN])
            actual=g.origin_price.to_numpy()*np.exp(g.y)
            rw=np.sqrt(np.mean((g.origin_price-actual)**2))
            row.update(rmse_usd=float(np.sqrt(np.mean((pred-actual)**2))),mae_usd=float(np.mean(abs(pred-actual))),
                       rw_rmse_usd=float(rw),rmse_vs_rw=float(np.sqrt(np.mean((pred-actual)**2))/rw),
                       crps_log=float(np.mean(losses(g))),coverage90=float(np.mean((g.y>=g['q_0.050'])&(g.y<=g['q_0.950']))),
                       width90_usd=float(np.mean(g.origin_price*(np.exp(g['q_0.950'])-np.exp(g['q_0.050'])))))
        row['inference']='descriptive_only' if mode!='historical_crossfit' else 'conditional_development_validation'
        rows.append(row)
    return pd.DataFrame(rows)
