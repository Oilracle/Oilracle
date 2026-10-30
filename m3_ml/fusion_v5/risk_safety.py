"""Retain the incumbent interval when a new calibration would narrow it."""
import numpy as np
import pandas as pd
from m3_ml.distribution.tails import ALL_COLUMNS,ALL_TAUS,MAIN
from m3_ml.distribution.quantiles import crps
from m3_ml.fusion.model import MEDIAN
from .data import OUT,OLD,KEY,read,write_json
from .risk import LOW,HIGH


def specification():
    path=OUT/'risk_safety_design.json'
    if path.exists():return read(path)
    design=dict(development_amendment=True,reason='Raw coverage-constrained selection still undercovered short horizons after a regime change and doubled h60 width.',
        policy='Take an outward quantile envelope with the causally issued incumbent shape at the new center. Interpolate back toward the incumbent if the 90% price interval exceeds 1.5 times incumbent width.',
        price_width_cap=1.5,cap_selection='Fixed research safeguard, not selected using October or a grid of caps.',
        promotion_thresholds='Unchanged original v5 thresholds; all new results remain adaptive historical development.',october_used=False)
    write_json(path,design);return design


def envelope(new,old,center,cap=1.5):
    new=np.asarray(new);old=np.asarray(old);center=np.asarray(center)
    base=old-old[:,[MEDIAN]]+center[:,None]
    outward=np.where(ALL_TAUS[None,:]<.5,np.minimum(new,base),np.maximum(new,base))
    outward[:,MEDIAN]=center
    max_width=cap*(np.exp(base[:,HIGH])-np.exp(base[:,LOW]))
    lo=np.zeros(len(new));hi=np.ones(len(new))
    needs=(np.exp(outward[:,HIGH])-np.exp(outward[:,LOW]))>max_width
    for _ in range(40):
        t=(lo+hi)/2
        lower=base[:,LOW]+t*(outward[:,LOW]-base[:,LOW])
        upper=base[:,HIGH]+t*(outward[:,HIGH]-base[:,HIGH])
        too_wide=np.exp(upper)-np.exp(lower)>max_width
        hi=np.where(too_wide,t,hi);lo=np.where(too_wide,lo,t)
    fraction=np.where(needs,lo,1.)
    q=base+fraction[:,None]*(outward-base);q[:,MEDIAN]=center
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-10).all()
    return q,fraction


def main():
    design=specification();spec=read(OUT/'specification.json')
    raw=pd.read_csv(OUT/'risk_forward_proposal.csv',parse_dates=['origin_date'])
    prior=pd.read_csv(OLD/'risk_forward_forecasts.csv',parse_dates=['origin_date']).copy()
    prior=prior.set_index(KEY).reindex(pd.MultiIndex.from_frame(raw[KEY])).reset_index()
    assert len(prior)==len(raw) and prior[ALL_COLUMNS].notna().all().all()
    q,fraction=envelope(raw[ALL_COLUMNS],prior[ALL_COLUMNS],raw.center,design['price_width_cap'])
    result=raw.copy();result.loc[:,ALL_COLUMNS]=q;result['risk_safety_fraction']=fraction
    result.to_csv(OUT/'risk_forward_forecasts.csv',index=False)
    rows=[];loss=crps(q[:,MAIN],raw.y);old_loss=crps(prior[ALL_COLUMNS].to_numpy()[:,MAIN],prior.y)
    for h,indices in result.groupby('horizon').groups.items():
        g=result.loc[indices];old=prior.loc[indices]
        coverage=g.y.between(g['q_0.050'],g['q_0.950']).mean();old_cov=old.y.between(old['q_0.050'],old['q_0.950']).mean()
        width=(g.origin_price*(np.exp(g['q_0.950'])-np.exp(g['q_0.050']))).mean()
        old_width=(old.origin_price*(np.exp(old['q_0.950'])-np.exp(old['q_0.050']))).mean()
        rows.append(dict(horizon=int(h),n=len(g),crps_vs_v4=float(loss[indices].mean()/old_loss[indices].mean()),
            crps_vs_hs=float(loss[indices].mean()/g.hs_crps.mean()),coverage90=float(coverage),v4_coverage90=float(old_cov),
            coverage_gap_improvement=float(abs(old_cov-.9)-abs(coverage-.9)),width90_usd=float(width),width_vs_v4=float(width/old_width)))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'historical_risk_metrics.csv',index=False)
    thresholds=spec['risk_promotion']
    stable=bool(summary.crps_vs_v4.mean()<=thresholds['max_mean_crps_ratio'] and summary.crps_vs_v4.max()<=thresholds['max_horizon_crps_ratio']
        and summary.width_vs_v4.max()<=thresholds['max_horizon_width_ratio'] and summary.coverage_gap_improvement.min()>=-thresholds['max_horizon_gap_degradation'])
    passed=stable and summary.coverage_gap_improvement.mean()>=thresholds['min_mean_coverage_gap_improvement']
    point_passed=read(OUT/'point_promotion_gate.json')['passed'];promote=bool(passed or (point_passed and stable))
    (OUT/'raw_promotion_gate.json').write_bytes((OUT/'promotion_gate.json').read_bytes())
    decision=dict(passed=promote,point_passed=point_passed,risk_passed=bool(passed),risk_stable=stable,
        mean_coverage_gap_improvement=float(summary.coverage_gap_improvement.mean()),mean_crps_vs_v4=float(summary.crps_vs_v4.mean()),
        thresholds=thresholds,interval_safety=design,october_used=False)
    write_json(OUT/'promotion_gate.json',decision)
    print(summary.to_string(index=False),flush=True);print('Safety-envelope promotion:',decision,flush=True)


if __name__=='__main__':main()
