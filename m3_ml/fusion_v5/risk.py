"""Past-only tail-score calibration with explicit coverage and width limits."""
import numpy as np
import pandas as pd
from m3_ml.fusion.model import blocks,mature_before,shape_scale,MEDIAN,metrics
from m3_ml.fusion_v4.risk import sources
from m3_ml.distribution.tails import ALL_COLUMNS,ALL_TAUS,MAIN
from m3_ml.distribution.quantiles import crps
from .data import OUT,OLD,KEY,HORIZONS,read,write_json

LOW=int(np.flatnonzero(ALL_TAUS==.05)[0]);HIGH=int(np.flatnonzero(ALL_TAUS==.95)[0])


def definitions(spec):
    return [dict(id=f'r{i:03d}',method=method,window=window,target=target,sides=sides)
        for i,(method,window,target,sides) in enumerate((m,w,t,s) for m in spec['risk_sources']
        for w in spec['risk_windows'] for t in spec['risk_targets'] for s in spec['risk_sides'])]


def upper_quantile(values,probability):
    n=len(values);rank=min(n,max(1,int(np.ceil((n+1)*probability))))
    return float(np.sort(values)[rank-1])


def fit(frame,d):
    train=frame.sort_values('origin_date')
    if d['window']:train=train.tail(d['window'])
    if len(train)<40:raise ValueError('Need at least 40 released calibration outcomes')
    shape,scale=shape_scale(train[ALL_COLUMNS]);residual=train.y.to_numpy()-train.center.to_numpy()
    lower=np.maximum(-residual/np.maximum(1e-6,-shape[:,LOW]*scale),0.)
    upper=np.maximum(residual/np.maximum(1e-6,shape[:,HIGH]*scale),0.)
    if d['sides']=='symmetric':left=right=upper_quantile(np.maximum(lower,upper),d['target'])
    else:
        level=1-(1-d['target'])/2
        left=upper_quantile(lower,level);right=upper_quantile(upper,level)
    return dict(left=max(.25,left),right=max(.25,right),n_train=len(train),
        latest_label_release=str(train.label_available_date.max().date()))


def predict(frame,calibration):
    shape,scale=shape_scale(frame[ALL_COLUMNS])
    factors=np.where(ALL_TAUS<.5,calibration['left'],calibration['right'])
    q=frame.center.to_numpy()[:,None]+shape*scale[:,None]*factors[None,:]
    q[:,MEDIAN]=frame.center.to_numpy()
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-10).all()
    return q


def choose(losses,cutoff,h,spec):
    past=mature_before(losses[losses.horizon.eq(h)],cutoff);constraint=spec['risk_constraint']
    dates=past.origin_date.drop_duplicates().sort_values().tail(constraint['selection_window'])
    past=past[past.origin_date.isin(dates)]
    stats=past.groupby('candidate').agg(crps=('crps','mean'),coverage=('covered','mean'),
        lower_miss=('lower_miss','mean'),upper_miss=('upper_miss','mean'),width=('width','mean'),hs_width=('hs_width','mean'),n=('crps','size')).reset_index()
    stats=stats[stats.n.ge(20)].copy()
    if stats.empty:raise ValueError('Insufficient earlier risk validation outcomes')
    stats['width_ratio']=stats.width/stats.hs_width
    stats['shortfall']=np.maximum(constraint['min_coverage']-stats.coverage,0.)+np.maximum(stats.lower_miss-constraint['max_tail_miss'],0.)+np.maximum(stats.upper_miss-constraint['max_tail_miss'],0.)
    feasible=stats[stats.shortfall.le(1e-12)&stats.width_ratio.le(constraint['max_width_vs_hs'])]
    if len(feasible):chosen=feasible.sort_values(['crps','width','candidate']).iloc[0];satisfied=True
    else:
        bounded=stats[stats.width_ratio.le(constraint['max_width_vs_hs'])]
        if bounded.empty:bounded=stats.sort_values(['width_ratio']).head(1)
        chosen=bounded.sort_values(['shortfall','crps','candidate']).iloc[0];satisfied=False
    return chosen.candidate,dict(constraints_satisfied=satisfied,n=int(chosen.n),coverage=float(chosen.coverage),
        crps=float(chosen.crps),width_ratio=float(chosen.width_ratio),lower_miss=float(chosen.lower_miss),upper_miss=float(chosen.upper_miss),
        latest_label_release=str(past.label_available_date.max().date()))


def main():
    spec=read(OUT/'specification.json');defs=definitions(spec);by_id={d['id']:d for d in defs}
    points=pd.read_csv(OUT/'point_crossfit.csv',parse_dates=['origin_date','target_date','label_available_date'])
    frames=sources(points,spec);base=frames['hs']
    hs_crps=crps(base[ALL_COLUMNS].to_numpy()[:,MAIN],base.y)
    hs_width=base.origin_price*(np.exp(base['q_0.950'])-np.exp(base['q_0.050']))
    for frame in frames.values():frame['hs_crps']=hs_crps;frame['hs_width']=hs_width.to_numpy()
    losses=[];audit=[];calibrations={}
    for method,frame in frames.items():
        for cutoff,stop in blocks(spec['risk_start'],spec['fit_end']):
            for h,g in frame[frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)].groupby('horizon'):
                train=mature_before(frame[frame.horizon.eq(h)],cutoff)
                for d in (d for d in defs if d['method']==method):
                    calibration=fit(train,d);q=predict(g,calibration)
                    calibrations[(str(cutoff.date()),int(h),d['id'])]=calibration
                    row=g[KEY+['label_available_date','hs_crps','hs_width']].copy()
                    row['candidate']=d['id'];row['crps']=crps(q[:,MAIN],g.y)
                    row['covered']=(g.y.to_numpy()>=q[:,LOW])&(g.y.to_numpy()<=q[:,HIGH])
                    row['lower_miss']=g.y.to_numpy()<q[:,LOW];row['upper_miss']=g.y.to_numpy()>q[:,HIGH]
                    row['width']=g.origin_price.to_numpy()*(np.exp(q[:,HIGH])-np.exp(q[:,LOW]));losses.append(row)
                    audit.append(dict(cutoff=str(cutoff.date()),horizon=int(h),candidate=d['id'],**calibration))
        print('Coverage-constrained source crossfit:',method,flush=True)
    losses=pd.concat(losses,ignore_index=True);losses.to_parquet(OUT/'risk_candidate_scores.parquet',index=False)
    parts=[];selections=[]
    for cutoff,stop in blocks(spec['risk_outer_start'],spec['fit_end']):
        for h in HORIZONS:
            cid,evidence=choose(losses,cutoff,h,spec);d=by_id[cid]
            frame=frames[d['method']];g=frame[frame.horizon.eq(h)&frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)].copy()
            if g.empty:continue
            calibration=calibrations[(str(cutoff.date()),h,cid)]
            g.loc[:,ALL_COLUMNS]=predict(g,calibration);g['risk_candidate']=cid;g['risk_fit_cutoff']=cutoff;parts.append(g)
            selections.append(dict(cutoff=str(cutoff.date()),horizon=h,candidate=cid,**evidence))
    forward=pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    forward.to_csv(OUT/'risk_forward_proposal.csv',index=False)
    final={}
    for h in HORIZONS:
        cid,evidence=choose(losses,spec['fit_end'],h,spec);d=by_id[cid]
        train=mature_before(frames[d['method']][frames[d['method']].horizon.eq(h)],spec['fit_end'])
        final[str(h)]=dict(**d,calibration=fit(train,d),selection=evidence)
    write_json(OUT/'risk_final.json',final);write_json(OUT/'risk_audit.json',dict(fits=audit,selections=selections))
    prior=pd.read_csv(OLD/'risk_forward_forecasts.csv',parse_dates=['origin_date'])
    prior['v4_crps']=crps(prior[ALL_COLUMNS].to_numpy()[:,MAIN],prior.y)
    prior['v4_width']=prior.origin_price*(np.exp(prior['q_0.950'])-np.exp(prior['q_0.050']))
    prior['v4_covered']=prior.y.between(prior['q_0.050'],prior['q_0.950'])
    merged=forward.merge(prior[KEY+['v4_crps','v4_width','v4_covered']],on=KEY,validate='one_to_one')
    merged['loss']=crps(merged[ALL_COLUMNS].to_numpy()[:,MAIN],merged.y);rows=[]
    for h,g in merged.groupby('horizon'):
        coverage=g.y.between(g['q_0.050'],g['q_0.950']).mean();old_coverage=g.v4_covered.mean()
        width=(g.origin_price*(np.exp(g['q_0.950'])-np.exp(g['q_0.050']))).mean()
        rows.append(dict(horizon=int(h),n=len(g),crps_vs_v4=float(g.loss.mean()/g.v4_crps.mean()),crps_vs_hs=float(g.loss.mean()/g.hs_crps.mean()),
            coverage90=float(coverage),v4_coverage90=float(old_coverage),coverage_gap_improvement=float(abs(old_coverage-.9)-abs(coverage-.9)),
            width90_usd=float(width),width_vs_v4=float(width/g.v4_width.mean())))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'historical_risk_proposal_metrics.csv',index=False)
    gate=spec['risk_promotion']
    stable=bool(summary.crps_vs_v4.mean()<=gate['max_mean_crps_ratio'] and summary.crps_vs_v4.max()<=gate['max_horizon_crps_ratio']
        and summary.width_vs_v4.max()<=gate['max_horizon_width_ratio'] and summary.coverage_gap_improvement.min()>=-gate['max_horizon_gap_degradation'])
    passed=stable and summary.coverage_gap_improvement.mean()>=gate['min_mean_coverage_gap_improvement']
    point_passed=read(OUT/'point_promotion_gate.json')['passed'];promote=bool(passed or (point_passed and stable))
    write_json(OUT/'promotion_gate.json',dict(passed=promote,point_passed=point_passed,risk_passed=bool(passed),risk_stable=stable,
        mean_coverage_gap_improvement=float(summary.coverage_gap_improvement.mean()),mean_crps_vs_v4=float(summary.crps_vs_v4.mean()),
        thresholds=gate,october_used=False))
    print(summary.to_string(index=False),flush=True)
    print('Overall promotion:',promote,'point:',point_passed,'risk:',passed,'risk stable:',stable,flush=True)
    print('Final risk choices:',{h:dict(candidate=d['id'],source=d['method'],feasible=d['selection']['constraints_satisfied']) for h,d in final.items()},flush=True)


if __name__=='__main__':main()
