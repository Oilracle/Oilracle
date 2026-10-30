"""Incumbent-anchored shrinkage fitted only on already-public forecast errors."""
import json
import numpy as np
import pandas as pd
from m3_ml.fusion.model import blocks,mature_before
from .data import OUT,OLD,KEY,HORIZONS,write_json

PENALTIES=(.1,1.,10.)


def coefficient(history,penalty):
    history=history.sort_values('origin_date').tail(252)
    if len(history)<40:return 0.
    delta=history.proposal.to_numpy()-history.incumbent.to_numpy()
    residual=history.actual.to_numpy()-history.incumbent.to_numpy()
    denominator=np.mean(delta*delta)+penalty*max(np.mean(residual*residual),1e-10)
    return float(np.clip(np.mean(delta*residual)/denominator,0.,1.))


def choose(candidates,cutoff,h):
    # Both base policies are locked by September 2025; only subsequent honest
    # forecast errors can choose this additional penalty. Earlier errors may
    # estimate coefficients once both base policies have been chosen.
    past=mature_before(candidates[candidates.horizon.eq(h)&candidates.origin_date.ge('2025-09-01')],cutoff)
    if pd.Timestamp(cutoff)<pd.Timestamp('2026-03-01') or past.empty or past.groupby('penalty').size().min()<20:return 1.
    scores=past.assign(loss=(past.forecast-past.actual)**2).groupby('penalty').loss.mean()
    return float(scores.sort_values(kind='stable').index[0])


def main():
    if not (OUT/'guard_design.json').exists():
        write_json(OUT/'guard_design.json',dict(reason='The unguarded proposal failed the predeclared 2% per-horizon degradation gate at h1.',
            procedure='Ridge-shrink the new-minus-incumbent forecast difference toward zero; coefficients in [0,1], no minimum weight.',
            penalties=list(PENALTIES),history_window=252,default_penalty=1.,penalty_selection_start='2026-03-01',
            eligible_penalty_scores='Only public outcomes of forecasts issued after both point policies locked on 2025-09-01.',
            development_amendment=True,october_used=False))
    if not (OUT/'point_proposal.csv').exists():(OUT/'point_proposal.csv').write_bytes((OUT/'point_crossfit.csv').read_bytes())
    f=pd.read_csv(OUT/'point_proposal.csv',parse_dates=['origin_date','target_date','label_available_date'])
    old=pd.read_csv(OLD/'point_crossfit.csv',parse_dates=['origin_date'])
    f=f.merge(old[KEY+['forecast']].rename(columns={'forecast':'incumbent'}),on=KEY,validate='one_to_one')
    f['proposal']=f.forecast;candidates=[];fit_audit=[]
    for cutoff,stop in blocks('2024-09-01','2026-09-30'):
        for h,g in f[f.origin_date.ge(cutoff)&f.origin_date.lt(stop)].groupby('horizon'):
            train=mature_before(f[f.horizon.eq(h)],cutoff)
            for penalty in PENALTIES:
                w=coefficient(train,penalty);p=g.copy()
                p['forecast']=p.incumbent+w*(p.proposal-p.incumbent);p['penalty']=penalty;p['upgrade_weight']=w
                p['guard_fit_cutoff']=cutoff;candidates.append(p)
                fit_audit.append(dict(cutoff=str(cutoff.date()),horizon=int(h),penalty=penalty,weight=w,
                    latest_label_release=None if train.empty else str(train.label_available_date.max().date())))
    candidates=pd.concat(candidates,ignore_index=True);parts=[];selection_audit=[]
    for cutoff,stop in blocks('2024-09-01','2026-09-30'):
        for h in HORIZONS:
            penalty=choose(candidates,cutoff,h)
            parts.append(candidates[candidates.origin_date.ge(cutoff)&candidates.origin_date.lt(stop)
                                    &candidates.horizon.eq(h)&candidates.penalty.eq(penalty)])
            selection_audit.append(dict(cutoff=str(cutoff.date()),horizon=h,penalty=penalty))
    result=pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    result['center']=np.log(result.forecast/result.origin_price)
    result.to_csv(OUT/'point_crossfit.csv',index=False)
    candidates.to_parquet(OUT/'guard_candidate_forecasts.parquet',index=False)
    final={}
    for h in HORIZONS:
        penalty=choose(candidates,'2026-09-30',h)
        train=mature_before(f[f.horizon.eq(h)],'2026-09-30')
        final[str(h)]=dict(penalty=penalty,upgrade_weight=coefficient(train,penalty),
                          n_train=min(252,len(train)),latest_label_release=str(train.label_available_date.max().date()))
    write_json(OUT/'guard_final.json',final)
    write_json(OUT/'guard_audit.json',dict(fits=fit_audit,selections=selection_audit))
    print('Incumbent-anchored final coefficients:',json.dumps(final),flush=True)


if __name__=='__main__':main()
