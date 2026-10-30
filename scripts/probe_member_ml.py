"""Historical-only feasibility probe: ML corrections to actual M1/M2 forecasts.

This experiment does not fit or evaluate October, or replace the sealed system.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table
from m3_ml.common.features import build_features
from m3_ml.common.execution import ValidationCache,run_jobs
from m3_ml.common.protocol import FIT_END,HORIZONS
from m3_ml.fusion.model import blocks,mature_before
from m3_ml.tabular.models import MODELS

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'m3_ml/fusion_v2/outputs/frozen_october_2026'
OUT=ROOT/'m3_ml/residual/outputs/member_ml_probe'
GRID=[('elastic_net',dict(alpha_ratio=a,l1_ratio=.5)) for a in (.3,.1,.03)]
GRID += [('lightgbm',dict(num_leaves=n,n_estimators=100)) for n in (4,8)]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def error_states(frame,member):
    """At an origin, use only earlier base errors whose outcomes are public."""
    error=frame.origin_price.to_numpy()*(np.exp(frame.y)-np.exp(frame[member]))
    rows=[]
    for date in frame.origin_date:
        eligible=frame.origin_date.lt(date)&frame.label_available_date.le(date)
        seen=error[eligible]
        rows.append([seen.iloc[-1] if len(seen) else 0.,seen.iloc[-20:].mean() if len(seen) else 0.,
                     seen.iloc[-60:].mean() if len(seen) else 0.,seen.iloc[-60:].std(ddof=0) if len(seen) else 0.,
                     min(len(seen),60)])
    return pd.DataFrame(rows,columns=['error_last_public','error_mean20_public','error_mean60_public','error_std60_public','error_history_count'])


def feature_sets(frame,table,shared,member):
    row=pd.Series(table.index,index=table.date).reindex(frame.origin_date).to_numpy()
    state=shared.iloc[row].reset_index(drop=True).copy()
    state['base_increment_usd']=frame.origin_price.to_numpy()*np.expm1(frame[member].to_numpy())
    state=pd.concat([state,error_states(frame,member)],axis=1)
    date=frame.origin_date.dt
    angle=2*np.pi*(date.month.to_numpy()-1)/12
    cal=pd.DataFrame({f'month_{fn}_{k}':getattr(np,fn)(k*angle) for k in (1,2) for fn in ('sin','cos')})
    for weekday in range(1,5):cal[f'weekday_{weekday}']=(date.dayofweek.to_numpy()==weekday).astype(float)
    combined=pd.concat([state,cal],axis=1)
    for season in ('month_sin_1','month_cos_1'):
        for condition in ('oil_implied_volatility','commercial_crude_stocks_yoy_chg','wti_ret_20d'):
            combined[f'{season}_x_{condition}']=cal[season]*state[condition]
    return {'calendar':cal,'state':state,'calendar_state':combined}


def job(context,task):
    panels,features=context
    member,h,feature,method,params=task
    frame=panels[h];X=features[(member,h)][feature] if feature!='none' else None
    target=frame.origin_price.to_numpy()*(np.exp(frame.y)-np.exp(frame[member]))
    parts=[]
    for cutoff,stop in blocks('2024-03-01',FIT_END):
        train=np.flatnonzero((frame.origin_date.lt(cutoff)&frame.label_available_date.le(cutoff)).to_numpy())
        test=np.flatnonzero((frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)).to_numpy())
        if not len(test):continue
        if len(train)<40:raise ValueError('Need at least 40 mature base residuals')
        if method=='none':prediction=np.zeros(len(test))
        elif method=='mean_bias':prediction=np.full(len(test),target.iloc[train].mean())
        else:
            model=MODELS[method][0](**params).fit(X.iloc[train],target.iloc[train])
            prediction=model.predict(X.iloc[test])
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite ML correction')
        parts.append(np.column_stack([test,prediction]))
    return np.concatenate(parts)


def select_previous(candidates,cutoff):
    past=mature_before(candidates,cutoff);options=[]
    for candidate,g in past.groupby('candidate'):
        if len(g)<20:continue
        options.append(dict(candidate=candidate,mse=float(np.mean((g.forecast-g.actual)**2)),n=len(g)))
    if not options:return 'none/none/0',[]
    winner=min(options,key=lambda x:(round(x['mse'],10),0 if x['candidate']=='none/none/0' else 1,x['candidate']))
    return winner['candidate'],options


def score(frame,label,member,h):
    rmse=lambda x:float(np.mean((x-frame.actual)**2)**.5)
    base=rmse(frame.base);rw=rmse(frame.origin_price);value=rmse(frame.forecast)
    return dict(member=member,horizon=int(h),evaluation=label,n=len(frame),rmse_usd=value,base_rmse_usd=base,
                rw_rmse_usd=rw,rmse_vs_base=value/base,rmse_vs_rw=value/rw,
                nonzero_correction_fraction=float(np.mean(abs(frame.forecast-frame.base)>1e-8)))


def feature_ablation(candidates):
    rows=[]
    for (member,h),g in candidates.groupby(['member','horizon']):
        for family in ('bias_only','calendar','state','calendar_state'):
            keep=g.candidate.str.startswith('none/')
            if family!='bias_only':keep=keep|g.candidate.str.startswith(family+'/')
            subset=g[keep];parts=[]
            for cutoff,stop in blocks('2024-09-01',FIT_END):
                choice,_=select_previous(subset,cutoff)
                parts.append(subset[subset.candidate.eq(choice)&subset.origin_date.ge(cutoff)&subset.origin_date.lt(stop)])
            rows.append(score(pd.concat(parts),'restricted_'+family,member,h))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[SOURCE/'selected.json',SOURCE/'test_forecasts.csv',ROOT/'m4_reports/final_strategy_manifest.json']
    before={str(p.relative_to(ROOT)):digest(p) for p in protected}
    spec=dict(scope='Historical feasibility only; no October fit/evaluation and no replacement of sealed strategy',
              members=['m1','m2'],horizons=list(HORIZONS),candidate_forecasts_start='2024-03-01',
              forward_selection_start='2024-09-01',end=FIT_END,minimum_mature_training_errors=40,
              minimum_mature_selection_scores=20,grid=GRID,features=['calendar','state','calendar_state'],
              correction_weights=[0.,.25,.5,1.],constant_bias_control=True,
              error_state_policy='Only earlier origin errors with label_available_date <= current origin; zero/count=0 cold start',
              limitation='Base specs were selected on the full development window. These are conditional historical development results, not a new independent test.')
    (OUT/'specification.json').write_text(json.dumps(spec,indent=2)+'\n')
    table=load_table().loc[lambda x:x.date.le(FIT_END)].reset_index(drop=True)
    panel=pd.read_csv(SOURCE/'validation_member_panel.csv',parse_dates=['origin_date','target_date','label_available_date'])
    assert panel.target_date.max()<=pd.Timestamp(FIT_END) and panel.label_available_date.max()<=pd.Timestamp(FIT_END)
    shared=build_features(table);panels={h:panel[panel.horizon.eq(h)].reset_index(drop=True) for h in HORIZONS}
    features={(member,h):feature_sets(f,table,shared,member) for member in spec['members'] for h,f in panels.items()}
    assert all(np.isfinite(x).all().all() for group in features.values() for x in group.values())
    # Mutation test on data-derived residual-state features: no future outcome
    # can affect inputs at earlier origins.
    for member in spec['members']:
        for h,f in panels.items():
            changed=f.copy();changed.loc[changed.label_available_date.gt('2025-03-01'),'y']+=1.
            got=error_states(changed,member);early=f.origin_date.le('2025-03-01')
            np.testing.assert_array_equal(got.loc[early],error_states(f,member).loc[early])
    tasks=[]
    for member in spec['members']:
        for h in HORIZONS:
            tasks.extend([(member,h,'none','none',{}),(member,h,'none','mean_bias',{})])
            for feature in spec['features']:
                tasks.extend([(member,h,feature,method,params) for method,params in GRID])
    cache=ValidationCache('actual_member_ml_probe',{'panel':panel,'table':table,'spec':spec,'source':Path(__file__).read_text()})
    outputs=run_jobs(job,tasks,(panels,features),'Actual M1/M2 ML correction probe',cache)
    forecasts=[];candidate_map={}
    for task,values in zip(tasks,outputs):
        member,h,feature,method,params=task
        params_id=','.join(f'{k}={v}' for k,v in sorted(params.items()))
        for weight in ([0.] if method=='none' else [.25,.5,1.]):
            candidate='none/none/0' if method=='none' else f'{feature}/{method}/{params_id}/w={weight:g}'
            candidate_map[candidate]=dict(feature=feature,method=method,params=params,weight=weight)
            f=panels[h].iloc[values[:,0].astype(int)].copy()
            f['actual']=f.origin_price*np.exp(f.y);f['base']=f.origin_price*np.exp(f[member])
            f['forecast']=f.base+weight*values[:,1]
            f['candidate']=candidate;f['member']=member
            forecasts.append(f[['origin_date','horizon','label_available_date','origin_price','actual','base','forecast','candidate','member']])
    candidates=pd.concat(forecasts,ignore_index=True)
    rows=[];adaptive=[];selection_audit=[];final_choices={};candidate_scores=[];quarters=[]
    for (member,h),all_candidates in candidates.groupby(['member','horizon']):
        parts=[]
        for cutoff,stop in blocks('2024-09-01',FIT_END):
            choice,options=select_previous(all_candidates,cutoff)
            test=all_candidates[all_candidates.candidate.eq(choice)&all_candidates.origin_date.ge(cutoff)&all_candidates.origin_date.lt(stop)].copy()
            if test.empty:continue
            test['selection_cutoff']=cutoff;parts.append(test)
            selection_audit.append(dict(member=member,horizon=int(h),cutoff=str(cutoff.date()),candidate=choice,
                                        latest_selection_release=str(mature_before(all_candidates,cutoff).label_available_date.max().date())))
        result=pd.concat(parts,ignore_index=True);adaptive.append(result)
        rows.append(score(result,'past_only_adaptive_selection',member,h))
        for quarter,g in result.groupby(result.origin_date.dt.to_period('Q')):
            quarters.append(dict(**score(g,'past_only_adaptive_selection',member,h),quarter=str(quarter)))
        final,options=select_previous(all_candidates,FIT_END)
        final_choices[f'{member}/h{h}']=dict(candidate=final,**candidate_map[final])
        for option in options:candidate_scores.append(dict(member=member,horizon=int(h),**option))
        # This score is a development-selected comparison, not the adaptive check.
        development=all_candidates[all_candidates.candidate.eq(final)&all_candidates.origin_date.ge('2024-09-01')]
        rows.append(score(development,'full_history_selected_candidate',member,h))
    pd.concat(adaptive,ignore_index=True).to_csv(OUT/'forward_forecasts.csv',index=False)
    candidates.to_csv(OUT/'candidate_forecasts.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(quarters).to_csv(OUT/'quarterly_comparison.csv',index=False)
    pd.DataFrame(candidate_scores).to_csv(OUT/'candidate_scores.csv',index=False)
    feature_ablation(candidates).to_csv(OUT/'feature_ablation.csv',index=False)
    (OUT/'historical_choices.json').write_text(json.dumps(final_choices,indent=2)+'\n')
    (OUT/'selection_audit.json').write_text(json.dumps(selection_audit,indent=2)+'\n')
    # Check the forward selector itself, not just features, against future labels.
    first=select_previous(candidates[(candidates.member=='m1')&(candidates.horizon==1)],'2025-03-01')
    changed=candidates[(candidates.member=='m1')&(candidates.horizon==1)].copy()
    changed.loc[changed.label_available_date.gt('2025-03-01'),'actual']+=9999.
    assert first==select_previous(changed,'2025-03-01')
    assert before=={str(p.relative_to(ROOT)):digest(p) for p in protected}
    audit=dict(status='passed',no_october_evaluation=True,sealed_v2_unchanged=True,
               future_error_mutation_invariant=True,future_selection_loss_mutation_invariant=True,
               max_label_release=str(panel.label_available_date.max().date()),task_count=len(tasks),
               candidates_per_member_horizon=len(candidate_map),protected_sha256=before)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(json.dumps(final_choices,indent=2),flush=True)


if __name__=='__main__':main()
