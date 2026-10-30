"""Separate historical selection and sealed October evaluation for v2."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table,observable_price,label_available_dates
from m3_ml.common.walkforward import final_training_positions
from m3_ml.common.protocol import FIT_END,VALIDATION_START,HORIZONS,PROTOCOL_ID
from m3_ml.fusion.run import join_member,read_csv,historical_distributions,october_frame
from m3_ml.fusion.model import mature_before
from .prepare import ROOT,OUT,OLD,PATH_H,write_json
from .model import (MEMBERS,CHANNELS,KEY,MEDIAN,ALL_COLUMNS,point_candidates,adaptive_points,
                    choose_penalty,fit_weights,center_from_weights,point_score,risk_grid,choose_risk,
                    selected_risk_frames,adaptive_risk,fit_calibrator,calibrate,metrics,risk_readout)


def validation_frame():
    table=load_table().loc[lambda x:x.date.le(FIT_END)].reset_index(drop=True)
    anchor=observable_price(table);release=label_available_dates(table);parts=[]
    for h in PATH_H:
        pos=final_training_positions(table.date,np.ones(len(table),bool),table.target_wti.gt(0),h,FIT_END)
        pos=pos[table.date.iloc[pos].ge(VALIDATION_START)]
        target=pos+h
        parts.append(pd.DataFrame(dict(origin_date=table.date.iloc[pos].to_numpy(),horizon=h,target_row=target,
                     target_date=table.date.iloc[target].to_numpy(),origin_price=anchor.iloc[pos].to_numpy(),
                     y=np.log(table.target_wti.iloc[target].to_numpy()/anchor.iloc[pos].to_numpy()),
                     label_available_date=[release.iloc[t:t+h+1].max() for t in pos])))
    f=pd.concat(parts,ignore_index=True)
    for member in MEMBERS:f=join_member(f,read_csv(OUT/f'{member}_validation.csv'),member)
    if f.target_date.max()>pd.Timestamp(FIT_END) or f.label_available_date.max()>pd.Timestamp(FIT_END):
        raise ValueError('Future selection label')
    return f.sort_values(KEY).reset_index(drop=True)


def seal_inputs():
    paths=[OUT/'specification.json',OUT/'base_folds.json',*[OUT/f'{m}_validation.csv' for m in MEMBERS],
           ROOT/'m3_ml/distribution/outputs/frozen_october_2026/validation_selected_quantiles.csv',
           ROOT/'data/processed/wti_daily.parquet',ROOT/'data/processed/wti_publication.csv']
    for folder in ('fusion_v2','fusion','common','distribution'):
        paths.extend(sorted((ROOT/'m3_ml'/folder).glob('*.py')))
    paths.extend([ROOT/'m3_ml/fusion_v2/replay_members.R',ROOT/'m3_ml/lstm/src/oil_lstm.py',
                  ROOT/'m1_time_series/for_test_set.R',ROOT/'m1_time_series/validation_selection.R',
                  ROOT/'m2_seasonality/seasonality.R'])
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def point_metrics(points):
    rows=[]
    for h,g in points.groupby('horizon'):
        for member in ['rw',*MEMBERS,'center']:
            pred=g.copy();pred['center']=0. if member=='rw' else g[member]
            rows.append(dict(horizon=int(h),member=member,n=len(g),rmse_vs_rw=point_score(pred)))
    return pd.DataFrame(rows)


def select():
    spec=json.loads((OUT/'specification.json').read_text())
    f=validation_frame();distributions=historical_distributions(f,spec)
    f.to_csv(OUT/'validation_member_panel.csv',index=False)
    candidates,candidate_audit=point_candidates(f,spec)
    points,point_audit=adaptive_points(f,candidates,spec)
    choices,point_scores=choose_penalty(candidates,FIT_END)
    pd.DataFrame(point_scores).to_csv(OUT/'point_selection.csv',index=False)
    candidates.to_csv(OUT/'point_candidate_forecasts.csv',index=False)
    points.to_csv(OUT/'point_crossfit.csv',index=False)
    point_metrics(points).to_csv(OUT/'point_validation_metrics.csv',index=False)
    print('Point meta-selection complete; now risk calibration',flush=True)
    losses,frames,risk_audit=risk_grid(points,distributions,spec)
    risk_choices,risk_scores=choose_risk(losses,FIT_END)
    pd.DataFrame(risk_scores).to_csv(OUT/'risk_selection.csv',index=False)
    losses.to_csv(OUT/'risk_candidate_losses.csv',index=False)
    historical=selected_risk_frames(frames,risk_choices)
    historical['mode']='historical_crossfit'
    historical.to_csv(OUT/'validation_forecasts.csv',index=False)
    metrics(historical,'integrated_v2').to_csv(OUT/'validation_metrics.csv',index=False)
    outer,outer_audit=adaptive_risk(losses,frames,spec)
    outer['mode']='historical_crossfit'
    outer.to_csv(OUT/'outer_validation_forecasts.csv',index=False)
    metrics(outer,'integrated_v2').to_csv(OUT/'outer_validation_metrics.csv',index=False)
    weights={};calibration={}
    for h in PATH_H:
        key=str(h);tr=mature_before(f,FIT_END);tr=tr[tr.horizon.eq(h)]
        w=fit_weights(tr,choices[key]['penalty'])
        weights[key]=dict(weights=dict(zip(CHANNELS,w.tolist())),penalty=choices[key]['penalty'],n_train=len(tr),
                          latest_label_release=str(tr.label_available_date.max().date()))
        method=risk_choices[key]['method']
        cp=points[points.horizon.eq(h)].merge(distributions[method][KEY+ALL_COLUMNS],on=KEY,validate='one_to_one')
        calibration[key]=fit_calibrator(mature_before(cp,FIT_END))[key]
    model=dict(protocol_id=PROTOCOL_ID,version=spec['version'],name=spec['name'],fit_end=FIT_END,
               point_by_horizon=weights,risk_by_horizon=risk_choices,calibration=calibration,
               point_meaning=spec['point_meaning'],historical_panel_rows=len(f),point_crossfit_rows=len(points),
               risk_validation_rows=len(historical),risk_outer_rows=len(outer),
               max_label_release=str(f.label_available_date.max().date()),
               input_sha256=seal_inputs(),base_selection_limitation=spec['base_selection_limitation'],
               october_status=spec['october_status'])
    write_json(OUT/'selected.json',model)
    write_json(OUT/'crossfit_audit.json',dict(point_candidates=candidate_audit,point=point_audit,
                                            risk_calibration=risk_audit,risk_outer=outer_audit))
    print('SEALED before October evaluation:',hashlib.sha256((OUT/'selected.json').read_bytes()).hexdigest(),flush=True)
    # Re-run the entire point selector without each member; no October data.
    rows=[]
    for h,g in points.groupby('horizon'):rows.append(dict(removed='none',horizon=int(h),n=len(g),rmse_vs_rw=point_score(g)))
    for removed in MEMBERS:
        members=tuple(m for m in MEMBERS if m!=removed)
        ab_candidates,_=point_candidates(f,spec,members)
        ab,_=adaptive_points(f,ab_candidates,spec,members)
        for h,g in ab.groupby('horizon'):rows.append(dict(removed=removed,horizon=int(h),n=len(g),rmse_vs_rw=point_score(g)))
        print('Historical point ablation complete:',removed,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'ablation.csv',index=False)


def final_forecasts(spec,model):
    f=october_frame(spec)
    for h,indices in f.groupby('horizon').groups.items():
        entry=model['point_by_horizon'][str(int(h))]
        w=[entry['weights'][m] for m in CHANNELS]
        f.loc[indices,'center']=center_from_weights(f.loc[indices],w)
    distributions=read_csv(ROOT/'m3_ml/distribution/outputs/frozen_october_2026/test_quantiles.csv')
    parts=[]
    for h,g in f.groupby('horizon'):
        entry=model['risk_by_horizon'][str(int(h))]
        d=distributions[distributions.method.eq(entry['method'])]
        pred=g.merge(d[['mode',*KEY,*ALL_COLUMNS]],on=['mode',*KEY],how='left',validate='one_to_one')
        Q=calibrate(pred,model['calibration'],entry['mixture'],entry['width'])
        pred.loc[:,ALL_COLUMNS]=Q
        pred['risk_method']=entry['method'];pred['risk_mixture']=entry['mixture'];pred['risk_width']=entry['width']
        pred['forecast']=pred.origin_price*np.exp(pred.center)
        pred['lower90']=pred.origin_price*np.exp(pred['q_0.050']);pred['upper90']=pred.origin_price*np.exp(pred['q_0.950'])
        for column,value in risk_readout(Q).items():pred[column]=value
        parts.append(pred)
    result=pd.concat(parts,ignore_index=True).sort_values(['mode',*KEY]).reset_index(drop=True)
    result['method']=model['name'];result['protocol_id']=PROTOCOL_ID;result['version']=model['version'];result['fit_end']=FIT_END
    result['status']=np.where(result.y.notna(),'realized','unreached')
    return result


def test():
    model=json.loads((OUT/'selected.json').read_text());spec=json.loads((OUT/'specification.json').read_text())
    if model['input_sha256']!=seal_inputs():raise ValueError('Selection sources changed; reselection required before test')
    selected_hash=hashlib.sha256((OUT/'selected.json').read_bytes()).hexdigest()
    f=final_forecasts(spec,model)
    if len(f)!=87 or f.duplicated(['mode',*KEY]).any():raise ValueError('Incomplete October grid')
    Q=f[ALL_COLUMNS].to_numpy()
    assert np.isfinite(Q).all() and (np.diff(Q,axis=1)>=-1e-10).all()
    np.testing.assert_array_equal(Q[:,MEDIAN],f.center)
    for h in HORIZONS:
        a=f[f['mode'].eq('fixed_origin')&f.horizon.eq(h)]
        b=f[f['mode'].eq('daily_update')&f.origin_date.eq(FIT_END)&f.horizon.eq(h)]
        np.testing.assert_allclose(a[ALL_COLUMNS],b[ALL_COLUMNS],atol=1e-12)
    f.to_csv(OUT/'test_forecasts.csv',index=False)
    scores=metrics(f,'integrated_v2');scores.to_csv(OUT/'test_metrics.csv',index=False)
    fixed=f[f['mode'].eq('fixed_origin')&f.actual.notna()]
    path_rmse=float(np.mean((fixed.forecast-fixed.actual)**2)**.5)
    path_rw=float(np.mean((fixed.origin_price-fixed.actual)**2)**.5)
    path=dict(n=len(fixed),rmse_usd=path_rmse,rw_rmse_usd=path_rw,rmse_vs_rw=path_rmse/path_rw,
              coverage90=float(((fixed.actual>=fixed.lower90)&(fixed.actual<=fixed.upper90)).mean()),
              interpretation='17 correlated targets from one origin; descriptive evaluation only')
    assert selected_hash==hashlib.sha256((OUT/'selected.json').read_bytes()).hexdigest()
    write_json(OUT/'test_audit.json',dict(status='passed',selected_sha256=selected_hash,rows=len(f),
                                        median_preserved=True,quantiles_monotone=True,first_origins_equal=True,
                                        actual_counts=scores[['mode','horizon','n']].to_dict('records'),fixed_path_metrics=path))
    print('OCTOBER EVALUATION ONLY',json.dumps(path),flush=True)
    print(scores[scores.horizon.isin(HORIZONS)].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['select','test']);args=parser.parse_args()
    (select if args.stage=='select' else test)()
