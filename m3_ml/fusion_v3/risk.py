"""Cross-fitted shape/width calibration, including causal futures volatility."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from m3_ml.fusion.model import blocks,mature_before
from m3_ml.fusion_v2.model import fit_calibrator,calibrate
from m3_ml.distribution.quantiles import crps
from m3_ml.distribution.tails import ALL_COLUMNS,ALL_TAUS,MAIN
from .data import OUT,ROOT,KEY,write_json


def source_frames(points,spec,final=False):
    filename='test_quantiles.csv' if final else 'validation_selected_quantiles.csv'
    original=pd.read_csv(ROOT/'m3_ml/distribution/outputs/frozen_october_2026'/filename,parse_dates=['origin_date'])
    keys=['mode',*KEY] if final else KEY
    result={}
    for method in spec['risk_methods']:
        if method=='futures_vol':
            f=points.copy()
            scale=np.maximum(.005,f.futures_vol.to_numpy()*np.sqrt(f.horizon.to_numpy()+2))
            q=scale[:,None]*norm.ppf(ALL_TAUS)[None,:]
            f=pd.concat([f,pd.DataFrame(q,columns=ALL_COLUMNS,index=f.index)],axis=1)
        else:
            data=original[original.method.eq(method)]
            f=points.merge(data[keys+ALL_COLUMNS],on=keys,validate='one_to_one',how='left')
        if f[ALL_COLUMNS].isna().any().any():raise ValueError('Missing risk source '+method)
        pd.testing.assert_frame_equal(f[keys].reset_index(drop=True),points[keys].reset_index(drop=True))
        result[method]=f
    return result


def calibration(frame,window):
    train=frame.sort_values('origin_date')
    if window:train=train.groupby('horizon',group_keys=False).tail(window)
    return fit_calibrator(train)


def definitions(spec):
    return [dict(id=f'r{i:03d}',method=method,window=window,mixture=mix,width=width)
            for i,(method,window,mix,width) in enumerate((method,window,mix,width)
            for method in spec['risk_methods'] for window in spec['risk_windows']
            for mix in spec['risk_mixtures'] for width in spec['risk_widths'])]


def choose(losses,cutoff):
    past=mature_before(losses,cutoff)
    scores=past.groupby(['horizon','candidate']).agg(loss=('loss','mean'),baseline=('hs_crps','mean'),n=('loss','size')).reset_index()
    scores=scores[scores.n.ge(20)].copy();scores['relative']=scores.loss/scores.baseline
    choices={str(int(h)):g.sort_values(['loss','candidate']).iloc[0].to_dict() for h,g in scores.groupby('horizon')}
    return choices,scores


def render_history(points,sources,spec,choices_by_cutoff,defs):
    parts=[];audit=[];by_id={d['id']:d for d in defs}
    for cutoff,stop in blocks(spec['risk_start'],spec['fit_end']):
        key=str(cutoff.date())
        if key not in choices_by_cutoff:continue
        choices=choices_by_cutoff[key]
        for h,choice in choices.items():
            d=by_id[choice['candidate']]
            frame=sources[d['method']]
            frame=frame[frame.horizon.eq(int(h))]
            test=frame[frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)].copy()
            if test.empty:continue
            train=mature_before(frame,cutoff)
            cal=calibration(train,d['window'])
            q=calibrate(test,cal,d['mixture'],d['width'])
            test.loc[:,ALL_COLUMNS]=q;test['risk_candidate']=d['id'];test['risk_fit_cutoff']=cutoff
            parts.append(test)
            audit.append(dict(cutoff=key,horizon=int(h),candidate=d['id'],calibration=cal[h]))
    return pd.concat(parts,ignore_index=True),audit


def select():
    spec=json.loads((OUT/'specification.json').read_text())
    points=pd.read_csv(OUT/'point_crossfit.csv',parse_dates=['origin_date','target_date','label_available_date'])
    sources=source_frames(points,spec)
    baseline=crps(sources['hs'][ALL_COLUMNS].to_numpy()[:,MAIN],points.y)
    points['hs_crps']=baseline
    for method in sources:sources[method]['hs_crps']=baseline
    defs=definitions(spec);losses=[];fit_audit=[]
    for method,frame in sources.items():
        for cutoff,stop in blocks(spec['risk_start'],spec['fit_end']):
            train=mature_before(frame,cutoff);test=frame[frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)].copy()
            if test.empty:continue
            for window in spec['risk_windows']:
                cal=calibration(train,window)
                fit_audit.append(dict(method=method,window=window,cutoff=str(cutoff.date()),
                                      latest_training_release=max(c['latest_label_release'] for c in cal.values())))
                for d in [d for d in defs if d['method']==method and d['window']==window]:
                    q=calibrate(test,cal,d['mixture'],d['width'])
                    row=test[KEY+['label_available_date','hs_crps']].copy()
                    row['candidate']=d['id'];row['loss']=crps(q[:,MAIN],test.y)
                    losses.append(row)
        print('V3 risk crossfit complete',method,flush=True)
    losses=pd.concat(losses,ignore_index=True)
    losses.to_parquet(OUT/'risk_candidate_losses.parquet',index=False)
    choices,scores=choose(losses,spec['fit_end'])
    scores.to_csv(OUT/'risk_selection.csv',index=False)
    global_choices={str(c.date()):choices for c,_ in blocks(spec['risk_start'],spec['fit_end'])}
    selected,selected_audit=render_history(points,sources,spec,global_choices,defs)
    selected.to_csv(OUT/'risk_development_forecasts.csv',index=False)
    outer_choices={};selection_audit=[]
    for cutoff,stop in blocks(spec['risk_outer_start'],spec['fit_end']):
        picked,_=choose(losses,cutoff);outer_choices[str(cutoff.date())]=picked
        selection_audit.append(dict(cutoff=str(cutoff.date()),choices=picked,
                                    latest_selection_release=str(mature_before(losses,cutoff).label_available_date.max().date())))
    outer,outer_audit=render_history(points,sources,spec,outer_choices,defs)
    outer.to_csv(OUT/'risk_forward_forecasts.csv',index=False)
    final={};by_id={d['id']:d for d in defs}
    for h,choice in choices.items():
        d=by_id[choice['candidate']]
        train=mature_before(sources[d['method']][sources[d['method']].horizon.eq(int(h))],spec['fit_end'])
        final[h]=dict(**d,calibration=calibration(train,d['window'])[h],validation_score=choice['relative'])
    write_json(OUT/'risk_final.json',final)
    write_json(OUT/'risk_crossfit_audit.json',dict(fits=fit_audit,selections=selection_audit,forward=outer_audit))
    write_json(OUT/'risk_candidates.json',defs)
    print('V3 risk fitted; historical forward rows',len(outer),flush=True)


if __name__=='__main__':select()
