"""Rolling risk upgrades with observable implied-volatility scale proxies."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from m3_ml.fusion.model import blocks,mature_before,metrics
from m3_ml.fusion_v3.risk import source_frames as old_sources,calibration,choose
from m3_ml.fusion_v2.model import calibrate
from m3_ml.distribution.tails import ALL_COLUMNS,ALL_TAUS,MAIN
from m3_ml.distribution.quantiles import crps
from .data import OUT,OLD,KEY,write_json


def sources(points,spec,final=False):
    known=[m for m in spec['risk_sources'] if m not in ('implied_vol','hybrid_vol')]
    result=old_sources(points,dict(risk_methods=known),final)
    for method in ('implied_vol','hybrid_vol'):
        f=points.copy();daily=f.implied_vol.to_numpy()/np.sqrt(252.)
        if method=='hybrid_vol':daily=np.maximum(daily,f.futures_vol.to_numpy())
        scale=np.maximum(.005,daily*np.sqrt(f.horizon.to_numpy()+2))
        q=scale[:,None]*norm.ppf(ALL_TAUS)[None,:]
        result[method]=pd.concat([f,pd.DataFrame(q,index=f.index,columns=ALL_COLUMNS)],axis=1)
    return result


def definitions(spec):
    return [dict(id=f'r{i:03d}',method=m,window=w,mixture=a,width=b,calibration_mode=c)
        for i,(m,w,a,b,c) in enumerate((m,w,a,b,c) for m in spec['risk_sources'] for w in spec['risk_windows']
            for a in spec['risk_mixtures'] for b in spec['risk_widths'] for c in ('centered','asymmetric'))]


def tail_calibration(frame,window,mode):
    cal=calibration(frame,window)
    if mode=='centered':return cal
    lo=int(np.flatnonzero(ALL_TAUS==.05)[0]);hi=int(np.flatnonzero(ALL_TAUS==.95)[0])
    for entry in cal.values():
        q=np.asarray(entry['quantiles']);raw=q+entry['residual_median_diagnostic']
        left=max(-raw[lo],.05)/max(-q[lo],1e-8)
        right=max(raw[hi],.05)/max(q[hi],1e-8)
        # Stretch each side independently to retain observed tail error levels,
        # while leaving the point forecast at the median. Never shift its center.
        q=np.where(ALL_TAUS<.5,q*left,q*right)
        entry['quantiles']=q.tolist();entry['tail_factors']=[float(left),float(right)]
    return cal


def render_history(points,frames,spec,choices_by_cutoff,defs):
    parts=[];audit=[];by_id={d['id']:d for d in defs}
    for cutoff,stop in blocks(spec['risk_start'],spec['fit_end']):
        key=str(cutoff.date())
        if key not in choices_by_cutoff:continue
        for h,choice in choices_by_cutoff[key].items():
            d=by_id[choice['candidate']];frame=frames[d['method']]
            frame=frame[frame.horizon.eq(int(h))]
            test=frame[frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)].copy()
            if test.empty:continue
            cal=tail_calibration(mature_before(frame,cutoff),d['window'],d['calibration_mode'])
            test.loc[:,ALL_COLUMNS]=calibrate(test,cal,d['mixture'],d['width'])
            test['risk_candidate']=d['id'];test['risk_fit_cutoff']=cutoff;parts.append(test)
            audit.append(dict(cutoff=key,horizon=int(h),candidate=d['id'],calibration=cal[h]))
    return pd.concat(parts,ignore_index=True),audit


def main():
    spec=json.loads((OUT/'specification.json').read_text())
    if not (OUT/'risk_tail_design.json').exists():
        archive=OUT/'risk_symmetric_proposal';archive.mkdir(exist_ok=True)
        for filename in ('historical_risk_metrics.csv','risk_final.json','risk_forward_forecasts.csv'):
            if (OUT/filename).exists():(archive/filename).write_bytes((OUT/filename).read_bytes())
        write_json(OUT/'risk_tail_design.json',dict(reason='Historical symmetric residual calibration improves CRPS but still undercovers; investigate directional bias without moving point forecasts.',
            calibration_modes=['centered','asymmetric'],tail_anchor_quantiles=[.05,.95],minimum_standardized_tail=.05,
            selection='Same past-only CRPS grid and forward-selection schedule; no October inputs.',development_amendment=True))
    points=pd.read_csv(OUT/'point_crossfit.csv',parse_dates=['origin_date','target_date','label_available_date'])
    frames=sources(points,spec);defs=definitions(spec);losses=[];fit_audit=[]
    baseline=crps(frames['hs'][ALL_COLUMNS].to_numpy()[:,MAIN],points.y)
    for f in frames.values():f['hs_crps']=baseline
    for method,frame in frames.items():
        for cutoff,stop in blocks(spec['risk_start'],spec['fit_end']):
            train=mature_before(frame,cutoff);test=frame[frame.origin_date.ge(cutoff)&frame.origin_date.lt(stop)].copy()
            if test.empty:continue
            for window in spec['risk_windows']:
                for mode in ('centered','asymmetric'):
                    cal=tail_calibration(train,window,mode)
                    fit_audit.append(dict(method=method,window=window,calibration_mode=mode,cutoff=str(cutoff.date()),
                        latest_label_release=max(c['latest_label_release'] for c in cal.values())))
                    for d in [d for d in defs if d['method']==method and d['window']==window and d['calibration_mode']==mode]:
                        q=calibrate(test,cal,d['mixture'],d['width'])
                        f=test[KEY+['label_available_date','hs_crps']].copy()
                        f['candidate']=d['id'];f['loss']=crps(q[:,MAIN],test.y);losses.append(f)
        print('V4 risk source crossfit:',method,flush=True)
    losses=pd.concat(losses,ignore_index=True);losses.to_parquet(OUT/'risk_candidate_losses.parquet',index=False)
    final_choices,score=choose(losses,spec['fit_end']);score.to_csv(OUT/'risk_selection.csv',index=False)
    write_json(OUT/'risk_candidates.json',defs)
    forward_choices={};selection_audit=[]
    for cutoff,stop in blocks(spec['risk_outer_start'],spec['fit_end']):
        picked,_=choose(losses,cutoff);forward_choices[str(cutoff.date())]=picked
        selection_audit.append(dict(cutoff=str(cutoff.date()),choices=picked,
            latest_label_release=str(mature_before(losses,cutoff).label_available_date.max().date())))
    forward,cal_audit=render_history(points,frames,spec,forward_choices,defs)
    forward.to_csv(OUT/'risk_forward_forecasts.csv',index=False)
    development,_=render_history(points,frames,spec,{str(c.date()):final_choices for c,_ in blocks(spec['risk_start'],spec['fit_end'])},defs)
    development.to_csv(OUT/'risk_development_forecasts.csv',index=False)
    final={};by_id={d['id']:d for d in defs}
    for h,choice in final_choices.items():
        d=by_id[choice['candidate']]
        train=mature_before(frames[d['method']][frames[d['method']].horizon.eq(int(h))],spec['fit_end'])
        final[h]=dict(**d,calibration=tail_calibration(train,d['window'],d['calibration_mode'])[h],validation_score=choice['relative'])
    write_json(OUT/'risk_final.json',final)
    write_json(OUT/'risk_audit.json',dict(fits=fit_audit,selections=selection_audit,calibration=cal_audit,
        ovx_source='https://www.cboe.com/us/indices/dashboard/ovx/',
        ovx_interpretation='USO-option 30-day implied volatility proxy, not a direct WTI physical-distribution forecast; sqrt-time scaling and calibration are modeling assumptions.'))
    rows=[]
    for period,frame in [('development',development),('forward',forward)]:
        prior=pd.read_csv(OLD/('risk_forward_forecasts.csv' if period=='forward' else 'risk_development_forecasts.csv'),parse_dates=['origin_date'])
        prior['v3_loss']=crps(prior[ALL_COLUMNS].to_numpy()[:,MAIN],prior.y)
        f=frame.merge(prior[KEY+['v3_loss']],on=KEY,validate='one_to_one')
        f['loss']=crps(f[ALL_COLUMNS].to_numpy()[:,MAIN],f.y)
        for h,g in f.groupby('horizon'):
            rows.append(dict(period=period,horizon=int(h),n=len(g),crps_vs_hs=float(g.loss.mean()/g.hs_crps.mean()),
                crps_vs_v3=float(g.loss.mean()/g.v3_loss.mean()),coverage90=float(g.y.between(g['q_0.050'],g['q_0.950']).mean()),
                mean_width90_usd=float((g.origin_price*(np.exp(g['q_0.950'])-np.exp(g['q_0.050']))).mean())))
        metrics(frame,'integrated_v4').to_csv(OUT/f'risk_{period}_metrics.csv',index=False)
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'historical_risk_metrics.csv',index=False)
    print(scores[scores.period.eq('forward')].to_string(index=False),flush=True)


if __name__=='__main__':main()
