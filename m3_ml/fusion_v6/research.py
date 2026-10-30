import inspect
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from m3_ml.common.execution import ValidationCache,run_jobs
from m3_ml.fusion.model import mature_before,MEDIAN
from m3_ml.distribution.tails import ALL_COLUMNS,ALL_TAUS,MAIN
from m3_ml.distribution.quantiles import crps
from m3_ml.fusion_v3.evaluate import paired_blocks
from m3_ml.fusion_v5 import data as previous,run as previous_run
from m3_ml.fusion_v5.risk_safety import envelope

ROOT=previous.ROOT;OLD=previous.OUT;OUT=ROOT/'m3_ml/fusion_v6/outputs/frozen_october_2026'
KEY=previous.KEY;HORIZONS=previous.HORIZONS;END='2026-09-30'
read=previous.read;write_json=previous.write_json;sha=previous.sha
STATE=previous.STATE+['state_season_cos','state_vol_ratio','state_basis_slope']
LO=int(np.flatnonzero(ALL_TAUS==.05)[0]);HI=int(np.flatnonzero(ALL_TAUS==.95)[0])


def protect():
    previous.protected();assert read(OLD/'selected.json')['input_sha256']==previous_run.input_hashes()
    if (OUT/'protected.json').exists():
        d=read(OUT/'protected.json')
        assert sha(OLD/'selected.json')==d['selected'] and sha(OLD/'test_forecasts.csv')==d['forecasts']


def months(start,end=END):
    dates=pd.date_range(start,end,freq='MS')
    for i,d in enumerate(dates):yield d,dates[i+1] if i+1<len(dates) else pd.Timestamp(end)+pd.Timedelta(days=1)


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'specification.json').exists():return
    protect();defs=[dict(id='p000',kind='none',window=0,target='dollar',fraction=0.)]
    def add(**d):defs.append(dict(id=f'p{len(defs):03d}',**d))
    for w in (20,63,126,252):
        for kind in ('mean','median'):
            for target in ('dollar','scaled'):
                for fraction in (.25,.5,1.):add(kind=kind,window=w,target=target,fraction=fraction)
    for w in (126,252):
        for alpha in (100.,1000.,10000.):
            for target in ('dollar','scaled'):
                for fraction in (.5,1.):add(kind='ridge',window=w,alpha=alpha,target=target,fraction=fraction)
    spec=dict(name='Integrated WTI Forecast',version='short_bias_v6',fit_end=END,forecast_mode='daily_update',horizons=list(HORIZONS),
        candidates=defs,candidate_start='2025-03-01',outer_start='2025-09-01',selector_window=126,min_selection=40,
        selector_min_mse_gain=.01,last63_max_mse_ratio=1.02,
        point_gate=dict(max_rmse_ratio=.995,max_mae_ratio=1.02,min_improved_quarters=2),
        risk=dict(start='2026-06-01',window=126,target=.95,price_width_cap=1.5,min_calibration=40,
            min_evaluation=60,min_coverage_gap_gain=.02,max_crps_ratio=1.05,max_mean_width_ratio=1.4),
        policy='Optimize only h1/h5; retain h20/h60. Monthly historical refits; nested selection uses released errors only. Per-horizon promotion gates are specified before results.',
        calibration_policy='One predeclared conservative asymmetric 95% tail-score target, no risk candidate search. Only v5 forward-issued risk forecasts from March 2026 feed calibration; evaluate from June 2026.',
        october='Previously observed October is retrospective evaluation only, never tuning or promotion selection. All models and calibration are frozen using September 30 available information.',
        limitation='Inherited base specifications and repeated historical research remain conditional; unchanged vintage and publication assumptions. No theoretical coverage guarantee for dependent observations.')
    write_json(OUT/'specification.json',spec);write_json(OUT/'protected.json',dict(selected=sha(OLD/'selected.json'),forecasts=sha(OLD/'test_forecasts.csv')))
    print('Registered',len(defs),'short-horizon candidates and one fixed tail-calibration rule.',flush=True)


def features(f):
    return np.column_stack([f[STATE],(f.forecast-f.futures_price)/f.scale,(f.forecast-f.origin_price)/f.scale])


def fit(f,d):
    f=f.sort_values('origin_date')
    if d['window']:f=f.tail(d['window'])
    m=dict(definition=d,n=len(f),last_release=str(f.label_available_date.max().date()))
    y=(f.actual-f.forecast).to_numpy()
    if d['target']=='scaled':y=y/f.scale.to_numpy()
    if d['kind']=='none':m['value']=0.
    elif d['kind']=='mean':m['value']=float(y.mean())
    elif d['kind']=='median':m['value']=float(np.median(y))
    else:
        m['scaler']=StandardScaler().fit(features(f));m['model']=Ridge(alpha=d['alpha']).fit(m['scaler'].transform(features(f)),y)
    m['correction_cap']=float(np.quantile(np.abs(y),.90))
    return m


def predict(f,m):
    d=m['definition']
    correction=m['model'].predict(m['scaler'].transform(features(f))) if d['kind']=='ridge' else np.full(len(f),m['value'])
    correction=np.clip(correction,-m['correction_cap'],m['correction_cap'])*d['fraction']
    if d['target']=='scaled':correction*=f.scale.to_numpy()
    return np.maximum(.01,f.forecast.to_numpy()+correction)


def job(context,task):
    panel,spec=context;h,d=task;f=panel[panel.horizon.eq(h)];rows=[]
    for cutoff,stop in months(spec['candidate_start']):
        train=mature_before(f,cutoff);test=f[f.origin_date.ge(cutoff)&f.origin_date.lt(stop)]
        if test.empty:continue
        m=fit(train,d)
        rows.append(np.column_stack([test.index,predict(test,m),np.full(len(test),cutoff.value),
            np.full(len(test),pd.Timestamp(m['last_release']).value)]))
    return np.concatenate(rows)


def choose(candidates,cutoff,h,spec):
    past=mature_before(candidates[candidates.horizon.eq(h)],cutoff)
    dates=past.origin_date.drop_duplicates().sort_values().tail(spec['selector_window']);past=past[past.origin_date.isin(dates)]
    scores=past.assign(loss=(past.prediction-past.actual)**2).groupby('candidate').loss.agg(['mean','size'])
    if scores.empty or scores['size'].min()<spec['min_selection']:return 'p000',dict(n=0,last_release=None)
    recent=dates.tail(63);short=past[past.origin_date.isin(recent)].assign(loss=lambda x:(x.prediction-x.actual)**2).groupby('candidate').loss.mean()
    eligible=scores[(scores['mean']<=(1-spec['selector_min_mse_gain'])*scores.loc['p000','mean'])&(short<=spec['last63_max_mse_ratio']*short.loc['p000'])]
    cid=eligible['mean'].sort_values(kind='stable').index[0] if len(eligible) else 'p000'
    return cid,dict(n=int(scores.loc[cid,'size']),last_release=str(past.label_available_date.max().date()),mse_ratio=float(scores.loc[cid,'mean']/scores.loc['p000','mean']))


def point_research():
    spec=read(OUT/'specification.json');protect()
    panel=pd.read_csv(OLD/'point_crossfit.csv',parse_dates=['origin_date','target_date','label_available_date'])
    assert panel.label_available_date.max()<=pd.Timestamp(END)
    tasks=[(h,d) for h in (1,5) for d in spec['candidates']]
    cache=ValidationCache('short_bias_v6',dict(panel=panel,spec=spec,code=inspect.getsource(inspect.getmodule(fit))))
    values=run_jobs(job,tasks,(panel,spec),'V6 short bias fits',cache);parts=[];audits=[]
    for (h,d),a in zip(tasks,values):
        f=panel.loc[a[:,0].astype(int)].copy();f['prediction']=a[:,1];f['candidate']=d['id'];parts.append(f)
        for cutoff,release in np.unique(a[:,2:4],axis=0):audits.append(dict(horizon=h,candidate=d['id'],cutoff=str(pd.Timestamp(int(cutoff)).date()),last_release=str(pd.Timestamp(int(release)).date())))
    candidates=pd.concat(parts,ignore_index=True);candidates.to_parquet(OUT/'candidates.parquet',index=False)
    selected=[];selection=[]
    for cutoff,stop in months(spec['outer_start']):
        for h in (1,5):
            cid,details=choose(candidates,cutoff,h,spec)
            g=candidates[candidates.horizon.eq(h)&candidates.candidate.eq(cid)&candidates.origin_date.ge(cutoff)&candidates.origin_date.lt(stop)].copy()
            selected.append(g);selection.append(dict(horizon=h,cutoff=str(cutoff.date()),candidate=cid,**details))
    proposal=pd.concat(selected,ignore_index=True);proposal.to_csv(OUT/'point_proposal.csv',index=False)
    gates={};rows=[];intervals=[];final={};models={};output=panel.copy();output['v5_forecast']=output.forecast
    defs={d['id']:d for d in spec['candidates']}
    for h,g in proposal.groupby('horizon'):
        new=(g.prediction-g.actual)**2;old=(g.forecast-g.actual)**2
        ratio=float(np.sqrt(new.mean()/old.mean()));mae=float(np.abs(g.prediction-g.actual).mean()/np.abs(g.forecast-g.actual).mean())
        quarter=g.assign(new=new,old=old).groupby(g.origin_date.dt.to_period('Q'))[['new','old']].mean()
        count=int((quarter.new<quarter.old).sum());limits=spec['point_gate']
        passed=ratio<=limits['max_rmse_ratio'] and mae<=limits['max_mae_ratio'] and count>=limits['min_improved_quarters']
        gates[str(h)]=dict(passed=bool(passed),rmse_ratio=ratio,mae_ratio=mae,improved_quarters=count)
        rows.append(dict(horizon=int(h),n=len(g),v6_rmse=float(np.sqrt(new.mean())),v5_rmse=float(np.sqrt(old.mean())),rmse_ratio=ratio,mae_ratio=mae,adopted=bool(passed)))
        for block in (20,60):intervals.append(dict(horizon=int(h),**paired_blocks(new,old,block)))
        cid,evidence=choose(candidates,END,h,spec)
        if not passed:cid='p000'
        final[str(h)]=dict(candidate=cid,selection=evidence);models[int(h)]=fit(mature_before(panel[panel.horizon.eq(h)],END),defs[cid])
        if passed:
            ix=pd.MultiIndex.from_frame(output[KEY]);values=g.set_index(KEY).prediction
            mask=output.horizon.eq(h)&output.origin_date.ge(spec['outer_start'])
            output.loc[mask,'forecast']=values.reindex(ix[mask]).to_numpy()
    output['center']=np.log(output.forecast/output.origin_price);output.to_csv(OUT/'point_crossfit.csv',index=False)
    (OUT/'models').mkdir(exist_ok=True);joblib.dump(models,OUT/'models/correction.joblib')
    write_json(OUT/'point_final.json',final);write_json(OUT/'point_gate.json',gates);write_json(OUT/'point_audit.json',dict(fits=audits,selections=selection))
    write_json(OUT/'point_inference.json',intervals);pd.DataFrame(rows).to_csv(OUT/'point_metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print('Final correction:',final,flush=True)


def tail_fit(f,spec):
    f=f.sort_values('origin_date').tail(spec['window'])
    if len(f)<spec['min_calibration']:raise ValueError('Insufficient released tail errors')
    lower=np.maximum((f.center-f.y)/np.maximum(1e-6,f.center-f['q_0.050']),0.)
    upper=np.maximum((f.y-f.center)/np.maximum(1e-6,f['q_0.950']-f.center),0.)
    rank=min(len(f),int(np.ceil((len(f)+1)*(1-(1-spec['target'])/2))))-1
    return dict(left=max(1.,float(np.sort(lower)[rank])),right=max(1.,float(np.sort(upper)[rank])),n=len(f),last_release=str(f.label_available_date.max().date()))


def tail_predict(f,cal,cap):
    old=f[ALL_COLUMNS].to_numpy();center=f.center.to_numpy()
    q=center[:,None]+(old-center[:,None])*np.where(ALL_TAUS<.5,cal['left'],cal['right'])[None,:]
    return envelope(q,old,center,cap)[0]


def risk_research():
    spec=read(OUT/'specification.json')['risk'];points=pd.read_csv(OUT/'point_crossfit.csv',parse_dates=['origin_date'])
    if (OUT/'width_amendment.json').exists():spec['price_width_cap']=read(OUT/'width_amendment.json')['price_width_cap']
    old=pd.read_csv(OLD/'risk_forward_forecasts.csv',parse_dates=['origin_date','label_available_date']).copy()
    p=points.set_index(KEY).reindex(pd.MultiIndex.from_frame(old[KEY]));base=old.copy()
    delta=p.center.to_numpy()-base.center.to_numpy()
    base.loc[:,ALL_COLUMNS]=base[ALL_COLUMNS].to_numpy()+delta[:,None]
    base['center']=p.center.to_numpy();base['forecast']=p.forecast.to_numpy()
    parts=[];audit=[]
    for cutoff,stop in months(spec['start']):
        for h,g in base[base.origin_date.ge(cutoff)&base.origin_date.lt(stop)].groupby('horizon'):
            g=g.copy()
            if h in (1,5):
                cal=tail_fit(mature_before(base[base.horizon.eq(h)],cutoff),spec)
                g.loc[:,ALL_COLUMNS]=tail_predict(g,cal,spec['price_width_cap'])
                audit.append(dict(horizon=int(h),cutoff=str(cutoff.date()),**cal))
            parts.append(g)
    proposal=pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    ref=old.set_index(KEY).reindex(pd.MultiIndex.from_frame(proposal[KEY])).reset_index()
    loss=crps(proposal[ALL_COLUMNS].to_numpy()[:,MAIN],proposal.y);ref_loss=crps(ref[ALL_COLUMNS].to_numpy()[:,MAIN],ref.y)
    rows=[];gates={};final={}
    for h,indices in proposal.groupby('horizon').groups.items():
        g=proposal.loc[indices];b=ref.loc[indices]
        coverage=float(g.y.between(g['q_0.050'],g['q_0.950']).mean());old_cov=float(b.y.between(b['q_0.050'],b['q_0.950']).mean())
        width=float((g.origin_price*(np.exp(g['q_0.950'])-np.exp(g['q_0.050']))).mean());old_width=float((b.origin_price*(np.exp(b['q_0.950'])-np.exp(b['q_0.050']))).mean())
        ratio=float(loss[indices].mean()/ref_loss[indices].mean());gain=abs(old_cov-.9)-abs(coverage-.9)
        passed=h in (1,5) and len(g)>=spec['min_evaluation'] and gain>=spec['min_coverage_gap_gain'] and ratio<=spec['max_crps_ratio'] and width/old_width<=spec['max_mean_width_ratio']
        if h in (1,5):
            gates[str(h)]=dict(passed=bool(passed),coverage_gap_gain=gain,crps_ratio=ratio,width_ratio=width/old_width)
            final[str(h)]=dict(enabled=bool(passed),calibration=tail_fit(mature_before(base[base.horizon.eq(h)],END),spec))
        rows.append(dict(horizon=int(h),n=len(g),coverage90=coverage,v5_coverage90=old_cov,crps_ratio=ratio,width_ratio=width/old_width,width90_usd=width,adopted=bool(passed)))
    proposal.to_csv(OUT/'risk_proposal.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'risk_metrics.csv',index=False)
    write_json(OUT/'risk_final.json',final);write_json(OUT/'risk_gate.json',gates);write_json(OUT/'risk_audit.json',audit)
    point=read(OUT/'point_gate.json');promote=any(d['passed'] for d in point.values()) or any(d['passed'] for d in gates.values())
    write_json(OUT/'promotion_gate.json',dict(passed=promote,point=point,risk=gates,october_used=False))
    print(pd.DataFrame(rows).to_string(index=False),flush=True);print('PROMOTE:',promote,flush=True)


if __name__=='__main__':
    register();point_research();risk_research()
