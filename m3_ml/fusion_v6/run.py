import argparse
import joblib
import numpy as np
import pandas as pd
from m3_ml.fusion.model import MEDIAN,metrics,risk_readout
from m3_ml.distribution.tails import ALL_COLUMNS
from . import research as r


def hashes():
    files=[*sorted((r.ROOT/'m3_ml/fusion_v6').glob('*.py')),*[r.OUT/n for n in (
        'specification.json','point_final.json','point_gate.json','point_audit.json','point_inference.json','point_crossfit.csv',
        'risk_final.json','risk_gate.json','risk_audit.json','risk_proposal.csv','point_metrics.csv','risk_metrics.csv',
        'promotion_gate.json','width_amendment.json','models/correction.joblib')],r.OLD/'selected.json',r.OLD/'test_forecasts.csv',r.OLD/'risk_forward_forecasts.csv']
    return {**r.previous_run.input_hashes(),**{p.relative_to(r.ROOT).as_posix():r.sha(p) for p in files}}


def seal():
    if (r.OUT/'test_audit.json').exists():raise ValueError('Cannot replace an evaluated model')
    r.protect();gate=r.read(r.OUT/'promotion_gate.json');assert gate['passed']
    spec=r.read(r.OUT/'specification.json');old=r.read(r.OLD/'selected.json')
    spec['risk']['price_width_cap']=r.read(r.OUT/'width_amendment.json')['price_width_cap']
    model=dict(name=spec['name'],version=spec['version'],protocol_id=old['protocol_id'],fit_end=r.END,
        forecast_mode='daily_update',horizons=list(r.HORIZONS),point=r.read(r.OUT/'point_final.json'),risk=r.read(r.OUT/'risk_final.json'),
        risk_rule=spec['risk'],promotion=gate,limitations=spec['limitation'],october=spec['october'],input_sha256=hashes())
    r.write_json(r.OUT/'selected.json',model);print('SEALED BEFORE OCTOBER:',r.sha(r.OUT/'selected.json'),flush=True)


def predict(frame,model,models):
    f=frame.copy();f['v5_forecast']=f.forecast
    for h in (1,5):
        ix=f.index[f.horizon.eq(h)];f.loc[ix,'forecast']=r.predict(frame.loc[ix],models[h])
    delta=np.log(f.forecast/f.origin_price)-f.center
    f.loc[:,ALL_COLUMNS]=f[ALL_COLUMNS].to_numpy()+delta.to_numpy()[:,None]
    f['center']=np.log(f.forecast/f.origin_price)
    for h in (1,5):
        d=model['risk'][str(h)]
        if d['enabled']:
            ix=f.index[f.horizon.eq(h)]
            f.loc[ix,ALL_COLUMNS]=r.tail_predict(f.loc[ix],d['calibration'],model['risk_rule']['price_width_cap'])
    f['lower90']=f.origin_price*np.exp(f['q_0.050']);f['upper90']=f.origin_price*np.exp(f['q_0.950'])
    for key,value in risk_readout(f[ALL_COLUMNS].to_numpy()).items():f[key]=value
    return f


def test():
    r.protect();model=r.read(r.OUT/'selected.json');assert model['input_sha256']==hashes()
    digest=r.sha(r.OUT/'selected.json')
    baseline=pd.read_csv(r.OLD/'test_forecasts.csv',parse_dates=['origin_date','target_date'])
    result=predict(baseline,model,joblib.load(r.OUT/'models/correction.joblib'))
    result['version']=model['version'];result['method']=model['name']
    assert len(result)==68 and result['mode'].eq('daily_update').all()
    q=result[ALL_COLUMNS].to_numpy();assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-12).all()
    np.testing.assert_allclose(q[:,MEDIAN],result.center,atol=1e-14)
    long=result.horizon.isin([20,60]);np.testing.assert_allclose(result.loc[long,['forecast',*ALL_COLUMNS]],baseline.loc[long,['forecast',*ALL_COLUMNS]],atol=1e-12)
    result.to_csv(r.OUT/'test_forecasts.csv',index=False);score=metrics(result,'integrated_v6')
    for i,row in score.iterrows():
        g=result[result.horizon.eq(row.horizon)&result.y.notna()]
        if len(g):
            for name,col in [('v5','v5_forecast'),('futures','futures_price')]:
                value=float(np.sqrt(np.mean((g[col]-g.actual)**2)));score.loc[i,name+'_rmse_usd']=value;score.loc[i,'rmse_vs_'+name]=row.rmse_usd/value
    score.to_csv(r.OUT/'test_metrics.csv',index=False)
    assert digest==r.sha(r.OUT/'selected.json')
    r.write_json(r.OUT/'test_audit.json',dict(status='passed',selected_sha256=digest,rows=68,october_tuning=False,long_horizons_unchanged=True))
    print(score.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['seal','test']);a=p.parse_args();dict(seal=seal,test=test)[a.stage]()
