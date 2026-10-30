"""Seal the pre-October system, then evaluate its frozen October forecasts."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table
from m3_ml.common.protocol import HORIZONS,FIT_END,PROTOCOL_ID
from m3_ml.fusion.run import october_frame
from m3_ml.fusion.model import MEDIAN,metrics,risk_readout
from m3_ml.fusion_v2.model import calibrate
from m3_ml.fusion_v2.run import seal_inputs as v2_inputs
from m3_ml.distribution.tails import ALL_COLUMNS
from .data import OUT,BASE,ROOT,RAW,KEY,sha,write_json,feature_frame
from .point import BLEND
from .risk import source_frames


def seal_inputs():
    paths=[RAW,OUT/'specification.json',OUT/'market_source_audit.json',OUT/'validation_member_panel.csv',
           OUT/'point_crossfit.csv',OUT/'locked_point_rules.json',OUT/'point_weights.json',
           OUT/'risk_final.json',OUT/'models/point_models.joblib',OUT/'historical_inference.json',
           OUT/'point_final_fit_audit.json',OUT/'point_crossfit_audit.json',OUT/'risk_crossfit_audit.json']
    for folder in ('fusion_v3','tabular'):
        paths.extend(sorted((ROOT/'m3_ml'/folder).glob('*.py')))
    # Native module outputs are fixed inputs to the October inference stage;
    # hashes attest to their identity, not their use during model selection.
    paths.extend([ROOT/'m1_time_series/outputs/frozen_october_2026/forecasts_2026-10-01_2026-10-23_asof_2026-10-23.csv',
                  ROOT/'m2_seasonality/outputs/frozen_october_2026/final_predictions.csv',
                  ROOT/'m3_ml/lstm/outputs/frozen_october_2026/test_predictions.csv',
                  ROOT/'m3_ml/distribution/outputs/frozen_october_2026/test_quantiles.csv'])
    paths.extend(sorted((ROOT/'m3_ml/tabular/outputs/frozen_october_2026').glob('*_forecasts.csv')))
    return {**v2_inputs(),**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}


def predict_points(frame,features,spec,weights,models):
    """Prediction only: no target values or fitting operations are consumed."""
    price=np.empty(len(frame));definitions={d['id']:d for d in spec['candidates']}
    for h,indices in frame.groupby('horizon').groups.items():
        g=frame.loc[indices];value=np.zeros(len(g));total=0.
        for cid,weight in weights[str(int(h))].items():
            if weight<1e-12:continue
            d=definitions[cid];model=models[(h,cid)]
            if d['kind']=='carry':p=g[d['base']+'_price'].to_numpy()
            elif d['kind']=='convex':p=g[[m+'_price' for m in BLEND]].to_numpy()@model['weights']
            else:
                p=np.maximum(.01,g[d['base']+'_price'].to_numpy()+model.predict(features[d['feature']].loc[indices]))
            value+=weight*p;total+=weight
        price[indices]=value/total
    if not np.isfinite(price).all() or np.any(price<=0):raise ValueError('Invalid frozen point predictions')
    return price


def seal():
    if (OUT/'test_audit.json').exists():raise ValueError('October already evaluated; do not silently replace its sealed selection')
    spec=json.loads((OUT/'specification.json').read_text())
    older=json.loads((BASE/'selected.json').read_text())
    if older['input_sha256']!=v2_inputs():raise ValueError('Protected v2 inputs changed')
    point_audit=json.loads((OUT/'point_crossfit_audit.json').read_text())
    final_audit=json.loads((OUT/'point_final_fit_audit.json').read_text())
    risk_audit=json.loads((OUT/'risk_crossfit_audit.json').read_text())
    for row in point_audit+risk_audit['selections']:
        assert row['latest_selection_release']<=row['cutoff']<=spec['fit_end']
    for row in final_audit:assert row['latest_label_release']<=spec['fit_end']
    for row in risk_audit['fits']:assert row['latest_training_release']<=row['cutoff']<=spec['fit_end']
    model=dict(name=spec['name'],version=spec['version'],protocol_id=PROTOCOL_ID,fit_end=FIT_END,
        point_weights=json.loads((OUT/'point_weights.json').read_text()),
        point_rules=json.loads((OUT/'locked_point_rules.json').read_text()),
        risk=json.loads((OUT/'risk_final.json').read_text()),
        point_meaning=spec['point_meaning'],source_audit=json.loads((OUT/'market_source_audit.json').read_text()),
        base_selection_limitation=spec['base_selection_limitation'],october_status=spec['october_status'],
        rule_lock_date=spec['rule_lock_date'],primary_historical=json.loads((OUT/'historical_inference.json').read_text()),
        point_final_fit=final_audit,input_sha256=seal_inputs())
    write_json(OUT/'selected.json',model)
    print('SEALED before October evaluation:',sha(OUT/'selected.json'),flush=True)


def final_forecasts(spec,model):
    frame,features=feature_frame(october_frame(spec),load_table())
    fitted=joblib.load(OUT/'models/point_models.joblib')
    frame['forecast']=predict_points(frame,features,spec,model['point_weights'],fitted)
    frame['center']=np.log(frame.forecast/frame.origin_price)
    sources=source_frames(frame,spec,final=True);parts=[]
    for h,g in frame.groupby('horizon'):
        entry=model['risk'][str(int(h))]
        f=sources[entry['method']].loc[g.index].copy()
        q=calibrate(f,{str(int(h)):entry['calibration']},entry['mixture'],entry['width'])
        f.loc[:,ALL_COLUMNS]=q;f['risk_method']=entry['method'];f['risk_window']=entry['window']
        f['risk_mixture']=entry['mixture'];f['risk_width']=entry['width']
        f['lower90']=f.origin_price*np.exp(f['q_0.050']);f['upper90']=f.origin_price*np.exp(f['q_0.950'])
        for column,value in risk_readout(q).items():f[column]=value
        parts.append(f)
    result=pd.concat(parts,ignore_index=True).sort_values(['mode',*KEY]).reset_index(drop=True)
    result['method']=model['name'];result['protocol_id']=PROTOCOL_ID;result['version']=model['version'];result['fit_end']=FIT_END
    result['status']=np.where(result.y.notna(),'realized','unreached')
    return result


def test():
    model=json.loads((OUT/'selected.json').read_text());spec=json.loads((OUT/'specification.json').read_text())
    if model['input_sha256']!=seal_inputs():raise ValueError('Sealed inputs changed; cannot evaluate')
    digest=sha(OUT/'selected.json')
    if (OUT/'test_audit.json').exists():
        assert json.loads((OUT/'test_audit.json').read_text())['selected_sha256']==digest
    f=final_forecasts(spec,model)
    if len(f)!=87 or f.duplicated(['mode',*KEY]).any():raise ValueError('Incomplete October grid')
    q=f[ALL_COLUMNS].to_numpy()
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-10).all()
    np.testing.assert_array_equal(q[:,MEDIAN],f.center)
    for h in HORIZONS:
        a=f[f['mode'].eq('fixed_origin')&f.horizon.eq(h)]
        b=f[f['mode'].eq('daily_update')&f.origin_date.eq(FIT_END)&f.horizon.eq(h)]
        np.testing.assert_allclose(a[ALL_COLUMNS],b[ALL_COLUMNS],atol=1e-12)
    f.to_csv(OUT/'test_forecasts.csv',index=False)
    scores=metrics(f,'integrated_v3')
    for i,row in scores.iterrows():
        g=f[f['mode'].eq(row['mode'])&f.horizon.eq(row.horizon)&f.y.notna()]
        if len(g):
            scores.loc[i,'futures_rmse_usd']=np.sqrt(np.mean((g.futures_price-g.actual)**2))
            scores.loc[i,'rmse_vs_futures']=row.rmse_usd/scores.loc[i,'futures_rmse_usd']
    scores.to_csv(OUT/'test_metrics.csv',index=False)
    g=f[f['mode'].eq('fixed_origin')&f.y.notna()]
    rmse=float(np.sqrt(np.mean((g.forecast-g.actual)**2)))
    rw=float(np.sqrt(np.mean((g.origin_price-g.actual)**2)))
    future=float(np.sqrt(np.mean((g.futures_price-g.actual)**2)))
    path=dict(n=len(g),rmse_usd=rmse,rw_rmse_usd=rw,futures_rmse_usd=future,
              rmse_vs_rw=rmse/rw,rmse_vs_futures=rmse/future,
              coverage90=float(g.actual.between(g.lower90,g.upper90).mean()),
              interpretation='17 correlated targets from one origin; descriptive evaluation only')
    assert digest==sha(OUT/'selected.json')
    write_json(OUT/'test_audit.json',dict(status='passed',selected_sha256=digest,rows=len(f),
        median_preserved=True,quantiles_monotone=True,first_origins_equal=True,
        actual_counts=scores[['mode','horizon','n']].to_dict('records'),fixed_path_metrics=path,
        futures_timing='Every consumed quote is at least two US federal business days old at its forecast origin',
        october_used_for_fitting=False))
    print('OCTOBER EVALUATION ONLY',json.dumps(path),flush=True)
    print(scores[scores.horizon.isin(HORIZONS)].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['seal','test']);args=parser.parse_args()
    (seal if args.stage=='seal' else test)()
