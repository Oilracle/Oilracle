"""Fit, seal and evaluate one state-dependent daily rolling forecast system."""
import argparse
import joblib
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table
from m3_ml.fusion.model import mature_before,metrics,risk_readout,MEDIAN
from m3_ml.fusion.run import october_frame
from m3_ml.fusion_v2.model import calibrate as incumbent_calibrate
from m3_ml.fusion_v3 import data as v3_data,point as v3_point
from m3_ml.fusion_v4 import data as v4_data,run as v4_run
from m3_ml.fusion_v4.long_models import fit_predict
from m3_ml.fusion_v4.risk import sources
from m3_ml.distribution.tails import ALL_COLUMNS
from .data import OUT,OLD,V3,ROOT,KEY,HORIZONS,FIT_END,EXPERTS,read,write_json,sha,protected,historical_inputs,add_state
from . import point,risk,risk_safety


def fit():
    if (OUT/'test_audit.json').exists():raise ValueError('October already evaluated; frozen v5 cannot be replaced')
    protected();gate=read(OUT/'promotion_gate.json')
    if not gate['passed']:raise ValueError('Historical promotion gate failed; keep v4 canonical')
    panel=historical_inputs();spec=read(OUT/'specification.json')
    definitions={d['id']:d for d in read(OUT/'supported_definitions.json')}
    selected=read(OUT/'adopted_point_final.json');models={};audit=[]
    for h in HORIZONS:
        choice=selected[str(h)];definition=dict(definitions[choice['candidate']])
        definition['update_fraction']=choice.get('update_fraction',1.)
        train=mature_before(panel[panel.horizon.eq(h)],FIT_END)
        models[h]=point.fit(train,definition)
        audit.append(dict(layer='meta',horizon=h,n_train=models[h]['n_train'],candidate=definition['id'],
            latest_label_release=models[h]['latest_label_release']))
    # Fit every selected expert family, including families receiving zero weight
    # in v4: the conditional gate may use them at a later input state.
    _,long_panel,long_X=v4_data.historical_inputs();_,short_panel,short_X=v3_data.historical_inputs()
    long_defs={d['id']:d for d in read(OLD/'specification.json')['new_candidates']}
    short_defs={d['id']:d for d in read(V3/'specification.json')['candidates']}
    choices=read(OLD/'final_families.json');families={}
    for h in HORIZONS:
        tr=mature_before(long_panel[long_panel.horizon.eq(h)],FIT_END)
        sr=mature_before(short_panel[short_panel.horizon.eq(h)],FIT_END)
        for family,cid in choices[str(h)].items():
            if family=='v3':_,fitted=v3_point.fit_predict(sr,sr.iloc[:1],short_X,short_defs[cid])
            else:_,fitted=fit_predict(tr,tr.iloc[:1],long_X,long_defs[cid])
            families[(h,family)]=fitted
        audit.append(dict(layer='expert',horizon=h,n_long_eligible=len(tr),n_member_eligible=len(sr),
            latest_label_release=str(tr.label_available_date.max().date())))
    (OUT/'models').mkdir(exist_ok=True);joblib.dump(dict(meta=models,families=families),OUT/'models/upgrade.joblib')
    write_json(OUT/'final_fit_audit.json',audit)
    print('Fitted final state/reliability functions and expert families using released pre-October labels.',flush=True)


def input_hashes():
    paths=[*sorted((ROOT/'m3_ml/fusion_v5').glob('*.py')),
        *[OUT/name for name in ('specification.json','support_design.json','damping_design.json','risk_safety_design.json',
            'supported_definitions.json','adopted_point_final.json','risk_final.json','point_promotion_gate.json','promotion_gate.json',
            'point_crossfit.csv','historical_point_proposal_metrics.csv','historical_risk_metrics.csv','historical_inference.json',
            'point_selection_audit.json','risk_audit.json','risk_forward_forecasts.csv',
            'final_fit_audit.json','models/upgrade.joblib')],OLD/'selected.json',OLD/'final_families.json',OLD/'risk_forward_forecasts.csv']
    return {**v4_run.inputs(),**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}


def seal():
    if (OUT/'test_audit.json').exists():raise ValueError('Cannot reseal after October evaluation')
    protected();spec=read(OUT/'specification.json');gate=read(OUT/'promotion_gate.json')
    assert gate['passed']
    model=dict(name=spec['name'],version=spec['version'],protocol_id=read(OLD/'selected.json')['protocol_id'],
        fit_end=FIT_END,forecast_mode='daily_update',horizons=list(HORIZONS),experts=EXPERTS,
        point_policy=read(OUT/'adopted_point_final.json'),risk=read(OUT/'risk_final.json'),
        risk_safety=read(OUT/'risk_safety_design.json'),promotion_gate=gate,incumbent_sha256=sha(OLD/'selected.json'),
        limitations=spec['limitations'],october_status=spec['october'],input_sha256=input_hashes())
    write_json(OUT/'selected.json',model)
    print('SEALED V5 BEFORE OCTOBER EVALUATION:',sha(OUT/'selected.json'),flush=True)


def experts(native,table,fitted):
    frame,short_X=v3_data.feature_frame(native,table)
    market,long_X=v4_data.features(table)
    positions=pd.Series(table.index,index=table.date).reindex(frame.origin_date).to_numpy().astype(int)
    frame['feature_row']=positions;frame['implied_vol']=table.oil_implied_volatility.iloc[positions].to_numpy()/100
    frame['scale']=np.maximum(1.,frame.futures_price*frame.futures_vol*np.sqrt(frame.horizon+2))
    for c in ('basis20','basis60','basis250'):frame[c]=market[c].iloc[positions].to_numpy()
    old_spec=read(OLD/'specification.json');old_model=read(OLD/'selected.json')
    v3_spec=read(V3/'specification.json');v3_model=read(V3/'selected.json')
    incumbent=v4_run.predict(frame,long_X,short_X,old_spec,old_model,v3_spec,v3_model,
        joblib.load(OLD/'models/upgrade.joblib'),joblib.load(V3/'models/point_models.joblib'))[0]
    frame['v4_forecast']=incumbent;frame['incumbent_expert']=incumbent
    for name in ('futures','m1','m2','lstm','ml'):frame[name+'_expert']=frame[name+'_price']
    long_defs={d['id']:d for d in old_spec['new_candidates']};short_defs={d['id']:d for d in v3_spec['candidates']}
    for h,indices in frame.groupby('horizon').groups.items():
        g=frame.loc[indices]
        for family,cid in old_model['family_choices'][str(int(h))].items():
            estimator=fitted['families'][(h,family)]
            if family=='v3':
                d=short_defs[cid]
                if d['kind']=='carry':price=g[d['base']+'_price'].to_numpy()
                elif d['kind']=='convex':price=g[[m+'_price' for m in v3_point.BLEND]].to_numpy()@estimator['weights']
                else:price=np.maximum(.01,g[d['base']+'_price'].to_numpy()+estimator.predict(short_X[d['feature']].loc[indices]))
            else:
                d=long_defs[cid]
                if d['kind']=='basis':price=g.futures_price.to_numpy()+g[d['basis']].to_numpy()
                elif d['kind']=='mean':price=g.futures_price.to_numpy()+estimator
                else:
                    correction=estimator.predict(long_X[d['feature']].iloc[g.feature_row])
                    if d['target']=='scaled':correction*=g.scale.to_numpy()
                    price=np.maximum(.01,g.futures_price.to_numpy()+correction)
            frame.loc[indices,family+'_expert']=price
    return add_state(frame,table,market)


def predict_points(frame,fitted):
    result=np.zeros(len(frame));weights=np.zeros((len(frame),len(EXPERTS)))
    for h,indices in frame.groupby('horizon').groups.items():
        p,w=point.predict(frame.loc[indices],fitted['meta'][int(h)],True)
        result[indices]=p;weights[indices]=w
    return result,weights


def predict_risk(frame,model,final):
    spec=read(OUT/'specification.json');frames=sources(frame,spec,final);old=read(OLD/'selected.json');parts=[]
    for h,indices in frame.groupby('horizon').groups.items():
        d=model['risk'][str(int(h))];g=frame.loc[indices].copy()
        new=risk.predict(frames[d['method']].loc[indices],d['calibration'])
        legacy=old['risk'][str(int(h))]
        base=incumbent_calibrate(frames[legacy['method']].loc[indices],{str(int(h)):legacy['calibration']},legacy['mixture'],legacy['width'])
        q,fraction=risk_safety.envelope(new,base,g.center,model['risk_safety']['price_width_cap'])
        g=pd.concat([g,pd.DataFrame(q,index=g.index,columns=ALL_COLUMNS)],axis=1)
        g['risk_method']=d['method'];g['risk_candidate']=d['id'];g['risk_safety_fraction']=fraction
        g['lower90']=g.origin_price*np.exp(g['q_0.050']);g['upper90']=g.origin_price*np.exp(g['q_0.950'])
        for name,value in risk_readout(q).items():g[name]=value
        parts.append(g)
    return pd.concat(parts).sort_index()


def test():
    protected();model=read(OUT/'selected.json');digest=sha(OUT/'selected.json')
    assert model['input_sha256']==input_hashes()
    native=october_frame(read(V3/'specification.json'));native=native[native['mode'].eq('daily_update')]
    fitted=joblib.load(OUT/'models/upgrade.joblib');frame=experts(native,load_table(),fitted)
    frame['forecast'],weights=predict_points(frame,fitted);frame['center']=np.log(frame.forecast/frame.origin_price)
    for i,e in enumerate(EXPERTS):frame[e+'_weight']=weights[:,i]
    result=predict_risk(frame,model,True).sort_values(KEY).reset_index(drop=True)
    result['version']=model['version'];result['method']=model['name'];result['protocol_id']=model['protocol_id'];result['fit_end']=FIT_END
    result['status']=np.where(result.y.notna(),'realized','unreached')
    q=result[ALL_COLUMNS].to_numpy();assert len(result)==68 and result['mode'].eq('daily_update').all()
    assert not result.duplicated(KEY).any() and np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-12).all()
    np.testing.assert_array_equal(q[:,MEDIAN],result.center)
    result.to_csv(OUT/'test_forecasts.csv',index=False);score=metrics(result,'integrated_v5')
    for i,row in score.iterrows():
        g=result[result.horizon.eq(row.horizon)&result.y.notna()]
        if len(g):
            for name,column in [('v4','v4_forecast'),('futures','futures_price')]:
                value=np.sqrt(np.mean((g[column]-g.actual)**2));score.loc[i,name+'_rmse_usd']=value;score.loc[i,'rmse_vs_'+name]=row.rmse_usd/value
    score.to_csv(OUT/'test_metrics.csv',index=False)
    assert sha(OUT/'selected.json')==digest
    write_json(OUT/'test_audit.json',dict(status='passed',selected_sha256=digest,rows=68,forecast_mode='daily_update',
        quantiles_monotone=True,median_preserved=True,october_used_for_fitting=False,actual_counts=score[['horizon','n']].to_dict('records')))
    print('OCTOBER EVALUATION ONLY',flush=True);print(score.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','seal','test']);args=parser.parse_args()
    dict(fit=fit,seal=seal,test=test)[args.stage]()
