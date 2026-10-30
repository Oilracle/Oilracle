"""Fit and seal an upgrade on history, then infer only daily rolling forecasts."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table
from m3_ml.fusion.model import mature_before,MEDIAN,metrics,risk_readout
from m3_ml.fusion.run import october_frame
from m3_ml.fusion_v3 import data as previous_data,point as previous_point,run as previous_run
from m3_ml.fusion_v2.model import calibrate
from m3_ml.distribution.tails import ALL_COLUMNS
from .data import OUT,OLD,ROOT,KEY,HORIZONS,FIT_END,write_json,sha,historical_inputs,features
from .long_models import fit_predict
from .risk import sources


def protected():
    expected=json.loads((OUT/'protected_v3.json').read_text())
    assert sha(OLD/'selected.json')==expected['selected_sha256']
    assert sha(OLD/'test_forecasts.csv')==expected['forecasts_sha256']
    selected=json.loads((OLD/'selected.json').read_text())
    assert selected['input_sha256']==previous_run.seal_inputs()


def inputs():
    paths=[*sorted((ROOT/'m3_ml/fusion_v4').glob('*.py')),
        *[OUT/name for name in ('specification.json','guard_design.json','risk_tail_design.json',
            'locked_rules.json','point_weights.json','final_families.json','guard_final.json','risk_final.json',
            'historical_inference.json','historical_risk_metrics.csv','point_promotion_gate.json','point_crossfit.csv',
            'models/upgrade.joblib','final_fit_audit.json')],OLD/'selected.json']
    return {**previous_run.seal_inputs(),**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}


def fit_and_seal():
    if (OUT/'test_audit.json').exists():raise ValueError('October already evaluated; do not replace the sealed upgrade')
    protected();gate=json.loads((OUT/'point_promotion_gate.json').read_text())
    if not gate['passed']:raise ValueError('The historical point promotion gate did not pass')
    spec=json.loads((OUT/'specification.json').read_text())
    prior_spec=json.loads((OLD/'specification.json').read_text())
    weights=json.loads((OUT/'point_weights.json').read_text());choices=json.loads((OUT/'final_families.json').read_text())
    _,panel,X=historical_inputs();_,old_panel,old_X=previous_data.historical_inputs()
    long_defs={d['id']:d for d in spec['new_candidates']};old_defs={d['id']:d for d in prior_spec['candidates']}
    fitted={};audit=[]
    for h in HORIZONS:
        train=mature_before(panel[panel.horizon.eq(h)],FIT_END)
        old_train=mature_before(old_panel[old_panel.horizon.eq(h)],FIT_END)
        for family,w in weights[str(h)].items():
            if w<1e-12 or family not in choices[str(h)]:continue
            cid=choices[str(h)][family]
            if family=='v3':_,model=previous_point.fit_predict(old_train,old_train.iloc[:1],old_X,old_defs[cid])
            else:_,model=fit_predict(train,train.iloc[:1],X,long_defs[cid])
            fitted[(h,family)]=model
        audit.append(dict(horizon=h,n_long_history=len(train),n_member_history=len(old_train),
            latest_label_release=str(train.label_available_date.max().date()),
            earliest_long_origin=str(train.origin_date.min().date())))
    (OUT/'models').mkdir(exist_ok=True);joblib.dump(fitted,OUT/'models/upgrade.joblib')
    write_json(OUT/'final_fit_audit.json',audit)
    model=dict(name=spec['name'],version=spec['version'],protocol_id=prior_spec['protocol_id'],
        fit_end=FIT_END,forecast_mode='daily_update',horizons=list(HORIZONS),
        family_weights=weights,family_choices=choices,
        guard=json.loads((OUT/'guard_final.json').read_text()),risk=json.loads((OUT/'risk_final.json').read_text()),
        historical_promotion_gate=gate,incumbent_sha256=sha(OLD/'selected.json'),
        source_limits=spec['source_limits'],october_status=spec['october'],input_sha256=inputs())
    write_json(OUT/'selected.json',model)
    print('SEALED V4 before October evaluation:',sha(OUT/'selected.json'),flush=True)


def predict(frame,X,old_X,spec,model,old_spec,old_model,fitted,old_fitted):
    incumbent=previous_run.predict_points(frame,old_X,old_spec,old_model['point_weights'],old_fitted)
    proposal=np.zeros(len(frame));defs={d['id']:d for d in spec['new_candidates']}
    old_defs={d['id']:d for d in old_spec['candidates']}
    for h,indices in frame.groupby('horizon').groups.items():
        g=frame.loc[indices];price=np.zeros(len(g));total=0.
        for family,w in model['family_weights'][str(int(h))].items():
            if w<1e-12:continue
            if family in ('futures','m1','m2','lstm','ml'):p=g[family+'_price'].to_numpy()
            else:
                cid=model['family_choices'][str(int(h))][family];estimator=fitted[(h,family)]
                if family=='v3':
                    d=old_defs[cid]
                    if d['kind']=='carry':p=g[d['base']+'_price'].to_numpy()
                    elif d['kind']=='convex':p=g[[m+'_price' for m in previous_point.BLEND]].to_numpy()@estimator['weights']
                    else:p=np.maximum(.01,g[d['base']+'_price'].to_numpy()+estimator.predict(old_X[d['feature']].loc[indices]))
                else:
                    d=defs[cid]
                    if d['kind']=='basis':p=g.futures_price.to_numpy()+g[d['basis']].to_numpy()
                    elif d['kind']=='mean':p=g.futures_price.to_numpy()+estimator
                    else:
                        correction=estimator.predict(X[d['feature']].iloc[g.feature_row])
                        if d['target']=='scaled':correction*=g.scale.to_numpy()
                        p=np.maximum(.01,g.futures_price.to_numpy()+correction)
            price+=w*p;total+=w
        proposal[indices]=price/total
    strength=frame.horizon.map({int(h):e['upgrade_weight'] for h,e in model['guard'].items()}).to_numpy()
    return incumbent+strength*(proposal-incumbent),proposal,incumbent


def final_forecasts(spec,model):
    table=load_table();old_spec=json.loads((OLD/'specification.json').read_text())
    old_model=json.loads((OLD/'selected.json').read_text())
    native=october_frame(old_spec);native=native[native['mode'].eq('daily_update')]
    frame,old_X=previous_data.feature_frame(native,table)
    market,X=features(table)
    pos=pd.Series(table.index,index=table.date).reindex(frame.origin_date).to_numpy().astype(int)
    frame['feature_row']=pos;frame['implied_vol']=table.oil_implied_volatility.iloc[pos].to_numpy()/100
    frame['scale']=np.maximum(1.,frame.futures_price*frame.futures_vol*np.sqrt(frame.horizon+2))
    for c in ('basis20','basis60','basis250'):frame[c]=market[c].iloc[pos].to_numpy()
    frame['forecast'],frame['proposal_forecast'],frame['incumbent_forecast']=predict(frame,X,old_X,spec,model,old_spec,old_model,
        joblib.load(OUT/'models/upgrade.joblib'),joblib.load(OLD/'models/point_models.joblib'))
    frame['center']=np.log(frame.forecast/frame.origin_price)
    frames=sources(frame,spec,True);parts=[]
    for h,g in frame.groupby('horizon'):
        d=model['risk'][str(int(h))];f=frames[d['method']].loc[g.index].copy()
        q=calibrate(f,{str(int(h)):d['calibration']},d['mixture'],d['width'])
        f.loc[:,ALL_COLUMNS]=q;f['risk_method']=d['method'];f['risk_calibration_mode']=d['calibration_mode']
        f['lower90']=f.origin_price*np.exp(f['q_0.050']);f['upper90']=f.origin_price*np.exp(f['q_0.950'])
        for key,value in risk_readout(q).items():f[key]=value
        parts.append(f)
    result=pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    result['version']=model['version'];result['method']=model['name'];result['protocol_id']=model['protocol_id'];result['fit_end']=FIT_END
    result['status']=np.where(result.y.notna(),'realized','unreached')
    return result


def test():
    protected();model=json.loads((OUT/'selected.json').read_text());spec=json.loads((OUT/'specification.json').read_text())
    assert model['input_sha256']==inputs();digest=sha(OUT/'selected.json')
    if (OUT/'test_audit.json').exists():assert json.loads((OUT/'test_audit.json').read_text())['selected_sha256']==digest
    f=final_forecasts(spec,model);q=f[ALL_COLUMNS].to_numpy()
    assert len(f)==68 and f['mode'].eq('daily_update').all() and not f.duplicated(KEY).any()
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-12).all()
    np.testing.assert_array_equal(q[:,MEDIAN],f.center)
    f.to_csv(OUT/'test_forecasts.csv',index=False)
    score=metrics(f,'integrated_v4')
    for i,row in score.iterrows():
        g=f[f.horizon.eq(row.horizon)&f.y.notna()]
        if len(g):
            for name,column in [('v3','incumbent_forecast'),('futures','futures_price')]:
                value=np.sqrt(np.mean((g[column]-g.actual)**2))
                score.loc[i,name+'_rmse_usd']=value;score.loc[i,'rmse_vs_'+name]=row.rmse_usd/value
    score.to_csv(OUT/'test_metrics.csv',index=False)
    assert sha(OUT/'selected.json')==digest
    write_json(OUT/'test_audit.json',dict(status='passed',selected_sha256=digest,rows=68,forecast_mode='daily_update',
        median_preserved=True,quantiles_monotone=True,october_used_for_fitting=False,
        actual_counts=score[['horizon','n']].to_dict('records')))
    print('OCTOBER EVALUATION ONLY',flush=True);print(score.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['seal','test']);args=parser.parse_args()
    (fit_and_seal if args.stage=='seal' else test)()
