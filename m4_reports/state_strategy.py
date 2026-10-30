"""Deliver the single sealed v5 strategy with full historical tradeoffs."""
import joblib
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table
from m3_ml.fusion.run import october_frame
from m3_ml.fusion_v5.data import OUT,OLD,V3,ROOT,KEY,HORIZONS,EXPERTS,read,write_json,sha,protected
from m3_ml.fusion_v5.run import input_hashes,experts,predict_points
from m4_reports.rolling_upgrade import records,figures

DELIVERY=OUT/'rolling'


def contribution_audit(f):
    fitted=joblib.load(OUT/'models/upgrade.joblib');table=load_table()
    native=october_frame(read(V3/'specification.json'));native=native[native['mode'].eq('daily_update')].copy()
    frame=experts(native,table,fitted);baseline,_=predict_points(frame,fitted)
    expected=f.set_index(KEY).forecast.reindex(pd.MultiIndex.from_frame(frame[KEY]))
    np.testing.assert_allclose(baseline,expected,rtol=1e-12,atol=1e-12)
    rows=[]
    for member in ('m1','m2','lstm','ml'):
        changed=native.copy()
        changed[member]=np.log((changed.origin_price*np.exp(changed[member])+.01)/changed.origin_price)
        perturbed=experts(changed,table,fitted);prediction,_=predict_points(perturbed,fitted)
        for date,h,value in zip(frame.origin_date,frame.horizon,(prediction-baseline)/.01):
            rows.append(dict(origin_date=date,horizon=int(h),member=member,local_price_sensitivity=float(value)))
    sensitivity=pd.DataFrame(rows);sensitivity.to_csv(DELIVERY/'module_input_sensitivity.csv',index=False)
    weights=f[KEY+[e+'_weight' for e in EXPERTS]].melt(id_vars=KEY,var_name='expert',value_name='weight')
    weights['expert']=weights.expert.str.removesuffix('_weight');weights.to_csv(DELIVERY/'expert_weights.csv',index=False)
    return sensitivity,weights


def main():
    protected();model=read(OUT/'selected.json');digest=sha(OUT/'selected.json')
    assert model['input_sha256']==input_hashes() and read(OUT/'test_audit.json')['selected_sha256']==digest
    DELIVERY.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(OUT/'test_forecasts.csv',parse_dates=['origin_date','target_date']);test=pd.read_csv(OUT/'test_metrics.csv')
    history=pd.read_csv(OUT/'historical_point_proposal_metrics.csv').assign(period='later_history')
    risks=pd.read_csv(OUT/'historical_risk_metrics.csv').assign(period='forward')
    previous_risk=pd.read_csv(OLD/'historical_risk_metrics.csv');inference=read(OUT/'historical_inference.json')
    assert model['promotion_gate']['point_passed'] and model['promotion_gate']['risk_passed']
    for source,destination in [('test_forecasts.csv','forecasts.csv'),('test_metrics.csv','metrics.csv'),('historical_inference.json','historical_inference.json')]:
        (DELIVERY/destination).write_bytes((OUT/source).read_bytes())
    history.to_csv(DELIVERY/'historical_point_metrics.csv',index=False);risks.to_csv(DELIVERY/'historical_risk_metrics.csv',index=False)
    sensitivity,weights=contribution_audit(f)
    figures(f,history,risks,previous_risk,inference,DELIVERY/'figures','v5','v4','forward')
    scope=dict(forecast_mode='daily_update',horizons=list(HORIZONS),parameter_policy='All fitted functions, preprocessing and risk calibration frozen as of September 30.',
        input_policy='Previously published input states can change each day and generate different weights through the frozen reliability function.',
        selection_policy='Nested time selection for the added layer, then historical promotion gates; October only evaluates.',
        research_amendments=['support_design.json','damping_design.json','risk_safety_design.json'],evidence_status=model['october_status'])
    write_json(DELIVERY/'delivery_scope.json',scope)
    write_json(DELIVERY/'protocol.json',dict(protocol_id=model['protocol_id'],fit_end=model['fit_end'],mode='daily_update',horizons=list(HORIZONS)))
    roles={'m1':'Actual VECM expert and inherited residual features; its influence is checked by perturbing the actual input.',
           'm2':'Actual monthly STL expert, inherited residual features and calendar state variables.',
           'lstm':'Actual three-seed LSTM expert and inherited residual model features.',
           'ml_point':'Original point ML and long-history residual experts; learned state gates and conditional error models assess reliability.',
           'ml_distribution':'ML shape/scale candidates, released-error tail calibration and empirical coverage constraints; guarded 63-quantile distribution.',
           'incumbent':'Frozen v4 remains the anchor; new meta-policy updates have a fixed 50% maximum contribution.'}
    limits=[model['limitations'],model['october_status'],
        'Historical incremental point RMSE gain is 1.356%; its 95% block interval includes zero. A borderline one-sided p-value is not robust significance after adaptive research.',
        'The historical primary gain over lagged-futures carry is 0.898%, also not statistically established.',
        'The initial new policy failed h1 protection. Support protection, half-step damping and an interval safeguard were added after historical inspection; all originals are archived.',
        'Forward 90% coverage remains below nominal at every horizon; the final raw risk selector has infeasible constraints at some horizons, which are reported explicitly.',
        '60-day average interval width rises 29.2% as coverage improves; the CRPS reduction is 4.9%.',
        'Local member sensitivity demonstrates use rather than isolated beneficial contribution.',
        'October has 17/13 realized targets at h1/h5 and none at h20/h60; no long-horizon October errors exist.',
        'Current-vintage data and assumed publication dates prevent a complete point-in-time data guarantee.']
    previous={}
    for version,path in [('v4',OLD),('v3',V3),('v2',ROOT/'m3_ml/fusion_v2/outputs/frozen_october_2026')]:
        previous[version]=records(pd.read_csv(path/'test_metrics.csv').loc[lambda x:x['mode'].eq('daily_update')])
    payload=dict(name=model['name'],version=model['version'],protocol_id=model['protocol_id'],fit_end=model['fit_end'],selected_sha256=digest,
        active_output=DELIVERY.relative_to(ROOT).as_posix(),forecast_mode='daily_update',horizons=list(HORIZONS),delivery_scope=scope,
        point_meaning='Median of the final coherent predictive distribution.',point_policy=model['point_policy'],risk_by_horizon=model['risk'],
        risk_safety=model['risk_safety'],module_roles=roles,module_sensitivity=records(sensitivity),expert_weights=records(weights),
        test_metrics=records(test),previous_rolling_metrics=previous,forecasts=records(f),historical_point_metrics=records(history),
        historical_risk_metrics=records(risks),historical_inference=inference,promotion_gate=model['promotion_gate'],
        source_audit=read(V3/'selected.json')['source_audit'],limitations=limits)
    write_json(DELIVERY/'presentation_data.json',payload)
    manifest=dict(active_version=model['version'],name=model['name'],output=DELIVERY.relative_to(ROOT).as_posix(),model_output=OUT.relative_to(ROOT).as_posix(),
        forecast_mode='daily_update',horizons=list(HORIZONS),delivery_profile='daily_rolling',selected_sha256=digest,
        presentation_data=(DELIVERY/'presentation_data.json').relative_to(ROOT).as_posix(),forecasts=(DELIVERY/'forecasts.csv').relative_to(ROOT).as_posix(),
        metrics=(DELIVERY/'metrics.csv').relative_to(ROOT).as_posix(),delivery_scope=(DELIVERY/'delivery_scope.json').relative_to(ROOT).as_posix(),
        selection_reason='Pre-October historical gates passed: point RMSE improves 1.356% vs v4 with maximum horizon degradation 0.518%; risk coverage gap improves 5.01 percentage points on average with CRPS and width limits.',
        superseded=[f'm3_ml/{name}/outputs/frozen_october_2026 (audit only)' for name in ('fusion','fusion_v2','fusion_v3','fusion_v4')])
    write_json(ROOT/'m4_reports/final_strategy_manifest.json',manifest)
    print('Canonical v5 rolling output:',DELIVERY,flush=True)
    print(test[['horizon','n','rmse_usd','rmse_vs_rw','rmse_vs_v4','coverage90']].to_string(index=False),flush=True)


if __name__=='__main__':main()
