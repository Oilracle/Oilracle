import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from m3_ml.fusion.model import mature_before,MEDIAN
from m3_ml.distribution.tails import ALL_COLUMNS,ALL_TAUS
from m3_ml.fusion_v5 import data,point,support,risk,risk_safety,run


@pytest.fixture(scope='module')
def panel():return data.historical_inputs()


def test_historical_features_do_not_consume_october(monkeypatch,panel):
    table=data.load_table()
    table.loc[table.date.gt('2026-09-30'),table.select_dtypes('number').columns]=9999.
    monkeypatch.setattr(data,'load_table',lambda:table.copy())
    pd.testing.assert_frame_equal(panel,data.historical_inputs())


@pytest.mark.parametrize('kind',['regime','reliability'])
def test_gates_use_only_released_training_labels_and_prediction_inputs(panel,kind):
    d=next(d for d in data.read(data.OUT/'specification.json')['point_candidates'] if d['kind']==kind)
    f=panel[panel.horizon.eq(5)];cutoff='2025-09-01';train=mature_before(f,cutoff)
    test=f[f.origin_date.ge(cutoff)].head(30)
    model=point.fit(train,d);before,w=point.predict(test,model,True)
    np.testing.assert_allclose(w.sum(axis=1),1.,atol=1e-10);assert (w>=0).all()
    mutated=f.copy();mutated.loc[mutated.label_available_date.gt(cutoff),'actual']=9999.
    model2=point.fit(mature_before(mutated,cutoff),d)
    after=point.predict(test.assign(actual=-9999.,y=-9999.),model2)
    np.testing.assert_array_equal(before,after)
    columns=[e+'_expert' for e in data.EXPERTS]
    assert np.all(before>=test[columns].min(axis=1)-1e-8) and np.all(before<=test[columns].max(axis=1)+1e-8)


def test_nested_selection_ignores_future_realized_errors():
    f=pd.read_parquet(data.OUT/'point_candidate_forecasts.parquet');spec=data.read(data.OUT/'specification.json')
    before=point.choose(f,'2026-03-01',20,spec)
    f.loc[f.label_available_date.gt('2026-03-01'),['actual','forecast']]=9999.
    assert point.choose(f,'2026-03-01',20,spec)==before


def test_extrapolation_protection_shrinks_when_states_leave_training_support(panel):
    train=mature_before(panel[panel.horizon.eq(20)],'2025-09-01');calibration=support.bounds(train)
    sample=train.tail(1).copy()
    inside=support.trust(sample,calibration,1.)[0]
    sample['state_vol']=10000.
    outside=support.trust(sample,calibration,1.)[0]
    assert 0<=outside<inside<=1


def synthetic_risk():
    rng=np.random.default_rng(5210);dates=pd.bdate_range('2023-01-01',periods=600)
    f=pd.DataFrame(dict(origin_date=dates,label_available_date=dates+pd.offsets.BDay(15),
        horizon=5,center=.02,y=.06+.08*rng.standard_t(6,len(dates))))
    q=np.tile(norm.ppf(ALL_TAUS)*.08,(len(f),1))
    return pd.concat([f,pd.DataFrame(q,columns=ALL_COLUMNS)],axis=1)


def test_tail_score_calibration_preserves_median_and_ignores_unpublished_labels():
    f=synthetic_risk();d=dict(window=126,target=.90,sides='asymmetric');cutoff='2025-01-01'
    train=mature_before(f,cutoff);calibration=risk.fit(train,d)
    changed=f.copy();changed.loc[changed.label_available_date.gt(cutoff),'y']=9999.
    assert risk.fit(mature_before(changed,cutoff),d)==calibration
    q=risk.predict(f,calibration)
    np.testing.assert_array_equal(q,risk.predict(f.assign(y=9999.),calibration))
    np.testing.assert_array_equal(q[:,MEDIAN],f.center)
    assert (np.diff(q,axis=1)>=0).all()
    calibrated=train.tail(126);qc=risk.predict(calibrated,calibration)
    assert np.mean((calibrated.y>=qc[:,risk.LOW])&(calibrated.y<=qc[:,risk.HIGH]))>=.90


def test_interval_safety_controls_width_and_never_narrows_incumbent():
    old=np.tile(norm.ppf(ALL_TAUS)*.12,(30,1));new=old*np.linspace(.4,5.,30)[:,None]
    center=np.linspace(-.2,.2,30);new+=center[:,None]
    q,fraction=risk_safety.envelope(new,old,center,1.5);base=old+center[:,None]
    np.testing.assert_array_equal(q[:,MEDIAN],center)
    assert (np.diff(q,axis=1)>=0).all() and (fraction>=0).all() and (fraction<=1).all()
    assert np.all(q[:,risk.LOW]<=base[:,risk.LOW]+1e-12) and np.all(q[:,risk.HIGH]>=base[:,risk.HIGH]-1e-12)
    width=np.exp(q[:,risk.HIGH])-np.exp(q[:,risk.LOW])
    baseline=np.exp(base[:,risk.HIGH])-np.exp(base[:,risk.LOW])
    assert np.all(width<=1.5*baseline+1e-10)


def test_coverage_constrained_selector_does_not_read_future_scores():
    f=pd.read_parquet(data.OUT/'risk_candidate_scores.parquet');spec=data.read(data.OUT/'specification.json')
    before=risk.choose(f,'2026-06-01',5,spec)
    future=f.label_available_date.gt('2026-06-01')
    f.loc[future,['crps','width']]=9999.;f.loc[future,'covered']=False
    assert risk.choose(f,'2026-06-01',5,spec)==before


def test_training_and_selection_audits_respect_publication_cutoffs():
    data.protected()
    for name in ('point_fit_audit.json','point_selection_audit.json','support_audit.json'):
        for row in data.read(data.OUT/name):
            if row.get('latest_label_release') is not None:assert row['latest_label_release']<=row['cutoff']<='2026-09-30'
    audit=data.read(data.OUT/'risk_audit.json')
    for row in audit['fits']+audit['selections']:assert row['latest_label_release']<=row['cutoff']<='2026-09-30'
    for row in data.read(data.OUT/'final_fit_audit.json'):assert row['latest_label_release']<='2026-09-30'
    assert data.read(data.OUT/'promotion_gate.json')['passed']


def test_final_fitted_pipeline_ignores_realized_targets():
    from m3_ml.fusion_v3 import data as old_data
    table,panel,_=old_data.historical_inputs()
    native=panel[panel.horizon.isin(data.HORIZONS)].groupby('horizon').tail(1).reset_index(drop=True)
    fitted=joblib.load(data.OUT/'models/upgrade.joblib')
    first=run.experts(native,table,fitted)
    second=run.experts(native.assign(y=9999.,actual=9999.),table,fitted)
    names=[e+'_expert' for e in data.EXPERTS]+data.STATE
    np.testing.assert_array_equal(first[names],second[names])
    before,weights=run.predict_points(first,fitted);after,_=run.predict_points(second,fitted)
    np.testing.assert_array_equal(before,after);np.testing.assert_allclose(weights.sum(axis=1),1.,atol=1e-10)
    first['forecast']=before;first['center']=np.log(before/first.origin_price)
    model=dict(risk=data.read(data.OUT/'risk_final.json'),risk_safety=data.read(data.OUT/'risk_safety_design.json'))
    q1=run.predict_risk(first,model,False)
    q2=run.predict_risk(first.assign(y=-9999.,actual=-9999.),model,False)
    np.testing.assert_array_equal(q1[ALL_COLUMNS],q2[ALL_COLUMNS])
