import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
import pytest
from m3_ml.fusion_v3 import data,point,risk,run
from m3_ml.fusion.model import mature_before,MEDIAN
from m3_ml.distribution.tails import ALL_COLUMNS


@lru_cache(maxsize=1)
def historical():return data.historical_inputs()


def test_federal_holiday_availability():
    assert pd.Timestamp('2026-07-02')+data.LAG==pd.Timestamp('2026-07-07')
    assert pd.Timestamp('2026-09-04')+data.LAG==pd.Timestamp('2026-09-09')
    table=historical()[0];features=data.market_features(table)
    assert features.available_date.le(features.date).all()
    assert features.quote_date.lt(features.date).all()


def test_future_quotes_and_unpublished_spot_values_cannot_change_past_market_features():
    table=historical()[0];raw=pd.read_csv(data.RAW,parse_dates=['quote_date'])
    before=data.market_features(table,raw)
    cutoff=pd.Timestamp('2025-06-02')
    unavailable=pd.Series([d+data.LAG for d in raw.quote_date]).gt(cutoff)
    changed=raw.copy();changed.loc[unavailable,'close']*=4
    new_table=table.copy()
    unavailable_spot=data.label_available_dates(table).gt(cutoff)
    new_table.loc[unavailable_spot,'target_wti']*=3
    after=data.market_features(new_table,changed)
    keep=before.date.le(cutoff)
    pd.testing.assert_frame_equal(before.loc[keep],after.loc[keep])


def test_october_rows_cannot_enter_historical_training(monkeypatch):
    before=historical();table=data.load_table()
    table.loc[table.date.gt('2026-09-30'),table.select_dtypes('number').columns]=9999.
    monkeypatch.setattr(data,'load_table',lambda:table.copy())
    after=data.historical_inputs()
    pd.testing.assert_frame_equal(before[1],after[1])
    for key in before[2]:pd.testing.assert_frame_equal(before[2][key],after[2][key])


def candidate_panel():
    dates=pd.bdate_range('2023-01-01','2026-09-30');base=pd.DataFrame(dict(origin_date=dates,horizon=1,
        label_available_date=dates+pd.offsets.BDay(12),actual=80.+np.sin(np.arange(len(dates))/13)))
    return pd.concat([base.assign(candidate='a',forecast=79.8),base.assign(candidate='b',forecast=80.3)],ignore_index=True)


def test_past_point_selection_and_risk_selection_ignore_future_losses():
    panel=candidate_panel();rule=dict(mode='soft8',window=252)
    before,audit=point.apply_rule(panel,rule,'2024-03-01','2026-09-30')
    changed=panel.copy();changed.loc[changed.label_available_date.gt('2025-03-01'),'actual']=9999.
    after,_=point.apply_rule(changed,rule,'2024-03-01','2026-09-30')
    past=before.origin_date.lt('2025-06-01')
    np.testing.assert_array_equal(before.loc[past,'forecast'],after.loc[past,'forecast'])
    for entry in audit:assert entry['latest_selection_release']<=entry['cutoff']
    losses=panel.assign(loss=(panel.actual-panel.forecast)**2,hs_crps=1.)
    a,_=risk.choose(losses,'2025-03-01')
    losses.loc[losses.label_available_date.gt('2025-03-01'),'loss']=9999.
    b,_=risk.choose(losses,'2025-03-01')
    assert a==b


def test_candidate_refit_ignores_unpublished_training_targets():
    table,panel,features=historical()
    sample=panel[panel.horizon.eq(1)].copy();spec=json.loads((data.OUT/'specification.json').read_text())
    definition=next(d for d in spec['candidates'] if d['kind']=='ridge')
    tr=mature_before(sample,'2025-03-01');te=sample[sample.origin_date.between('2025-03-01','2025-05-31')]
    first,_=point.fit_predict(tr,te,features,definition)
    changed=sample.copy();changed.loc[changed.label_available_date.gt('2025-03-01'),'actual']+=1000.
    second,_=point.fit_predict(mature_before(changed,'2025-03-01'),te,features,definition)
    np.testing.assert_array_equal(first,second)


def test_serialized_point_models_do_not_read_outcomes_and_use_all_member_features():
    table,panel,_=historical();spec=json.loads((data.OUT/'specification.json').read_text())
    frame,features=data.feature_frame(panel.groupby('horizon').tail(1),table)
    models=joblib.load(data.OUT/'models/point_models.joblib')
    weights=json.loads((data.OUT/'point_weights.json').read_text())
    first=run.predict_points(frame,features,spec,weights,models)
    changed=frame.copy();changed['actual']=9999.;changed['y']=-9999.
    second=run.predict_points(changed,features,spec,weights,models)
    np.testing.assert_array_equal(first,second)
    for member in data.MEMBERS:assert member+'_minus_futures' in features['members']
    for values in weights.values():assert min(values.values())>=0 and abs(sum(values.values())-1)<1e-10


def test_risk_calibration_preserves_point_and_ignores_prediction_targets():
    spec=json.loads((data.OUT/'specification.json').read_text())
    points=pd.read_csv(data.OUT/'point_crossfit.csv',parse_dates=['origin_date','label_available_date'])
    frame=risk.source_frames(points,spec)['futures_vol']
    train=mature_before(frame,'2025-09-01');cal=risk.calibration(train,126)
    test=frame[frame.origin_date.ge('2025-09-01')]
    for mixture in (0.,.5,1.):
        q=risk.calibrate(test,cal,mixture,1.25)
        np.testing.assert_array_equal(q[:,MEDIAN],test.center)
        assert (np.diff(q,axis=1)>=-1e-12).all()
        np.testing.assert_array_equal(q,risk.calibrate(test.assign(y=9999.),cal,mixture,1.25))
    changed=frame.copy();changed.loc[changed.label_available_date.gt('2025-09-01'),'y']=9999.
    assert cal==risk.calibration(mature_before(changed,'2025-09-01'),126)


def test_saved_fit_audits_and_protected_v2():
    for entry in json.loads((data.OUT/'point_crossfit_audit.json').read_text()):
        assert entry['latest_selection_release']<=entry['cutoff']<='2026-09-30'
    evidence=json.loads((data.OUT/'risk_crossfit_audit.json').read_text())
    for entry in evidence['fits']:assert entry['latest_training_release']<=entry['cutoff']<='2026-09-30'
    for entry in evidence['selections']:assert entry['latest_selection_release']<=entry['cutoff']<='2026-09-30'
    spec=json.loads((data.OUT/'specification.json').read_text())
    assert spec['risk_start']>=spec['rule_lock_date']
    assert data.sha(data.BASE/'selected.json')=='fb5dede1373a6c63e488ddab6c7e1704dd41f0d0f6eaa725f8917207343b6dff'
    assert data.sha(data.BASE/'test_forecasts.csv')=='38f40fc56f1bd4c1c3f07160edfd1e93d6144482b6b3851355f118a7fbf6c454'
    older=json.loads((data.BASE/'selected.json').read_text())
    assert older['input_sha256']==run.v2_inputs()


def test_sealed_october_grid_when_evaluated():
    if not (data.OUT/'test_forecasts.csv').exists():pytest.skip('October not evaluated before seal')
    f=pd.read_csv(data.OUT/'test_forecasts.csv');q=f[ALL_COLUMNS].to_numpy()
    assert len(f)==87 and not f.duplicated(['mode',*data.KEY]).any()
    np.testing.assert_allclose(q[:,MEDIAN],f.center,atol=1e-14)
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-10).all()
    for h in (1,5,20,60):
        a=f[f['mode'].eq('fixed_origin')&f.horizon.eq(h)]
        b=f[f['mode'].eq('daily_update')&f.origin_date.eq('2026-09-30')&f.horizon.eq(h)]
        np.testing.assert_allclose(a[ALL_COLUMNS],b[ALL_COLUMNS],atol=1e-12)
    selected=json.loads((data.OUT/'selected.json').read_text())
    assert selected['input_sha256']==run.seal_inputs()
    assert json.loads((data.OUT/'test_audit.json').read_text())['selected_sha256']==data.sha(data.OUT/'selected.json')
