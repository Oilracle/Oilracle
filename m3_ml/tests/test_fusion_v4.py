import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from m3_ml.fusion_v4 import data,long_models,stack,guard,risk,run
from m3_ml.fusion.model import mature_before,MEDIAN
from m3_ml.distribution.tails import ALL_COLUMNS,ALL_TAUS


def test_long_features_and_training_ignore_october_rows(monkeypatch):
    before=data.historical_inputs();table=data.load_table()
    table.loc[table.date.gt('2026-09-30'),table.select_dtypes('number').columns]=9999.
    monkeypatch.setattr(data,'load_table',lambda:table.copy())
    after=data.historical_inputs()
    pd.testing.assert_frame_equal(before[1],after[1])
    for k in before[2]:pd.testing.assert_frame_equal(before[2][k],after[2][k])
    assert before[1].label_available_date.max()<=pd.Timestamp('2026-09-30')


def test_long_model_training_purges_unavailable_targets():
    _,panel,X=data.historical_inputs();spec=json.loads((data.OUT/'specification.json').read_text())
    d=next(d for d in spec['new_candidates'] if d['kind']=='ridge')
    panel=panel[panel.horizon.eq(5)];cutoff=pd.Timestamp('2025-03-01')
    train=mature_before(panel,cutoff);test=panel[panel.origin_date.between('2025-03-01','2025-05-31')]
    first,_=long_models.fit_predict(train,test,X,d)
    modified=panel.copy();modified.loc[modified.label_available_date.gt(cutoff),'actual']=9999.
    second,_=long_models.fit_predict(mature_before(modified,cutoff),test,X,d)
    np.testing.assert_array_equal(first,second)


def test_family_selection_ignores_future_scores():
    spec=json.loads((data.OUT/'specification.json').read_text());long,old,native=stack.inputs(spec)
    chosen,_=stack.family_choices(long,old,spec,'2025-03-01')
    for frame in (long,old):frame.loc[frame.label_available_date.gt('2025-03-01'),'actual']=9999.
    after,_=stack.family_choices(long,old,spec,'2025-03-01')
    assert chosen==after


def test_guard_zero_is_allowed_and_future_losses_do_not_change_selection():
    dates=pd.bdate_range('2024-01-01','2026-09-30')
    f=pd.DataFrame(dict(origin_date=dates,horizon=5,label_available_date=dates+pd.offsets.BDay(15),
        incumbent=90.,proposal=100.,actual=85.))
    assert guard.coefficient(f,.1)==0.
    candidates=pd.concat([f.assign(penalty=p,forecast=90+p) for p in guard.PENALTIES],ignore_index=True)
    before=guard.choose(candidates,'2026-06-01',5)
    candidates.loc[candidates.label_available_date.gt('2026-06-01'),'actual']=9999.
    assert guard.choose(candidates,'2026-06-01',5)==before
    assert guard.choose(candidates.iloc[:0],'2026-06-01',5)==1.


def test_asymmetric_tail_fit_preserves_center_and_publication_cutoff():
    dates=pd.bdate_range('2023-09-01','2026-09-30');rng=np.random.default_rng(5210)
    f=pd.DataFrame(dict(origin_date=dates,horizon=60,label_available_date=dates+pd.offsets.BDay(70),
        center=.01,y=.06+rng.standard_t(5,len(dates))*.08))
    f=pd.concat([f,pd.DataFrame(np.tile(norm.ppf(ALL_TAUS)*.12,(len(f),1)),columns=ALL_COLUMNS)],axis=1)
    cal=risk.tail_calibration(mature_before(f,'2025-09-01'),126,'asymmetric')
    q=risk.calibrate(f,cal,1.,1.25)
    np.testing.assert_array_equal(q[:,MEDIAN],f.center)
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=0).all()
    changed=f.copy();changed.loc[changed.label_available_date.gt('2025-09-01'),'y']=9999.
    assert risk.tail_calibration(mature_before(changed,'2025-09-01'),126,'asymmetric')==cal
    np.testing.assert_array_equal(q,risk.calibrate(f.assign(y=9999.),cal,1.,1.25))


def test_observable_implied_volatility_scale_ignores_outcomes():
    spec=json.loads((data.OUT/'specification.json').read_text())
    f=pd.read_csv(data.OUT/'point_crossfit.csv',parse_dates=['origin_date']).head(40)
    first=risk.sources(f,spec)
    second=risk.sources(f.assign(y=9999.,actual=9999.),spec)
    for name in ('implied_vol','hybrid_vol'):
        np.testing.assert_array_equal(first[name][ALL_COLUMNS],second[name][ALL_COLUMNS])
        assert np.isfinite(first[name][ALL_COLUMNS]).all().all()


def test_saved_fits_and_source_archives():
    run.protected()
    for name in ('long_fit_audit.json','family_selection_audit.json','stack_fit_audit.json'):
        for row in json.loads((data.OUT/name).read_text()):
            if row.get('latest_label_release') is not None:assert row['latest_label_release']<=row['cutoff']<='2026-09-30'
    for row in json.loads((data.OUT/'guard_audit.json').read_text())['fits']:
        if row['latest_label_release'] is not None:assert row['latest_label_release']<=row['cutoff']
    for row in json.loads((data.OUT/'risk_audit.json').read_text())['fits']:
        assert row['latest_label_release']<=row['cutoff']<='2026-09-30'
    assert json.loads((data.OUT/'point_promotion_gate.json').read_text())['passed']


def test_frozen_prediction_does_not_read_realized_targets():
    from m3_ml.fusion_v3 import data as old_data
    table,panel,_=old_data.historical_inputs()
    sample=panel[panel.horizon.isin(data.HORIZONS)].groupby('horizon').tail(1)
    frame,old_X=old_data.feature_frame(sample,table)
    market,X=data.features(table)
    pos=pd.Series(table.index,index=table.date).reindex(frame.origin_date).to_numpy().astype(int)
    frame['feature_row']=pos;frame['implied_vol']=table.oil_implied_volatility.iloc[pos].to_numpy()/100
    frame['scale']=np.maximum(1.,frame.futures_price*frame.futures_vol*np.sqrt(frame.horizon+2))
    for c in ('basis20','basis60','basis250'):frame[c]=market[c].iloc[pos].to_numpy()
    spec=json.loads((data.OUT/'specification.json').read_text());model=json.loads((data.OUT/'selected.json').read_text())
    old_spec=json.loads((data.OLD/'specification.json').read_text());old_model=json.loads((data.OLD/'selected.json').read_text())
    fitted=joblib.load(data.OUT/'models/upgrade.joblib');old_fitted=joblib.load(data.OLD/'models/point_models.joblib')
    before=run.predict(frame,X,old_X,spec,model,old_spec,old_model,fitted,old_fitted)
    after=run.predict(frame.assign(y=9999.,actual=9999.),X,old_X,spec,model,old_spec,old_model,fitted,old_fitted)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    assert np.isfinite(before[0]).all() and (before[0]>0).all()
    assert model['input_sha256']==run.inputs()
