import joblib
import numpy as np
import pandas as pd
import pytest
from m3_ml.fusion.model import mature_before,MEDIAN
from m3_ml.distribution.tails import ALL_COLUMNS
from m3_ml.fusion_v6 import research as r,run


@pytest.mark.parametrize('kind',['mean','median','ridge'])
def test_bias_fit_uses_only_released_labels_and_predict_ignores_targets(kind):
    panel=pd.read_csv(r.OLD/'point_crossfit.csv',parse_dates=['origin_date','label_available_date'])
    panel=panel[panel.horizon.eq(5)];d=next(d for d in r.read(r.OUT/'specification.json')['candidates'] if d['kind']==kind)
    cutoff='2026-03-01';tr=mature_before(panel,cutoff);test=panel[panel.origin_date.ge(cutoff)].head(30)
    first=r.predict(test,r.fit(tr,d))
    panel.loc[panel.label_available_date.gt(cutoff),'actual']=9999.
    second=r.predict(test.assign(actual=-9999.,y=-9999.),r.fit(mature_before(panel,cutoff),d))
    np.testing.assert_array_equal(first,second)


def test_nested_selector_cannot_use_unpublished_losses():
    f=pd.read_parquet(r.OUT/'candidates.parquet');spec=r.read(r.OUT/'specification.json')
    before=r.choose(f,'2026-03-01',1,spec)
    f.loc[f.label_available_date.gt('2026-03-01'),['prediction','actual']]=9999.
    assert r.choose(f,'2026-03-01',1,spec)==before


def test_tail_calibration_preserves_center_cap_and_publication_cutoff():
    f=pd.read_csv(r.OLD/'risk_forward_forecasts.csv',parse_dates=['origin_date','label_available_date'])
    f=f[f.horizon.eq(5)];spec=r.read(r.OUT/'specification.json')['risk'];cutoff='2026-07-01'
    cap=r.read(r.OUT/'width_amendment.json')['price_width_cap']
    cal=r.tail_fit(mature_before(f,cutoff),spec);q=r.tail_predict(f,cal,cap)
    changed=f.copy();changed.loc[changed.label_available_date.gt(cutoff),'y']=9999.
    assert r.tail_fit(mature_before(changed,cutoff),spec)==cal
    np.testing.assert_array_equal(q,r.tail_predict(f.assign(y=9999.),cal,cap))
    np.testing.assert_array_equal(q[:,MEDIAN],f.center)
    assert (np.diff(q,axis=1)>=0).all()
    new_width=np.exp(q[:,r.HI])-np.exp(q[:,r.LO]);old_width=np.exp(f['q_0.950'])-np.exp(f['q_0.050'])
    assert (new_width<=cap*old_width+1e-10).all()


def test_audit_times_and_predecessor_seal():
    r.protect();audit=r.read(r.OUT/'point_audit.json')
    for row in audit['fits']+audit['selections']:
        if row.get('last_release') is not None:assert row['last_release']<=row['cutoff']<=r.END
    for row in r.read(r.OUT/'risk_audit.json'):assert row['last_release']<=row['cutoff']<=r.END


def test_full_frozen_correction_pipeline_ignores_realized_targets():
    base=pd.read_csv(r.OLD/'risk_forward_forecasts.csv')
    points=pd.read_csv(r.OLD/'point_crossfit.csv').set_index(r.KEY)
    # Historical risk rows lack some point-state columns; add only observable inputs.
    for c in r.STATE+['scale','futures_price']:
        if c not in base:base[c]=points[c].reindex(pd.MultiIndex.from_frame(base[r.KEY])).to_numpy()
    model=dict(risk=r.read(r.OUT/'risk_final.json'),risk_rule=r.read(r.OUT/'specification.json')['risk'])
    model['risk_rule']['price_width_cap']=r.read(r.OUT/'width_amendment.json')['price_width_cap']
    fitted=joblib.load(r.OUT/'models/correction.joblib')
    first=run.predict(base,model,fitted);second=run.predict(base.assign(y=9999.,actual=9999.),model,fitted)
    np.testing.assert_array_equal(first[['forecast',*ALL_COLUMNS]],second[['forecast',*ALL_COLUMNS]])
    long=base.horizon.isin([20,60])
    np.testing.assert_allclose(first.loc[long,['forecast',*ALL_COLUMNS]],base.loc[long,['forecast',*ALL_COLUMNS]],atol=1e-12)
