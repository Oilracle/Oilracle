import numpy as np
import pandas as pd
from scipy.stats import norm
from m3_ml.fusion.model import (MEMBERS, MEDIAN, ALL_COLUMNS, ALL_TAUS, fit_weights,
                                mature_before, crossfit_points, fit_calibrator, calibrate, risk_readout)

def panel():
    rng=np.random.default_rng(5210)
    frames=[]
    dates=pd.bdate_range('2023-09-01','2026-09-30')
    for h in (1,5,20,60):
        y=rng.normal(0,.03*np.sqrt(h),len(dates))
        f=pd.DataFrame({'origin_date':dates,'horizon':h,'y':y,
                        'label_available_date':dates+pd.offsets.BDay(h+8)})
        for i,m in enumerate(MEMBERS): f[m]=y*.1+rng.normal(0,.02*(i+1)*np.sqrt(h),len(dates))
        frames.append(f)
    return pd.concat(frames,ignore_index=True)

def test_maturity_and_weight_constraints():
    f=mature_before(panel(),'2024-09-01')
    assert f.label_available_date.le('2024-09-01').all()
    w=fit_weights(f,.1)
    assert np.min(w)>=.05-1e-10
    np.testing.assert_allclose(w.sum(),1,atol=1e-10)
    # The least noisy channel should not receive the least weight.
    assert w[0]>w[-1]

def test_future_label_mutation_does_not_change_earlier_meta_forecasts():
    f=panel(); a,_=crossfit_points(f,.1,'2024-09-01','2026-09-30')
    mutated=f.copy()
    mutated.loc[mutated.label_available_date.gt('2025-03-01'),'y']+=100
    b,_=crossfit_points(mutated,.1,'2024-09-01','2026-09-30')
    np.testing.assert_array_equal(a.loc[a.origin_date.lt('2025-06-01'),'center'],
                                  b.loc[b.origin_date.lt('2025-06-01'),'center'])

def residual_frame():
    f=panel(); f['center']=0.
    f=pd.concat([f,pd.DataFrame(np.tile(norm.ppf(ALL_TAUS)*.1,(len(f),1)),columns=ALL_COLUMNS)],axis=1)
    return f

def test_calibrated_distribution_and_risk_are_coherent():
    f=residual_frame(); fit=fit_calibrator(mature_before(f,'2025-03-01'))
    out=calibrate(f,fit,.5)
    assert np.isfinite(out).all()
    assert (np.diff(out,axis=1)>=0).all()
    risk=risk_readout(out)
    for name,value in risk.items():
        if name.startswith('prob_'): assert ((value>=0)&(value<=1)).all()
    assert (risk['es_loss_log_0.05']>=risk['var_loss_log_0.05']).all()
    # All candidate shapes share the residual-calibrated median.
    np.testing.assert_allclose(out[:,MEDIAN],calibrate(f,fit,1)[:,MEDIAN])

def test_test_outcomes_cannot_change_frozen_quantiles():
    f=residual_frame(); fit=fit_calibrator(mature_before(f,'2025-03-01'))
    a=calibrate(f,fit,1.)
    f['y']=999.
    np.testing.assert_array_equal(a,calibrate(f,fit,1.))

def test_unreleased_residuals_cannot_change_calibrator():
    f=residual_frame(); train=mature_before(f,'2025-03-01'); a=fit_calibrator(train)
    f.loc[f.label_available_date.gt('2025-03-01'),'y']=999
    assert a==fit_calibrator(mature_before(f,'2025-03-01'))

def test_saved_system_is_invariant_to_test_outcomes(monkeypatch):
    """Integration check across member adapters, frozen weights and risk output."""
    import json
    import pytest
    from m3_ml.fusion import run
    if not (run.OUT/'selected.json').exists(): pytest.skip('Run fusion selection first')
    spec=json.loads((run.OUT/'specification.json').read_text())
    model=json.loads((run.OUT/'selected.json').read_text())
    first=run.final_forecasts(spec,model)
    table=run.load_table()
    table.loc[table.date.gt('2026-09-30'),'target_wti']+=1000
    monkeypatch.setattr(run,'load_table',lambda:table.copy())
    second=run.final_forecasts(spec,model)
    columns=['forecast',*ALL_COLUMNS,'prob_up','prob_drop_5pct']
    np.testing.assert_array_equal(first[columns],second[columns])
    assert not first.actual.fillna(0).equals(second.actual.fillna(0))
