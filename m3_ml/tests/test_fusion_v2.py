import json
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from m3_ml.fusion_v2 import model as m


def panel():
    rng=np.random.default_rng(5210)
    dates=pd.bdate_range('2023-09-01','2026-09-30');parts=[]
    for h in (1,5,20,60):
        anchor=rng.uniform(50,120,len(dates));y=rng.normal(0,.025*np.sqrt(h),len(dates))
        f=pd.DataFrame(dict(origin_date=dates,horizon=h,origin_price=anchor,y=y,
                            label_available_date=dates+pd.offsets.BDay(h+8)))
        for i,member in enumerate(m.MEMBERS):f[member]=.15*y+rng.normal(0,.02*(i+1)*np.sqrt(h),len(dates))
        parts.append(f)
    return pd.concat(parts,ignore_index=True)


def specification():
    return dict(fit_end='2026-09-30',point_candidate_start='2024-03-01',point_validation_start='2024-09-01',ridge_candidates=[0.,.1,1.])


def test_exact_price_blend_and_zero_weights():
    f=panel().query('horizon == 1').copy()
    x,_=m.price_design(f);weights=np.array([.3,.4,.2,.1,0.])
    f['y']=np.log1p((x@weights)/f.origin_price)
    learned=m.fit_weights(f,0.)
    np.testing.assert_allclose(learned,weights,atol=2e-5)
    np.testing.assert_allclose(m.center_from_weights(f,learned),f.y,atol=2e-6)
    assert learned[-1]<1e-5 and learned.min()>=0
    np.testing.assert_allclose(learned.sum(),1,atol=1e-12)


def test_rw_is_explicit_available_baseline():
    f=panel().query('horizon == 1')
    np.testing.assert_array_equal(m.fit_weights(f,'rw'),[1.,0.,0.,0.,0.])
    np.testing.assert_array_equal(m.center_from_weights(f,m.fit_weights(f,'rw')),np.zeros(len(f)))


def test_future_labels_do_not_change_past_selection_or_weights():
    f=panel();spec=specification()
    candidates,_=m.point_candidates(f,spec);a,_=m.adaptive_points(f,candidates,spec)
    changed=f.copy();changed.loc[changed.label_available_date.gt('2025-03-01'),'y']+=.7
    new_candidates,_=m.point_candidates(changed,spec);b,_=m.adaptive_points(changed,new_candidates,spec)
    before=a.origin_date.lt('2025-06-01')
    np.testing.assert_array_equal(a.loc[before,'center'],b.loc[before,'center'])
    np.testing.assert_array_equal(a.loc[before,'selected_penalty'],b.loc[before,'selected_penalty'])


def test_risk_never_moves_selected_center_and_ignores_test_outcomes():
    f=panel();f['center']=.01
    f=pd.concat([f,pd.DataFrame(np.tile(norm.ppf(m.ALL_TAUS)*.12,(len(f),1)),columns=m.ALL_COLUMNS)],axis=1)
    calibration=m.fit_calibrator(m.mature_before(f,'2025-03-01'))
    for mixture in (0.,.5,1.):
        for width in (.75,1.,1.25):
            q=m.calibrate(f,calibration,mixture,width)
            np.testing.assert_array_equal(q[:,m.MEDIAN],f.center)
            assert np.isfinite(q).all() and (np.diff(q,axis=1)>=0).all()
            changed=f.assign(y=999.)
            np.testing.assert_array_equal(q,m.calibrate(changed,calibration,mixture,width))
            for key,value in m.risk_readout(q).items():
                if key.startswith('prob_'):assert ((value>=0)&(value<=1)).all()
    changed=f.copy();changed.loc[changed.label_available_date.gt('2025-03-01'),'y']=999.
    assert calibration==m.fit_calibrator(m.mature_before(changed,'2025-03-01'))


def test_risk_candidate_choice_excludes_unpublished_losses():
    f=panel().query('horizon == 1')[['origin_date','horizon','label_available_date']]
    losses=pd.concat([f.assign(method='a',mixture=0.,width=1.,loss=1.,hs_crps=2.),
                      f.assign(method='b',mixture=.5,width=1.25,loss=1.2,hs_crps=2.)])
    before=m.choose_risk(losses,'2025-03-01')
    losses.loc[losses.label_available_date.gt('2025-03-01'),'loss']=1e10
    assert before==m.choose_risk(losses,'2025-03-01')


def test_saved_historical_fits_have_valid_information_cutoffs():
    from m3_ml.fusion_v2.prepare import OUT
    if not (OUT/'lstm_refit_audit.json').exists():pytest.skip('Historical replay not generated')
    for row in json.loads((OUT/'lstm_refit_audit.json').read_text()):
        assert row['latest_label_release']<=row['cutoff'] and row['n_train']>0
    for member in ('m1','m2','lstm'):
        f=pd.read_csv(OUT/f'{member}_validation.csv',parse_dates=['origin_date','base_fit_cutoff'])
        assert f.base_fit_cutoff.le(f.origin_date).all()
        assert f.base_fit_cutoff.nunique()==13


def test_october_target_mutation_cannot_change_sealed_forecasts(monkeypatch):
    from m3_ml.fusion_v2 import run
    from m3_ml.fusion import run as original
    if not (run.OUT/'selected.json').exists():pytest.skip('Selection has not finished')
    spec=json.loads((run.OUT/'specification.json').read_text())
    selected=json.loads((run.OUT/'selected.json').read_text())
    before=run.final_forecasts(spec,selected)
    table=original.load_table()
    table.loc[table.date.gt('2026-09-30'),'target_wti']+=1000
    monkeypatch.setattr(original,'load_table',lambda:table.copy())
    after=run.final_forecasts(spec,selected)
    np.testing.assert_array_equal(before[['forecast',*m.ALL_COLUMNS,'prob_up']],after[['forecast',*m.ALL_COLUMNS,'prob_up']])
    assert not before.actual.fillna(0).equals(after.actual.fillna(0))


def test_october_rows_cannot_change_historical_training_panel(monkeypatch):
    from m3_ml.fusion_v2 import run
    if not (run.OUT/'lstm_validation.csv').exists():pytest.skip('Historical replay not generated')
    before=run.validation_frame();table=run.load_table()
    numeric=table.select_dtypes(include='number').columns
    table.loc[table.date.gt('2026-09-30'),numeric]=9999.
    monkeypatch.setattr(run,'load_table',lambda:table.copy())
    pd.testing.assert_frame_equal(before,run.validation_frame())
