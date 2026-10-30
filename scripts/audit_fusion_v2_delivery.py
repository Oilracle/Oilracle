"""Verify delivery identity, arithmetic, information cutoffs and v1 preservation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from m3_ml.fusion_v2.prepare import ROOT,OUT,OLD
from m3_ml.fusion_v2.run import seal_inputs
from m3_ml.fusion_v2.model import CHANNELS,MEMBERS,ALL_COLUMNS,MEDIAN


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selected=json.loads((OUT/'selected.json').read_text())
    test_audit=json.loads((OUT/'test_audit.json').read_text())
    presentation=json.loads((OUT/'presentation_data.json').read_text())
    manifest=json.loads((ROOT/'m4_reports/final_strategy_manifest.json').read_text())
    checks={}
    assert selected['input_sha256']==seal_inputs()
    assert sha(OUT/'selected.json')==test_audit['selected_sha256']==presentation['selected_sha256']==manifest['selected_sha256']
    assert selected['version']==presentation['version']==manifest['active_version']=='horizon_rw_v2'
    checks['single_active_identity_and_sealed_inputs']=True
    f=pd.read_csv(OUT/'test_forecasts.csv')
    assert len(f)==87 and not f.duplicated(['mode','origin_date','horizon']).any()
    assert np.isfinite(f[ALL_COLUMNS]).all().all()
    np.testing.assert_array_equal(f['q_0.500'],f.center)
    for h,g in f.groupby('horizon'):
        w=selected['point_by_horizon'][str(int(h))]['weights']
        assert all(v>=0 for v in w.values())
        np.testing.assert_allclose(sum(w.values()),1.,atol=1e-12)
        prices=np.column_stack([g.origin_price,g.origin_price.to_numpy()[:,None]*np.exp(g[list(MEMBERS)])])
        expected=prices@np.array([w[m] for m in CHANNELS])
        np.testing.assert_allclose(g.forecast,expected,rtol=1e-12,atol=1e-10)
        assert (g.forecast>=prices.min(axis=1)-1e-10).all() and (g.forecast<=prices.max(axis=1)+1e-10).all()
    checks['arithmetic_price_blend_and_preserved_median']=True
    fixed=f[f['mode'].eq('fixed_origin')&f.actual.notna()]
    ratio=np.mean((fixed.forecast-fixed.actual)**2)**.5/np.mean((fixed.origin_price-fixed.actual)**2)**.5
    np.testing.assert_allclose(ratio,test_audit['fixed_path_metrics']['rmse_vs_rw'],rtol=1e-12)
    assert len(fixed)==17
    for h,n in [(1,17),(5,13),(20,0),(60,0)]:
        assert f.loc[f['mode'].eq('daily_update')&f.horizon.eq(h),'actual'].notna().sum()==n
    checks['evaluation_counts_and_full_path_metric']=True
    records=pd.DataFrame(presentation['forecasts'])
    np.testing.assert_allclose(records.forecast,f.forecast,rtol=1e-9,atol=1e-8)
    checks['presentation_predictions_match_sealed_csv']=True
    crossfit=json.loads((OUT/'crossfit_audit.json').read_text())
    for section in ('point_candidates','point'):
        for row in crossfit[section]:
            assert row['latest_label_release']<=row['cutoff']
            if 'latest_selection_release' in row:assert row['latest_selection_release']<=row['cutoff']
    for methods in crossfit['risk_calibration'].values():
        for row in methods:
            assert all(v['latest_label_release']<=row['cutoff'] for v in row['calibration'].values())
    for row in crossfit['risk_outer']:assert row['latest_selection_release']<=row['cutoff']
    historical=pd.read_csv(OUT/'validation_member_panel.csv')
    assert historical.label_available_date.max()<='2026-09-30' and historical.target_date.max()<='2026-09-30'
    checks['all_meta_fit_and_selection_labels_mature']=True
    expected={'selected.json':'72f7cf8302ed1b9afbe22c72bb8f7b8729cf274c7e8e8581bc9b3620236b97de',
              'test_forecasts.csv':'bbbe159d959ae300ac77724e10fa81d2687f7d198ac4e77863956334662ad290'}
    assert all(sha(OLD/name)==value for name,value in expected.items())
    checks['unfavorable_v1_results_preserved']=True
    report=dict(status='passed',checks=checks,selected_sha256=sha(OUT/'selected.json'),
                test_forecasts_sha256=sha(OUT/'test_forecasts.csv'),version=selected['version'],
                retained_point_members={m:[int(h) for h,v in selected['point_by_horizon'].items() if v['weights'][m]>1e-8] for m in CHANNELS},
                note='No parameter or candidate changes after October evaluation. ML direct-point weights are zero; all modules were evaluated.')
    (OUT/'delivery_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
