"""Read-only model diagnosis; write evidence separately, never change selected fits.

October is used for attribution and alignment checks, not model selection.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table, observable_price, label_available_dates
from m3_ml.common.features import build_features
from m3_ml.common.walkforward import forecast_origins
from m3_ml.fusion.model import MEMBERS, MEDIAN
from m3_ml.distribution.tails import ALL_COLUMNS, ALL_TAUS, MAIN
from m3_ml.distribution.quantiles import crps

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'m3_ml/fusion/outputs/frozen_october_2026'
OUT=BASE/'diagnostics'
KEY=['mode','origin_date','horizon']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(pred,actual):return float(np.sqrt(np.mean((np.asarray(pred)-np.asarray(actual))**2)))

def plot_diagnosis(fixed,historical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    ax=axes[0]
    for label,column,color,style in [('Observed WTI','actual','#111827','-'),('Random walk','origin_price','#64748b','--'),
                                    ('LSTM member','lstm_price','#9333ea','-'),('Fusion before calibration','center_price','#2563eb','-'),
                                    ('Final calibrated median','forecast','#ea580c','-')]:
        ax.plot(fixed.target_date,fixed[column],label=label,color=color,lw=2,ls=style)
    ax.set_title('Fixed-origin path: downward LSTM, then location shift',loc='left',fontsize=11)
    ax.set_ylabel('WTI, USD / barrel');ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'));ax.legend(frameon=False,fontsize=8,loc='lower left')
    ax.grid(axis='y',alpha=.15)
    ax=axes[1];h=pd.DataFrame(historical)
    h=h[h.stage.eq('same_risk_crossfit_period')&h.horizon.isin([1,5,20,60])]
    for i,hor in enumerate([1,5,20,60]):
        g=h[h.horizon.eq(hor)].set_index('component').relative_rmse
        ax.plot([g.center,g.calibrated],[i,i],color='#94a3b8',lw=3)
        ax.scatter(g.center,i,color='#2563eb',s=65,label='Before location calibration' if i==0 else None,zorder=3)
        ax.scatter(g.calibrated,i,color='#ea580c',s=65,label='After location calibration' if i==0 else None,zorder=3)
    ax.axvline(1,color='#64748b',ls='--',lw=1)
    ax.set_yticks(range(4),['1 day','5 days','20 days','60 days']);ax.invert_yaxis()
    ax.set_xlabel('Price RMSE / random-walk RMSE (lower is better)')
    ax.set_title('Same historical window, before October',loc='left',fontsize=11)
    ax.legend(frameon=False,loc='lower right',fontsize=9);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Diagnosis only | existing weights and parameters remain frozen',fontsize=15,fontweight='bold',x=.01,ha='left')
    fig.savefig(OUT/'diagnosis.png',dpi=170);plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    protected=[BASE/'selected.json',BASE/'test_forecasts.csv',ROOT/'data/processed/wti_daily.csv',
               ROOT/'data/processed/wti_daily.parquet',ROOT/'data/processed/wti_publication.csv',
               *list((ROOT/'m3_ml/lstm/outputs/frozen_october_2026/models').glob('*'))]
    before={str(p.relative_to(ROOT)):digest(p) for p in protected}
    checks={}
    table=load_table();dates=table.date;anchor=observable_price(table)
    csv=pd.read_csv(ROOT/'data/processed/wti_daily.csv',parse_dates=['date'])
    pd.testing.assert_frame_equal(csv,table,check_exact=False,rtol=1e-12,atol=1e-10)
    publication=pd.read_csv(ROOT/'data/processed/wti_publication.csv',parse_dates=['date','available_date'])
    np.testing.assert_array_equal(publication.date,dates)
    np.testing.assert_array_equal(publication.target_wti,table.target_wti)
    raw=pd.read_csv(ROOT/'data/raw/fred/wti_spot_daily.csv',parse_dates=['observation_date']).set_index('observation_date').DCOILWTICO
    np.testing.assert_allclose(pd.to_numeric(raw.reindex(dates)),table.target_wti,atol=1e-12)
    checks['raw_fred_csv_parquet_publication_values_match']=True
    independent=[];anchor_dates=[]
    for d in dates:
        known=publication[publication.date.le(d)&publication.available_date.le(d)]
        independent.append(known.target_wti.iloc[-1] if len(known) else np.nan)
        anchor_dates.append(known.date.iloc[-1] if len(known) else pd.NaT)
    known=np.isfinite(independent)
    np.testing.assert_array_equal(anchor[known],np.asarray(independent)[known])
    checks['independent_asof_anchor_matches']=True
    audit=table[['date','target_wti','wti_spot_last_published']].copy()
    audit['available_date']=publication.available_date;audit['evidence']=publication.evidence
    audit['canonical_anchor']=anchor;audit['anchor_quote_date']=anchor_dates
    audit[audit.date.ge('2026-09-01')].to_csv(OUT/'publication_audit.csv',index=False)
    f=pd.read_csv(BASE/'test_forecasts.csv',parse_dates=['origin_date','target_date'])
    spec=json.loads((BASE/'specification.json').read_text()); model=json.loads((BASE/'selected.json').read_text())
    index=pd.Series(table.index,index=dates);pos=f.origin_date.map(index).to_numpy();target=pos+f.horizon.to_numpy()
    np.testing.assert_array_equal(f.target_row,target)
    expected_dates=dates.reindex(target).to_numpy()
    assert pd.to_datetime(expected_dates).equals(pd.DatetimeIndex(f.target_date))
    expected_actual=table.target_wti.reindex(target).to_numpy()
    np.testing.assert_allclose(f.actual,expected_actual,equal_nan=True,atol=1e-10)
    checks['fusion_origin_plus_horizon_and_actual_match_raw_grid']=True
    paths={'m1':ROOT/'m1_time_series/outputs/frozen_october_2026/forecasts_2026-10-01_2026-10-23_asof_2026-10-23.csv',
           'm2':ROOT/'m2_seasonality/outputs/frozen_october_2026/final_predictions.csv',
           'lstm':ROOT/'m3_ml/lstm/outputs/frozen_october_2026/test_predictions.csv',
           'ml':ROOT/'m3_ml/tabular/outputs/frozen_october_2026/direct_forecasts.csv',
           'distribution':ROOT/'m3_ml/distribution/outputs/frozen_october_2026/test_quantiles.csv'}
    for name,path in paths.items():
        src=pd.read_csv(path,parse_dates=['origin_date','target_date'])
        method={'m1':spec['m1_method'],'m2':spec['m2_method'],'lstm':'LSTM','ml':'elastic_net','distribution':model['risk_method']}[name]
        src=src[src.method.eq(method)]
        if name=='m2':src=src[src.target_type.eq('price')]
        common=f[KEY+['target_row','target_date','actual','origin_price']].merge(src,on=KEY,how='left',validate='one_to_one',suffixes=('_fusion','_source'))
        base=1 if name=='m2' else 0
        np.testing.assert_array_equal(common.target_row_source-base,common.target_row_fusion)
        pd.testing.assert_series_equal(common.target_date_source,common.target_date_fusion,check_names=False)
        np.testing.assert_allclose(common.actual_source,common.actual_fusion,atol=1e-9,equal_nan=True)
        np.testing.assert_allclose(common.origin_price_source,common.origin_price_fusion,atol=1e-9)
        if name in MEMBERS:
            np.testing.assert_allclose(common.forecast,f.origin_price*np.exp(f[name]),atol=1e-9)
        checks[f'{name}_dates_rows_actuals_anchors_and_units']=True
    hist=pd.read_csv(BASE/'validation_member_panel.csv',parse_dates=['origin_date','target_date','label_available_date'])
    dist=pd.read_csv(paths['distribution'].with_name('validation_selected_quantiles.csv'),parse_dates=['origin_date','target_date'])
    dist=dist[dist.method.eq(model['risk_method'])]
    match=hist.merge(dist,on=['origin_date','horizon'],how='left',validate='one_to_one',suffixes=('_fusion','_source'))
    np.testing.assert_array_equal(match.target_row_fusion,match.target_row_source)
    np.testing.assert_allclose(match.origin_price_fusion,match.origin_price_source,atol=1e-9)
    np.testing.assert_allclose(match.y,match.actual_log_return,atol=1e-12)
    checks['historical_distribution_and_fusion_targets_match']=True
    # Exact numerical decomposition, with no alternative model selection.
    fixed=f[f['mode'].eq('fixed_origin')&f.actual.notna()].copy()
    for m in MEMBERS:fixed[m+'_price']=fixed.origin_price*np.exp(fixed[m])
    fixed['center_price']=fixed.origin_price*np.exp(fixed.center)
    fixed['calibration_usd']=fixed.forecast-fixed.center_price
    fixed.to_csv(OUT/'fixed_path_decomposition.csv',index=False)
    summaries=[]
    for mode,h in [('fixed_origin',None),('daily_update',1),('daily_update',5)]:
        g=f[f['mode'].eq(mode)&f.actual.notna()]
        if h:g=g[g.horizon.eq(h)]
        for m in ['rw',*MEMBERS,'center','calibrated']:
            pred=g.origin_price if m=='rw' else g.forecast if m=='calibrated' else g.origin_price*np.exp(g[m])
            summaries.append(dict(mode=mode,horizon='full_path' if h is None else h,component=m,n=len(g),rmse=rmse(pred,g.actual),bias=float(np.mean(pred-g.actual))))
    pd.DataFrame(summaries).to_csv(OUT/'component_metrics.csv',index=False)
    c=fixed.center.to_numpy();a=fixed.origin_price.to_numpy();e=a-fixed.actual.to_numpy();delta=fixed.forecast.to_numpy()-a
    factor=a*np.divide(np.expm1(c),c,out=np.ones(len(c)),where=abs(c)>1e-12)
    shifts={m:factor*model['weights'][m]*fixed[m].to_numpy() for m in MEMBERS}
    shifts['calibration']=fixed.calibration_usd.to_numpy()
    attribution={m:float(np.mean(d*(2*e+delta))) for m,d in shifts.items()}
    excess=float(np.mean((fixed.forecast-fixed.actual)**2)-np.mean(e**2))
    np.testing.assert_allclose(sum(attribution.values()),excess,atol=1e-10)
    # Historical-only evidence for the evaluation-objective mismatch.
    points=pd.read_csv(BASE/'point_crossfit.csv');risk=pd.read_csv(BASE/'validation_forecasts.csv')
    historical=[]
    for h,g in points.groupby('horizon'):
        for m in [*MEMBERS,'center']:
            historical.append(dict(stage='point_crossfit',horizon=h,component=m,n=len(g),
                                   relative_rmse=rmse(g[m],g.y)/rmse(np.zeros(len(g)),g.y),unit='log return'))
    for h,g in risk.groupby('horizon'):
        actual=g.origin_price*np.exp(g.y);baseline=rmse(g.origin_price,actual)
        for m,column in [('center','center'),('calibrated','q_0.500')]:
            historical.append(dict(stage='same_risk_crossfit_period',horizon=h,component=m,n=len(g),
                                   relative_rmse=rmse(g.origin_price*np.exp(g[column]),actual)/baseline,unit='USD'))
    pd.DataFrame(historical).to_csv(OUT/'historical_objective_diagnostics.csv',index=False)
    controls=[]
    for h,g in risk.groupby('horizon'):
        q=g[ALL_COLUMNS].to_numpy()
        keep=q-(q[:,MEDIAN]-g.center.to_numpy())[:,None]
        controls.append(dict(horizon=int(h),n=len(g),
                             preserve_center_crps_ratio=float(np.mean(crps(keep[:,MAIN],g.y))/np.mean(g.hs_crps)),
                             calibrated_crps_ratio=float(np.mean(crps(q[:,MAIN],g.y))/np.mean(g.hs_crps))))
    pd.DataFrame(controls).to_csv(OUT/'historical_calibration_control.csv',index=False)
    plot_diagnosis(fixed,historical)
    # Native checkpoint/estimator replay, with no fitting on October labels.
    from m3_ml.lstm.src import oil_lstm as lstm
    lstm.seed_everything(42)
    df,cols=lstm.read_data(ROOT/'data/processed/wti_daily.csv',ROOT/'data/processed/features.csv')
    model_dir=ROOT/'m3_ml/lstm/outputs/frozen_october_2026/models'
    replay=lstm.forecast_checkpoint_frame(df,cols,model_dir,modes=('fixed_origin',))
    replay=replay[replay.method.eq('LSTM')]
    native=pd.read_csv(paths['lstm']);native=native[native['mode'].eq('fixed_origin')&native.method.eq('LSTM')]
    np.testing.assert_allclose(replay.forecast,native.forecast,atol=3e-5,rtol=0)
    mutated=df.copy();mutated.loc[mutated.date.gt('2026-09-30'),cols]=9999.
    future=lstm.forecast_checkpoint_frame(mutated,cols,model_dir,modes=('fixed_origin',))
    np.testing.assert_array_equal(replay.forecast,future.loc[future.method.eq('LSTM'),'forecast'])
    checks['native_lstm_checkpoint_reproduces_all_60_steps']=True
    checks['lstm_fixed_path_invariant_to_october_predictor_mutation']=True
    from m3_ml.tabular.run_direct import frozen_forecast
    from m3_ml.tabular.models import MODELS
    from m3_ml.common.data import targets
    X=build_features(table);Y=targets(table,tuple(int(h) for h in spec['ml']))
    fixed_origin=forecast_origins(dates,'fixed_origin')
    for horizon,entry in spec['ml'].items():
        h=int(horizon)
        assert entry['variant']=='direct' and entry['method']=='elastic_net'
        prediction=frozen_forecast(MODELS[entry['method']][0],entry['params'],X,Y[h],dates,h,fixed_origin)
        emitted=f[f['mode'].eq('fixed_origin')&f.horizon.eq(h)].ml.to_numpy()
        np.testing.assert_allclose(prediction,emitted,atol=1e-10,rtol=0)
    checks['native_ml_final_refit_reproduces_all_19_horizons']=True
    after={str(p.relative_to(ROOT)):digest(p) for p in protected}
    assert before==after
    report=dict(status='diagnosis_completed_no_retuning',checks=checks,protected_files_unchanged=True,
                fixed_path_excess_mse=excess,exact_excess_mse_attribution=attribution,
                historical_key_horizon_crps_control=pd.DataFrame(controls).query('horizon in [1,5,20,60]')[['preserve_center_crps_ratio','calibrated_crps_ratio']].mean().to_dict(),
                note='Attribution uses the original frozen October forecasts. Historical checks use only pre-October outcomes. No parameter/weight or publication-policy changes.',protected_sha256=after)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2),flush=True)

if __name__=='__main__':main()
