"""Render the user-selected daily rolling product from sealed v3 predictions."""
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from m3_ml.fusion_v3.data import OUT,ROOT,BASE,KEY,MEMBERS,sha,write_json,feature_frame
from m3_ml.fusion_v3.run import seal_inputs,predict_points
from m3_ml.fusion.run import october_frame
from m3_ml.common.data import load_table
from m3_ml.common.protocol import HORIZONS

DELIVERY=OUT/'rolling'
MODE='daily_update'


def records(frame):return json.loads(frame.to_json(orient='records',date_format='iso'))


def contribution_audit(spec,model):
    """Local input sensitivities, not re-optimized out-of-sample ablations."""
    native=october_frame(spec)
    frame,features=feature_frame(native[native['mode'].eq(MODE)],load_table())
    models=joblib.load(OUT/'models/point_models.joblib')
    baseline=predict_points(frame,features,spec,model['point_weights'],models)
    rows=[]
    for member in MEMBERS:
        changed=frame.copy();changed[member+'_price']+=.01
        x={k:v.copy() for k,v in features.items()}
        for value in x.values():value[member+'_minus_futures']+=.01
        forecast=predict_points(changed,x,spec,model['point_weights'],models)
        for date,h,value in zip(frame.origin_date,frame.horizon,(forecast-baseline)/.01):
            rows.append(dict(origin_date=date,horizon=int(h),member=member,local_price_sensitivity=float(value)))
    sensitivity=pd.DataFrame(rows)
    sensitivity.to_csv(DELIVERY/'module_input_sensitivity.csv',index=False)
    definitions={d['id']:d for d in spec['candidates']};rows=[]
    for h,weights in model['point_weights'].items():
        if int(h) not in HORIZONS:continue
        for cid,weight in weights.items():
            definition=definitions[cid]
            rows.append(dict(horizon=int(h),candidate=cid,weight=weight,**{k:v for k,v in definition.items() if k!='id'}))
    weights=pd.DataFrame(rows);weights.to_csv(DELIVERY/'candidate_weights.csv',index=False)
    return sensitivity,weights


def main():
    model=json.loads((OUT/'selected.json').read_text());spec=json.loads((OUT/'specification.json').read_text())
    audit=json.loads((OUT/'test_audit.json').read_text());digest=sha(OUT/'selected.json')
    assert model['input_sha256']==seal_inputs() and digest==audit['selected_sha256']
    DELIVERY.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(OUT/'test_forecasts.csv',parse_dates=['origin_date','target_date']).loc[lambda x:x['mode'].eq(MODE)].copy()
    test=pd.read_csv(OUT/'test_metrics.csv').loc[lambda x:x['mode'].eq(MODE)].copy()
    history=pd.read_csv(OUT/'historical_point_metrics.csv').loc[lambda x:x.horizon.isin(HORIZONS)].copy()
    risks=pd.read_csv(OUT/'historical_risk_comparison.csv').loc[lambda x:x.horizon.isin(HORIZONS)].copy()
    uncertainty=json.loads((OUT/'historical_inference.json').read_text())
    uncertainty={k:v for k,v in uncertainty.items() if k not in ('primary','comparisons')}
    uncertainty['comparisons']=[r for r in model['primary_historical']['comparisons'] if r['scope'] in {f'h{h}' for h in HORIZONS}]
    uncertainty['scope']='Separate rolling horizons 1, 5, 20, 60; historical origins from 2025-09 onward.'
    assert len(f)==68 and set(f.horizon)==set(HORIZONS) and len(test)==4
    f.to_csv(DELIVERY/'forecasts.csv',index=False)
    test.to_csv(DELIVERY/'metrics.csv',index=False)
    history.to_csv(DELIVERY/'historical_point_metrics.csv',index=False)
    risks.to_csv(DELIVERY/'historical_risk_metrics.csv',index=False)
    write_json(DELIVERY/'historical_inference.json',uncertainty)
    sensitivity,weights=contribution_audit(spec,model)
    daily=test[test.horizon.isin([1,5])].set_index('horizon')
    old=pd.read_csv(BASE/'test_metrics.csv').loc[lambda x:x['mode'].eq(MODE)].set_index('horizon')
    figdir=DELIVERY/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
    blue='#2563eb';orange='#dc6b20';gray='#64748b';green='#0d9488';purple='#7c3aed'
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.1),layout='constrained')
    for ax,h in zip(axes,(1,5)):
        g=f[f.horizon.eq(h)&f.actual.notna()].sort_values('target_date')
        ax.fill_between(g.target_date,g.lower90,g.upper90,color=blue,alpha=.12,label='90% model interval')
        for col,label,color,style in [('forecast','Integrated median',blue,'-'),('origin_price','Published-spot RW',gray,'--'),
                                     ('futures_price','Lagged-futures carry',green,':'),('actual','Observed WTI',orange,'-')]:
            ax.plot(g.target_date,g[col],color=color,lw=2,ls=style,label=label)
        ax.set_title(f'Rolling {h}-day forecast | n={len(g)}',loc='left');ax.set_ylabel('WTI, USD / barrel')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8,loc='upper left')
        ax.set_xlabel('Target date')
    fig.suptitle('Daily rolling WTI forecasts | parameters frozen on 30 September',fontsize=14,fontweight='bold')
    fig.savefig(figdir/'october_forecasts.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    entries=[r for r in uncertainty['comparisons'] if r['reference']=='rw' and r['block_length']==60]
    for i,row in enumerate(entries):
        lo,hi=row['improvement_pct_ci95'];value=row['improvement_pct']
        axes[0].errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color=blue if value>0 else orange,capsize=5,lw=2)
        axes[0].text(hi+1,i,f'{value:+.1f}%',va='center',fontsize=9)
    axes[0].set_yticks(range(len(entries)),[f"{r['scope'][1:]} {'day' if r['scope']=='h1' else 'days'} | n={r['n_origins']}" for r in entries]);axes[0].invert_yaxis()
    low=min(r['improvement_pct_ci95'][0] for r in entries);high=max(r['improvement_pct_ci95'][1] for r in entries)
    axes[0].axvline(0,color=gray,ls='--');axes[0].set_xlim(min(0,low)-3,high+10)
    axes[0].set_xlabel('RMSE reduction vs published-spot RW (%)')
    axes[0].set_title('Rolling validation: September 2025 onward',loc='left',fontsize=10)
    axes[0].text(.01,-.23,'95% paired block intervals (60 origins); conditional historical evidence',transform=axes[0].transAxes,fontsize=8)
    values=[daily.loc[1,'rmse_vs_rw'],daily.loc[5,'rmse_vs_rw']]
    ref=[daily.loc[1,'futures_rmse_usd']/daily.loc[1,'rw_rmse_usd'],
         daily.loc[5,'futures_rmse_usd']/daily.loc[5,'rw_rmse_usd']]
    previous=[old.loc[h,'rmse_vs_rw'] for h in (1,5)];x=np.arange(2)
    axes[1].bar(x-.24,values,.23,color=blue,label='Integrated v3')
    axes[1].bar(x,ref,.23,color=green,label='Lagged-futures carry')
    axes[1].bar(x+.24,previous,.23,color=purple,label='Previous fusion (v2)')
    for i,value in enumerate(values):axes[1].text(i-.24,value+.025,f'{value:.3f}',ha='center',fontsize=9)
    axes[1].axhline(1,color=gray,ls='--',label='Published-spot RW = 1')
    axes[1].set_xticks(x,['Rolling 1 day\nn=17','Rolling 5 days\nn=13'])
    axes[1].set_ylim(0,max(ref+values)*1.40);axes[1].set_ylabel('USD price RMSE / RW RMSE')
    axes[1].set_title('October rolling forecasts: descriptive evaluation',loc='left',fontsize=10)
    axes[1].legend(frameon=False,fontsize=8,loc='upper left',ncol=2);axes[1].grid(axis='y',alpha=.15)
    fig.suptitle('One daily rolling forecast system | history selects, October evaluates',fontsize=13,fontweight='bold')
    fig.savefig(figdir/'strategy_summary.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12.8,4.5),layout='constrained');keys=[1,5,20,60]
    h=history[history.period.eq('later_history')].set_index('horizon').loc[keys]
    for col,label,color in [('v3_rmse_usd','Integrated v3',blue),('v2_rmse_usd','Previous fusion',gray),('futures_rmse_usd','Lagged-futures carry',green)]:
        axes[0].plot(range(4),h[col]/h.rw_rmse_usd,marker='o',label=label,color=color,lw=2)
    axes[0].axhline(1,color=gray,ls='--');axes[0].set_ylabel('USD price RMSE / RW RMSE')
    axes[0].set_title('Point forecast: September 2025 onward',loc='left',fontsize=10)
    for period,label,color in [('development_selection','Development-selected calibration',blue),('forward_selection','Forward risk selection',orange)]:
        r=risks[risks.period.eq(period)].set_index('horizon').loc[keys]
        axes[1].plot(range(4),r.crps_vs_hs,marker='o',color=color,label=label)
    axes[1].axhline(1,color=gray,ls='--');axes[1].set_ylabel('CRPS / historical-simulation CRPS')
    axes[1].set_title('Risk forecast: 60-day weakness remains',loc='left',fontsize=10)
    for ax in axes:
        ax.set_xticks(range(4),keys);ax.set_xlabel('Horizon (quoted rows)');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Historical checks | current-vintage data and conditional base specifications',fontsize=13,fontweight='bold')
    fig.savefig(figdir/'historical_validation.png',dpi=170);plt.close(fig)
    scope=dict(forecast_mode=MODE,horizons=list(HORIZONS),scope_requested_on='2026-10-30',
        parameter_policy='Model coefficients, weights, calibration and preprocessing remain frozen as of 2026-09-30.',
        input_policy='At each daily origin, update only observations already available under the publication rules.',
        scope_change='User narrowed the delivery scope after reviewing October; this is not a new independent test or a new model selection.',
        archived_evaluation=(OUT/'test_audit.json').relative_to(ROOT).as_posix())
    write_json(DELIVERY/'delivery_scope.json',scope)
    write_json(DELIVERY/'protocol.json',dict(protocol_id=model['protocol_id'],fit_end=model['fit_end'],mode=MODE,horizons=list(HORIZONS)))
    limitations=[spec['base_selection_limitation'],spec['october_status'],model['source_audit']['limitation'],
        'Historical rolling h1 is better than lagged-futures carry; h5/h20/h60 remain worse than that stronger benchmark.',
        'October rolling h1/h5 improve vs RW but do not beat v2; parameters were not changed using October outcomes.',
        'Only 17/13 realized October outcomes at h1/h5. h20/h60 have no realized October outcomes; report no test error for them.',
        'The 60-day point and risk forecasts remain weak; 90% interval coverage is not guaranteed.',
        'All four point modules enter the fitted residual/combination system. Local sensitivity measures use, not beneficial out-of-sample contribution.',
        'Historical confidence intervals do not correct for all prior research or full-window base-model specification selection.']
    roles={'m1':'Actual VECM forecast; residual-learning base, feature and convex candidate.',
           'm2':'Actual monthly STL forecast; residual-learning base, feature and convex candidate, plus calendar/state interactions.',
           'lstm':'Actual three-seed sequence forecast enters residual features and convex candidates.',
           'ml_point':'Existing selected ML prediction plus newly fitted Ridge, Elastic Net and LightGBM residual corrections.',
           'ml_distribution':'Historically selected ML scale/shape, fused-error calibration and 63 coherent quantiles; median unchanged.',
           'futures':'Existing CL=F raw Close, delayed two US federal business days; separate stronger benchmark.'}
    payload=dict(name=model['name'],version=model['version'],protocol_id=model['protocol_id'],fit_end=model['fit_end'],
        selected_sha256=digest,active_output=DELIVERY.relative_to(ROOT).as_posix(),point_meaning=model['point_meaning'],
        delivery_scope=scope,forecast_mode=MODE,horizons=list(HORIZONS),
        point_weights={str(h):model['point_weights'][str(h)] for h in HORIZONS},
        point_rules={str(h):model['point_rules'][str(h)] for h in HORIZONS},
        risk_by_horizon={str(h):model['risk'][str(h)] for h in HORIZONS},
        candidate_definitions=spec['candidates'],module_roles=roles,module_sensitivity=records(sensitivity),
        test_metrics=records(test),previous_rolling_metrics=records(old.reset_index()),forecasts=records(f),
        historical_point_metrics=records(history),historical_risk_metrics=records(risks),historical_inference=uncertainty,
        source_audit=model['source_audit'],limitations=limitations)
    write_json(DELIVERY/'presentation_data.json',payload)
    manifest=dict(active_version=model['version'],name=model['name'],output=DELIVERY.relative_to(ROOT).as_posix(),
        model_output=OUT.relative_to(ROOT).as_posix(),forecast_mode=MODE,horizons=list(HORIZONS),delivery_profile='daily_rolling',
        presentation_data=(DELIVERY/'presentation_data.json').relative_to(ROOT).as_posix(),selected_sha256=digest,
        forecasts=(DELIVERY/'forecasts.csv').relative_to(ROOT).as_posix(),metrics=(DELIVERY/'metrics.csv').relative_to(ROOT).as_posix(),
        delivery_scope=(DELIVERY/'delivery_scope.json').relative_to(ROOT).as_posix(),
        selection_reason='Pre-October later-history primary loss: 14.1% RMSE improvement over v2; not selected on October results.',
        superseded=['m3_ml/fusion/outputs/frozen_october_2026 (audit only)','m3_ml/fusion_v2/outputs/frozen_october_2026 (audit only)'])
    write_json(ROOT/'m4_reports/final_strategy_manifest.json',manifest)
    print('Active daily rolling dataset:',DELIVERY/'presentation_data.json',flush=True)
    print(test[['horizon','n','rmse_usd','rmse_vs_rw','rmse_vs_futures']].to_string(index=False),flush=True)


if __name__=='__main__':main()
