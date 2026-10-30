"""Publish the sealed v4 rolling forecasts and their historical evidence."""
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from m3_ml.common.data import load_table
from m3_ml.fusion.run import october_frame
from m3_ml.fusion_v3 import data as old_data
from m3_ml.fusion_v4.data import OUT,OLD,ROOT,KEY,HORIZONS,sha,write_json,features
from m3_ml.fusion_v4.run import inputs,protected,predict

DELIVERY=OUT/'rolling'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def records(frame):return json.loads(frame.to_json(orient='records',date_format='iso'))


def contribution_audit(spec,model,forecasts):
    """Measure input sensitivity; this is not an out-of-sample benefit estimate."""
    table=load_table();old_spec=read(OLD/'specification.json');old_model=read(OLD/'selected.json')
    native=october_frame(old_spec)
    frame,old_X=old_data.feature_frame(native[native['mode'].eq('daily_update')],table)
    market,X=features(table)
    pos=pd.Series(table.index,index=table.date).reindex(frame.origin_date).to_numpy().astype(int)
    frame['feature_row']=pos;frame['implied_vol']=table.oil_implied_volatility.iloc[pos].to_numpy()/100
    frame['scale']=np.maximum(1.,frame.futures_price*frame.futures_vol*np.sqrt(frame.horizon+2))
    for column in ('basis20','basis60','basis250'):frame[column]=market[column].iloc[pos].to_numpy()
    fitted=joblib.load(OUT/'models/upgrade.joblib');old_fitted=joblib.load(OLD/'models/point_models.joblib')
    baseline=predict(frame,X,old_X,spec,model,old_spec,old_model,fitted,old_fitted)[0]
    expected=forecasts.set_index(KEY).forecast.reindex(pd.MultiIndex.from_frame(frame[KEY]))
    np.testing.assert_allclose(baseline,expected,rtol=1e-12,atol=1e-12)
    rows=[]
    for member in old_data.MEMBERS:
        changed=frame.copy();changed[member+'_price']+=.01
        changed_X={k:v.copy() for k,v in old_X.items()}
        for value in changed_X.values():value[member+'_minus_futures']+=.01
        value=predict(changed,X,changed_X,spec,model,old_spec,old_model,fitted,old_fitted)[0]
        for date,h,s in zip(frame.origin_date,frame.horizon,(value-baseline)/.01):
            rows.append(dict(origin_date=date,horizon=int(h),member=member,local_price_sensitivity=float(s)))
    sensitivity=pd.DataFrame(rows);sensitivity.to_csv(DELIVERY/'module_input_sensitivity.csv',index=False)
    rows=[]
    for h,weights in model['family_weights'].items():
        beta=model['guard'][h]['upgrade_weight']
        rows.append(dict(horizon=int(h),component='incumbent_v3',within_proposal_weight=np.nan,
                         effective_weight=1-beta,candidate='sealed_v3'))
        for family,weight in weights.items():
            rows.append(dict(horizon=int(h),component=family,within_proposal_weight=weight,
                             effective_weight=beta*weight,candidate=model['family_choices'][h].get(family,family)))
    weights=pd.DataFrame(rows);weights.to_csv(DELIVERY/'component_weights.csv',index=False)
    return sensitivity,weights


def figures(f,history,risks,old_risks,inference,folder=None,current='v4',previous='v3',previous_risk_period='forward_selection'):
    folder=folder or DELIVERY/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
    blue='#2563eb';orange='#dc6b20';gray='#64748b';green='#0d9488'
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.1),layout='constrained')
    for ax,h in zip(axes,(1,5)):
        g=f[f.horizon.eq(h)&f.actual.notna()].sort_values('target_date')
        ax.fill_between(g.target_date,g.lower90,g.upper90,color=blue,alpha=.12,label='90% model interval')
        for column,label,color,style in [('forecast',f'Integrated {current} median',blue,'-'),('origin_price','Published-spot RW',gray,'--'),
                                        ('futures_price','Lagged-futures carry',green,':'),('actual','Observed WTI',orange,'-')]:
            ax.plot(g.target_date,g[column],color=color,lw=2,ls=style,label=label)
        ax.set_title(f'Rolling {h}-day forecast | n={len(g)}',loc='left');ax.set_ylabel('WTI, USD / barrel')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8,loc='upper left');ax.set_xlabel('Target date')
    fig.suptitle('Daily rolling WTI forecasts | parameters frozen on 30 September',fontsize=14,fontweight='bold')
    fig.savefig(folder/'october_forecasts.png',dpi=170);plt.close(fig)

    hist=history[history.period.eq('later_history')].set_index('horizon').loc[list(HORIZONS)]
    fig,axes=plt.subplots(1,2,figsize=(13.5,4.9),layout='constrained')
    for column,label,color in [(current+'_rmse_usd',f'Integrated {current}',blue),(previous+'_rmse_usd',f'Previous {previous}',gray),('futures_rmse_usd','Lagged-futures carry',green)]:
        axes[0].plot(range(4),hist[column]/hist.rw_rmse_usd,marker='o',label=label,color=color,lw=2)
    axes[0].axhline(1,color=gray,ls='--');axes[0].set_xticks(range(4),[f"{h} {'day' if h==1 else 'days'}\nn={n}" for h,n in zip(HORIZONS,hist.n)])
    axes[0].set_ylabel('USD price RMSE / published-spot RW RMSE');axes[0].set_title('Each rolling horizon',loc='left')
    axes[0].legend(frameon=False,fontsize=8);axes[0].grid(axis='y',alpha=.15)
    entries=[r for r in inference['comparisons'] if r['scope']=='primary_four_horizons' and r['block_length']==60]
    labels={previous:f'vs previous {previous}','rw':'vs published-spot RW','futures':'vs lagged futures'}
    for i,row in enumerate(entries):
        lo,hi=row['improvement_pct_ci95'];value=row['improvement_pct']
        axes[1].errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',color=blue,capsize=5,lw=2)
        axes[1].text(hi+.4,i,f'{value:+.1f}%',va='center',fontsize=9)
    axes[1].set_yticks(range(len(entries)),[labels[r['reference']] for r in entries]);axes[1].invert_yaxis()
    axes[1].axvline(0,color=gray,ls='--');axes[1].set_xlim(-8,21)
    axes[1].set_xlabel('RMSE reduction (%) | 95% block intervals')
    axes[1].set_title('Four-horizon primary score | 202 shared origins',loc='left',fontsize=10)
    fig.suptitle('Historical point validation | September 2025 onward',fontsize=14,fontweight='bold')
    fig.supxlabel(f'Conditional research evidence; 60-origin blocks. The incremental {current} gain is not robustly significant.',fontsize=9)
    fig.savefig(folder/'strategy_summary.png',dpi=170);plt.close(fig)

    risk=risks[risks.period.eq('forward')].set_index('horizon').loc[list(HORIZONS)]
    prior=old_risks[old_risks.period.eq(previous_risk_period)].set_index('horizon').loc[list(HORIZONS)]
    fig,axes=plt.subplots(1,2,figsize=(13.2,4.8),layout='constrained')
    for data,label,color in [(prior,f'Previous {previous}',gray),(risk,f'Integrated {current}',blue)]:
        axes[0].plot(range(4),data.crps_vs_hs,marker='o',color=color,label=label,lw=2)
        axes[1].plot(range(4),100*data.coverage90,marker='o',color=color,label=label,lw=2)
    axes[0].axhline(1,color=orange,ls='--',label='Historical simulation');axes[0].set_ylabel('CRPS / historical-simulation CRPS')
    axes[0].set_title('Distribution accuracy | lower is better',loc='left',fontsize=10)
    axes[1].axhline(90,color=orange,ls='--',label='Nominal 90%');axes[1].set_ylim(45,100)
    axes[1].set_ylabel('90% interval coverage (%)');axes[1].set_title('Coverage remains below nominal',loc='left',fontsize=10)
    for ax in axes:
        ax.set_xticks(range(4),[f"{h} {'day' if h==1 else 'days'}\nn={n}" for h,n in zip(HORIZONS,risk.n)])
        ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Forward risk selection | historical origins from March 2026',fontsize=14,fontweight='bold')
    gain=100*(1-risk.loc[60,'crps_vs_'+previous])
    fig.supxlabel(f'60-day CRPS improves {gain:.1f}% vs {previous}; interval coverage remains below 90%.',fontsize=9)
    fig.savefig(folder/'historical_validation.png',dpi=170);plt.close(fig)


def main():
    protected();model=read(OUT/'selected.json');spec=read(OUT/'specification.json');audit=read(OUT/'test_audit.json')
    digest=sha(OUT/'selected.json');assert model['input_sha256']==inputs() and digest==audit['selected_sha256']
    DELIVERY.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(OUT/'test_forecasts.csv',parse_dates=['origin_date','target_date'])
    test=pd.read_csv(OUT/'test_metrics.csv');history=pd.read_csv(OUT/'historical_point_metrics.csv')
    risks=pd.read_csv(OUT/'historical_risk_metrics.csv');old_risks=pd.read_csv(OLD/'historical_risk_comparison.csv')
    inference=read(OUT/'historical_inference.json');old_model=read(OLD/'selected.json')
    assert len(f)==68 and f['mode'].eq('daily_update').all() and len(test)==4
    for source,destination in [('test_forecasts.csv','forecasts.csv'),('test_metrics.csv','metrics.csv'),
                               ('historical_point_metrics.csv','historical_point_metrics.csv'),
                               ('historical_risk_metrics.csv','historical_risk_metrics.csv'),
                               ('historical_inference.json','historical_inference.json')]:
        (DELIVERY/destination).write_bytes((OUT/source).read_bytes())
    sensitivity,weights=contribution_audit(spec,model,f)
    figures(f,history,risks,old_risks,inference)
    scope=dict(forecast_mode='daily_update',horizons=list(HORIZONS),scope_requested_on='2026-10-30',
        parameter_policy='Coefficients, preprocessing, weights and risk calibration frozen as of 2026-09-30.',
        input_policy='Update each daily origin using only observations available under the existing publication rules.',
        selection_policy='Pre-October historical development only. No October ranking, tuning or fallback selection.',
        evidence_status=spec['october'],historical_research_amendments=['guard_design.json','risk_tail_design.json'])
    write_json(DELIVERY/'delivery_scope.json',scope)
    write_json(DELIVERY/'protocol.json',dict(protocol_id=model['protocol_id'],fit_end=model['fit_end'],mode='daily_update',horizons=list(HORIZONS)))
    limitations=[spec['source_limits'],spec['october'],
        'Guard and asymmetric-tail research amendments followed inspection of pre-October historical results; this is adaptive development.',
        'The 2.47% historical aggregate gain over v3 is not statistically significant; its 95% improvement interval is -3.54% to +5.73%.',
        'The primary historical score is 0.46% worse than lagged-futures carry, although better than published-spot RW.',
        'Forward risk coverage remains below 90% for every horizon; h5 coverage fell from 83.7% to 72.6%.',
        'Forward h60 CRPS improved 20.0% vs v3 but remains 5.6% worse than historical simulation.',
        'October has only 17/13 realized outcomes at h1/h5, and none at h20/h60; October h1/h5 do not beat v2.',
        'Local member-input sensitivity proves model use, not beneficial out-of-sample contribution of each member.',
        'Current-vintage data and full-window base-specification selection prevent a claim of a fully nested point-in-time blind test.']
    roles={'m1':'Actual VECM forecasts enter the inherited residual system and the new convex family pool.',
           'm2':'Actual monthly STL forecasts enter inherited residual models; long-history models add calendar and state interactions.',
           'lstm':'Actual three-seed LSTM forecasts enter inherited residual models and the new family pool.',
           'ml_point':'Existing point ML plus 2012-onward Ridge, Elastic Net, LightGBM and Huber residual candidates.',
           'ml_distribution':'Selected ML conditional scale/shape plus historical fused-error calibration; 63 monotone quantiles preserve the point median.',
           'futures':'Causal raw CL=F Close with a two-federal-business-day lag, also reported as a separate benchmark.'}
    previous={}
    for version,path in [('v3',OLD),('v2',old_data.BASE)]:
        previous[version]=records(pd.read_csv(path/'test_metrics.csv').loc[lambda x:x['mode'].eq('daily_update')])
    payload=dict(name=model['name'],version=model['version'],protocol_id=model['protocol_id'],fit_end=model['fit_end'],
        selected_sha256=digest,active_output=DELIVERY.relative_to(ROOT).as_posix(),point_meaning='Price median of the final calibrated predictive distribution.',
        delivery_scope=scope,forecast_mode='daily_update',horizons=list(HORIZONS),
        upgrade_guard=model['guard'],family_weights=model['family_weights'],family_choices=model['family_choices'],
        incumbent_point_weights={str(h):old_model['point_weights'][str(h)] for h in HORIZONS},
        risk_by_horizon=model['risk'],component_weights=records(weights),module_roles=roles,module_sensitivity=records(sensitivity),
        test_metrics=records(test),previous_rolling_metrics=previous,forecasts=records(f),
        historical_point_metrics=records(history),historical_risk_metrics=records(risks),historical_inference=inference,
        promotion_gate=model['historical_promotion_gate'],source_audit=old_model['source_audit'],limitations=limitations)
    write_json(DELIVERY/'presentation_data.json',payload)
    manifest=dict(active_version=model['version'],name=model['name'],output=DELIVERY.relative_to(ROOT).as_posix(),
        model_output=OUT.relative_to(ROOT).as_posix(),forecast_mode='daily_update',horizons=list(HORIZONS),delivery_profile='daily_rolling',
        presentation_data=(DELIVERY/'presentation_data.json').relative_to(ROOT).as_posix(),selected_sha256=digest,
        forecasts=(DELIVERY/'forecasts.csv').relative_to(ROOT).as_posix(),metrics=(DELIVERY/'metrics.csv').relative_to(ROOT).as_posix(),
        delivery_scope=(DELIVERY/'delivery_scope.json').relative_to(ROOT).as_posix(),
        selection_reason='Pre-October historical gate: primary four-horizon RMSE improves 2.465% over v3; maximum per-horizon degradation 0.458% (<2%). No October selection.',
        superseded=[f'm3_ml/{version}/outputs/frozen_october_2026 (audit only)' for version in ('fusion','fusion_v2','fusion_v3')])
    write_json(ROOT/'m4_reports/final_strategy_manifest.json',manifest)
    print('Active daily rolling dataset:',DELIVERY/'presentation_data.json',flush=True)
    print(test[['horizon','n','rmse_usd','rmse_vs_rw','rmse_vs_v3','rmse_vs_futures']].to_string(index=False),flush=True)


if __name__=='__main__':main()
