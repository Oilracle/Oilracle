"""Read the sealed v2 system and render the sole active presentation dataset."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from m3_ml.fusion_v2.prepare import OUT,ROOT
from m3_ml.fusion_v2.run import seal_inputs
from m3_ml.fusion_v2.model import CHANNELS,ALL_COLUMNS,MAIN
from m3_ml.distribution.quantiles import crps


def records(frame):return json.loads(frame.to_json(orient='records',date_format='iso'))


def risk_summary(frame):
    rows=[]
    for h,g in frame.groupby('horizon'):
        rows.append(dict(horizon=int(h),n=len(g),crps_vs_hs=float(crps(g[ALL_COLUMNS].to_numpy()[:,MAIN],g.y).mean()/g.hs_crps.mean())))
    return pd.DataFrame(rows)


def main():
    model=json.loads((OUT/'selected.json').read_text());spec=json.loads((OUT/'specification.json').read_text())
    audit=json.loads((OUT/'test_audit.json').read_text())
    digest=hashlib.sha256((OUT/'selected.json').read_bytes()).hexdigest()
    assert model['input_sha256']==seal_inputs() and digest==audit['selected_sha256']
    f=pd.read_csv(OUT/'test_forecasts.csv',parse_dates=['origin_date','target_date'])
    test=pd.read_csv(OUT/'test_metrics.csv');points=pd.read_csv(OUT/'point_validation_metrics.csv')
    history=pd.read_csv(OUT/'validation_forecasts.csv');outer=pd.read_csv(OUT/'outer_validation_forecasts.csv')
    selected_risk=risk_summary(history);outer_risk=risk_summary(outer)
    selected_risk.to_csv(OUT/'risk_validation_summary.csv',index=False)
    outer_risk.to_csv(OUT/'risk_outer_summary.csv',index=False)
    ablation=pd.read_csv(OUT/'ablation.csv')
    weight_rows=[dict(horizon=int(h),**entry['weights'],penalty=entry['penalty']) for h,entry in model['point_by_horizon'].items()]
    weights=pd.DataFrame(weight_rows).sort_values('horizon');weights.to_csv(OUT/'weights.csv',index=False)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
    blue='#2563eb';orange='#ea580c';gray='#64748b'
    fixed=f[f['mode'].eq('fixed_origin')&f.actual.notna()].sort_values('target_date')
    fig,axes=plt.subplots(1,2,figsize=(13.2,4.8),layout='constrained')
    ax=axes[0]
    ax.fill_between(fixed.target_date,fixed.lower90,fixed.upper90,color=blue,alpha=.12,label='90% model interval')
    ax.plot(fixed.target_date,fixed.forecast,color=blue,lw=2,label='Integrated median')
    ax.plot(fixed.target_date,fixed.origin_price,color=gray,lw=1.7,ls='--',label='Random walk')
    ax.plot(fixed.target_date,fixed.actual,color=orange,lw=2,label='Observed WTI')
    ax.set_title('Fixed origin: 30 September',loc='left');ax.set_ylabel('WTI, USD / barrel')
    ax=axes[1]
    for h,color in [(1,blue),(5,'#7c3aed')]:
        g=f[f['mode'].eq('daily_update')&f.horizon.eq(h)&f.actual.notna()].sort_values('target_date')
        ax.plot(g.target_date,g.forecast,lw=2,color=color,label=f'{h}-day integrated median')
    ax.plot(fixed.target_date,fixed.actual,color=orange,lw=2,label='Observed WTI')
    ax.set_title('Daily inputs, frozen coefficients',loc='left')
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8,loc='upper left')
    fig.suptitle('Integrated WTI Forecast | sealed horizon policy | October evaluation',fontsize=15,fontweight='bold')
    fig.savefig(figdir/'october_forecasts.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13.2,5.8),layout='constrained')
    ax=axes[0];bottom=np.zeros(len(weights))
    colors={'rw':gray,'m1':blue,'m2':'#0d9488','lstm':'#7c3aed','ml':'#f59e0b'}
    for member in CHANNELS:
        values=100*weights[member].to_numpy()
        ax.barh(np.arange(len(weights)),values,left=bottom,color=colors[member],label=member.upper(),height=.75)
        bottom+=values
    ax.set_yticks(np.arange(len(weights)),[str(h) for h in weights.horizon]);ax.invert_yaxis()
    ax.set_ylabel('Forecast horizon (quoted rows)');ax.set_xlabel('Price weight (%)');ax.set_xlim(0,100)
    ax.set_title('Weights chosen using history only',loc='left')
    ax.legend(frameon=False,ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.10))
    ax=axes[1]
    daily=test[test['mode'].eq('daily_update')&test.horizon.isin([1,5])].set_index('horizon')
    values=[daily.loc[1,'rmse_vs_rw'],daily.loc[5,'rmse_vs_rw'],audit['fixed_path_metrics']['rmse_vs_rw']]
    ax.bar(['Daily 1 day\nn=17','Daily 5 days\nn=13','Fixed path\n17 targets'],values,color=[blue,blue,orange],width=.55)
    for i,value in enumerate(values):ax.text(i,value+.025,f'{value:.3f}',ha='center',fontweight='bold')
    ax.axhline(1,color=gray,ls='--',label='Random walk = 1')
    ax.set_ylim(0,max(values)*1.18);ax.set_ylabel('USD price RMSE / RW RMSE');ax.grid(axis='y',alpha=.15)
    ax.set_title('Daily forecasts improve; fixed path still trails RW',loc='left',fontsize=10)
    ax.legend(frameon=False,loc='lower right',fontsize=9)
    fig.suptitle('One policy across horizons | no minimum member weight',fontsize=15,fontweight='bold')
    fig.savefig(figdir/'strategy_summary.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12.6,4.5),layout='constrained')
    key=[1,5,20,60]
    for member in ['m1','m2','lstm','ml','center']:
        g=points[points.member.eq(member)&points.horizon.isin(key)].set_index('horizon').reindex(key)
        axes[0].plot(range(4),g.rmse_vs_rw,marker='o',lw=2 if member=='center' else 1.2,
                     color='#111827' if member=='center' else colors[member],
                     label='Integrated' if member=='center' else member.upper())
    axes[0].axhline(1,color=gray,ls='--');axes[0].set_ylabel('USD price RMSE / RW RMSE')
    axes[0].set_title('Past-only point selection: Sep 2024–Sep 2026',loc='left',fontsize=10)
    for data,label,color in [(selected_risk,'Selected calibration (Sep 2025–Sep 2026)',blue),(outer_risk,'Forward selection (Mar–Sep 2026)',orange)]:
        axes[1].plot(range(4),data.set_index('horizon').loc[key,'crps_vs_hs'],marker='o',label=label,color=color)
    axes[1].axhline(1,color=gray,ls='--');axes[1].set_ylabel('CRPS / historical-simulation CRPS')
    axes[1].set_title('Risk selection: development vs forward check',loc='left',fontsize=10)
    for ax in axes:
        ax.set_xticks(range(4),[str(h) for h in key]);ax.set_xlabel('Forecast horizon');ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Historical evidence | current-vintage, base-specification-conditional validation',fontsize=13,fontweight='bold')
    fig.savefig(figdir/'historical_validation.png',dpi=170);plt.close(fig)

    payload=dict(name=model['name'],version=model['version'],protocol_id=model['protocol_id'],fit_end=model['fit_end'],
                 selected_sha256=digest,active_output=OUT.relative_to(ROOT).as_posix(),point_meaning=model['point_meaning'],
                 weights_by_horizon=model['point_by_horizon'],risk_by_horizon=model['risk_by_horizon'],
                 test_metrics=records(test),fixed_path_metrics=audit['fixed_path_metrics'],
                 point_validation_metrics=records(points),risk_development_metrics=records(selected_risk),
                 risk_forward_metrics=records(outer_risk),ablation=records(ablation),forecasts=records(f),
                 module_roles={'m1':'Selected VECM contributes to horizons 1–20.',
                               'm2':'Selected monthly STL contributes at horizons 13,14,15,16,20.',
                               'lstm':'Three-seed sequence forecasts contribute to horizons 1–20.',
                               'ml_point':'Elastic Net enters selection and ablation; final direct weights are zero.',
                               'ml_distribution':'Historically selected ML conditional quantiles provide risk distributions at horizons 1–20.',
                               'rw':'Explicit price fallback, selected alone at horizon 60.'},
                 limitations=[spec['base_selection_limitation'],spec['october_status'],
                              'Original train/test are development data; outer meta-checks do not remove full-window base-specification selection.',
                              'ML direct-point weights are zero. Inclusion in the selection system is not evidence of positive forecast contribution.',
                              'October daily forecasts improve but the single fixed path remains 16.4% worse than RW; only 17/13 daily outcomes.',
                              'Historical risk forward check is weaker at 60 days; interval coverage is not guaranteed.',
                              'All publication/vintage limitations remain unchanged.'])
    (OUT/'presentation_data.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    manifest=dict(active_version=model['version'],name=model['name'],output=OUT.relative_to(ROOT).as_posix(),
                  presentation_data=(OUT/'presentation_data.json').relative_to(ROOT).as_posix(),selected_sha256=digest,
                  superseded='m3_ml/fusion/outputs/frozen_october_2026 (audit only)')
    (ROOT/'m4_reports/final_strategy_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Active presentation dataset:',OUT/'presentation_data.json',flush=True)


if __name__=='__main__':main()
