"""Render the single sealed strategy's figures and portable presentation data.

No fitting or selection occurs here. Source: m3_ml/fusion/outputs/frozen_october_2026.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'m3_ml/fusion/outputs/frozen_october_2026'
FIG=OUT/'figures'

def records(frame):
    return json.loads(frame.to_json(orient='records',date_format='iso'))

def main():
    FIG.mkdir(exist_ok=True)
    model=json.loads((OUT/'selected.json').read_text())
    spec=json.loads((OUT/'specification.json').read_text())
    f=pd.read_csv(OUT/'test_forecasts.csv',parse_dates=['origin_date','target_date'])
    test=pd.read_csv(OUT/'test_metrics.csv');history=pd.read_csv(OUT/'validation_metrics.csv')
    ablation=pd.read_csv(OUT/'ablation.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#334155','text.color':'#172554',
                         'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
    blue='#2563eb';orange='#ea580c';muted='#64748b'
    fig,axes=plt.subplots(1,2,figsize=(13.2,4.8),layout='constrained')
    fixed=f[f['mode'].eq('fixed_origin')&f.target_date.notna()].sort_values('target_date')
    ax=axes[0]
    ax.fill_between(fixed.target_date,fixed.lower90,fixed.upper90,color=blue,alpha=.14,label='90% model interval')
    ax.plot(fixed.target_date,fixed.forecast,color=blue,lw=2,label='Integrated median')
    ax.plot(fixed.target_date,fixed.actual,color=orange,lw=2,label='Observed WTI')
    ax.set_title('One path issued on 30 September',loc='left');ax.set_ylabel('WTI, USD / barrel')
    ax=axes[1]
    for h,color in [(1,blue),(5,'#7c3aed')]:
        g=f[f['mode'].eq('daily_update')&f.horizon.eq(h)&f.target_date.notna()].sort_values('target_date')
        ax.plot(g.target_date,g.forecast,lw=2,color=color,label=f'{h}-day integrated median')
    actual=fixed[['target_date','actual']]
    ax.plot(actual.target_date,actual.actual,color=orange,lw=2,label='Observed WTI')
    ax.set_title('Daily inputs, frozen model parameters',loc='left')
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.grid(axis='y',alpha=.16);ax.legend(frameon=False,fontsize=9,loc='lower left')
    fig.suptitle('Integrated WTI Forecast | October 2026 evaluation',fontsize=16,fontweight='bold',x=.01,ha='left')
    fig.savefig(FIG/'october_forecasts.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    labels=['M1: VECM','M2: monthly STL','LSTM: three seeds','ML point: Elastic Net']
    weights=[100*model['weights'][k] for k in ('m1','m2','lstm','ml')]
    axes[0].barh(labels[::-1],weights[::-1],color=[blue,blue,blue,muted][::-1])
    for i,w in enumerate(weights[::-1]):axes[0].text(w+.6,i,f'{w:.1f}%',va='center')
    axes[0].set_xlim(0,50);axes[0].set_xlabel('Weight (%)');axes[0].set_title('Historical validation selects the weights',loc='left')
    axes[0].text(0,-.25,'Minimum 5% per member was fixed before evaluation.',transform=axes[0].transAxes,color=muted,fontsize=9)
    g=test[test['mode'].eq('daily_update')&test.horizon.isin([1,5])]
    axes[1].bar(['1 day (n=17)','5 days (n=13)'],g.rmse_vs_rw,color=orange,width=.5)
    axes[1].axhline(1,color=muted,ls='--',lw=1,label='Random walk = 1')
    for i,v in enumerate(g.rmse_vs_rw):axes[1].text(i,v+.02,f'{v:.3f}',ha='center')
    axes[1].set_ylim(0,1.22);axes[1].set_ylabel('Price RMSE / random-walk RMSE')
    axes[1].set_title('October daily forecasts did not beat RW',loc='left')
    axes[1].legend(frameon=False,loc='lower right')
    for ax in axes:ax.grid(axis='x' if ax is axes[0] else 'y',alpha=.13);ax.set_axisbelow(True)
    fig.suptitle('One selected system, with its limitations retained',fontsize=16,fontweight='bold',x=.01,ha='left')
    fig.savefig(FIG/'strategy_summary.png',dpi=170);plt.close(fig)
    path_rmse=float(np.sqrt(np.mean((fixed.forecast-fixed.actual)**2)))
    path_rw=float(np.sqrt(np.mean((fixed.origin_price-fixed.actual)**2)))
    full_path=dict(n=len(fixed),rmse_usd=path_rmse,rw_rmse_usd=path_rw,rmse_vs_rw=path_rmse/path_rw,
                   coverage90=float(((fixed.actual>=fixed.lower90)&(fixed.actual<=fixed.upper90)).mean()),
                   interpretation='17 targets from one fixed origin, not 17 independent forecasts')
    payload=dict(name=model['name'],protocol_id=model['protocol_id'],fit_end=model['fit_end'],
                 members={'m1':spec['m1_method'],'m2':spec['m2_method'],'lstm':spec['lstm'],'ml':spec['ml']},
                 weights=model['weights'],risk_method=model['risk_method'],risk_mixture=model['risk_mixture'],
                 price_interpretation=model['point_meaning'],test_metrics=records(test),
                 history_metrics=records(history),ablation=records(ablation),fixed_path_metrics=full_path,
                 forecasts=records(f),limitations=[spec['base_selection_limitation'],spec['october_status'],
                 spec['member_constraint'],'October daily h=1/5 did not beat random walk; long horizons have no realized test targets.'])
    (OUT/'presentation_data.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Rendered',FIG,'and presentation_data.json')

if __name__=='__main__':main()
