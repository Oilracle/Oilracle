"""Publish the v6 risk-only upgrade and retain rejected point experiments."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from m3_ml.fusion.model import MEDIAN,metrics
from m3_ml.distribution.tails import ALL_COLUMNS,MAIN
from m3_ml.distribution.quantiles import crps
from m3_ml.fusion_v6 import research as r,run
from m4_reports.rolling_upgrade import records

OUT=r.OUT;ROOT=r.ROOT;DELIVERY=OUT/'rolling'


def interval_score(frame):
    lo=frame.origin_price*np.exp(frame['q_0.050']);hi=frame.origin_price*np.exp(frame['q_0.950'])
    actual=frame.origin_price*np.exp(frame.y)
    return hi-lo+20*np.maximum(lo-actual,0.)+20*np.maximum(actual-hi,0.)


def historical():
    proposed=pd.read_csv(OUT/'risk_proposal.csv',parse_dates=['origin_date'])
    old=pd.read_csv(r.OLD/'risk_forward_forecasts.csv',parse_dates=['origin_date'])
    old=old.set_index(r.KEY).reindex(pd.MultiIndex.from_frame(proposed[r.KEY])).reset_index()
    adopted=old.copy();rules=r.read(OUT/'risk_final.json')
    for h in (1,5):
        if rules[str(h)]['enabled']:
            ix=adopted.horizon.eq(h);adopted.loc[ix,ALL_COLUMNS]=proposed.loc[ix,ALL_COLUMNS].to_numpy()
    rows=[];newloss=crps(adopted[ALL_COLUMNS].to_numpy()[:,MAIN],adopted.y);oldloss=crps(old[ALL_COLUMNS].to_numpy()[:,MAIN],old.y)
    for h in (1,5):
        ix=adopted.horizon.eq(h);g=adopted[ix];b=old[ix]
        rows.append(dict(horizon=h,n=len(g),period='2026-06 onward',coverage90=float(g.y.between(g['q_0.050'],g['q_0.950']).mean()),
            v5_coverage90=float(b.y.between(b['q_0.050'],b['q_0.950']).mean()),crps_vs_v5=float(newloss[ix].mean()/oldloss[ix].mean()),
            width_vs_v5=float((g.origin_price*(np.exp(g['q_0.950'])-np.exp(g['q_0.050']))).mean()/(b.origin_price*(np.exp(b['q_0.950'])-np.exp(b['q_0.050']))).mean()),
            interval_score_vs_v5=float(interval_score(g).mean()/interval_score(b).mean())))
    return adopted,pd.DataFrame(rows)


def figures(f,risk,proposals):
    folder=DELIVERY/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    blue='#2563eb';gray='#64748b';orange='#dc6b20'
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.5),layout='constrained')
    x=np.arange(2);axes[0].bar(x-.17,risk.v5_coverage90*100,.32,color=gray,label='v5')
    axes[0].bar(x+.17,risk.coverage90*100,.32,color=blue,label='Adopted v6')
    axes[0].axhline(90,color=orange,ls='--',label='Nominal 90%');axes[0].set_ylim(0,110)
    axes[0].set_xticks(x,[f'{h} day | n={n}' for h,n in zip(risk.horizon,risk.n)]);axes[0].set_ylabel('Coverage (%)')
    axes[0].set_title('Historical coverage | June–September 2026',fontsize=11);axes[0].legend(frameon=False,fontsize=8)
    axes[1].bar(x-.22,risk.crps_vs_v5,.21,color=blue,label='CRPS / v5')
    axes[1].bar(x,risk.width_vs_v5,.21,color=orange,label='Interval width / v5')
    axes[1].bar(x+.22,risk.interval_score_vs_v5,.21,color=gray,label='Interval score / v5')
    axes[1].axhline(1,color=gray,ls='--');axes[1].set_xticks(x,['1 day','5 days']);axes[1].set_ylim(0,1.65)
    axes[1].set_title('Coverage improvement has a width cost',fontsize=11);axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('V6: accept the 5-day interval upgrade; retain v5 point forecasts',fontweight='bold')
    fig.savefig(folder/'historical_validation.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.5),layout='constrained')
    ax.bar(x-.18,proposals.rmse_ratio,.34,color=orange,label='Rejected bias-correction policy')
    ax.bar(x+.18,[1,1],.34,color=blue,label='Adopted v6 = v5 points')
    ax.axhline(1,color=gray,ls='--');ax.set_ylim(0,1.85);ax.set_xticks(x,['1 day | n=261','5 days | n=257'])
    ax.set_ylabel('Historical RMSE / v5 RMSE');ax.legend(frameon=False,fontsize=9)
    ax.set_title('Short-horizon corrections failed the historical promotion gates',fontsize=11)
    fig.savefig(folder/'strategy_summary.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,h in zip(axes,(1,5)):
        g=f[f.horizon.eq(h)&f.actual.notna()].sort_values('target_date')
        ax.fill_between(g.target_date,g.lower90,g.upper90,color=blue,alpha=.15,label='90% model interval')
        for c,label,color,style in [('forecast','Integrated median',blue,'-'),('origin_price','Published-spot RW',gray,'--'),('actual','Observed WTI',orange,'-')]:
            ax.plot(g.target_date,g[c],color=color,ls=style,lw=2,label=label)
        ax.set_title(f'Rolling {h} day | n={len(g)}',loc='left');ax.set_ylabel('USD / barrel')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5));ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('October evaluation only | model frozen before scoring',fontweight='bold')
    fig.savefig(folder/'october_forecasts.png',dpi=170);plt.close(fig)


def main():
    r.protect();model=r.read(OUT/'selected.json');digest=r.sha(OUT/'selected.json')
    assert model['input_sha256']==run.hashes() and r.read(OUT/'test_audit.json')['selected_sha256']==digest
    DELIVERY.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(OUT/'test_forecasts.csv',parse_dates=['origin_date','target_date']);score=pd.read_csv(OUT/'test_metrics.csv')
    history,risk=historical();risk.to_csv(DELIVERY/'historical_risk_metrics.csv',index=False)
    history.to_csv(DELIVERY/'historical_risk_forecasts.csv',index=False)
    for src,dest in [('test_forecasts.csv','forecasts.csv'),('test_metrics.csv','metrics.csv')]:
        (DELIVERY/dest).write_bytes((OUT/src).read_bytes())
    for name in ('module_input_sensitivity.csv','expert_weights.csv','historical_point_metrics.csv'):
        (DELIVERY/name).write_bytes((r.OLD/'rolling'/name).read_bytes())
    figures(f,risk,pd.read_csv(OUT/'point_metrics.csv'))
    scope=dict(forecast_mode='daily_update',horizons=list(r.HORIZONS),parameter_policy='Frozen as of September 30; only published daily inputs update.',
        change='All v5 point forecasts and h1/h20/h60 distributions retained. Only h5 tail recalibration adopted.',
        selection_policy=model['october'],risk_history_start='2026-06-01',development_amendment='width_amendment.json')
    r.write_json(DELIVERY/'delivery_scope.json',scope)
    payload=r.read(r.OLD/'rolling/presentation_data.json')
    payload.update(version=model['version'],selected_sha256=digest,active_output=DELIVERY.relative_to(ROOT).as_posix(),
        delivery_scope=scope,test_metrics=records(score),forecasts=records(f),historical_risk_metrics=records(risk),
        inherited_v5_historical_risk_metrics=payload['historical_risk_metrics'],
        historical_inference=dict(point_increment='Zero: v5 points retained.',inherited_v5=payload['historical_inference']),
        rejected_point_experiments=records(pd.read_csv(OUT/'point_metrics.csv')),point_policy=model['point'],
        tail_recalibration=model['risk'],promotion_gate=model['promotion'],
        limitations=[model['limitations'],model['october'],
            'No 1-day or 5-day point improvement was validated. The proposed corrections worsened RMSE by 32.6%/51.5% and were rejected.',
            'New interval evidence compares the same June–September origins (76/72 at h1/h5), not the earlier March–September totals.',
            'The h5 interval is 35% wider; historical coverage remains below 90%. October has only 13 h5 targets.',
            'One narrower cap was introduced after historical inspection; original thresholds and unsuccessful experiments are preserved.',
            'No realized October outcomes exist for h20/h60. Their forecasts and intervals are unchanged.'])
    payload['previous_rolling_metrics']['v5']=records(pd.read_csv(r.OLD/'test_metrics.csv'))
    r.write_json(DELIVERY/'presentation_data.json',payload)
    manifest=dict(active_version=model['version'],name=model['name'],output=DELIVERY.relative_to(ROOT).as_posix(),model_output=OUT.relative_to(ROOT).as_posix(),
        forecast_mode='daily_update',horizons=list(r.HORIZONS),delivery_profile='daily_rolling',selected_sha256=digest,
        presentation_data=(DELIVERY/'presentation_data.json').relative_to(ROOT).as_posix(),forecasts=(DELIVERY/'forecasts.csv').relative_to(ROOT).as_posix(),
        metrics=(DELIVERY/'metrics.csv').relative_to(ROOT).as_posix(),delivery_scope=(DELIVERY/'delivery_scope.json').relative_to(ROOT).as_posix(),
        selection_reason='Pre-October gate accepts only h5 tail recalibration: matched-history coverage 65.28% to 79.17%, CRPS -0.87%, width +35%; all point corrections rejected.',
        superseded=[f'm3_ml/{v}/outputs/frozen_october_2026 (audit only)' for v in ('fusion','fusion_v2','fusion_v3','fusion_v4','fusion_v5')])
    r.write_json(ROOT/'m4_reports/final_strategy_manifest.json',manifest)
    print('Published v6 risk-only upgrade.',flush=True);print(risk.to_string(index=False),flush=True)


def audit():
    r.protect();model=r.read(OUT/'selected.json');assert model['input_sha256']==run.hashes()
    digest=r.sha(OUT/'selected.json');manifest=r.read(ROOT/'m4_reports/final_strategy_manifest.json');payload=r.read(DELIVERY/'presentation_data.json')
    assert manifest['active_version']==payload['version']==model['version']=='short_bias_v6'
    assert digest==manifest['selected_sha256']==payload['selected_sha256']==r.read(OUT/'test_audit.json')['selected_sha256']
    assert r.sha(OUT/'test_forecasts.csv')==r.sha(DELIVERY/'forecasts.csv')
    f=pd.read_csv(DELIVERY/'forecasts.csv',parse_dates=['origin_date','target_date']);old=pd.read_csv(r.OLD/'test_forecasts.csv',parse_dates=['origin_date','target_date'])
    assert len(f)==68 and f['mode'].eq('daily_update').all() and not f.duplicated(r.KEY).any()
    pd.testing.assert_frame_equal(f[r.KEY+['target_date']],old[r.KEY+['target_date']])
    np.testing.assert_allclose(f.forecast,old.forecast,rtol=0,atol=1e-12)
    unchanged=~f.horizon.eq(5);np.testing.assert_allclose(f.loc[unchanged,ALL_COLUMNS],old.loc[unchanged,ALL_COLUMNS],atol=1e-12)
    q=f[ALL_COLUMNS].to_numpy();assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-12).all()
    np.testing.assert_allclose(q[:,MEDIAN],f.center,atol=1e-14)
    width=f.origin_price*(np.exp(f['q_0.950'])-np.exp(f['q_0.050']))
    old_width=old.origin_price*(np.exp(old['q_0.950'])-np.exp(old['q_0.050']))
    assert (width<=1.35*old_width+1e-9).all()
    scores=pd.read_csv(DELIVERY/'metrics.csv').set_index('horizon');fresh=metrics(f,'integrated_v6').set_index('horizon')
    cols=['n','rmse_usd','mae_usd','rmse_vs_rw','coverage90','crps_log','width90_usd']
    np.testing.assert_allclose(scores[cols],fresh[cols],atol=1e-10,equal_nan=True);assert scores.n.tolist()==[17,13,0,0]
    hist,hs=historical();pd.testing.assert_frame_equal(pd.read_csv(DELIVERY/'historical_risk_metrics.csv'),hs,atol=1e-10,rtol=1e-10)
    assert model['promotion']['passed'] and model['promotion']['risk']['5']['passed'] and not model['promotion']['risk']['1']['passed']
    assert all(not d['passed'] for d in model['promotion']['point'].values())
    for name in ('module_input_sensitivity.csv','expert_weights.csv','historical_point_metrics.csv'):
        assert r.sha(DELIVERY/name)==r.sha(r.OLD/'rolling'/name)
    for h,n in ((1,17),(5,13)):
        path=ROOT/f'data/video/fan_integrated_h{h}.csv';meta=r.read(path.with_suffix('.json'))
        assert meta['selected_sha256']==digest and meta['n']==n and meta['version']==model['version']
        v=pd.read_csv(path,parse_dates=['date']);g=f[f.horizon.eq(h)&f.y.notna()].sort_values('target_date')
        np.testing.assert_array_equal(v.date,g.target_date)
        for c in (c for c in ALL_COLUMNS if c.startswith('q_')):
            np.testing.assert_allclose(v[c.replace('q_','p_')],g.origin_price*np.exp(g[c]),rtol=0,atol=.000051)
    r.write_json(DELIVERY/'delivery_audit.json',dict(status='passed',selected_sha256=digest,forecast_sha256=r.sha(DELIVERY/'forecasts.csv'),
        rows=68,point_forecasts_unchanged=True,only_h5_intervals_changed=True,metrics_recomputed=True,matched_historical_windows=True,
        cap_verified=True,predecessor_preserved=True,video_data_verified=True,october_tuning=False))
    print('V6 delivery audit passed:',digest,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    (audit if args.audit else main)()
