"""Pre-October evaluation with paired, origin-level block uncertainty."""
import json
import numpy as np
import pandas as pd
from m3_ml.fusion.model import mature_before,metrics
from m3_ml.distribution.quantiles import crps
from m3_ml.distribution.tails import ALL_COLUMNS,MAIN
from .data import OUT,BASE,KEY,write_json


def paired_blocks(a,b,block,draws=4000,seed=5210):
    """Circular moving-block bootstrap; paired horizons stay in one origin row.

    The one-sided p-value uses a zero-centered bootstrap of the loss difference.
    This is conditional exploratory inference, not correction for all previous
    research choices or a claim of prospective out-of-sample significance.
    """
    a=np.asarray(a,float);b=np.asarray(b,float);n=len(a)
    if len(b)!=n or n<block or not np.isfinite(a+b).all():raise ValueError('Invalid paired loss series')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,(draws,int(np.ceil(n/block))))
    indices=((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
    am=a[indices].mean(axis=1);bm=b[indices].mean(axis=1)
    ratio=np.sqrt(am/bm);difference=a-b
    centered=(difference[indices]-difference.mean()).mean(axis=1)
    return dict(n_origins=n,block_length=block,draws=draws,
                rmse_ratio=float(np.sqrt(a.mean()/b.mean())),
                rmse_ratio_ci95=np.quantile(ratio,[.025,.975]).tolist(),
                improvement_pct=float(100*(1-np.sqrt(a.mean()/b.mean()))),
                improvement_pct_ci95=(100*(1-np.quantile(ratio,[.975,.025]))).tolist(),
                one_sided_centered_bootstrap_p=float((1+np.sum(centered<=difference.mean()))/(draws+1)))


def point_evaluation():
    spec=json.loads((OUT/'specification.json').read_text())
    f=pd.read_csv(OUT/'point_crossfit.csv',parse_dates=['origin_date','label_available_date'])
    old=pd.read_csv(BASE/'point_crossfit.csv',parse_dates=['origin_date'])
    old['v2_price']=old.origin_price*np.exp(old.center)
    f=f.merge(old[KEY+['v2_price']],on=KEY,validate='one_to_one')
    names={'v3':'forecast','rw':'rw_price','futures':'futures_price','v2':'v2_price'}
    for name,col in names.items():f[name+'_loss']=(f[col]-f.actual)**2
    early=mature_before(f,spec['rule_lock_date'])
    scales=early.groupby('horizon').rw_loss.mean()
    late=f[f.origin_date.ge(spec['rule_lock_date'])].copy()
    rows=[];quarter=[];inference=[]
    for period,data in [('development_all',f),('later_history',late)]:
        for h,g in data.groupby('horizon'):
            row=dict(period=period,horizon=int(h),n=len(g))
            for name in names:row[name+'_rmse_usd']=float(np.sqrt(g[name+'_loss'].mean()))
            for ref in ('rw','futures','v2'):row['v3_vs_'+ref]=row['v3_rmse_usd']/row[ref+'_rmse_usd']
            rows.append(row)
    for (period,h),g in late.groupby([late.origin_date.dt.to_period('Q'),'horizon']):
        quarter.append(dict(quarter=str(period),horizon=int(h),n=len(g),
            **{'v3_vs_'+ref:float(np.sqrt(g.v3_loss.mean()/g[ref+'_loss'].mean())) for ref in ('rw','futures','v2')}))
    complete=late[late.horizon.between(1,17)].groupby('origin_date').filter(lambda g:len(g)==17).copy()
    complete['loss_scale']=complete.horizon.map(scales)
    for name in names:complete[name+'_normalized']=complete[name+'_loss']/complete.loss_scale
    origin=complete.groupby('origin_date')[[n+'_normalized' for n in names]].mean().sort_index()
    origin.to_csv(OUT/'primary_origin_losses.csv')
    for ref in ('rw','futures','v2'):
        for block in (20,60):
            inference.append(dict(scope='primary_h1_17',reference=ref,
                **paired_blocks(origin.v3_normalized,origin[ref+'_normalized'],block)))
    for h in (1,5,20,60):
        g=late[late.horizon.eq(h)].sort_values('origin_date')
        for ref in ('rw','futures','v2'):
            for block in (20,60):
                inference.append(dict(scope='h'+str(h),reference=ref,
                    **paired_blocks(g.v3_loss,g[ref+'_loss'],block)))
    pd.DataFrame(rows).to_csv(OUT/'historical_point_metrics.csv',index=False)
    pd.DataFrame(quarter).to_csv(OUT/'historical_quarter_metrics.csv',index=False)
    result=dict(primary=dict(start=str(origin.index.min().date()),end=str(origin.index.max().date()),
                            n_origins=len(origin),n_horizons=17,
                            loss_scale='Per-horizon RW MSE on labels public before the 2025-09-01 rule lock'),
                comparisons=inference,interpretation=spec['inference'],
                base_selection_limitation=spec['base_selection_limitation'],no_october_rows=True)
    write_json(OUT/'historical_inference.json',result)
    print(pd.DataFrame(rows).query("period == 'later_history' and horizon in [1,5,20,60]").to_string(index=False),flush=True)
    for row in inference:
        if row['scope']=='primary_h1_17':print(json.dumps(row),flush=True)


def risk_evaluation():
    from .risk import source_frames
    spec=json.loads((OUT/'specification.json').read_text());rows=[]
    for period,filename in [('development_selection','risk_development_forecasts.csv'),('forward_selection','risk_forward_forecasts.csv')]:
        frame=pd.read_csv(OUT/filename,parse_dates=['origin_date'])
        scores=metrics(frame,'integrated_v3');scores['period']=period
        scores.to_csv(OUT/('risk_'+period+'_metrics.csv'),index=False)
        # Drop calibrated quantiles before joining the independent raw HS source.
        sources=source_frames(frame.drop(columns=ALL_COLUMNS),spec)
        raw=crps(sources['hs'][ALL_COLUMNS].to_numpy()[:,MAIN],frame.y)
        fused=crps(frame[ALL_COLUMNS].to_numpy()[:,MAIN],frame.y)
        frame['hs_loss']=raw;frame['v3_loss']=fused
        old_name='outer_validation_forecasts.csv' if period=='forward_selection' else 'validation_forecasts.csv'
        old=pd.read_csv(BASE/old_name,parse_dates=['origin_date'])
        old['v2_loss']=crps(old[ALL_COLUMNS].to_numpy()[:,MAIN],old.y)
        joined=frame.merge(old[KEY+['v2_loss']],on=KEY,how='left',validate='one_to_one')
        for h,g in joined.groupby('horizon'):
            valid=g.v2_loss.notna()
            rows.append(dict(period=period,horizon=int(h),n=len(g),crps_vs_hs=float(g.v3_loss.mean()/g.hs_loss.mean()),
                             crps_vs_v2=float(g.loc[valid,'v3_loss'].mean()/g.loc[valid,'v2_loss'].mean()),
                             coverage90=float(g.y.between(g['q_0.050'],g['q_0.950']).mean()),
                             coverage50=float(g.y.between(g['q_0.250'],g['q_0.750']).mean())))
    result=pd.DataFrame(rows);result.to_csv(OUT/'historical_risk_comparison.csv',index=False)
    print(result.query("period == 'forward_selection' and horizon in [1,5,20,60]").to_string(index=False),flush=True)


if __name__=='__main__':
    point_evaluation()
    if (OUT/'risk_forward_forecasts.csv').exists():risk_evaluation()
