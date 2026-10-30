import json
import numpy as np
import pandas as pd
from m3_ml.fusion.model import mature_before
from m3_ml.fusion_v3.evaluate import paired_blocks
from .data import OUT,OLD,KEY,HORIZONS,write_json


def point_evaluation():
    spec=json.loads((OUT/'specification.json').read_text())
    f=pd.read_csv(OUT/'point_crossfit.csv',parse_dates=['origin_date','label_available_date'])
    old=pd.read_csv(OLD/'point_crossfit.csv',parse_dates=['origin_date'])
    f=f.merge(old[KEY+['forecast']].rename(columns={'forecast':'v3_forecast'}),on=KEY,validate='one_to_one')
    for name,column in [('v4','forecast'),('v3','v3_forecast'),('futures','futures_price'),('rw','origin_price')]:
        f[name+'_loss']=(f[column]-f.actual)**2
    early=mature_before(f,spec['rule_lock_date']);scales=early.groupby('horizon').rw_loss.mean()
    late=f[f.origin_date.ge(spec['rule_lock_date'])].copy();rows=[];quarters=[]
    for period,data in [('development_all',f),('later_history',late)]:
        for h,g in data.groupby('horizon'):
            rows.append(dict(period=period,horizon=int(h),n=len(g),
                **{n+'_rmse_usd':float(np.sqrt(g[n+'_loss'].mean())) for n in ('v4','v3','rw','futures')},
                **{'v4_vs_'+n:float(np.sqrt(g.v4_loss.mean()/g[n+'_loss'].mean())) for n in ('v3','rw','futures')}))
    for (quarter,h),g in late.groupby([late.origin_date.dt.to_period('Q'),'horizon']):
        quarters.append(dict(quarter=str(quarter),horizon=int(h),n=len(g),
            **{'v4_vs_'+n:float(np.sqrt(g.v4_loss.mean()/g[n+'_loss'].mean())) for n in ('v3','rw','futures')}))
    complete=late.groupby('origin_date').filter(lambda g:len(g)==len(HORIZONS)).copy()
    complete['scale']=complete.horizon.map(scales)
    for name in ('v4','v3','rw','futures'):complete[name+'_normalized']=complete[name+'_loss']/complete.scale
    origin=complete.groupby('origin_date')[[n+'_normalized' for n in ('v4','v3','rw','futures')]].mean().sort_index()
    origin.to_csv(OUT/'primary_origin_losses.csv')
    inference=[]
    for name in ('v3','rw','futures'):
        for block in (20,60):
            inference.append(dict(scope='primary_four_horizons',reference=name,
                **paired_blocks(origin.v4_normalized,origin[name+'_normalized'],block)))
    for h,g in late.groupby('horizon'):
        for name in ('v3','rw','futures'):
            for block in (20,60):
                inference.append(dict(scope=f'h{h}',reference=name,**paired_blocks(g.v4_loss,g[name+'_loss'],block)))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'historical_point_metrics.csv',index=False)
    pd.DataFrame(quarters).to_csv(OUT/'historical_quarters.csv',index=False)
    relative=float(np.sqrt(origin.v4_normalized.mean()/origin.v3_normalized.mean()))
    late_metrics=metrics[metrics.period.eq('later_history')]
    passed=bool(relative<1 and late_metrics.v4_vs_v3.max()<=1.02)
    write_json(OUT/'point_promotion_gate.json',dict(passed=passed,aggregate_rmse_vs_v3=relative,
        worst_horizon_rmse_vs_v3=float(late_metrics.v4_vs_v3.max()),rule=spec['promotion'],october_used=False))
    write_json(OUT/'historical_inference.json',dict(primary=dict(n_origins=len(origin),horizons=list(HORIZONS)),
        comparisons=inference,interpretation=spec['inference'],source_limits=spec['source_limits']))
    print(late_metrics.to_string(index=False),flush=True)
    print('Point promotion gate:',passed,'aggregate RMSE/v3',relative,flush=True)
    for row in inference:
        if row['scope']=='primary_four_horizons':print(json.dumps(row),flush=True)


if __name__=='__main__':point_evaluation()
