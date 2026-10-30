"""Evaluate the new nested point policy before deciding whether to adopt it."""
import numpy as np
import pandas as pd
from m3_ml.fusion.model import mature_before
from m3_ml.fusion_v3.evaluate import paired_blocks
from .data import OUT,OLD,KEY,HORIZONS,read,write_json


def main():
    spec=read(OUT/'specification.json')
    proposed=pd.read_csv(OUT/'point_proposal.csv',parse_dates=['origin_date','target_date','label_available_date'])
    old=pd.read_csv(OLD/'point_crossfit.csv',parse_dates=['origin_date','target_date','label_available_date'])
    rows=[]
    for name,col in [('v5','forecast'),('v4','v4_forecast'),('rw','origin_price'),('futures','futures_price')]:
        proposed[name+'_loss']=(proposed[col]-proposed.actual)**2
    for h,g in proposed.groupby('horizon'):
        rows.append(dict(horizon=int(h),n=len(g),**{name+'_rmse_usd':float(np.sqrt(g[name+'_loss'].mean())) for name in ('v5','v4','rw','futures')},
            **{'v5_vs_'+name:float(np.sqrt(g.v5_loss.mean()/g[name+'_loss'].mean())) for name in ('v4','rw','futures')}))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'historical_point_proposal_metrics.csv',index=False)
    early=mature_before(old,spec['outer_start']);early['loss']=(early.origin_price-early.actual)**2
    scales=early.groupby('horizon').loss.mean()
    complete=proposed.groupby('origin_date').filter(lambda g:len(g)==len(HORIZONS)).copy()
    for name in ('v5','v4','rw','futures'):complete[name+'_normalized']=complete[name+'_loss']/complete.horizon.map(scales)
    origin=complete.groupby('origin_date')[[name+'_normalized' for name in ('v5','v4','rw','futures')]].mean()
    origin.to_csv(OUT/'primary_proposal_losses.csv');inference=[]
    for reference in ('v4','rw','futures'):
        for block in (20,60):inference.append(dict(scope='primary_four_horizons',reference=reference,**paired_blocks(origin.v5_normalized,origin[reference+'_normalized'],block)))
    for h,g in proposed.groupby('horizon'):
        for reference in ('v4','rw','futures'):
            for block in (20,60):inference.append(dict(scope=f'h{h}',reference=reference,**paired_blocks(g.v5_loss,g[reference+'_loss'],block)))
    ratio=float(np.sqrt(origin.v5_normalized.mean()/origin.v4_normalized.mean()))
    gate=spec['point_promotion'];passed=bool(ratio<gate['max_aggregate_ratio'] and metrics.v5_vs_v4.max()<=gate['max_horizon_ratio'])
    decision=dict(passed=passed,aggregate_rmse_vs_v4=ratio,worst_horizon_ratio=float(metrics.v5_vs_v4.max()),
        thresholds=gate,october_used=False,action='adopt_nested_point_policy' if passed else 'retain_v4_point_policy')
    write_json(OUT/'point_promotion_gate.json',decision)
    write_json(OUT/'historical_inference.json',dict(primary=dict(n_origins=len(origin),horizons=list(HORIZONS)),comparisons=inference,
        point_policy_adopted=passed,interpretation=spec['limitations']))
    panel=pd.read_parquet(OUT/'expert_panel.parquet').set_index(KEY)
    if passed:
        values=proposed.set_index(KEY).forecast
        panel.loc[values.index,'forecast']=values
    panel['center']=np.log(panel.forecast/panel.origin_price)
    panel.reset_index().to_csv(OUT/'point_crossfit.csv',index=False)
    final=read(OUT/'point_final.json') if passed else {str(h):dict(candidate='p000',reason='Point promotion gate failed.') for h in HORIZONS}
    write_json(OUT/'adopted_point_final.json',final)
    print(metrics.to_string(index=False),flush=True);print('Point gate:',decision,flush=True)
    print('Primary 60-origin interval:',next(x for x in inference if x['scope']=='primary_four_horizons' and x['reference']=='v4' and x['block_length']==60),flush=True)


if __name__=='__main__':main()
