"""Conservative half-step for a newly estimated nested meta policy."""
import numpy as np
import pandas as pd
from .data import OUT,read,write_json,sha


def main():
    if not (OUT/'damping_design.json').exists():
        (OUT/'supported_point_proposal.csv').write_bytes((OUT/'point_proposal.csv').read_bytes())
        (OUT/'supported_point_metrics.csv').write_bytes((OUT/'historical_point_proposal_metrics.csv').read_bytes())
        write_json(OUT/'damping_design.json',dict(development_amendment=True,
            reason='Support-only protection did not satisfy the original h1 degradation gate; reduce the impact of the new meta policy.',
            update_fraction=.5,policy='Apply a fixed half-step to the same causally selected policy at every horizon and every historical origin. No re-ranking on October.',
            coefficient_selection='One conservative half-step, registered before its evaluation; no coefficient grid or adjusted promotion thresholds.',
            undamped_proposal_sha256=sha(OUT/'supported_point_proposal.csv'),october_used=False))
    design=read(OUT/'damping_design.json');f=pd.read_csv(OUT/'supported_point_proposal.csv')
    f['undamped_forecast']=f.forecast;f['forecast']=f.v4_forecast+design['update_fraction']*(f.forecast-f.v4_forecast)
    f['center']=np.log(f.forecast/f.origin_price);f['update_fraction']=design['update_fraction']
    f.to_csv(OUT/'point_proposal.csv',index=False)
    final=read(OUT/'point_final.json')
    for d in final.values():d['update_fraction']=design['update_fraction']
    write_json(OUT/'point_final.json',final)
    print('Fixed half-step applied to the nested historical policy; ready for unchanged promotion gate.',flush=True)


if __name__=='__main__':main()
