"""Research amendment: shrink extrapolated state gates toward the incumbent."""
import numpy as np
import pandas as pd
from .data import OUT,KEY,HORIZONS,STATE,read,write_json,sha
from .point import choose
from m3_ml.fusion.model import blocks,mature_before


def bounds(train):
    z=train[STATE[:5]].to_numpy()
    return dict(low=np.quantile(z,.01,axis=0).tolist(),high=np.quantile(z,.99,axis=0).tolist(),
        spread=np.maximum(np.quantile(z,.75,axis=0)-np.quantile(z,.25,axis=0),.1).tolist())


def trust(frame,calibration,decay):
    z=frame[STATE[:5]].to_numpy();spread=np.asarray(calibration['spread'])
    excess=np.maximum(np.maximum(np.asarray(calibration['low'])-z,z-np.asarray(calibration['high'])),0.)/spread
    return np.exp(-float(decay)*excess.max(axis=1))


def main():
    design=OUT/'support_design.json'
    if not design.exists():
        archive=OUT/'initial_proposal';archive.mkdir(exist_ok=True)
        for name in ('point_proposal.csv','point_final.json','historical_point_proposal_metrics.csv','historical_inference.json',
                     'point_promotion_gate.json','risk_forward_proposal.csv','historical_risk_proposal_metrics.csv','risk_final.json','promotion_gate.json'):
            if (OUT/name).exists():(archive/name).write_bytes((OUT/name).read_bytes())
        write_json(design,dict(development_amendment=True,reason='Initial nested policy failed h1 degradation gate; test protection against states outside mature training experience.',
            features=STATE[:5],support_quantiles=[.01,.99],spread='training IQR with floor 0.1',decays=[.5,1.,2.],
            rule='Trust=exp(-decay*maximum normalized distance outside training support). Blend toward incumbent, using no outcomes at the prediction origin.',
            validation='Same nested selector and unchanged promotion thresholds.',october_used=False,
            initial_proposal_sha256=sha(archive/'point_proposal.csv')))
    spec=read(OUT/'specification.json');definitions=spec['point_candidates'];design=read(design)
    panel=pd.read_parquet(OUT/'expert_panel.parquet');candidates=pd.read_parquet(OUT/'point_candidate_forecasts.parquet')
    parts=[candidates];audit=[];extended=list(definitions)
    for d in definitions:
        if d['kind']=='incumbent':continue
        original=candidates[candidates.candidate.eq(d['id'])]
        for decay in design['decays']:
            definition={**d,'id':f"{d['id']}_support{decay:g}",'base_candidate':d['id'],'support_decay':decay}
            extended.append(definition);pieces=[]
            for (h,cutoff),g in original.groupby(['horizon','point_fit_cutoff']):
                train=mature_before(panel[panel.horizon.eq(h)],cutoff)
                if d['window']:train=train.sort_values('origin_date').tail(d['window'])
                calibration=bounds(train);f=g.copy();weight=trust(f,calibration,decay)
                f['forecast']=f.v4_forecast+weight*(f.forecast-f.v4_forecast);f['candidate']=definition['id'];pieces.append(f)
                audit.append(dict(horizon=int(h),cutoff=str(cutoff.date()),candidate=definition['id'],
                    latest_label_release=str(train.label_available_date.max().date()),mean_trust=float(weight.mean()),bounds=calibration))
            parts.append(pd.concat(pieces,ignore_index=True))
    extended_candidates=pd.concat(parts,ignore_index=True);extended_candidates.to_parquet(OUT/'point_candidate_supported.parquet',index=False)
    selected=[];evidence=[]
    for cutoff,stop in blocks(spec['outer_start'],spec['fit_end']):
        for h in HORIZONS:
            cid,details=choose(extended_candidates,cutoff,h,spec)
            g=extended_candidates[extended_candidates.horizon.eq(h)&extended_candidates.candidate.eq(cid)
                &extended_candidates.origin_date.ge(cutoff)&extended_candidates.origin_date.lt(stop)].copy()
            g['selection_cutoff']=cutoff;selected.append(g);evidence.append(dict(cutoff=str(cutoff.date()),horizon=h,candidate=cid,**details))
    result=pd.concat(selected,ignore_index=True).sort_values(KEY).reset_index(drop=True)
    result['center']=np.log(result.forecast/result.origin_price);result.to_csv(OUT/'point_proposal.csv',index=False)
    final={str(h):dict(candidate=cid,**details) for h in HORIZONS for cid,details in [choose(extended_candidates,spec['fit_end'],h,spec)]}
    write_json(OUT/'point_final.json',final);write_json(OUT/'point_selection_audit.json',evidence)
    write_json(OUT/'supported_definitions.json',extended);write_json(OUT/'support_audit.json',audit)
    print('Support-aware nested candidates:',len(extended),'final IDs:',{h:d['candidate'] for h,d in final.items()},flush=True)


if __name__=='__main__':main()
