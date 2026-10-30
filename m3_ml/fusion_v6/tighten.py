"""Historical amendment: reduce tail expansion cost without relaxing gates."""
from . import research as r


def main():
    if not (r.OUT/'width_amendment.json').exists():
        archive=r.OUT/'initial_proposal';archive.mkdir(exist_ok=True)
        for n in ('risk_proposal.csv','risk_metrics.csv','risk_final.json','risk_gate.json','promotion_gate.json'):
            (archive/n).write_bytes((r.OUT/n).read_bytes())
        r.write_json(r.OUT/'width_amendment.json',dict(development_amendment=True,
            reason='The initial 1.5 expansion exceeded the predeclared 1.4 mean-width gate, despite improving h5 coverage.',
            price_width_cap=1.35,policy='One tighter expansion cap with margin below the unchanged width gate; no cap grid, no October selection.',
            october_used=False))
    r.risk_research()


if __name__=='__main__':main()
