"""Summarize approved historical-validation selection, never old 2021 folds."""
import json
import pandas as pd

from m3_ml.common.protocol import PROTOCOL_ID
from .run import OUT


def main():
    scores = pd.read_csv(OUT / 'validation_scores.csv')
    selected = json.loads((OUT / 'selected.json').read_text())
    if selected.get('protocol_id') != PROTOCOL_ID or not scores.protocol_id.eq(PROTOCOL_ID).all():
        raise ValueError('Diagnosis requires approved-protocol validation outputs.')
    setting = selected['sampler']
    chosen = scores[(scores['T'] == setting['T']) & (scores.top_p == setting['top_p'])].copy()
    chosen['selection_objective'] = 'mean CRPS across h1,5,20,60; one coherent sampler'
    chosen.to_csv(OUT / 'validation_diagnostic.csv', index=False)
    print(chosen.to_string(index=False))


if __name__ == '__main__':
    main()
