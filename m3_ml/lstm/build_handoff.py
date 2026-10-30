"""Generate a truthful protocol handoff from completed frozen-October outputs.

Run from any directory: python m3_ml/lstm/build_handoff.py [--execute].
No training or model download occurs. Existing historical presentation binaries
are not overwritten. Missing approved outputs cause an error, never fake tables.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from m3_ml.common.protocol import PROTOCOL_ID

OUT = ROOT / 'outputs' / 'frozen_october_2026'


def markdown_table(frame):
    rows = [list(frame.columns)] + [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(['| ' + ' | '.join(rows[0]) + ' |',
                      '| ' + ' | '.join(['---'] * len(rows[0])) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows[1:]])


def main(execute=False):
    summary = json.loads((OUT / 'run_summary.json').read_text())
    if summary.get('protocol_id') != PROTOCOL_ID:
        raise ValueError('Only completed approved-protocol runs may generate the current handoff.')
    metrics = pd.read_csv(OUT / 'test_metrics_key_horizons.csv')
    selection = pd.read_csv(OUT / 'validation_search.csv')
    report = f'''# LSTM frozen-October results

Protocol: `{PROTOCOL_ID}`. This table is generated from an actual saved training
run; it supersedes historical claims in the old `outputs/` files and slides.
It is a retrospective current-vintage experiment, not a prospective untouched
October test: earlier model development inspected portions of this period.

The actual saved fit uses {summary['train_origins']} initial-training origins,
{summary['validation_origins']} historical-validation origins and
{summary['final_fit_origins']} final-fit origins. Final ensemble seeds are
{summary['selected']['seeds']}, trained for {summary['selected']['epochs']} fixed
epochs. The latest mature supervised label is
{summary['selected']['last_observed_fit_label']}; all final coefficients and
preprocessing remain frozen afterward. Endpoint counts and errors below, not
historical long-horizon claims, determine the conclusion.

## Protocol

Initial fit labels available by 2023-08-31; historical validation target dates
2023-09-01 through 2026-09-30 with labels available by September 30. All 60 labels
of every supervised path are purged against the appropriate availability cutoff.
Final weights, X/Y scalers, selected architecture, seed ensemble and epoch count
are frozen using only information available by 2026-09-30. Publication dates use
matched WPSR archives where supplied, otherwise the explicitly retrospective
weekly-release assumption in `data/processed/wti_publication.csv`.

The observable published-WTI channel is reconstructed in memory from the common
publication schedule, choosing the newest legally scheduled observation and
never rewinding when an older quote arrives late. Residual labels and feature
windows use the same anchor transformation. Only leading pre-table cold-start
rows retain their supplied published quote; source CSV files stay unchanged.

The direct 60-output LSTM needs just one causal feature window. It never uses
future actual exogenous values or future targets to construct that path.
`fixed_origin` uses the Sep30 information set once; `daily_update` moves the
history window using features already public before the origin row date, without
re-estimating weights or scalers. Horizon h targets quoted row origin+h.
The supplied feature publication conventions remain upstream assumptions, not
independently verified vintages for every covariate.

## Realized-target metrics

Only October 1–23 target dates are scored, separately by mode. Horizons count
quoted rows, not calendar days. A horizon with no realized target has `n=0`, not
an invented performance estimate. Unknown future quotation dates remain blank;
`target_row` retains the numerical target position. Seed min/max measures
initialization sensitivity and is not a calibrated prediction interval.

{markdown_table(metrics.round(6))}

## Historical-validation selection

{markdown_table(selection.round(6))}

## Reproduction

From the repository root:

```powershell
uv run python m3_ml/lstm/src/oil_lstm.py
uv run python m3_ml/lstm/src/oil_lstm.py --test-only
uv run python m3_ml/lstm/src/oil_lstm.py --forecast-only
uv run python -m unittest discover -s m3_ml/lstm/tests -v
uv run python m3_ml/lstm/build_handoff.py --execute
```

New output directory: `m3_ml/lstm/outputs/frozen_october_2026/`. Old checkpoints
lack approved provenance and cannot be silently reused. Daily refresh requires
upstream ingestion/alignment plus the forecast-only command; it does not
retrain the model. Existing binary slides and their source remain historical
presentation material; do not quote their old scores as this protocol's results.
'''
    (ROOT / 'results_summary.md').write_text(report, encoding='utf-8')
    import nbformat as nb
    notebook = nb.v4.new_notebook(cells=[
        nb.v4.new_markdown_cell('# Frozen-October LSTM walkthrough\n\n' +
                                'This notebook reads genuine completed outputs. ' +
                                'Legacy slides/results use a superseded protocol. ' +
                                'See results_summary.md for release assumptions and limitations.'),
        nb.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\n"
                            "ROOT = Path.cwd()\n"
                            "if not (ROOT/'src/oil_lstm.py').exists(): ROOT = ROOT/'m3_ml/lstm'\n"
                            "OUT = ROOT/'outputs/frozen_october_2026'\n"
                            "summary = json.loads((OUT/'run_summary.json').read_text())\n"
                            "print(summary['protocol_id'])\nprint(summary['selected'])"),
        nb.v4.new_code_cell("display(pd.read_csv(OUT/'validation_search.csv'))\n"
                            "display(pd.read_csv(OUT/'test_metrics_key_horizons.csv'))"),
        nb.v4.new_code_cell("paths = pd.read_csv(OUT/'test_predictions.csv')\n"
                            "display(paths[(paths.method=='LSTM') & paths.horizon.isin([1,5,20,60])])"),
        nb.v4.new_markdown_cell('## Interpretation\n\nCompare modes separately and always report n. '
                                'Unreached h20/60 forecasts are numerical forecasts, not accuracy evidence. '
                                'No October weight/preprocessing fit is permitted.')],
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    if execute:
        from nbclient import NotebookClient
        notebook = NotebookClient(notebook, timeout=600, resources={'metadata': {'path': str(ROOT)}}).execute()
    nb.write(notebook, ROOT / 'Oil_LSTM_Walkthrough.ipynb')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    main(parser.parse_args().execute)
