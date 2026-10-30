"""Direct 1--60 quoted-row WTI paths under the frozen October 2026 protocol.

Feature windows end at the reported origin row; h predicts row origin+h.
All 60 outputs use that one causal window, never future realized features.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from m3_ml.common.protocol import (
    TRAIN_END, VALIDATION_START, VALIDATION_END, FIT_END, TEST_START, TEST_END,
    HORIZONS, MODES, PROTOCOL_ID,
)
from m3_ml.common.data import label_available_dates, observable_price

HORIZON = 60
ANCHOR = "wti_spot_last_published"
SEEDS = [42, 43, 44]
CONFIGS = [
    dict(lookback=20, hidden=16, dropout=0.1, lr=0.001),
    dict(lookback=60, hidden=16, dropout=0.1, lr=0.001),
    dict(lookback=60, hidden=32, dropout=0.2, lr=0.001),
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def read_data(path, dictionary):
    df = pd.read_csv(path, parse_dates=["date"])
    meta = pd.read_csv(dictionary)
    cols = meta.loc[~meta.group.isin(["index", "target"]), "column"].tolist()
    if df.date.isna().any() or df.date.duplicated().any() or not df.date.is_monotonic_increasing:
        raise ValueError("Dates must be valid, unique and strictly increasing; do not silently sort.")
    if not len(df) or not all(c in df for c in cols) or ANCHOR not in cols:
        raise ValueError("Missing required predictor columns or empty data.")
    if "target_wti" in cols:
        raise ValueError("Target leakage in feature dictionary.")
    # Use the common publication-evidence information set for BOTH the residual
    # anchor and its predictor channel; never repair just the output baseline.
    df[ANCHOR] = observable_price(df).to_numpy()
    x = df[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Predictors must be finite. Fix upstream causal alignment; no backfill is allowed.")
    if "target_wti" not in df:
        df["target_wti"] = np.nan
    return df, cols


def boundary(df, date, embargo):
    """Last usable target index, withholding embargo observations before boundary."""
    before = np.flatnonzero((df.date <= pd.Timestamp(date)).to_numpy())
    if len(before) <= embargo:
        raise ValueError("Insufficient data before split boundary.")
    return int(before[-1] - embargo)


def origins_between(n, lookback, start, last_label):
    # i+HORIZON <= last_label. All 60 next-row outputs obey the split cutoff.
    return np.arange(max(lookback - 1, start), min(n - HORIZON - 1, last_label - HORIZON) + 1)


def published_origins(df, origins, cutoff):
    """Keep complete label paths whose publication schedule has matured."""
    available = label_available_dates(df).to_numpy(dtype="datetime64[ns]")
    known = ~np.isnat(available) & (available <= np.datetime64(cutoff))
    return np.asarray([i for i in origins if known[i + 1:i + HORIZON + 1].all()], dtype=int)


def windows(x, origins, lookback):
    return np.stack([x[i-lookback+1:i+1] for i in origins]).astype(np.float32)


def targets(df, origins):
    y = df.target_wti.to_numpy(float)
    a = df[ANCHOR].to_numpy(float)
    return np.stack([y[i + 1:i + HORIZON + 1] - a[i] for i in origins])


def fit_scaler(x):
    mu, sd = np.mean(x, axis=0), np.std(x, axis=0)
    return mu, np.where(sd < 1e-8, 1.0, sd)


class OilLSTM(nn.Module):
    def __init__(self, n_features, hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, HORIZON))

    def forward(self, x):
        states, _ = self.lstm(x)
        return self.head(states[:, -1, :])


def train_one(config, seed, xtrain, ytrain, xval=None, yval=None, epochs=80, patience=12):
    seed_everything(seed)
    model = OilLSTM(xtrain.shape[-1], config["hidden"], config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.001)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(xtrain), torch.from_numpy(ytrain.astype(np.float32))),
                        batch_size=128, shuffle=True, generator=generator)
    best, best_epoch, state, history, stale = float("inf"), 0, None, [], 0
    xv = None if xval is None else torch.from_numpy(xval)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = None if xv is None else float(np.mean((model(xv).numpy() - yval)**2))
        history.append(dict(epoch=epoch, train_mse=total/len(xtrain), validation_mse=vl))
        if vl is None:
            best_epoch = epoch
        elif vl < best - 1e-5:
            best, best_epoch, state, stale = vl, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if xv is not None and stale >= patience:
            break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model, history, best_epoch, best


def predict(model, x):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+256])).numpy() for i in range(0, len(x), 256)])


def metric_values(actual, pred, anchor):
    error = pred - actual
    active = np.abs(actual-anchor) > 1e-8
    true_up, pred_up = actual > anchor, pred > anchor
    tp = np.sum(true_up & pred_up)
    precision = tp / max(int(pred_up.sum()), 1)
    recall = tp / max(int(true_up.sum()), 1)
    positive = actual > 1.0
    return dict(n=len(actual), mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                mape_positive_only_pct=float(np.mean(np.abs(error[positive])/actual[positive])*100) if positive.any() else None,
                mape_n=int(positive.sum()), direction_vs_published_pct=float(np.mean(np.sign(pred[active]-anchor[active]) == np.sign(actual[active]-anchor[active]))*100) if active.any() else None,
                directional_n=int(active.sum()), neutral_forecast_pct=float(np.mean(np.abs(pred-anchor)<1e-8)*100),
                up_precision=float(precision), up_recall=float(recall),
                up_f1=float(2*precision*recall/max(precision+recall, 1e-12)))


def evaluate(predictions):
    """Only realized October targets are scored; unreached horizons retain n=0."""
    rows = []
    for mode in MODES:
        for name in predictions.model.unique():
            for h in HORIZONS:
                g = predictions[(predictions["mode"] == mode) &
                                (predictions.model == name) &
                                (predictions.horizon == h)].dropna(subset=["actual"])
                values = metric_values(g.actual.to_numpy(), g.prediction.to_numpy(),
                                       g.anchor.to_numpy()) if len(g) else dict(
                                           n=0, mae=np.nan, rmse=np.nan)
                rows.append(dict(mode=mode, model=name, method=name, horizon=h,
                                 protocol_id=PROTOCOL_ID, fit_end=FIT_END, **values))
    return pd.DataFrame(rows)




def target_calendar(df, first, count=HORIZON):
    """Only observed quoted dates are dates; unknown future targets remain NaT."""
    observed = list(pd.to_datetime(df.date.iloc[first:first + count]))
    return observed + [pd.NaT] * (count - len(observed)), len(observed)


def forecast_checkpoint_frame(df, cols, model_dir, modes=MODES, latest=False):
    """Load genuine frozen checkpoints; never estimate weights/scalers at inference."""
    saved = json.loads((model_dir / "metadata.json").read_text())
    if (saved.get("protocol_id") != PROTOCOL_ID or saved["fit_label_end"] != FIT_END or
            saved.get("anchor_policy") != "scheduled_quote_asof"):
        raise ValueError("Checkpoint predates approved protocol; run training into a new output directory.")
    if cols != saved["features"]:
        raise ValueError("Feature names/order differ from fitted model.")
    cfg, lb = saved["config"], saved["config"]["lookback"]
    scale = np.load(model_dir / "scalers.npz")
    x = ((df[cols].to_numpy(float) - scale["x_mean"]) / scale["x_std"]).astype(np.float32)
    first = int(df.date.searchsorted(pd.Timestamp(FIT_END), side="right")) - 1
    stop = int(df.date.searchsorted(pd.Timestamp(TEST_END), side="right")) - 1
    if not latest and df.date.iloc[first] != pd.Timestamp(FIT_END):
        raise ValueError("Missing September 30 fixed-origin feature row.")
    requests = []
    for mode in modes:
        indices = [len(df) - 1] if latest else ([first] if mode == "fixed_origin" else range(first, stop))
        for i in indices:
            if i < lb - 1 or i >= len(df) or i == 0:
                raise ValueError("Insufficient aligned October feature history.")
            requests.append((mode, i))
    if not requests:
        raise ValueError("No target rows in the approved test window.")
    xx = windows(x, [i for _, i in requests], lb)
    forecasts = []
    for seed in saved["seeds"]:
        model = OilLSTM(len(cols), cfg["hidden"], cfg["dropout"])
        model.load_state_dict(torch.load(model_dir / f"lstm_seed_{seed}.pt",
                                         map_location="cpu", weights_only=True))
        model.eval()
        forecasts.append(predict(model, xx) * scale["y_std"] + scale["y_mean"])
    forecasts = np.stack(forecasts)
    records = []
    for j, (mode, i) in enumerate(requests):
        dates, observed = target_calendar(df, i + 1)
        anchor = float(df[ANCHOR].iloc[i])
        origin = df.date.iloc[i]
        for k, date in enumerate(dates):
            realized = k < observed and pd.Timestamp(TEST_START) <= date <= pd.Timestamp(TEST_END)
            actual = float(df.target_wti.iloc[i + k + 1]) if realized else np.nan
            for name, delta in (("LSTM", float(forecasts[:, j, k].mean())), ("Persistence", 0.0)):
                records.append(dict(mode=mode, origin_date=origin, target_date=date, horizon=k + 1,
                                    method=name, model=name, anchor=anchor, origin_price=anchor,
                                    protocol_id=PROTOCOL_ID, target_row=i + k + 1, actual=actual,
                                    fit_end=FIT_END, train_end=TRAIN_END,
                                    validation_start=VALIDATION_START, validation_end=VALIDATION_END,
                                    forecast=anchor + delta, prediction=anchor + delta,
                                    seed_min=anchor + float(forecasts[:, j, k].min()) if name == "LSTM" else anchor,
                                    seed_max=anchor + float(forecasts[:, j, k].max()) if name == "LSTM" else anchor,
                                    target_calendar="quoted" if k < observed else "unknown"))
    return pd.DataFrame(records)


def forecast_from_checkpoint(data_path, dictionary, model_dir, output_path):
    df, cols = read_data(data_path, dictionary)
    result = forecast_checkpoint_frame(df, cols, model_dir, modes=("daily_update",), latest=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return result


def run(args):
    seed_everything(42)
    out, model_dir = args.output, args.output / "models"
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    df, cols = read_data(args.data, args.dictionary)
    n, xraw = len(df), df[cols].to_numpy(float)
    tr_end = boundary(df, TRAIN_END, 0)
    va_end = boundary(df, FIT_END, 0)
    val_start = int(df.date.searchsorted(pd.Timestamp(VALIDATION_START))) - 1
    common_lb = max(c["lookback"] for c in CONFIGS)
    train_idx = published_origins(df, origins_between(n, common_lb, 0, tr_end), TRAIN_END)
    val_idx = published_origins(df, origins_between(n, common_lb, val_start, va_end), FIT_END)
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Empty purged training or historical-validation sample.")
    xm, xs = fit_scaler(xraw[:train_idx[-1] + 1])
    x = (xraw - xm) / xs
    yt, yv = targets(df, train_idx), targets(df, val_idx)
    if not np.isfinite(yt).all() or not np.isfinite(yv).all():
        raise ValueError("Pre-October training and validation labels must be finite.")
    ym, ys = fit_scaler(yt)
    zt, zv = (yt - ym) / ys, (yv - ym) / ys
    selections, histories = [], {}
    for number, cfg in enumerate(CONFIGS):
        _, hist, epoch, score = train_one(
            cfg, 42, windows(x, train_idx, cfg["lookback"]), zt,
            windows(x, val_idx, cfg["lookback"]), zv, epochs=args.max_epochs)
        histories[f"candidate_{number}"] = hist
        selections.append(dict(candidate=number, **cfg, best_epoch=epoch, validation_mse=score))
        print(f"Candidate {number}: historical validation MSE={score:.5f}, epoch={epoch}", flush=True)
    selection = pd.DataFrame(selections)
    selection.to_csv(out / "validation_search.csv", index=False)
    winner = int(selection.validation_mse.idxmin())
    cfg, epochs = CONFIGS[winner], int(selection.iloc[winner].best_epoch)
    fit_idx = published_origins(df, origins_between(n, common_lb, 0, va_end), FIT_END)
    fm, fs = fit_scaler(xraw[:fit_idx[-1] + 1])
    xf = (xraw - fm) / fs
    yf = targets(df, fit_idx)
    if not np.isfinite(yf).all():
        raise ValueError("Final fit labels must be finite.")
    fym, fys = fit_scaler(yf)
    xfit, zf = windows(xf, fit_idx, cfg["lookback"]), (yf - fym) / fys
    for seed in SEEDS:
        model, _, _, _ = train_one(cfg, seed, xfit, zf, epochs=epochs)
        torch.save(model.state_dict(), model_dir / f"lstm_seed_{seed}.pt")
        print(f"Final seed {seed}: {epochs} fixed epochs; {len(fit_idx)} purged origins", flush=True)
    np.savez(model_dir / "scalers.npz", x_mean=fm, x_std=fs, y_mean=fym, y_std=fys)
    metadata = dict(protocol_id=PROTOCOL_ID, config=cfg, seeds=SEEDS, epochs=epochs,
                    features=cols, fit_label_end=FIT_END,
                    last_observed_fit_label=str(df.date.iloc[fit_idx[-1] + HORIZON].date()),
                    training_origin_end=str(df.date.iloc[fit_idx[-1]].date()),
                    train_end=TRAIN_END, validation_start=VALIDATION_START,
                    val_end=VALIDATION_END, test_start=TEST_START, test_end=TEST_END,
                    label_availability="verified WPSR dates, otherwise retrospective weekly-release assumption",
                    anchor_policy="scheduled_quote_asof",
                    horizon=HORIZON,
                    origin_convention="feature window ends at origin row; h predicts origin+h",
                    architecture="direct 60-output path; no future exogenous inputs")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    predictions = forecast_checkpoint_frame(df, cols, model_dir)
    predictions.to_csv(out / "test_predictions.csv", index=False, date_format="%Y-%m-%d")
    metrics = evaluate(predictions)
    metrics.to_csv(out / "test_metrics_key_horizons.csv", index=False)
    forecast_from_checkpoint(args.data, args.dictionary, model_dir, out / "latest_origin_forecast.csv")
    audit = dict(protocol_id=PROTOCOL_ID, rows=n, predictors=len(cols),
                 first_date=str(df.date.iloc[0].date()), last_date=str(df.date.iloc[-1].date()),
                 train_origins=len(train_idx), validation_origins=len(val_idx),
                 final_fit_origins=len(fit_idx),
                 selection_train_label_end=str(df.date.iloc[tr_end].date()),
                 validation_label_end=str(df.date.iloc[va_end].date()),
                 data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                 python=platform.python_version(), torch=torch.__version__,
                 numpy=np.__version__, pandas=pd.__version__, selected=metadata)
    (out / "run_summary.json").write_text(json.dumps(audit, indent=2))
    (out / "training_histories.json").write_text(json.dumps(histories, indent=2))
    print(metrics.to_string(index=False), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=REPO/"data/processed/wti_daily.csv")
    parser.add_argument("--dictionary",type=Path,default=REPO/"data/processed/features.csv")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"frozen_october_2026")
    parser.add_argument("--test-only", action="store_true",
                        help="Score both modes from existing approved-protocol checkpoints.")
    parser.add_argument("--max-epochs",type=int,default=80)
    parser.add_argument("--forecast-only",action="store_true")
    args=parser.parse_args()
    if args.max_epochs < 1:
        parser.error("Max epochs must be positive.")
    if args.test_only:
        df, cols = read_data(args.data, args.dictionary)
        result = forecast_checkpoint_frame(df, cols, args.output / "models")
        result.to_csv(args.output / "test_predictions.csv", index=False, date_format="%Y-%m-%d")
        evaluate(result).to_csv(args.output / "test_metrics_key_horizons.csv", index=False)
    elif args.forecast_only:
        seed_everything(42)
        print(forecast_from_checkpoint(args.data,args.dictionary,args.output/"models",args.output/"latest_origin_forecast.csv").to_string(index=False))
    else:
        run(args)


if __name__=="__main__":
    main()
