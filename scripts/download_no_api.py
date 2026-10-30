"""Download public project data that does not require an API key."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from data_layout import source_path, MANIFESTS


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw"
MANIFEST_PATH = MANIFESTS / "public_download.json"

FRED_SERIES = {
    "DCOILWTICO": "wti_spot",
    "DCOILBRENTEU": "brent_spot",
    "DTWEXBGS": "broad_dollar_index",
    "DGS2": "us_treasury_2y",
    "DGS10": "us_treasury_10y",
}

YFINANCE_TICKERS = {
    "CL=F": "wti_nearby_futures",
    "BZ=F": "brent_futures",
    "^GSPC": "sp500",
    "^VIX": "vix",
    "XLE": "energy_select_sector_etf",
    "GC=F": "gold_futures",
    "NG=F": "natural_gas_futures",
    "DX-Y.NYB": "dollar_index_futures",
}

CFTC_WTI_CONTRACT_CODE = "067651"
CFTC_HISTORY_URL = (
    "https://www.cftc.gov/files/dea/history/"
    "fut_disagg_txt_hist_2006_2016.zip"
)
CFTC_ANNUAL_URL = (
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def download_bytes(session: requests.Session, url: str) -> bytes:
    print(f"Downloading {url}", flush=True)
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl:
        result = subprocess.run(
            [
                curl,
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                "300",
                url,
            ],
            check=True,
            capture_output=True,
        )
        return result.stdout

    last_error: requests.RequestException | None = None
    for attempt in range(4):
        try:
            response = session.get(url, timeout=(30, 300))
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            last_error = error
            if attempt < 3:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def download_fred(session: requests.Session) -> list[dict[str, object]]:
    output_dir = DATA_ROOT / "fred"
    records: list[dict[str, object]] = []

    for series_id, name in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        content = download_bytes(session, url)
        path = source_path(output_dir / f"{series_id}_{name}.csv")
        write_bytes(path, content)

        frame = pd.read_csv(io.BytesIO(content))
        values = pd.to_numeric(frame[series_id], errors="coerce")
        valid_dates = frame.loc[values.notna(), "observation_date"]
        if valid_dates.empty:
            raise ValueError(f"FRED series {series_id} contains no observations")

        records.append(
            {
                "source": "FRED",
                "series": series_id,
                "url": url,
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256_bytes(content),
                "bytes": len(content),
                "rows": int(values.notna().sum()),
                "first_observation": valid_dates.iloc[0],
                "last_observation": valid_dates.iloc[-1],
            }
        )

    return records


def download_yfinance() -> list[dict[str, object]]:
    output_dir = DATA_ROOT / "yfinance"
    records: list[dict[str, object]] = []
    end_exclusive = datetime.now(UTC).date().isoformat()

    for ticker, name in YFINANCE_TICKERS.items():
        frame = yf.download(
            ticker,
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            multi_level_index=False,
        )
        if frame.empty:
            raise ValueError(f"yfinance ticker {ticker} contains no observations")

        frame.index.name = "date"
        frame = frame.reset_index()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame[
            frame["date"].dt.date < datetime.fromisoformat(end_exclusive).date()
        ]
        path = source_path(output_dir / f"{name}.csv")
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        write_bytes(path, csv_bytes)

        records.append(
            {
                "source": "yfinance/Yahoo Finance",
                "series": ticker,
                "url": f"https://finance.yahoo.com/quote/{ticker}/history/",
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256_bytes(csv_bytes),
                "bytes": len(csv_bytes),
                "rows": len(frame),
                "first_observation": str(frame["date"].iloc[0].date()),
                "last_observation": str(frame["date"].iloc[-1].date()),
                "request_end_exclusive": end_exclusive,
            }
        )

    return records


def read_cftc_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError(f"Expected one CFTC text file, found {members}")
        with archive.open(members[0]) as stream:
            return pd.read_csv(stream, low_memory=False)


def normalize_cftc_code(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )


def download_cftc(session: requests.Session) -> list[dict[str, object]]:
    output_dir = DATA_ROOT / "cftc"
    current_year = datetime.now(UTC).year
    sources = [("2006_2016", CFTC_HISTORY_URL)] + [
        (str(year), CFTC_ANNUAL_URL.format(year=year))
        for year in range(2017, current_year + 1)
    ]

    frames: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for label, url in sources:
        content = download_bytes(session, url)
        path = source_path(output_dir / f"fut_disagg_txt_{label}.zip")
        write_bytes(path, content)
        frame = read_cftc_zip(content)
        frames.append(frame)
        records.append(
            {
                "source": "CFTC",
                "series": f"Disaggregated Futures Only {label}",
                "url": url,
                "file": str(path.relative_to(ROOT)),
                "sha256": sha256_bytes(content),
                "bytes": len(content),
                "rows": len(frame),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    codes = normalize_cftc_code(combined["CFTC_Contract_Market_Code"])
    wti = combined.loc[codes.eq(CFTC_WTI_CONTRACT_CODE)].copy()
    wti["Report_Date_as_YYYY-MM-DD"] = pd.to_datetime(
        wti["Report_Date_as_YYYY-MM-DD"], errors="raise"
    )
    wti = (
        wti.sort_values("Report_Date_as_YYYY-MM-DD")
        .drop_duplicates(subset=["Report_Date_as_YYYY-MM-DD"], keep="last")
        .reset_index(drop=True)
    )
    if wti.empty:
        raise ValueError("No NYMEX WTI CFTC observations were found")

    path = source_path(output_dir / "wti_067651_disaggregated_futures_only.csv")
    csv_bytes = wti.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
    write_bytes(path, csv_bytes)
    records.append(
        {
            "source": "CFTC",
            "series": "NYMEX WTI disaggregated futures only (067651)",
            "url": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/"
            "HistoricalCompressed/index.htm",
            "file": str(path.relative_to(ROOT)),
            "sha256": sha256_bytes(csv_bytes),
            "bytes": len(csv_bytes),
            "rows": len(wti),
            "first_observation": str(
                wti["Report_Date_as_YYYY-MM-DD"].iloc[0].date()
            ),
            "last_observation": str(
                wti["Report_Date_as_YYYY-MM-DD"].iloc[-1].date()
            ),
        }
    )
    return records


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "NUS-QF5210-oil-price-forecasting/0.1"

    records = []
    records.extend(download_fred(session))
    records.extend(download_yfinance())
    records.extend(download_cftc(session))

    manifest = {
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "records": records,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Downloaded {len(records)} datasets")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
