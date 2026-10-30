"""Download EIA weekly petroleum fundamentals using an API key from .env."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from data_layout import source_path, INTERIM, MANIFESTS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw" / "eia"
INTERIM_DIR = INTERIM
MANIFEST_PATH = MANIFESTS / "eia_api_download.json"
API_ROOT = "https://api.eia.gov/v2/seriesid"

# Official EIA legacy series IDs exposed through the API v2 seriesid route.
SERIES = {
    "PET.WCESTUS1.W": "commercial_crude_stocks_ex_spr",
    "PET.W_EPC0_SAX_YCUOK_MBBL.W": "cushing_crude_stocks",
    "PET.WCRFPUS2.W": "us_crude_field_production",
    "PET.WCRRIUS2.W": "us_refinery_crude_inputs",
    "PET.WPULEUS3.W": "us_refinery_utilization",
    "PET.WCRIMUS2.W": "us_crude_imports",
    "PET.WCREXUS2.W": "us_crude_exports",
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sanitize(value: Any, api_key: str) -> Any:
    """Remove the API key from EIA's echoed request metadata."""
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if key.lower() == "api_key"
            else sanitize(item, api_key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize(item, api_key) for item in value]
    if isinstance(value, str):
        return value.replace(api_key, "<redacted>")
    return value


def make_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "NUS-QF5210-oil-price-forecasting/0.1"
    return session


def fetch_series(
    session: requests.Session, series_id: str, api_key: str
) -> dict[str, Any]:
    url = f"{API_ROOT}/{series_id}"
    page_size = 5000
    offset = 0
    rows: list[dict[str, Any]] = []
    first_payload: dict[str, Any] | None = None

    while True:
        params = {
            "api_key": api_key,
            "offset": offset,
            "length": page_size,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        try:
            response = session.get(url, params=params, timeout=(30, 180))
        except requests.RequestException as error:
            raise RuntimeError(
                f"EIA request for {series_id} failed ({type(error).__name__})"
            ) from None
        if response.status_code != 200:
            safe_body = response.text.replace(api_key, "<redacted>")[:500]
            raise RuntimeError(
                f"EIA returned HTTP {response.status_code} for {series_id}: "
                f"{safe_body}"
            )

        payload = response.json()
        if first_payload is None:
            first_payload = payload
        response_data = payload.get("response", {})
        page = response_data.get("data", [])
        if not isinstance(page, list):
            raise ValueError(f"Unexpected EIA response structure for {series_id}")
        rows.extend(page)
        total = int(response_data.get("total", len(rows)))
        if len(rows) >= total or not page:
            break
        offset += len(page)

    if first_payload is None or not rows:
        raise ValueError(f"EIA series {series_id} contains no observations")
    first_payload["response"]["data"] = rows
    first_payload["response"]["total"] = str(len(rows))
    return sanitize(first_payload, api_key)


def normalize_rows(payload: dict[str, Any], series_id: str) -> pd.DataFrame:
    rows = payload["response"]["data"]
    frame = pd.DataFrame(rows)
    if "period" not in frame or "value" not in frame:
        raise ValueError(f"EIA series {series_id} lacks period or value")
    frame = frame.rename(columns={"period": "observation_date"})
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], errors="raise"
    )
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame.insert(0, "series_id", series_id)
    frame = (
        frame.sort_values("observation_date")
        .drop_duplicates(subset=["observation_date"], keep="last")
        .reset_index(drop=True)
    )
    if frame["value"].notna().sum() == 0:
        raise ValueError(f"EIA series {series_id} contains no numeric observations")
    return frame


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("EIA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("EIA_API_KEY is missing from .env")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    retrieved_at = datetime.now(UTC).isoformat()
    manifest_records: list[dict[str, Any]] = []
    combined: pd.DataFrame | None = None

    for series_id, short_name in SERIES.items():
        print(f"Downloading EIA {series_id} ({short_name})", flush=True)
        payload = fetch_series(session, series_id, api_key)
        frame = normalize_rows(payload, series_id)

        json_bytes = (
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        json_path = source_path(OUTPUT_DIR / f"{series_id}_{short_name}.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_bytes(json_bytes)

        csv_path = source_path(OUTPUT_DIR / f"{series_id}_{short_name}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_bytes = frame.to_csv(index=False, date_format="%Y-%m-%d").encode(
            "utf-8"
        )
        csv_path.write_bytes(csv_bytes)

        valid = frame.loc[frame["value"].notna(), ["observation_date", "value"]]
        feature = valid.rename(columns={"value": short_name})
        combined = feature if combined is None else combined.merge(
            feature, on="observation_date", how="outer", validate="one_to_one"
        )

        manifest_records.append(
            {
                "source": "U.S. Energy Information Administration",
                "series_id": series_id,
                "name": short_name,
                "url": f"{API_ROOT}/{series_id}",
                "raw_json_file": str(json_path.relative_to(ROOT)),
                "csv_file": str(csv_path.relative_to(ROOT)),
                "raw_json_sha256": sha256_bytes(json_bytes),
                "csv_sha256": sha256_bytes(csv_bytes),
                "rows": int(frame["value"].notna().sum()),
                "first_observation": str(valid["observation_date"].iloc[0].date()),
                "last_observation": str(valid["observation_date"].iloc[-1].date()),
            }
        )

    assert combined is not None
    combined = combined.sort_values("observation_date").reset_index(drop=True)
    combined_path = INTERIM_DIR / "eia_weekly_petroleum_fundamentals.csv"
    combined_bytes = combined.to_csv(
        index=False, date_format="%Y-%m-%d"
    ).encode("utf-8")
    combined_path.write_bytes(combined_bytes)

    manifest = {
        "retrieved_at_utc": retrieved_at,
        "api_key_saved": False,
        "records": manifest_records,
        "combined_file": str(combined_path.relative_to(ROOT)),
        "combined_sha256": sha256_bytes(combined_bytes),
        "availability_note": (
            "observation_date is the week-ending date, not the public release date; "
            "apply a publication lag before daily model alignment"
        ),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Downloaded {len(manifest_records)} EIA series", flush=True)
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}", flush=True)
    print(f"Combined: {combined_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
