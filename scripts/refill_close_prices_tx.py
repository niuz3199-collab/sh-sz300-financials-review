import argparse
import contextlib
import io
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.step5_fetch_close_prices import (
    DEFAULT_BASE_DIR,
    ClosePriceRow,
    load_kept_constituents,
    load_name_map,
    resolve_path,
    write_markdown,
)


def to_tx_symbol(code_name: str) -> str:
    text = str(code_name or "").strip().lower()
    if "." not in text:
        raise ValueError(f"invalid code_name: {code_name!r}")
    market_prefix, code = text.split(".", 1)
    if market_prefix not in {"sh", "sz"}:
        raise ValueError(f"unsupported market prefix: {code_name!r}")
    code = code.strip()
    if not code.isdigit():
        raise ValueError(f"invalid code digits: {code_name!r}")
    return f"{market_prefix}{code.zfill(6)}"


def suppress_output():
    return contextlib.ExitStack()


def load_history_from_tx(
    *,
    code_name: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    max_retries: int,
    retry_backoff: float,
) -> pd.DataFrame:
    tx_symbol = to_tx_symbol(code_name)
    cache_path = cache_dir / f"{tx_symbol}.pkl"
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if cached is not None and not cached.empty:
            return cached

    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, int(max_retries)) + 1):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                df = ak.stock_zh_a_hist_tx(
                    symbol=tx_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="",
                )
            if df is None or df.empty:
                result = pd.DataFrame(columns=["date", "close"])
            else:
                result = df.copy()
                result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
                result["close"] = pd.to_numeric(result["close"], errors="coerce")
                result = result[result["date"].notna() & result["close"].notna()].copy()
                result = result[["date", "close"]].sort_values("date").reset_index(drop=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            result.to_pickle(cache_path)
            return result
        except Exception as exc:
            last_error = exc
            if attempt >= int(max_retries):
                break
            time.sleep(float(retry_backoff) * attempt)

    if last_error is not None:
        raise last_error
    return pd.DataFrame(columns=["date", "close"])


def pick_close_on_or_before(
    history_df: pd.DataFrame,
    *,
    target: date,
    lookback_days: int,
) -> Optional[Tuple[str, float]]:
    if history_df is None or history_df.empty:
        return None
    lower_bound = target - timedelta(days=int(lookback_days))
    tmp = history_df[(history_df["date"] <= target) & (history_df["date"] >= lower_bound)].copy()
    if tmp.empty:
        return None
    tmp = tmp.sort_values("date")
    last = tmp.iloc[-1]
    return last["date"].isoformat(), float(last["close"])


@dataclass
class YearContext:
    kept_df: pd.DataFrame
    name_map: Dict[str, str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refill yearly 4/30 close prices with Tencent history API")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="project root")
    parser.add_argument("--start-year", type=int, default=2006, help="start year inclusive")
    parser.add_argument("--end-year", type=int, default=2024, help="end year inclusive")
    parser.add_argument("--status-csv", default="hs300_with_ipo_status.csv", help="kept constituent source")
    parser.add_argument("--history-json", default="hs300_history.json", help="stock name source")
    parser.add_argument("--cache-dir", default=".cache/tx_price_hist", help="cache dir for per-symbol history")
    parser.add_argument("--overwrite", action="store_true", help="overwrite year csv/md outputs")
    parser.add_argument("--lookback-days", type=int, default=120, help="fallback lookback days")
    parser.add_argument("--sleep", type=float, default=0.02, help="sleep between symbol fetches")
    parser.add_argument("--jitter", type=float, default=0.02, help="random jitter between symbol fetches")
    parser.add_argument("--max-retries", type=int, default=3, help="max retries per symbol history fetch")
    parser.add_argument("--retry-backoff", type=float, default=0.8, help="retry backoff multiplier")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    status_csv_path = resolve_path(base_dir, args.status_csv).resolve()
    history_json_path = resolve_path(base_dir, args.history_json).resolve()
    cache_dir = resolve_path(base_dir, args.cache_dir).resolve()
    stock_name_map_by_year = load_name_map(history_json_path)

    year_contexts: Dict[int, YearContext] = {}
    unique_code_names: List[str] = []
    seen = set()
    for year in range(int(args.start_year), int(args.end_year) + 1):
        kept_df = load_kept_constituents(status_csv_path, year)
        year_contexts[year] = YearContext(kept_df=kept_df, name_map=stock_name_map_by_year.get(str(year), {}))
        for code_name in kept_df["code_name"].astype(str).str.strip().tolist():
            if code_name and code_name not in seen:
                seen.add(code_name)
                unique_code_names.append(code_name)

    print(f"[symbols] {len(unique_code_names)} unique constituents", flush=True)

    history_cache: Dict[str, pd.DataFrame] = {}
    start_fetch = f"{int(args.start_year)}0101"
    end_fetch = f"{int(args.end_year)}1231"
    for idx, code_name in enumerate(unique_code_names, start=1):
        history_cache[code_name] = load_history_from_tx(
            code_name=code_name,
            start_date=start_fetch,
            end_date=end_fetch,
            cache_dir=cache_dir,
            max_retries=int(args.max_retries),
            retry_backoff=float(args.retry_backoff),
        )
        if idx % 20 == 0 or idx == len(unique_code_names):
            print(f"[history] {idx}/{len(unique_code_names)}", flush=True)
        time.sleep(max(0.0, float(args.sleep) + random.random() * float(args.jitter)))

    for year in range(int(args.start_year), int(args.end_year) + 1):
        year_dir = base_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        out_csv = year_dir / f"{year}_0430收盘价.csv"
        out_md = year_dir / f"{year}_0430收盘价.md"
        if not args.overwrite and out_csv.exists() and out_md.exists():
            print(f"[skip] existing outputs for {year}", flush=True)
            continue

        target_date = f"{year}-04-30"
        target = date.fromisoformat(target_date)
        rows: List[ClosePriceRow] = []
        ctx = year_contexts[year]
        for rec in ctx.kept_df.to_dict(orient="records"):
            stock_code = str(rec.get("stock_code", "")).zfill(6)
            code_name = str(rec.get("code_name", "")).strip()
            stock_name = ctx.name_map.get(code_name, "")
            picked = pick_close_on_or_before(
                history_cache.get(code_name, pd.DataFrame()),
                target=target,
                lookback_days=int(args.lookback_days),
            )
            if picked is None:
                rows.append(
                    ClosePriceRow(
                        year=year,
                        target_date=target_date,
                        stock_code=stock_code,
                        code_name=code_name,
                        stock_name=stock_name,
                        price_date="",
                        close=float("nan"),
                        note="no_price",
                    )
                )
            else:
                price_date, close = picked
                note = "" if price_date == target_date else f"fallback_to_{price_date}"
                rows.append(
                    ClosePriceRow(
                        year=year,
                        target_date=target_date,
                        stock_code=stock_code,
                        code_name=code_name,
                        stock_name=stock_name,
                        price_date=price_date,
                        close=close,
                        note=note,
                    )
                )

        out_df = pd.DataFrame([row.__dict__ for row in rows])
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        write_markdown(out_md, title=f"剔除后成分股收盘价 - {year}年4月30日", df=out_df)
        nonnull_count = int(out_df["close"].notna().sum()) if "close" in out_df.columns else 0
        print(f"[year] {year} close_nonnull={nonnull_count}/{len(out_df)} -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
