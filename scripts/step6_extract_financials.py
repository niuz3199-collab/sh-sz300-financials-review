import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent


@dataclass(frozen=True)
class FinancialRow:
    year: int
    stock_code: str
    code_name: str
    stock_name: str
    parent_netprofit: Optional[float]
    share_capital: Optional[float]
    netcash_operate: Optional[float]
    construct_long_asset: Optional[float]


def _parse_code_name(code_name: str) -> Tuple[str, str]:
    text = (code_name or "").strip()
    if "." not in text:
        raise ValueError(f"Invalid code_name: {code_name!r}")
    market_prefix, code = text.split(".", 1)
    market_prefix = market_prefix.lower().strip()
    code = code.strip()
    if market_prefix not in {"sh", "sz"}:
        raise ValueError(f"Unsupported market prefix: {code_name!r}")
    if not code.isdigit():
        raise ValueError(f"Invalid code: {code_name!r}")
    return market_prefix, code.zfill(6)


def code_name_to_ak_symbol(code_name: str) -> str:
    market_prefix, code = _parse_code_name(code_name)
    market = "SH" if market_prefix == "sh" else "SZ"
    return f"{market}{code}"


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, float):
            if pd.isna(value):
                return None
            return float(value)
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except Exception:
        return None


def load_kept_constituents(status_csv_path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(status_csv_path, dtype=str, encoding="utf-8")
    df_year = df[df["year"].astype(str) == str(year)].copy()
    status = df_year.get("剔除状态")
    if status is None:
        kept = df_year
    else:
        kept = df_year[status.isna() | (status.astype(str).str.strip() == "")]
    kept = kept[["stock_code", "code_name"]].copy()
    kept["stock_code"] = kept["stock_code"].astype(str).str.zfill(6)
    kept["code_name"] = kept["code_name"].astype(str).str.strip()
    kept = kept.drop_duplicates(subset=["code_name"]).sort_values("code_name")
    return kept.reset_index(drop=True)


def resolve_path(base_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def _cache_path(cache_dir: Path, symbol: str, kind: str) -> Path:
    safe = symbol.replace(".", "_")
    return cache_dir / f"{safe}_{kind}.pkl"


def _prepare_yearly_df(df: pd.DataFrame, *, keep_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["report_year"] + keep_cols)
    out = df.copy()
    # only annual reports
    if "REPORT_TYPE" in out.columns:
        out = out[out["REPORT_TYPE"].astype(str).str.strip() == "年报"].copy()
    if "REPORT_DATE" not in out.columns:
        return pd.DataFrame(columns=["report_year"] + keep_cols)
    out["report_year"] = out["REPORT_DATE"].astype(str).str.slice(0, 4)
    out = out[out["report_year"].str.fullmatch(r"\d{4}", na=False)].copy()
    out["report_year"] = out["report_year"].astype(int)
    cols = ["report_year"] + [c for c in keep_cols if c in out.columns]
    out = out[cols].drop_duplicates(subset=["report_year"], keep="first").sort_values("report_year")
    return out.reset_index(drop=True)


def load_or_fetch_profit(cache_dir: Path, symbol: str, *, force: bool = False) -> pd.DataFrame:
    cache_path = _cache_path(cache_dir, symbol, "profit_yearly")
    if cache_path.exists() and not force:
        return pd.read_pickle(cache_path)
    df = ak.stock_profit_sheet_by_yearly_em(symbol=symbol)
    keep_cols = ["SECURITY_NAME_ABBR", "PARENT_NETPROFIT"]
    out = _prepare_yearly_df(df, keep_cols=keep_cols)
    out.to_pickle(cache_path)
    return out


def load_or_fetch_cashflow(cache_dir: Path, symbol: str, *, force: bool = False) -> pd.DataFrame:
    cache_path = _cache_path(cache_dir, symbol, "cashflow_yearly")
    if cache_path.exists() and not force:
        return pd.read_pickle(cache_path)
    df = ak.stock_cash_flow_sheet_by_yearly_em(symbol=symbol)
    keep_cols = ["NETCASH_OPERATE", "CONSTRUCT_LONG_ASSET"]
    out = _prepare_yearly_df(df, keep_cols=keep_cols)
    out.to_pickle(cache_path)
    return out


def load_or_fetch_balance(cache_dir: Path, symbol: str, *, force: bool = False) -> pd.DataFrame:
    cache_path = _cache_path(cache_dir, symbol, "balance_yearly")
    if cache_path.exists() and not force:
        return pd.read_pickle(cache_path)
    df = ak.stock_balance_sheet_by_yearly_em(symbol=symbol)
    keep_cols = ["SHARE_CAPITAL"]
    out = _prepare_yearly_df(df, keep_cols=keep_cols)
    out.to_pickle(cache_path)
    return out


def _retry_fetch(fetch_fn, *, retries: int = 4, base_sleep: float = 0.8):
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return fetch_fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(base_sleep * attempt)
    raise RuntimeError(last_exc)  # type: ignore[arg-type]


def get_value_for_year(df: pd.DataFrame, year: int, col: str) -> Optional[float]:
    if df is None or df.empty:
        return None
    if "report_year" not in df.columns or col not in df.columns:
        return None
    hit = df[df["report_year"] == int(year)]
    if hit.empty:
        return None
    return _safe_float(hit.iloc[0][col])


def get_name_for_year(df_profit: pd.DataFrame, year: int) -> str:
    if df_profit is None or df_profit.empty:
        return ""
    if "report_year" not in df_profit.columns or "SECURITY_NAME_ABBR" not in df_profit.columns:
        return ""
    hit = df_profit[df_profit["report_year"] == int(year)]
    if hit.empty:
        return ""
    return str(hit.iloc[0]["SECURITY_NAME_ABBR"] or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Step6: 提取年报关键字段（归母净利润/总股本/经营现金流/CapEx）")
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="项目根目录（包含 2006/2007/... 年度文件夹与 hs300_with_ipo_status.csv 的目录）",
    )
    parser.add_argument("--start-year", type=int, default=2006, help="起始年份（含）")
    parser.add_argument("--end-year", type=int, default=2024, help="结束年份（含）")
    parser.add_argument("--status-csv", default="hs300_with_ipo_status.csv", help="剔除状态文件")
    parser.add_argument("--cache-dir", default=".cache/financials", help="缓存目录（避免重复拉取）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在年度输出")
    parser.add_argument("--resume", action="store_true", help="若年度输出已存在则续跑（跳过已完成股票）")
    parser.add_argument("--force-refresh-cache", action="store_true", help="强制刷新缓存（慢）")
    parser.add_argument("--sleep", type=float, default=0.2, help="每只股票抓取之间的基础休眠（秒）")
    parser.add_argument("--jitter", type=float, default=0.5, help="随机抖动（秒）")
    parser.add_argument("--start", type=int, default=0, help="从第 N 只股票开始（用于断点）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少只股票（0 表示全部）")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    status_csv_path = resolve_path(base_dir, args.status_csv).resolve()
    cache_dir = resolve_path(base_dir, args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for year in range(int(args.start_year), int(args.end_year) + 1):
        year_dir = base_dir / str(year)
        if not year_dir.exists():
            year_dir.mkdir(parents=True, exist_ok=True)
            print(f"[创建] 年度目录: {year_dir}")

        out_csv = year_dir / f"{year}_财报数据.csv"
        existing_df: Optional[pd.DataFrame] = None
        done_code_names: set[str] = set()
        if out_csv.exists() and not args.overwrite:
            if not args.resume:
                print(f"[跳过] 已存在: {out_csv}")
                continue
            try:
                existing_df = pd.read_csv(out_csv, dtype=str, encoding="utf-8-sig")
                if "code_name" in existing_df.columns:
                    done_code_names = set(existing_df["code_name"].dropna().astype(str).str.strip().tolist())
                print(f"[续跑] 已存在 {len(done_code_names)} 条记录: {out_csv}")
            except Exception as exc:
                print(f"[警告] 读取既有文件失败，将重新生成: {out_csv} - {exc}")
                existing_df = None
                done_code_names = set()

        kept = load_kept_constituents(status_csv_path, year)
        if done_code_names:
            kept = kept[~kept["code_name"].astype(str).str.strip().isin(done_code_names)].reset_index(drop=True)

        start = max(0, int(args.start))
        end = len(kept) if int(args.limit) <= 0 else min(len(kept), start + int(args.limit))
        kept = kept.iloc[start:end].reset_index(drop=True)

        print(f"\n--- {year}: 提取财报字段（待处理股票数: {len(kept)}） ---")

        out_rows: List[FinancialRow] = []

        for i, rec in enumerate(kept.to_dict(orient="records"), start=1):
            stock_code = str(rec.get("stock_code", "")).zfill(6)
            code_name = str(rec.get("code_name", "")).strip()
            symbol = code_name_to_ak_symbol(code_name)

            df_profit = pd.DataFrame()
            df_cash = pd.DataFrame()
            df_bal = pd.DataFrame()

            profit_err: Optional[str] = None
            cash_err: Optional[str] = None
            bal_err: Optional[str] = None

            # If cache is already present for all 3 tables, avoid sleeping (fast path).
            # If cache is missing (or forced refresh), we likely hit the network, so keep a delay.
            profit_cached = _cache_path(cache_dir, symbol, "profit_yearly").exists()
            cash_cached = _cache_path(cache_dir, symbol, "cashflow_yearly").exists()
            bal_cached = _cache_path(cache_dir, symbol, "balance_yearly").exists()
            need_sleep = bool(args.force_refresh_cache) or (not (profit_cached and cash_cached and bal_cached))

            try:
                df_profit = _retry_fetch(
                    lambda: load_or_fetch_profit(cache_dir, symbol, force=bool(args.force_refresh_cache)),
                    retries=4,
                )
            except Exception as exc:
                profit_err = str(exc)
            try:
                df_cash = _retry_fetch(
                    lambda: load_or_fetch_cashflow(cache_dir, symbol, force=bool(args.force_refresh_cache)),
                    retries=4,
                )
            except Exception as exc:
                cash_err = str(exc)
            try:
                df_bal = _retry_fetch(
                    lambda: load_or_fetch_balance(cache_dir, symbol, force=bool(args.force_refresh_cache)),
                    retries=4,
                )
            except Exception as exc:
                bal_err = str(exc)

            if profit_err or cash_err or bal_err:
                err_msg = "; ".join(
                    [m for m in [f"profit={profit_err}" if profit_err else "", f"cash={cash_err}" if cash_err else "", f"bal={bal_err}" if bal_err else ""] if m]
                )
                print(f"[警告] {symbol} 部分拉取失败: {err_msg}")

            stock_name = get_name_for_year(df_profit, year)
            parent_netprofit = get_value_for_year(df_profit, year, "PARENT_NETPROFIT")
            share_capital = get_value_for_year(df_bal, year, "SHARE_CAPITAL")
            netcash_operate = get_value_for_year(df_cash, year, "NETCASH_OPERATE")
            construct_long_asset = get_value_for_year(df_cash, year, "CONSTRUCT_LONG_ASSET")

            out_rows.append(
                FinancialRow(
                    year=year,
                    stock_code=stock_code,
                    code_name=code_name,
                    stock_name=stock_name,
                    parent_netprofit=parent_netprofit,
                    share_capital=share_capital,
                    netcash_operate=netcash_operate,
                    construct_long_asset=construct_long_asset,
                )
            )

            if i % 20 == 0:
                print(f"[进度] {i}/{len(kept)}")

            if need_sleep:
                time.sleep(max(0.0, float(args.sleep) + random.random() * float(args.jitter)))

        df_out = pd.DataFrame([r.__dict__ for r in out_rows])
        if existing_df is not None and not existing_df.empty:
            # 合并旧数据（旧数据可能为字符串，这里尽量保留原始字段并去重）
            df_old = existing_df.copy()
            df_old["year"] = pd.to_numeric(df_old.get("year"), errors="coerce").fillna(year).astype(int)
            for col in ["stock_code", "code_name", "stock_name"]:
                if col in df_old.columns:
                    df_old[col] = df_old[col].astype(str)
            df_out = pd.concat([df_old, df_out], ignore_index=True)
            df_out = df_out.drop_duplicates(subset=["year", "code_name"], keep="first")
            df_out = df_out.sort_values(["year", "code_name"]).reset_index(drop=True)
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[完成] 输出: {out_csv}")


if __name__ == "__main__":
    main()
