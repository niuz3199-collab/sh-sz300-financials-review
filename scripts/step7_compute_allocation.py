import argparse
import math
import random
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent


def resolve_path(base_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def _parse_code_name(code_name: str) -> Tuple[str, str]:
    """
    code_name examples:
    - sh.600000
    - sz.000001
    """
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


def _profit_cache_path(cache_dir: Path, code_name: str) -> Path:
    symbol = code_name_to_ak_symbol(code_name)
    return cache_dir / f"{symbol}_profit_yearly.pkl"


def load_profit_yearly(cache_dir: Path, code_name: str, memo: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    symbol = code_name_to_ak_symbol(code_name)
    if symbol in memo:
        return memo[symbol]
    path = _profit_cache_path(cache_dir, code_name)
    if not path.exists():
        memo[symbol] = pd.DataFrame()
        return memo[symbol]
    df = pd.read_pickle(path)
    if df is None or df.empty:
        memo[symbol] = pd.DataFrame()
        return memo[symbol]
    out = df.copy()
    if "report_year" in out.columns:
        out["report_year"] = pd.to_numeric(out["report_year"], errors="coerce")
    if "PARENT_NETPROFIT" in out.columns:
        out["PARENT_NETPROFIT"] = pd.to_numeric(out["PARENT_NETPROFIT"], errors="coerce")
    memo[symbol] = out
    return out


def compute_avg_real_profit(
    code_name: str,
    year: int,
    *,
    cpi_map: Dict[int, float],
    cache_dir: Path,
    profit_memo: Dict[str, pd.DataFrame],
) -> Optional[float]:
    """
    Use cached `*_profit_yearly.pkl` to build a 10-year average real earning (CAPE numerator data).
    Require continuous 10 fiscal years: [year-9 .. year].
    """
    cpi_current = cpi_map.get(year)
    if cpi_current is None:
        return None
    df = load_profit_yearly(cache_dir, code_name, profit_memo)
    if df is None or df.empty:
        return None
    if "report_year" not in df.columns or "PARENT_NETPROFIT" not in df.columns:
        return None

    year_range = list(range(year - 9, year + 1))
    tmp = df[df["report_year"].isin(year_range)].copy()
    tmp = tmp[tmp["report_year"].notna() & tmp["PARENT_NETPROFIT"].notna()].copy()
    if tmp.empty:
        return None
    tmp["report_year"] = tmp["report_year"].astype(int)
    tmp = tmp.sort_values("report_year").drop_duplicates(subset=["report_year"], keep="first")
    if tmp["report_year"].nunique() != 10:
        return None

    tmp["cpi"] = tmp["report_year"].map(cpi_map)
    tmp = tmp[tmp["cpi"].notna()].copy()
    if tmp["report_year"].nunique() != 10:
        return None

    tmp["real_profit"] = tmp["PARENT_NETPROFIT"] * (cpi_current / tmp["cpi"])
    return float(tmp["real_profit"].mean())


def load_cpi_map(path: Path) -> Dict[int, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df is None or df.empty or len(df.columns) < 2:
        return {}
    # Parse the first two columns by position so Chinese header encoding does not matter.
    df = df.iloc[:, :2].copy()
    df.columns = ["year", "CPI"]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["CPI"] = pd.to_numeric(df["CPI"], errors="coerce")
    out: Dict[int, float] = {}
    for _, row in df.iterrows():
        try:
            if pd.isna(row["year"]) or pd.isna(row["CPI"]):
                continue
            y = int(row["year"])
            cpi = float(row["CPI"])
        except Exception:
            continue
        out[y] = cpi
    return out


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


def _pick_latest_on_or_before(df: pd.DataFrame, target: date) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    if "日期" not in df.columns:
        return None
    tmp = df.copy()
    tmp["日期_dt"] = pd.to_datetime(tmp["日期"], errors="coerce").dt.date
    tmp = tmp[tmp["日期_dt"].notna()]
    tmp = tmp[tmp["日期_dt"] <= target]
    if tmp.empty:
        return None
    tmp = tmp.sort_values("日期_dt")
    return tmp.iloc[-1]


def get_10y_yield_real(year: int) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Return (real_rate_decimal, nominal_rate_decimal, used_date_str)
    - Use chinabond gov yield curve 10Y on or before 4/30.
    - If only nominal is available, estimate real = nominal - 0.02
    """
    target = date(year, 4, 30)
    start_date = f"{year}0401"
    end_date = f"{year}0430"
    df = ak.bond_china_yield(start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return None, None, None
    df_gov = df[df["曲线名称"].astype(str).str.contains("国债", na=False)].copy()
    if df_gov.empty:
        return None, None, None
    row = _pick_latest_on_or_before(df_gov, target)
    if row is None:
        return None, None, None
    used_date = str(row["日期"])[:10]
    y10 = _safe_float(row.get("10年"))
    if y10 is None:
        return None, None, used_date
    nominal = float(y10) / 100.0
    real = nominal - 0.02
    return real, nominal, used_date


def cape_percentile(cape: float, u_cape: float) -> float:
    if cape <= 4.78:
        return 100.0
    if cape >= u_cape:
        return 0.0
    return (u_cape - cape) / (u_cape - 4.78) * 100.0


def fcf_percentile(fcf_yield_pct: float) -> float:
    if fcf_yield_pct <= 3.0 or fcf_yield_pct >= 8.0:
        return 0.0
    return (fcf_yield_pct - 3.0) / 5.0 * 100.0


def load_year_close_prices(year_dir: Path, year: int) -> pd.DataFrame:
    path = year_dir / f"{year}_0430收盘价.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 Step5 输出: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["code_name"] = df["code_name"].astype(str).str.strip()
    return df


def load_year_financials(year_dir: Path, year: int) -> pd.DataFrame:
    path = year_dir / f"{year}_财报数据.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 Step6 输出: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["parent_netprofit"] = pd.to_numeric(df.get("parent_netprofit"), errors="coerce")
    df["share_capital"] = pd.to_numeric(df.get("share_capital"), errors="coerce")
    df["netcash_operate"] = pd.to_numeric(df.get("netcash_operate"), errors="coerce")
    df["construct_long_asset"] = pd.to_numeric(df.get("construct_long_asset"), errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["code_name"] = df["code_name"].astype(str).str.strip()
    return df


def compute_for_year(
    year: int,
    *,
    year_financials: pd.DataFrame,
    close_prices: pd.DataFrame,
    cpi_map: Dict[int, float],
    cache_dir: Path,
    profit_memo: Dict[str, pd.DataFrame],
) -> Tuple[Optional[float], Optional[float], int, int, int]:
    """
    Return (cape_index, fcf_yield_pct, cape_n, fcf_n, avg_profit_n).
    Either CAPE/FCF can be None if insufficient data.
    """
    cpi_current = cpi_map.get(year)
    if cpi_current is None:
        return None, None, 0, 0, 0
    if year_financials is None or year_financials.empty:
        return None, None, 0, 0, 0

    # join prices with current-year share capital & cashflow items
    merged = close_prices.merge(
        year_financials[
            [
                "code_name",
                "stock_code",
                "stock_name",
                "share_capital",
                "netcash_operate",
                "construct_long_asset",
            ]
        ],
        on=["code_name", "stock_code"],
        how="left",
        suffixes=("", "_fin"),
    )

    merged["market_cap"] = merged["close"] * merged["share_capital"]
    merged["fcf"] = merged["netcash_operate"] - merged["construct_long_asset"]

    # build 10-year earnings average per stock from cached yearly profit tables.
    avg_real_profit_map: Dict[str, float] = {}
    for code_name in merged["code_name"].dropna().astype(str).unique().tolist():
        code_name = str(code_name).strip()
        if not code_name:
            continue
        try:
            avg = compute_avg_real_profit(
                code_name,
                year,
                cpi_map=cpi_map,
                cache_dir=cache_dir,
                profit_memo=profit_memo,
            )
        except Exception:
            avg = None
        if avg is None:
            continue
        avg_real_profit_map[code_name] = float(avg)

    merged["avg_real_profit"] = merged["code_name"].map(avg_real_profit_map)

    # CAPE index (use stocks with avg_real_profit & market_cap)
    cape_set = merged[(merged["avg_real_profit"].notna()) & (merged["market_cap"].notna())].copy()
    cape_set = cape_set[cape_set["avg_real_profit"] != 0]
    if cape_set.empty:
        cape_index = None
    else:
        market_cap_sum = float(cape_set["market_cap"].sum())
        avg_profit_sum = float(cape_set["avg_real_profit"].sum())
        cape_index = market_cap_sum / avg_profit_sum if avg_profit_sum != 0 else None

    # FCF yield (use stocks with fcf & market_cap)
    fcf_set = merged[(merged["fcf"].notna()) & (merged["market_cap"].notna())].copy()
    if fcf_set.empty:
        fcf_yield_pct = None
    else:
        market_cap_sum = float(fcf_set["market_cap"].sum())
        fcf_sum = float(fcf_set["fcf"].sum())
        fcf_yield_pct = (fcf_sum / market_cap_sum * 100.0) if market_cap_sum != 0 else None

    return cape_index, fcf_yield_pct, int(len(cape_set)), int(len(fcf_set)), int(len(avg_real_profit_map))


def main() -> None:
    parser = argparse.ArgumentParser(description="Step7: 计算沪深300 CAPE/FCF Yield 并生成逐年配置比例")
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="项目根目录（包含 2006/2007/... 年度文件夹与 CPI指数汇总.csv 的目录）",
    )
    parser.add_argument("--start-year", type=int, default=2006, help="起始年份（含）")
    parser.add_argument("--end-year", type=int, default=2024, help="结束年份（含）")
    parser.add_argument("--cpi-csv", default="CPI指数汇总.csv", help="CPI 汇总表")
    parser.add_argument("--cache-dir", default=".cache/financials", help="Step6 缓存目录（用于取10年归母净利润）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在配置文件")
    parser.add_argument("--sleep", type=float, default=0.0, help="每年之间休眠（秒）")
    parser.add_argument("--jitter", type=float, default=0.0, help="随机抖动（秒）")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    cpi_path = resolve_path(base_dir, str(args.cpi_csv)).resolve()
    cache_dir = resolve_path(base_dir, str(args.cache_dir)).resolve()
    cpi_map = load_cpi_map(cpi_path)
    if not cpi_map:
        raise ValueError(f"CPI 汇总表为空或无法解析: {cpi_path}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"未找到 Step6 缓存目录: {cache_dir}（请先运行 step6_extract_financials.py）")
    profit_memo: Dict[str, pd.DataFrame] = {}

    for year in range(int(args.start_year), int(args.end_year) + 1):
        year_dir = base_dir / str(year)
        if not year_dir.exists():
            year_dir.mkdir(parents=True, exist_ok=True)
            print(f"[创建] 年度目录: {year_dir}")

        out_csv = year_dir / f"{year}_配置比例.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[跳过] 已存在: {out_csv}")
            continue

        close_prices = load_year_close_prices(year_dir, year)
        year_financials = load_year_financials(year_dir, year)

        real_rate, nominal_rate, used_yield_date = get_10y_yield_real(year)
        if real_rate is None:
            print(f"[警告] {year} 未获取到 10Y 国债收益率，实际利率将留空")

        cape_index, fcf_yield_pct, cape_n, fcf_n, avg_profit_n = compute_for_year(
            year,
            year_financials=year_financials,
            close_prices=close_prices,
            cpi_map=cpi_map,
            cache_dir=cache_dir,
            profit_memo=profit_memo,
        )
        print(f"[信息] {year}: CAPE样本={cape_n} FCF样本={fcf_n} 10年利润样本={avg_profit_n}")

        if cape_index is None or fcf_yield_pct is None or real_rate is None:
            # 仍然输出一行，便于后续定位缺失年份
            out = pd.DataFrame(
                [
                    {
                        "年份": year,
                        "沪深300CAPE": cape_index,
                        "沪深300FCFYield": fcf_yield_pct,
                        "实际利率": real_rate,
                        "CAPE危险阈值": None,
                        "CAPE百分位": None,
                        "FCF百分位": None,
                        "权益配置比例W": None,
                    }
                ]
            )
            out.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[完成] {year}: 数据不足，已输出空值占位 -> {out_csv}")
            time.sleep(max(0.0, float(args.sleep) + random.random() * float(args.jitter)))
            continue

        u_cape = 1.0 / (real_rate + 0.02)
        p_cape = cape_percentile(float(cape_index), float(u_cape))
        p_fcf = fcf_percentile(float(fcf_yield_pct))
        w = (p_cape + p_fcf) / 2.0

        out = pd.DataFrame(
            [
                {
                    "年份": year,
                    "沪深300CAPE": float(cape_index),
                    "沪深300FCFYield": float(fcf_yield_pct),
                    "实际利率": float(real_rate),
                    "CAPE危险阈值": float(u_cape),
                    "CAPE百分位": float(p_cape),
                    "FCF百分位": float(p_fcf),
                    "权益配置比例W": float(w),
                }
            ]
        )
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(
            f"[完成] {year}: CAPE={cape_index:.2f} FCFYield={fcf_yield_pct:.2f}% W={w:.1f}% -> {out_csv}"
        )
        time.sleep(max(0.0, float(args.sleep) + random.random() * float(args.jitter)))


if __name__ == "__main__":
    main()
