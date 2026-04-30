import argparse
import math
import random
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd
import requests


CHINABOND_SINGLE_INDEX_QUERY_URL = "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQuery"
# 经验：该 indexid 对应 ChinaBond Treasury Bond Aggregate Index（财富/总值）
DEFAULT_BOND_INDEX_ID = "2c9081e50e8767dc010e879acb220021"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent


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


def pick_latest_on_or_before(df: pd.DataFrame, target: date, *, date_col: str, value_col: str) -> Optional[Tuple[date, float]]:
    if df is None or df.empty:
        return None
    if date_col not in df.columns or value_col not in df.columns:
        return None
    tmp = df[[date_col, value_col]].copy()
    tmp["_d"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.date
    tmp["_v"] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp[tmp["_d"].notna() & tmp["_v"].notna()]
    tmp = tmp[tmp["_d"] <= target]
    if tmp.empty:
        return None
    tmp = tmp.sort_values("_d")
    last = tmp.iloc[-1]
    return last["_d"], float(last["_v"])


def fetch_h00300_series(start: str, end: str) -> pd.DataFrame:
    df = ak.stock_zh_index_hist_csindex(symbol="H00300", start_date=start, end_date=end)
    if df is None or df.empty:
        raise RuntimeError("无法获取 H00300.CSI 历史数据")
    # 标准化列名
    if "日期" not in df.columns or "收盘" not in df.columns:
        raise RuntimeError(f"H00300 数据列不符合预期: {df.columns.tolist()}")
    df = df[["日期", "收盘"]].copy()
    return df


def fetch_chinabond_index_series(indexid: str) -> pd.DataFrame:
    """
    从中债（yield.chinabond.com.cn）抓取财富指数（CFZS_00）。
    返回 DataFrame: date, value
    """
    params = {"indexid": indexid, "qxlxt": "00", "zslxt": "CFZS", "lx": "1", "locale": ""}
    r = requests.post(CHINABOND_SINGLE_INDEX_QUERY_URL, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    series = j.get("CFZS_00")
    if not isinstance(series, dict) or not series:
        raise RuntimeError(f"中债指数返回为空: indexid={indexid}")
    temp_df = pd.DataFrame(list(series.items()), columns=["ms", "value"])
    temp_df["ms"] = pd.to_numeric(temp_df["ms"], errors="coerce")
    temp_df["value"] = pd.to_numeric(temp_df["value"], errors="coerce")
    temp_df = temp_df[temp_df["ms"].notna() & temp_df["value"].notna()].copy()
    temp_df["date"] = (
        pd.to_datetime(temp_df["ms"], unit="ms", errors="coerce", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.date
    )
    temp_df = temp_df[temp_df["date"].notna()].copy()
    return temp_df[["date", "value"]].sort_values("date").reset_index(drop=True)


def load_gold_series() -> pd.DataFrame:
    """
    由于 SGE 历史行情数据在 akshare 中从 2016 年起较稳定，这里用 Au99.99 作为黄金现货代理。
    返回列: date, close
    """
    df = ak.spot_hist_sge(symbol="Au99.99")
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)


def resolve_path(base_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def load_w_map(base_dir: Path, start_year: int, end_year: int) -> Dict[int, Optional[float]]:
    """
    从 {year}/{year}_配置比例.csv 读取权益配置比例W（0-100），返回 {year: W_fraction(0-1)}。
    """
    out: Dict[int, Optional[float]] = {}
    for year in range(start_year, end_year + 1):
        path = base_dir / str(year) / f"{year}_配置比例.csv"
        if not path.exists():
            out[year] = None
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty or "权益配置比例W" not in df.columns:
            out[year] = None
            continue
        w = _safe_float(df.iloc[0]["权益配置比例W"])
        out[year] = None if w is None else max(0.0, min(1.0, float(w) / 100.0))
    return out


def load_real_rate_map(base_dir: Path, start_year: int, end_year: int) -> Dict[int, Optional[float]]:
    """
    从 {year}/{year}_配置比例.csv 读取 实际利率（小数，如 0.015），返回 {year: rate}。
    """
    out: Dict[int, Optional[float]] = {}
    for year in range(start_year, end_year + 1):
        path = base_dir / str(year) / f"{year}_配置比例.csv"
        if not path.exists():
            out[year] = None
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty or "实际利率" not in df.columns:
            out[year] = None
            continue
        r = _safe_float(df.iloc[0]["实际利率"])
        out[year] = None if r is None else float(r)
    return out


def annual_return(series_df: pd.DataFrame, *, date_col: str, value_col: str, d0: date, d1: date) -> Optional[float]:
    v0 = pick_latest_on_or_before(series_df, d0, date_col=date_col, value_col=value_col)
    v1 = pick_latest_on_or_before(series_df, d1, date_col=date_col, value_col=value_col)
    if v0 is None or v1 is None:
        return None
    _, a = v0
    _, b = v1
    if a == 0:
        return None
    return b / a - 1.0


@dataclass
class StrategyState:
    total: float = 0.0
    # for multi-asset strategies
    equity: float = 0.0
    bond: float = 0.0
    gold: float = 0.0
    cash: float = 0.0


def rebalance(state: StrategyState, weights: Dict[str, float]) -> None:
    total = state.equity + state.bond + state.gold + state.cash
    state.equity = total * weights.get("equity", 0.0)
    state.bond = total * weights.get("bond", 0.0)
    state.gold = total * weights.get("gold", 0.0)
    state.cash = total * weights.get("cash", 0.0)


def apply_returns(state: StrategyState, r: Dict[str, float]) -> None:
    state.equity *= 1.0 + r.get("equity", 0.0)
    state.bond *= 1.0 + r.get("bond", 0.0)
    state.gold *= 1.0 + r.get("gold", 0.0)
    state.cash *= 1.0 + r.get("cash", 0.0)


def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.fillna(0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step8: 策略回测模拟（按逐年配置比例 W）")
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="项目根目录（包含 2006/2007/... 年度文件夹与配置比例文件的目录）",
    )
    parser.add_argument("--start-year", type=int, default=2006, help="起始年份（含）")
    parser.add_argument("--end-year", type=int, default=2024, help="结束年份（含）")
    parser.add_argument("--initial-capital", type=float, default=0.0, help="起始时一次性投入金额")
    parser.add_argument("--annual-contribution", type=float, default=10000.0, help="每年新增投入金额")
    parser.add_argument("--bond-index-id", default=DEFAULT_BOND_INDEX_ID, help="中债财富指数 indexid")
    parser.add_argument("--out-dir", default="backtest_output", help="回测输出目录（默认在 base-dir 下）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖输出目录内已有结果文件")
    parser.add_argument(
        "--year-detail-suffix",
        default="",
        help="年度明细文件后缀，例如 _initial100k，避免覆盖原有文件",
    )
    args = parser.parse_args()

    start_year = int(args.start_year)
    end_year = int(args.end_year)
    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = resolve_path(base_dir, args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load W map
    w_map = load_w_map(base_dir, start_year, end_year)
    real_rate_map = load_real_rate_map(base_dir, start_year, end_year)

    # Fetch asset series
    eq_df = fetch_h00300_series(start=f"{start_year}0101", end=f"{end_year + 1}1231")
    bond_df = fetch_chinabond_index_series(indexid=str(args.bond_index_id))
    gold_df = load_gold_series()

    # Determine gold availability
    gold_start = None
    if not gold_df.empty:
        gold_start = pd.to_datetime(gold_df["date"], errors="coerce").min()
        gold_start = None if pd.isna(gold_start) else gold_start.date()

    # Strategy states
    A = StrategyState()
    B = StrategyState()
    # Strategy C starts with initial capital in equity, then uses the existing
    # half-year holding approximation for later contributions.
    C_total = float(args.initial_capital)
    D = StrategyState()

    initial_capital = float(args.initial_capital)
    if initial_capital:
        A.cash += initial_capital
        B.cash += initial_capital
        D.cash += initial_capital

    principal = initial_capital
    rows: List[Dict] = []

    for year in range(start_year, end_year + 1):
        d0 = date(year, 4, 30)
        d1 = date(year + 1, 4, 30)

        r_eq = annual_return(eq_df, date_col="日期", value_col="收盘", d0=d0, d1=d1)
        r_bond = annual_return(bond_df, date_col="date", value_col="value", d0=d0, d1=d1)
        r_gold = annual_return(gold_df, date_col="date", value_col="close", d0=d0, d1=d1) if not gold_df.empty else None

        if r_eq is None or r_bond is None:
            print(f"[跳过] {year}: 缺少股/债收益率数据（eq={r_eq} bond={r_bond}）")
            continue

        principal += float(args.annual_contribution)

        # Strategy A: dynamic equity/bond with W
        w = w_map.get(year)
        if w is None:
            # fallback: 50/50 if missing
            w = 0.5
        A.cash += float(args.annual_contribution)
        rebalance(A, {"equity": float(w), "bond": float(1.0 - w), "gold": 0.0, "cash": 0.0})
        apply_returns(A, {"equity": r_eq, "bond": r_bond, "gold": 0.0, "cash": 0.0})

        # Strategy B: yearly lump sum 100% equity
        B.cash += float(args.annual_contribution)
        rebalance(B, {"equity": 1.0, "bond": 0.0, "gold": 0.0, "cash": 0.0})
        apply_returns(B, {"equity": r_eq, "bond": 0.0, "gold": 0.0, "cash": 0.0})

        # Strategy C: monthly DCA approximation into equity
        # (yearly return data used; contribution assumed average holding time = 0.5 year)
        C_total = C_total * (1.0 + r_eq) + float(args.annual_contribution) * (1.0 + 0.5 * r_eq)

        # Strategy D: permanent portfolio (25/25/25/25)
        # If gold data is not available for this window, treat gold return as 0 and keep going.
        D.cash += float(args.annual_contribution)
        rebalance(D, {"equity": 0.25, "bond": 0.25, "gold": 0.25, "cash": 0.25})
        apply_returns(
            D,
            {
                "equity": r_eq,
                "bond": r_bond,
                "gold": 0.0 if r_gold is None else r_gold,
                "cash": 0.0,
            },
        )

        A_total = A.equity + A.bond + A.gold + A.cash
        B_total = B.equity + B.bond + B.gold + B.cash
        D_total = D.equity + D.bond + D.gold + D.cash

        rows.append(
            {
                "年份": year,
                "投入本金": principal,
                "策略A_资产": A_total,
                "策略B_资产": B_total,
                "策略C_资产": C_total,
                "策略D_资产": D_total,
                "当年股收益率": r_eq,
                "当年债收益率": r_bond,
                "当年金收益率": r_gold,
                "W": w,
            }
        )

        # also write per-year detail into the year folder (optional)
        year_dir = base_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        detail_suffix = str(args.year_detail_suffix or "")
        per_year_name = (
            f"{year}_策略资产明细{detail_suffix}.csv"
            if detail_suffix
            else f"{year}_策略资产明细.csv"
        )
        per_year_path = year_dir / per_year_name
        pd.DataFrame([rows[-1]]).to_csv(per_year_path, index=False, encoding="utf-8-sig")

    if not rows:
        raise RuntimeError("没有生成任何回测记录：请先确保 Step7 的配置比例文件齐全，并且可获取指数数据。")

    df = pd.DataFrame(rows).sort_values("年份").reset_index(drop=True)
    df.to_csv(out_dir / "策略资产明细.csv", index=False, encoding="utf-8-sig")

    # Compute metrics
    metrics_rows = []
    strategies = {
        "策略A": "策略A_资产",
        "策略B": "策略B_资产",
        "策略C": "策略C_资产",
        "策略D": "策略D_资产",
    }
    n_years = len(df)
    # 无风险利率：用 Step7 的“实际利率 + 2%”近似名义 10Y 国债收益率，再取均值
    nominal_candidates = []
    for y in df["年份"].tolist():
        rr = real_rate_map.get(int(y))
        if rr is None:
            continue
        nominal_candidates.append(rr + 0.02)
    risk_free = float(sum(nominal_candidates) / len(nominal_candidates)) if nominal_candidates else 0.0

    for name, col in strategies.items():
        series = pd.to_numeric(df[col], errors="coerce")
        series = series.ffill().fillna(0.0)
        principal_series = pd.to_numeric(df["投入本金"], errors="coerce").ffill()
        final_value = float(series.iloc[-1])
        final_principal = float(principal_series.iloc[-1])
        cum_ret = (final_value - final_principal) / final_principal if final_principal else float("nan")
        ann_ret = (final_value / final_principal) ** (1.0 / n_years) - 1.0 if final_principal else float("nan")

        # annual returns for volatility
        rets = series.pct_change().dropna()
        vol = float(rets.std(ddof=0)) if not rets.empty else float("nan")
        sharpe = (ann_ret - risk_free) / vol if vol and not math.isnan(vol) else float("nan")

        dd = compute_drawdown(series)
        max_dd = float(dd.min()) if not dd.empty else 0.0
        rr = (cum_ret / abs(max_dd)) if max_dd != 0 else float("nan")

        metrics_rows.append(
            {
                "策略": name,
                "期末总资产": final_value,
                "累计投入本金": final_principal,
                "累计收益率": cum_ret,
                "年化收益率": ann_ret,
                "最大回撤": max_dd,
                "夏普比率": sharpe,
                "收益风险比": rr,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "回测结果对比.csv", index=False, encoding="utf-8-sig")

    # Plots
    plt.figure(figsize=(10, 6))
    for name, col in strategies.items():
        plt.plot(df["年份"], df[col], label=name)
    plt.title("累计净值走势（总资产）")
    plt.xlabel("年份")
    plt.ylabel("资产（元）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "累计净值走势.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, col in strategies.items():
        series = pd.to_numeric(df[col], errors="coerce").ffill()
        dd = compute_drawdown(series)
        plt.plot(df["年份"], dd, label=name)
    plt.title("回撤对比")
    plt.xlabel("年份")
    plt.ylabel("回撤")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "回撤对比.png", dpi=150)
    plt.close()

    # Simple markdown report
    report_lines = []
    report_lines.append("# 策略表现分析报告")
    report_lines.append("")
    report_lines.append(f"- 回测区间: {int(df['年份'].min())} ~ {int(df['年份'].max())}（按 4/30 年度持有期近似）")
    report_lines.append(f"- 起始资金: {args.initial_capital:.2f} 元")
    report_lines.append(f"- 年度投入: {args.annual_contribution:.2f} 元")
    report_lines.append(f"- 债券指数: indexid={args.bond_index_id}（财富/总值）")
    if gold_start:
        report_lines.append(f"- 黄金数据起始: {gold_start}（早于该日期的黄金收益率用 0 近似）")
    report_lines.append("")
    report_lines.append("## 回测结果对比")
    report_lines.append("")
    report_lines.append(metrics_df.to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## 关键说明")
    report_lines.append("")
    report_lines.append("- 策略C（月度定投）在本实现中采用年度收益率的“半年度投入近似”。")
    report_lines.append("- 策略D（永久组合）黄金现货数据在 akshare 中可用区间有限，早期年份用 0 收益率近似，建议后续替换为更长历史的黄金价格序列。")

    (out_dir / "策略表现分析报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[完成] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
