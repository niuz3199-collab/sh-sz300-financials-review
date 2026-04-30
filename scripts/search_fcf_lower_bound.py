#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_latest_experiment_root() -> Optional[Path]:
    candidates = sorted(
        [p for p in REPO_ROOT.glob("allocation_transform_experiments_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def resolve_default_signal_csv() -> Path:
    latest_root = find_latest_experiment_root()
    if latest_root is None:
        raise FileNotFoundError("未找到 allocation_transform_experiments_* 目录，请先运行实验脚本。")
    path = latest_root / "base_signal_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到原始信号表: {path}")
    return path


def resolve_default_returns_csv() -> Path:
    path = REPO_ROOT / "backtest_output_initial100k_nodca" / "策略资产明细.csv"
    if path.exists():
        return path
    raise FileNotFoundError("未找到 backtest_output_initial100k_nodca/策略资产明细.csv")


def resolve_default_metrics_csv() -> Path:
    path = REPO_ROOT / "backtest_output_initial100k_nodca" / "回测结果对比.csv"
    if path.exists():
        return path
    raise FileNotFoundError("未找到 backtest_output_initial100k_nodca/回测结果对比.csv")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def frange(start: float, end: float, step: float) -> List[float]:
    values: List[float] = []
    cur = float(start)
    max_iters = 100000
    i = 0
    while cur <= end + 1e-12 and i < max_iters:
        values.append(round(cur, 10))
        cur += float(step)
        i += 1
    return values


def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.fillna(0.0)


@dataclass
class Metrics:
    final_asset: float
    cum_return: float
    ann_return: float
    max_drawdown: float
    sharpe: float
    risk_return_ratio: float


def compute_metrics(series: pd.Series, principal: float, risk_free: float) -> Metrics:
    series = pd.to_numeric(series, errors="coerce").ffill().fillna(0.0)
    n_years = len(series)
    final_value = float(series.iloc[-1])
    cum_ret = (final_value - principal) / principal if principal else float("nan")
    ann_ret = (final_value / principal) ** (1.0 / n_years) - 1.0 if principal else float("nan")
    rets = series.pct_change().dropna()
    vol = float(rets.std(ddof=0)) if not rets.empty else float("nan")
    sharpe = (ann_ret - risk_free) / vol if vol and not math.isnan(vol) else float("nan")
    dd = compute_drawdown(series)
    max_dd = float(dd.min()) if not dd.empty else 0.0
    rr = (cum_ret / abs(max_dd)) if max_dd != 0 else float("nan")
    return Metrics(
        final_asset=final_value,
        cum_return=cum_ret,
        ann_return=ann_ret,
        max_drawdown=max_dd,
        sharpe=sharpe,
        risk_return_ratio=rr,
    )


def fcf_percentile_with_bounds(fcf_yield_pct: Optional[float], lower: float, upper: float) -> Optional[float]:
    if fcf_yield_pct is None or pd.isna(fcf_yield_pct):
        return None
    if lower >= upper:
        return 0.0
    x = float(fcf_yield_pct)
    if x <= lower or x >= upper:
        return 0.0
    return (x - lower) / (upper - lower) * 100.0


def run_backtest_with_lower_bound(
    *,
    merged_df: pd.DataFrame,
    lower_bound: float,
    upper_bound: float,
    initial_capital: float,
) -> Dict[str, object]:
    principal = float(initial_capital)
    equity = 0.0
    bond = 0.0
    cash = float(initial_capital)
    rows: List[Dict[str, object]] = []

    for _, row in merged_df.iterrows():
        p_cape = row.get("p_cape")
        fcf_yield_pct = row.get("fcf_yield_pct")
        p_fcf = fcf_percentile_with_bounds(fcf_yield_pct, lower=lower_bound, upper=upper_bound)
        if p_cape is None or pd.isna(p_cape):
            w = 0.5
        else:
            w = clamp01((float(p_cape) + float(p_fcf or 0.0)) / 200.0)

        total = equity + bond + cash
        equity = total * w
        bond = total * (1.0 - w)
        cash = 0.0

        r_eq = float(row["当年股收益率"])
        r_bond = float(row["当年债收益率"])
        equity *= 1.0 + r_eq
        bond *= 1.0 + r_bond
        total = equity + bond + cash
        rows.append(
            {
                "年份": int(row["year"]),
                "策略A_资产": total,
                "W": w,
                "W_pct": w * 100.0,
                "p_cape": float(p_cape) if p_cape is not None and not pd.isna(p_cape) else None,
                "p_fcf_new": float(p_fcf) if p_fcf is not None else None,
                "fcf_yield_pct": float(fcf_yield_pct) if fcf_yield_pct is not None and not pd.isna(fcf_yield_pct) else None,
                "当年股收益率": r_eq,
                "当年债收益率": r_bond,
            }
        )

    detail_df = pd.DataFrame(rows)
    return {"detail_df": detail_df}


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid search the FCF lower bound under the current allocation mapping.")
    parser.add_argument("--signal-csv", default="", help="默认使用最新实验目录下的 base_signal_table.csv")
    parser.add_argument("--returns-csv", default="", help="默认使用 backtest_output_initial100k_nodca/策略资产明细.csv")
    parser.add_argument("--metrics-csv", default="", help="默认使用 backtest_output_initial100k_nodca/回测结果对比.csv")
    parser.add_argument("--lower-start", type=float, default=-6.0)
    parser.add_argument("--lower-end", type=float, default=4.0)
    parser.add_argument("--lower-step", type=float, default=0.1)
    parser.add_argument("--upper-bound", type=float, default=8.0)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--out-dir", default="", help="输出目录；默认按时间戳新建")
    args = parser.parse_args()

    signal_csv = Path(args.signal_csv).expanduser().resolve() if str(args.signal_csv or "").strip() else resolve_default_signal_csv()
    returns_csv = Path(args.returns_csv).expanduser().resolve() if str(args.returns_csv or "").strip() else resolve_default_returns_csv()
    metrics_csv = Path(args.metrics_csv).expanduser().resolve() if str(args.metrics_csv or "").strip() else resolve_default_metrics_csv()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir or "").strip()
        else (REPO_ROOT / f"fcf_lower_bound_search_{now_stamp()}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_df = pd.read_csv(signal_csv, encoding="utf-8-sig")
    returns_df = pd.read_csv(returns_csv, encoding="utf-8-sig")
    metrics_df = pd.read_csv(metrics_csv, encoding="utf-8-sig")

    signal_df["year"] = pd.to_numeric(signal_df["year"], errors="coerce")
    signal_df["fcf_yield_pct"] = pd.to_numeric(signal_df["fcf_yield_pct"], errors="coerce")
    signal_df["p_cape"] = pd.to_numeric(signal_df["p_cape"], errors="coerce")
    signal_df["real_rate"] = pd.to_numeric(signal_df["real_rate"], errors="coerce")

    returns_df["年份"] = pd.to_numeric(returns_df["年份"], errors="coerce")
    returns_df["当年股收益率"] = pd.to_numeric(returns_df["当年股收益率"], errors="coerce")
    returns_df["当年债收益率"] = pd.to_numeric(returns_df["当年债收益率"], errors="coerce")
    returns_df["策略D_资产"] = pd.to_numeric(returns_df["策略D_资产"], errors="coerce")

    merged_df = signal_df.merge(
        returns_df[["年份", "当年股收益率", "当年债收益率", "策略D_资产"]],
        left_on="year",
        right_on="年份",
        how="inner",
    )
    merged_df = merged_df.dropna(subset=["year", "当年股收益率", "当年债收益率"]).sort_values("year").reset_index(drop=True)

    nominal_candidates = [float(x) + 0.02 for x in signal_df["real_rate"].dropna().tolist()]
    risk_free = float(sum(nominal_candidates) / len(nominal_candidates)) if nominal_candidates else 0.0

    d_series = pd.to_numeric(merged_df["策略D_资产"], errors="coerce").ffill().fillna(0.0)
    d_metrics = compute_metrics(d_series, principal=float(args.initial_capital), risk_free=risk_free)

    result_rows: List[Dict[str, object]] = []
    best_detail_by_final: Optional[pd.DataFrame] = None
    best_final_value = -float("inf")

    for lower_bound in frange(float(args.lower_start), float(args.lower_end), float(args.lower_step)):
        backtest = run_backtest_with_lower_bound(
            merged_df=merged_df,
            lower_bound=lower_bound,
            upper_bound=float(args.upper_bound),
            initial_capital=float(args.initial_capital),
        )
        detail_df = backtest["detail_df"]
        metrics = compute_metrics(detail_df["策略A_资产"], principal=float(args.initial_capital), risk_free=risk_free)

        zeroed_positive = merged_df[
            (pd.to_numeric(merged_df["当年股收益率"], errors="coerce") > 0)
            & (pd.to_numeric(merged_df["fcf_yield_pct"], errors="coerce") <= lower_bound)
        ]
        active_fcf = detail_df["p_fcf_new"].fillna(0.0) > 0
        row_2007 = detail_df[detail_df["年份"] == 2007]
        row_2008 = detail_df[detail_df["年份"] == 2008]
        ret_2008 = None
        if not row_2007.empty and not row_2008.empty:
            v0 = float(row_2007.iloc[0]["策略A_资产"])
            v1 = float(row_2008.iloc[0]["策略A_资产"])
            if v0:
                ret_2008 = v1 / v0 - 1.0

        result_rows.append(
            {
                "lower_bound": lower_bound,
                "upper_bound": float(args.upper_bound),
                "final_asset_A": metrics.final_asset,
                "cum_return_A": metrics.cum_return,
                "ann_return_A": metrics.ann_return,
                "max_drawdown_A": metrics.max_drawdown,
                "sharpe_A": metrics.sharpe,
                "risk_return_ratio_A": metrics.risk_return_ratio,
                "return_2008_A": ret_2008,
                "active_fcf_years": int(active_fcf.sum()),
                "zeroed_positive_return_years": int(len(zeroed_positive)),
                "beats_D_final_asset": bool(metrics.final_asset > d_metrics.final_asset),
                "beats_D_drawdown": bool(metrics.max_drawdown >= d_metrics.max_drawdown),
            }
        )

        if metrics.final_asset > best_final_value:
            best_final_value = metrics.final_asset
            best_detail_by_final = detail_df.copy()

    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(["final_asset_A", "max_drawdown_A"], ascending=[False, False]).reset_index(drop=True)
    results_df.to_csv(out_dir / "fcf_lower_bound_search_results.csv", index=False, encoding="utf-8-sig")

    if best_detail_by_final is not None:
        best_detail_by_final.to_csv(out_dir / "best_final_asset_detail.csv", index=False, encoding="utf-8-sig")

    d_summary = pd.DataFrame(
        [
            {
                "strategy": "D",
                "final_asset": d_metrics.final_asset,
                "cum_return": d_metrics.cum_return,
                "ann_return": d_metrics.ann_return,
                "max_drawdown": d_metrics.max_drawdown,
                "sharpe": d_metrics.sharpe,
                "risk_return_ratio": d_metrics.risk_return_ratio,
            }
        ]
    )
    d_summary.to_csv(out_dir / "reference_strategy_D.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["lower_bound"], results_df["final_asset_A"], linewidth=2.0, label="策略A期末总资产")
    plt.axhline(d_metrics.final_asset, linestyle="--", color="black", label="策略D期末总资产")
    plt.xlabel("FCF 下限 (%)")
    plt.ylabel("期末总资产（元）")
    plt.title("FCF 下限 vs 策略A期末总资产")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "final_asset_vs_lower_bound.png", dpi=170)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["lower_bound"], results_df["max_drawdown_A"], linewidth=2.0, label="策略A最大回撤")
    plt.axhline(d_metrics.max_drawdown, linestyle="--", color="black", label="策略D最大回撤")
    plt.xlabel("FCF 下限 (%)")
    plt.ylabel("最大回撤")
    plt.title("FCF 下限 vs 策略A最大回撤")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "max_drawdown_vs_lower_bound.png", dpi=170)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["lower_bound"], results_df["zeroed_positive_return_years"], linewidth=2.0)
    plt.xlabel("FCF 下限 (%)")
    plt.ylabel("收益为正但被 FCF 归零的年份数")
    plt.title("FCF 下限 vs 误杀正收益年份数")
    plt.tight_layout()
    plt.savefig(out_dir / "zeroed_positive_years_vs_lower_bound.png", dpi=170)
    plt.close()

    top_final = results_df.nlargest(10, "final_asset_A")
    top_sharpe = results_df.nlargest(10, "sharpe_A")
    under_d_dd = results_df[results_df["max_drawdown_A"] >= d_metrics.max_drawdown].copy()
    best_under_d_dd = under_d_dd.nlargest(10, "final_asset_A") if not under_d_dd.empty else pd.DataFrame()

    report_lines = [
        "# FCF 下限搜索",
        "",
        f"- 原始信号表: `{signal_csv}`",
        f"- 收益率表: `{returns_csv}`",
        f"- 搜索区间: {float(args.lower_start):.2f}% ~ {float(args.lower_end):.2f}%，步长 {float(args.lower_step):.2f}%",
        f"- 固定上限: {float(args.upper_bound):.2f}%",
        f"- 初始资金: {float(args.initial_capital):.2f} 元",
        "",
        "## 策略D参考",
        "",
        d_summary.to_markdown(index=False),
        "",
        "## 终值最优 Top 10",
        "",
        top_final.to_markdown(index=False),
        "",
        "## 夏普最优 Top 10",
        "",
        top_sharpe.to_markdown(index=False),
        "",
        "## 回撤不差于 D 时的终值 Top 10",
        "",
        (best_under_d_dd.to_markdown(index=False) if not best_under_d_dd.empty else "无"),
        "",
        "## 说明",
        "",
        "- 当前搜索只改 FCF 的下限阈值，CAPE 映射、上限 8%、以及 W=(p_cape+p_fcf)/2 保持不变。",
        "- `zeroed_positive_return_years` 表示下一年股市收益为正，但因为 FCF 小于等于下限而被打成 0 分的年份数。",
        "- `beats_D_drawdown=True` 表示最大回撤不差于永久组合 D（数值上更接近 0）。",
    ]
    (out_dir / "搜索报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[out_dir] {out_dir}")
    print(f"[results_csv] {out_dir / 'fcf_lower_bound_search_results.csv'}")
    print(f"[best_final] {results_df.iloc[0]['lower_bound']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
