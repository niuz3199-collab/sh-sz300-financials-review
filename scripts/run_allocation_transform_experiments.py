#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.step7_compute_allocation import (  # noqa: E402
    cape_percentile,
    compute_for_year,
    fcf_percentile,
    get_10y_yield_real,
    load_cpi_map,
    load_year_close_prices,
    load_year_financials,
    resolve_path,
)
from scripts.step8_backtest import (  # noqa: E402
    DEFAULT_BOND_INDEX_ID,
    StrategyState,
    annual_return,
    apply_returns,
    compute_drawdown,
    fetch_chinabond_index_series,
    fetch_h00300_series,
    load_gold_series,
    rebalance,
)


@dataclass(frozen=True)
class Variant:
    slug: str
    display_name: str
    formula: str
    description: str
    weight_fn: Callable[[float], float]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def sigmoid_unit(x: float, k: float) -> float:
    z = float(k) * (float(x) - 0.5)
    return 1.0 / (1.0 + math.exp(-z))


def linear_band(x: float, floor: float, cap: float) -> float:
    return clamp01(float(floor) + (float(cap) - float(floor)) * clamp01(x))


def sigmoid_band(x: float, k: float, floor: float, cap: float) -> float:
    return clamp01(float(floor) + (float(cap) - float(floor)) * sigmoid_unit(x, k))


def build_variants() -> List[Variant]:
    return [
        Variant(
            slug="baseline_linear_0_100",
            display_name="Linear 0-100",
            formula="W = score",
            description="当前原始线性映射，不加上下限。",
            weight_fn=lambda x: clamp01(x),
        ),
        Variant(
            slug="linear_20_80",
            display_name="Linear 20-80",
            formula="W = 20% + 60% * score",
            description="给权益仓位加 20% 下限，同时把上限压到 80%。",
            weight_fn=lambda x: linear_band(x, 0.20, 0.80),
        ),
        Variant(
            slug="linear_30_70",
            display_name="Linear 30-70",
            formula="W = 30% + 40% * score",
            description="更高抬底、更窄波动区间，偏防守但避免长期贴近 0。",
            weight_fn=lambda x: linear_band(x, 0.30, 0.70),
        ),
        Variant(
            slug="sigmoid_k6_20_80",
            display_name="Sigmoid k=6 20-80",
            formula="W = 20% + 60% * sigmoid(6 * (score - 0.5))",
            description="S 形映射，温和拉开高低估区分，同时保留 20%-80% 区间。",
            weight_fn=lambda x: sigmoid_band(x, 6.0, 0.20, 0.80),
        ),
        Variant(
            slug="sigmoid_k10_20_80",
            display_name="Sigmoid k=10 20-80",
            formula="W = 20% + 60% * sigmoid(10 * (score - 0.5))",
            description="更陡的 S 形映射，信号强时更敢加仓，弱时更快降仓。",
            weight_fn=lambda x: sigmoid_band(x, 10.0, 0.20, 0.80),
        ),
    ]


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_signal_rows(
    *,
    data_base_dir: Path,
    start_year: int,
    end_year: int,
    cpi_csv: str,
    cache_dir: str,
) -> List[Dict[str, object]]:
    cpi_path = resolve_path(data_base_dir, str(cpi_csv)).resolve()
    cache_path = resolve_path(data_base_dir, str(cache_dir)).resolve()
    cpi_map = load_cpi_map(cpi_path)
    if not cpi_map:
        raise ValueError(f"CPI 汇总表为空或无法解析: {cpi_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"未找到 Step6 缓存目录: {cache_path}")

    profit_memo: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, object]] = []

    for year in range(int(start_year), int(end_year) + 1):
        year_dir = data_base_dir / str(year)
        close_prices = load_year_close_prices(year_dir, year)
        year_financials = load_year_financials(year_dir, year)

        real_rate, nominal_rate, used_yield_date = get_10y_yield_real(year)
        cape_index, fcf_yield_pct, cape_n, fcf_n, avg_profit_n = compute_for_year(
            year,
            year_financials=year_financials,
            close_prices=close_prices,
            cpi_map=cpi_map,
            cache_dir=cache_path,
            profit_memo=profit_memo,
        )

        row: Dict[str, object] = {
            "year": year,
            "cape_index": cape_index,
            "fcf_yield_pct": fcf_yield_pct,
            "real_rate": real_rate,
            "nominal_rate": nominal_rate,
            "used_yield_date": used_yield_date,
            "cape_n": cape_n,
            "fcf_n": fcf_n,
            "avg_profit_n": avg_profit_n,
            "u_cape": None,
            "p_cape": None,
            "p_fcf": None,
            "score_01": None,
            "baseline_w_pct": None,
        }

        if cape_index is not None and fcf_yield_pct is not None and real_rate is not None:
            u_cape = 1.0 / (float(real_rate) + 0.02)
            p_cape = cape_percentile(float(cape_index), float(u_cape))
            p_fcf = fcf_percentile(float(fcf_yield_pct))
            baseline_w_pct = (float(p_cape) + float(p_fcf)) / 2.0
            row.update(
                {
                    "u_cape": float(u_cape),
                    "p_cape": float(p_cape),
                    "p_fcf": float(p_fcf),
                    "score_01": float(baseline_w_pct) / 100.0,
                    "baseline_w_pct": float(baseline_w_pct),
                }
            )

        rows.append(row)

    return rows


def write_variant_year_configs(
    *,
    variant_dir: Path,
    variant: Variant,
    signal_rows: Sequence[Dict[str, object]],
) -> Dict[int, Optional[float]]:
    w_map: Dict[int, Optional[float]] = {}
    yearly_rows: List[Dict[str, object]] = []

    for row in signal_rows:
        year = int(row["year"])
        score_01 = row.get("score_01")
        w_fraction: Optional[float] = None
        w_pct: Optional[float] = None
        if score_01 is not None:
            w_fraction = clamp01(variant.weight_fn(float(score_01)))
            w_pct = float(w_fraction) * 100.0

        out_row = {
            "年份": year,
            "沪深300CAPE": row.get("cape_index"),
            "沪深300FCFYield": row.get("fcf_yield_pct"),
            "实际利率": row.get("real_rate"),
            "CAPE危险阈值": row.get("u_cape"),
            "CAPE百分位": row.get("p_cape"),
            "FCF百分位": row.get("p_fcf"),
            "权益配置比例W": w_pct,
            "原始线性W": row.get("baseline_w_pct"),
            "score_01": score_01,
            "variant_slug": variant.slug,
            "variant_name": variant.display_name,
            "variant_formula": variant.formula,
        }

        year_dir = variant_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([out_row]).to_csv(year_dir / f"{year}_配置比例.csv", index=False, encoding="utf-8-sig")
        yearly_rows.append(out_row)
        w_map[year] = w_fraction

    pd.DataFrame(yearly_rows).to_csv(variant_dir / "variant_weights_by_year.csv", index=False, encoding="utf-8-sig")
    return w_map


def run_backtest_for_variant(
    *,
    variant: Variant,
    variant_dir: Path,
    signal_rows: Sequence[Dict[str, object]],
    w_map: Dict[int, Optional[float]],
    start_year: int,
    end_year: int,
    initial_capital: float,
    annual_contribution: float,
    bond_index_id: str,
    eq_df: pd.DataFrame,
    bond_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> Dict[str, object]:
    out_dir = variant_dir / "backtest_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    real_rate_map = {int(row["year"]): row.get("real_rate") for row in signal_rows}
    gold_start = None
    if gold_df is not None and not gold_df.empty:
        gold_start_ts = pd.to_datetime(gold_df["date"], errors="coerce").min()
        gold_start = None if pd.isna(gold_start_ts) else gold_start_ts.date()

    a_state = StrategyState()
    b_state = StrategyState()
    d_state = StrategyState()
    c_total = float(initial_capital)

    if initial_capital:
        a_state.cash += float(initial_capital)
        b_state.cash += float(initial_capital)
        d_state.cash += float(initial_capital)

    principal = float(initial_capital)
    rows: List[Dict[str, object]] = []

    for year in range(int(start_year), int(end_year) + 1):
        d0 = date(year, 4, 30)
        d1 = date(year + 1, 4, 30)

        r_eq = annual_return(eq_df, date_col="日期", value_col="收盘", d0=d0, d1=d1)
        r_bond = annual_return(bond_df, date_col="date", value_col="value", d0=d0, d1=d1)
        r_gold = annual_return(gold_df, date_col="date", value_col="close", d0=d0, d1=d1) if not gold_df.empty else None
        if r_eq is None or r_bond is None:
            continue

        principal += float(annual_contribution)

        w = w_map.get(year)
        if w is None:
            w = 0.5

        a_state.cash += float(annual_contribution)
        rebalance(a_state, {"equity": float(w), "bond": float(1.0 - w), "gold": 0.0, "cash": 0.0})
        apply_returns(a_state, {"equity": r_eq, "bond": r_bond, "gold": 0.0, "cash": 0.0})

        b_state.cash += float(annual_contribution)
        rebalance(b_state, {"equity": 1.0, "bond": 0.0, "gold": 0.0, "cash": 0.0})
        apply_returns(b_state, {"equity": r_eq, "bond": 0.0, "gold": 0.0, "cash": 0.0})

        c_total = c_total * (1.0 + r_eq) + float(annual_contribution) * (1.0 + 0.5 * r_eq)

        d_state.cash += float(annual_contribution)
        rebalance(d_state, {"equity": 0.25, "bond": 0.25, "gold": 0.25, "cash": 0.25})
        apply_returns(
            d_state,
            {
                "equity": r_eq,
                "bond": r_bond,
                "gold": 0.0 if r_gold is None else r_gold,
                "cash": 0.0,
            },
        )

        a_total = a_state.equity + a_state.bond + a_state.gold + a_state.cash
        b_total = b_state.equity + b_state.bond + b_state.gold + b_state.cash
        d_total = d_state.equity + d_state.bond + d_state.gold + d_state.cash

        out_row = {
            "年份": year,
            "投入本金": principal,
            "策略A_资产": a_total,
            "策略B_资产": b_total,
            "策略C_资产": c_total,
            "策略D_资产": d_total,
            "当年股收益率": r_eq,
            "当年债收益率": r_bond,
            "当年金收益率": r_gold,
            "W": float(w),
            "W_pct": float(w) * 100.0,
        }
        rows.append(out_row)

        year_dir = variant_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([out_row]).to_csv(year_dir / f"{year}_策略资产明细.csv", index=False, encoding="utf-8-sig")

    if not rows:
        raise RuntimeError(f"{variant.slug}: 没有生成任何回测记录")

    detail_df = pd.DataFrame(rows).sort_values("年份").reset_index(drop=True)
    detail_df.to_csv(out_dir / "策略资产明细.csv", index=False, encoding="utf-8-sig")

    nominal_candidates = []
    for year in detail_df["年份"].tolist():
        real_rate = real_rate_map.get(int(year))
        if real_rate is None:
            continue
        nominal_candidates.append(float(real_rate) + 0.02)
    risk_free = float(sum(nominal_candidates) / len(nominal_candidates)) if nominal_candidates else 0.0

    metrics_rows: List[Dict[str, object]] = []
    strategies = {
        "策略A": "策略A_资产",
        "策略B": "策略B_资产",
        "策略C": "策略C_资产",
        "策略D": "策略D_资产",
    }
    n_years = len(detail_df)

    for name, col in strategies.items():
        series = pd.to_numeric(detail_df[col], errors="coerce").ffill().fillna(0.0)
        principal_series = pd.to_numeric(detail_df["投入本金"], errors="coerce").ffill()
        final_value = float(series.iloc[-1])
        final_principal = float(principal_series.iloc[-1])
        cum_ret = (final_value - final_principal) / final_principal if final_principal else float("nan")
        ann_ret = (final_value / final_principal) ** (1.0 / n_years) - 1.0 if final_principal else float("nan")
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

    plt.figure(figsize=(10, 6))
    for name, col in strategies.items():
        plt.plot(detail_df["年份"], detail_df[col], label=name)
    plt.title(f"累计净值走势（{variant.display_name}）")
    plt.xlabel("年份")
    plt.ylabel("资产（元）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "累计净值走势.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, col in strategies.items():
        series = pd.to_numeric(detail_df[col], errors="coerce").ffill()
        dd = compute_drawdown(series)
        plt.plot(detail_df["年份"], dd, label=name)
    plt.title(f"回撤对比（{variant.display_name}）")
    plt.xlabel("年份")
    plt.ylabel("回撤")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "回撤对比.png", dpi=150)
    plt.close()

    report_lines = [
        "# 策略表现分析报告",
        "",
        f"- 实验方案: {variant.display_name}",
        f"- 权重公式: {variant.formula}",
        f"- 方案说明: {variant.description}",
        f"- 回测区间: {int(detail_df['年份'].min())} ~ {int(detail_df['年份'].max())}（按 4/30 年度持有期近似）",
        f"- 起始资金: {float(initial_capital):.2f} 元",
        f"- 年度投入: {float(annual_contribution):.2f} 元",
        f"- 债券指数: indexid={bond_index_id}（财富/总值）",
    ]
    if gold_start:
        report_lines.append(f"- 黄金数据起始: {gold_start}（早于该日期的黄金收益率用 0 近似）")
    report_lines.extend(
        [
            "",
            "## 回测结果对比",
            "",
            metrics_df.to_markdown(index=False),
            "",
            "## 关键说明",
            "",
            "- 策略C（月度定投）在本实验中若年度投入为 0，会退化为一次性买入并持有。",
            "- 策略D（永久组合）黄金现货数据在 akshare 中可用区间有限，早期年份用 0 收益率近似。",
        ]
    )
    (out_dir / "策略表现分析报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    strategy_a_metrics = metrics_df[metrics_df["策略"] == "策略A"].iloc[0].to_dict()
    return {
        "detail_df": detail_df,
        "metrics_df": metrics_df,
        "strategy_a_metrics": strategy_a_metrics,
    }


def summarize_variant(
    *,
    variant: Variant,
    signal_rows: Sequence[Dict[str, object]],
    detail_df: pd.DataFrame,
    strategy_a_metrics: Dict[str, object],
) -> Dict[str, object]:
    score_last = None
    weight_last_pct = None
    weight_min_pct = None
    weight_max_pct = None

    w_series = pd.to_numeric(detail_df["W_pct"], errors="coerce")
    if not w_series.empty:
        weight_last_pct = float(w_series.iloc[-1])
        weight_min_pct = float(w_series.min())
        weight_max_pct = float(w_series.max())

    row_2007 = detail_df[detail_df["年份"] == 2007]
    row_2008 = detail_df[detail_df["年份"] == 2008]
    row_2021 = detail_df[detail_df["年份"] == 2021]
    row_2024 = detail_df[detail_df["年份"] == 2024]

    ret_2008 = None
    if not row_2007.empty and not row_2008.empty:
        v0 = float(row_2007.iloc[0]["策略A_资产"])
        v1 = float(row_2008.iloc[0]["策略A_资产"])
        if v0:
            ret_2008 = v1 / v0 - 1.0

    ret_2021 = None
    row_2020 = detail_df[detail_df["年份"] == 2020]
    if not row_2020.empty and not row_2021.empty:
        v0 = float(row_2020.iloc[0]["策略A_资产"])
        v1 = float(row_2021.iloc[0]["策略A_资产"])
        if v0:
            ret_2021 = v1 / v0 - 1.0

    for signal_row in reversed(list(signal_rows)):
        if signal_row.get("score_01") is not None:
            score_last = float(signal_row["score_01"])
            break

    return {
        "variant_slug": variant.slug,
        "variant_name": variant.display_name,
        "formula": variant.formula,
        "description": variant.description,
        "final_asset_A": strategy_a_metrics.get("期末总资产"),
        "cum_return_A": strategy_a_metrics.get("累计收益率"),
        "ann_return_A": strategy_a_metrics.get("年化收益率"),
        "max_drawdown_A": strategy_a_metrics.get("最大回撤"),
        "sharpe_A": strategy_a_metrics.get("夏普比率"),
        "risk_return_ratio_A": strategy_a_metrics.get("收益风险比"),
        "return_2008_A": ret_2008,
        "return_2021_A": ret_2021,
        "final_weight_pct": weight_last_pct,
        "min_weight_pct": weight_min_pct,
        "max_weight_pct": weight_max_pct,
        "last_signal_score": score_last,
        "asset_2024_A": None if row_2024.empty else float(row_2024.iloc[0]["策略A_资产"]),
    }


def make_summary_plots(
    *,
    root_dir: Path,
    experiment_results: Sequence[Dict[str, object]],
) -> None:
    plt.figure(figsize=(11, 6))
    d_plotted = False
    b_plotted = False
    for result in experiment_results:
        variant = result["variant"]
        detail_df = result["detail_df"]
        plt.plot(detail_df["年份"], detail_df["策略A_资产"], label=f"A - {variant.display_name}", linewidth=2.0)
        if not d_plotted:
            plt.plot(
                detail_df["年份"],
                detail_df["策略D_资产"],
                label="D - 永久组合",
                linewidth=2.2,
                linestyle="--",
                color="black",
            )
            d_plotted = True
        if not b_plotted:
            plt.plot(
                detail_df["年份"],
                detail_df["策略B_资产"],
                label="B - 100%权益",
                linewidth=1.8,
                linestyle=":",
                color="gray",
            )
            b_plotted = True
    plt.title("不同 W 变换下的策略A累计资产对比")
    plt.xlabel("年份")
    plt.ylabel("资产（元）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_dir / "A策略横向对比.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    for result in experiment_results:
        variant = result["variant"]
        detail_df = result["detail_df"]
        plt.plot(detail_df["年份"], detail_df["W_pct"], label=variant.display_name, linewidth=2.0)
    plt.title("不同 W 变换下的年度权益仓位")
    plt.xlabel("年份")
    plt.ylabel("W（%）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_dir / "W权重横向对比.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    summary_rows = [result["summary_row"] for result in experiment_results]
    x = [float(row["max_drawdown_A"]) for row in summary_rows]
    y = [float(row["final_asset_A"]) for row in summary_rows]
    labels = [str(row["variant_name"]) for row in summary_rows]
    plt.scatter(x, y, s=90)
    for px, py, label in zip(x, y, labels):
        plt.annotate(label, (px, py), textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.title("策略A：终值与最大回撤")
    plt.xlabel("最大回撤")
    plt.ylabel("期末总资产（元）")
    plt.tight_layout()
    plt.savefig(root_dir / "A策略终值_vs_回撤.png", dpi=160)
    plt.close()


def write_root_report(
    *,
    root_dir: Path,
    summary_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    variants: Sequence[Variant],
    start_year: int,
    end_year: int,
    initial_capital: float,
    annual_contribution: float,
) -> None:
    lines = [
        "# W 变换实验汇总",
        "",
        f"- 回测区间: {start_year} ~ {end_year}",
        f"- 起始资金: {initial_capital:.2f} 元",
        f"- 年度投入: {annual_contribution:.2f} 元",
        f"- 变体数量: {len(list(variants))}",
        "",
        "## 变体定义",
        "",
    ]
    for variant in variants:
        lines.append(f"- `{variant.slug}`: {variant.display_name}，{variant.formula}。{variant.description}")

    lines.extend(
        [
            "",
            "## 策略A汇总对比",
            "",
            summary_df.to_markdown(index=False),
            "",
            "## 原始信号概览",
            "",
            signal_df.to_markdown(index=False),
            "",
            "## 文件说明",
            "",
            "- 每个变体目录下包含年度配置、年度策略资产明细、回测 CSV、报告和两张图。",
            "- 顶层额外提供 A 策略横向对比图、W 权重横向对比图，以及终值/回撤散点图。",
        ]
    )
    (root_dir / "实验汇总报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple W-transform allocation experiments and backtests.")
    parser.add_argument("--data-base-dir", default=str(REPO_ROOT), help="读取年度财报数据和价格数据的项目根目录")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--cpi-csv", default="CPI指数汇总.csv")
    parser.add_argument("--cache-dir", default=".cache/financials")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--annual-contribution", type=float, default=0.0)
    parser.add_argument("--bond-index-id", default=DEFAULT_BOND_INDEX_ID)
    parser.add_argument("--out-root", default="", help="实验总目录；默认自动按时间戳新建")
    args = parser.parse_args()

    data_base_dir = Path(args.data_base_dir).expanduser().resolve()
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if str(args.out_root or "").strip()
        else (REPO_ROOT / f"allocation_transform_experiments_{now_stamp()}").resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)

    variants = build_variants()
    signal_rows = compute_signal_rows(
        data_base_dir=data_base_dir,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        cpi_csv=str(args.cpi_csv),
        cache_dir=str(args.cache_dir),
    )
    signal_df = pd.DataFrame(signal_rows)
    signal_df.to_csv(out_root / "base_signal_table.csv", index=False, encoding="utf-8-sig")

    eq_df = fetch_h00300_series(start=f"{int(args.start_year)}0101", end=f"{int(args.end_year) + 1}1231")
    bond_df = fetch_chinabond_index_series(indexid=str(args.bond_index_id))
    gold_df = load_gold_series()

    experiment_results: List[Dict[str, object]] = []

    for variant in variants:
        variant_dir = out_root / variant.slug
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            variant_dir / "variant_meta.json",
            {
                "slug": variant.slug,
                "display_name": variant.display_name,
                "formula": variant.formula,
                "description": variant.description,
                "initial_capital": float(args.initial_capital),
                "annual_contribution": float(args.annual_contribution),
                "start_year": int(args.start_year),
                "end_year": int(args.end_year),
            },
        )

        w_map = write_variant_year_configs(variant_dir=variant_dir, variant=variant, signal_rows=signal_rows)
        backtest_result = run_backtest_for_variant(
            variant=variant,
            variant_dir=variant_dir,
            signal_rows=signal_rows,
            w_map=w_map,
            start_year=int(args.start_year),
            end_year=int(args.end_year),
            initial_capital=float(args.initial_capital),
            annual_contribution=float(args.annual_contribution),
            bond_index_id=str(args.bond_index_id),
            eq_df=eq_df,
            bond_df=bond_df,
            gold_df=gold_df,
        )
        summary_row = summarize_variant(
            variant=variant,
            signal_rows=signal_rows,
            detail_df=backtest_result["detail_df"],
            strategy_a_metrics=backtest_result["strategy_a_metrics"],
        )
        experiment_results.append(
            {
                "variant": variant,
                "variant_dir": variant_dir,
                "detail_df": backtest_result["detail_df"],
                "metrics_df": backtest_result["metrics_df"],
                "summary_row": summary_row,
            }
        )

    summary_df = pd.DataFrame([result["summary_row"] for result in experiment_results]).sort_values(
        by=["final_asset_A", "max_drawdown_A"],
        ascending=[False, False],
    )
    summary_df.to_csv(out_root / "variant_comparison.csv", index=False, encoding="utf-8-sig")

    make_summary_plots(root_dir=out_root, experiment_results=experiment_results)
    write_root_report(
        root_dir=out_root,
        summary_df=summary_df,
        signal_df=signal_df[
            [
                "year",
                "cape_index",
                "fcf_yield_pct",
                "real_rate",
                "p_cape",
                "p_fcf",
                "score_01",
                "baseline_w_pct",
                "cape_n",
                "fcf_n",
                "avg_profit_n",
            ]
        ],
        variants=variants,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        initial_capital=float(args.initial_capital),
        annual_contribution=float(args.annual_contribution),
    )

    write_json(
        out_root / "experiment_manifest.json",
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "data_base_dir": str(data_base_dir),
            "out_root": str(out_root),
            "start_year": int(args.start_year),
            "end_year": int(args.end_year),
            "initial_capital": float(args.initial_capital),
            "annual_contribution": float(args.annual_contribution),
            "bond_index_id": str(args.bond_index_id),
            "variants": [
                {
                    "slug": variant.slug,
                    "display_name": variant.display_name,
                    "formula": variant.formula,
                    "description": variant.description,
                }
                for variant in variants
            ],
        },
    )

    print(f"[out_root] {out_root}")
    print(f"[variant_count] {len(variants)}")
    print(f"[summary_csv] {out_root / 'variant_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
