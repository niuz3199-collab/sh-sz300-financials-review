#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class StrategySource:
    family: str
    slug: str
    display_name: str
    detail_csv: Path
    strategy_col: str


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, float):
            if math.isnan(value):
                return None
            return float(value)
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except Exception:
        return None


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_dir(prefix: str) -> Path:
    candidates = sorted([p for p in REPO_ROOT.glob(f"{prefix}_*") if p.is_dir()], key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"未找到 {prefix}_* 目录")
    return candidates[-1]


def resolve_default_old_root() -> Path:
    return latest_dir("allocation_transform_experiments")


def resolve_default_new_root() -> Path:
    return latest_dir("drawdown_driven_allocation")


def resolve_default_benchmark_csv() -> Path:
    path = REPO_ROOT / "backtest_output_initial100k_nodca" / "策略资产明细.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到基准回测明细: {path}")
    return path


def resolve_default_real_rate_csv(old_root: Path) -> Optional[Path]:
    path = old_root / "base_signal_table.csv"
    return path if path.exists() else None


def load_variant_sources(root_dir: Path, family: str) -> List[StrategySource]:
    summary_csv = root_dir / "variant_comparison.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"未找到变体汇总表: {summary_csv}")
    summary_df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    sources: List[StrategySource] = []
    for _, row in summary_df.iterrows():
        slug = str(row["variant_slug"]).strip()
        display_name = str(row["variant_name"]).strip()
        detail_csv = root_dir / slug / "backtest_output" / "策略资产明细.csv"
        if not detail_csv.exists():
            raise FileNotFoundError(f"未找到策略明细: {detail_csv}")
        sources.append(
            StrategySource(
                family=family,
                slug=slug,
                display_name=display_name,
                detail_csv=detail_csv,
                strategy_col="策略A_资产",
            )
        )
    return sources


def load_benchmark_sources(detail_csv: Path) -> List[StrategySource]:
    return [
        StrategySource("benchmark", "benchmark_B", "策略B 100%权益", detail_csv, "策略B_资产"),
        StrategySource("benchmark", "benchmark_C", "策略C 月定投近似", detail_csv, "策略C_资产"),
        StrategySource("benchmark", "benchmark_D", "策略D 永久组合", detail_csv, "策略D_资产"),
    ]


def load_real_rate_map(path: Optional[Path]) -> Dict[int, float]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "year" not in df.columns or "real_rate" not in df.columns:
        return {}
    tmp = df.copy()
    tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
    tmp["real_rate"] = pd.to_numeric(tmp["real_rate"], errors="coerce")
    out: Dict[int, float] = {}
    for _, row in tmp.iterrows():
        year = safe_float(row["year"])
        rate = safe_float(row["real_rate"])
        if year is None or rate is None:
            continue
        out[int(year)] = float(rate)
    return out


def average_nominal_rate(real_rate_map: Dict[int, float], start_year: int, end_year: int) -> float:
    values = [float(real_rate_map[y]) + 0.02 for y in range(start_year, end_year + 1) if y in real_rate_map]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def extract_yearly_returns(detail_csv: Path, strategy_col: str) -> pd.DataFrame:
    df = pd.read_csv(detail_csv, encoding="utf-8-sig")
    if "年份" not in df.columns or strategy_col not in df.columns:
        raise ValueError(f"{detail_csv} 缺少列 年份/{strategy_col}")
    out = df.copy()
    out["年份"] = pd.to_numeric(out["年份"], errors="coerce")
    out[strategy_col] = pd.to_numeric(out[strategy_col], errors="coerce")
    out = out.dropna(subset=["年份", strategy_col]).copy()
    out["年份"] = out["年份"].astype(int)
    out = out.sort_values("年份").reset_index(drop=True)
    out["strategy_return"] = out[strategy_col].pct_change()
    return out[["年份", "strategy_return"]]


def build_rebased_series(
    return_df: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    initial_capital: float,
) -> pd.DataFrame:
    tmp = return_df.copy()
    tmp = tmp[(tmp["年份"] >= int(start_year)) & (tmp["年份"] <= int(end_year))].copy()
    tmp["strategy_return"] = pd.to_numeric(tmp["strategy_return"], errors="coerce")
    if tmp["strategy_return"].isna().any():
        bad_years = tmp[tmp["strategy_return"].isna()]["年份"].tolist()
        raise ValueError(f"存在无法用于重建净值的空收益率年份: {bad_years}")

    value = float(initial_capital)
    rows: List[Dict[str, float]] = []
    for _, row in tmp.iterrows():
        year = int(row["年份"])
        annual_return = float(row["strategy_return"])
        value *= 1.0 + annual_return
        rows.append({"年份": year, "annual_return": annual_return, "asset": value})
    return pd.DataFrame(rows)


def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.fillna(0.0)


def compute_metrics(rebased_df: pd.DataFrame, *, initial_capital: float, risk_free: float) -> Dict[str, float]:
    asset_series = pd.to_numeric(rebased_df["asset"], errors="coerce").ffill()
    return_series = pd.to_numeric(rebased_df["annual_return"], errors="coerce")
    final_value = float(asset_series.iloc[-1])
    n_years = len(rebased_df)
    cum_ret = final_value / float(initial_capital) - 1.0
    ann_ret = (final_value / float(initial_capital)) ** (1.0 / n_years) - 1.0
    vol = float(return_series.std(ddof=0)) if not return_series.empty else float("nan")
    sharpe = (ann_ret - risk_free) / vol if vol and not math.isnan(vol) else float("nan")
    dd = compute_drawdown(asset_series)
    max_dd = float(dd.min()) if not dd.empty else 0.0
    rr = (cum_ret / abs(max_dd)) if max_dd != 0 else float("nan")
    return {
        "final_asset": final_value,
        "cum_return": cum_ret,
        "ann_return": ann_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "risk_return_ratio": rr,
    }


def color_for_family(family: str) -> str:
    if family == "old_transform":
        return "#2f6db3"
    if family == "drawdown_driven":
        return "#c45b12"
    return "#3a3a3a"


def make_scatter_plot(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(11, 7))
    for family, grp in summary_df.groupby("family"):
        plt.scatter(
            grp["max_drawdown"],
            grp["final_asset"],
            s=90,
            label=family,
            color=color_for_family(str(family)),
            alpha=0.85,
        )
        for _, row in grp.iterrows():
            plt.annotate(
                str(row["display_name"]),
                (float(row["max_drawdown"]), float(row["final_asset"])),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )
    plt.title("2011-2024 Live 区间：期末资产 vs 最大回撤")
    plt.xlabel("最大回撤")
    plt.ylabel("期末资产（元）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_selected_curve_plot(curve_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    selected_names: List[str] = []
    for family in ["old_transform", "drawdown_driven"]:
        family_df = summary_df[summary_df["family"] == family].sort_values("final_asset", ascending=False)
        selected_names.extend(family_df.head(2)["display_name"].astype(str).tolist())
    benchmark_names = ["策略B 100%权益", "策略D 永久组合"]
    selected_names.extend(benchmark_names)
    selected_names = list(dict.fromkeys(selected_names))

    plt.figure(figsize=(11, 6))
    for name in selected_names:
        grp = curve_df[curve_df["display_name"] == name].sort_values("年份")
        if grp.empty:
            continue
        plt.plot(grp["年份"], grp["asset"], label=name, linewidth=2.0)
    plt.title("2011-2024 Live 区间：代表性策略净值对比")
    plt.xlabel("年份")
    plt.ylabel("资产（元）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_report(
    *,
    out_dir: Path,
    summary_df: pd.DataFrame,
    family_best_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    initial_capital: float,
    risk_free: float,
    old_root: Path,
    new_root: Path,
) -> None:
    lines = [
        "# 2011-2024 Live 区间策略评估",
        "",
        f"- 评估区间: {start_year} ~ {end_year}",
        f"- 起始资金: {initial_capital:.2f} 元",
        f"- 无风险利率近似: {risk_free:.4%}",
        f"- 旧 W 映射策略目录: `{old_root}`",
        f"- 新回撤驱动策略目录: `{new_root}`",
        "",
        "## 各家族最佳",
        "",
        family_best_df.to_markdown(index=False),
        "",
        "## 全策略汇总",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## 说明",
        "",
        "- 所有策略统一按 2011 年起重新以 10 万元起点复利重建净值，不沿用 2010 年末已有资产。",
        "- 新回撤驱动策略的 2011-2024 年全部属于 live 年份，因此这里正好对应其真实可交易区间。",
        "- 策略C 在当前无定投设定下会与策略B完全重合，这里仍保留一行，便于口径对照。",
    ]
    (out_dir / "评估报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate all discussed strategies only on the 2011-2024 live period.")
    parser.add_argument("--old-root", default="", help="旧 W 映射策略实验目录")
    parser.add_argument("--new-root", default="", help="新回撤驱动策略实验目录")
    parser.add_argument("--benchmark-csv", default="", help="基准回测明细 CSV")
    parser.add_argument("--real-rate-csv", default="", help="无风险利率近似来源 CSV")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--out-dir", default="", help="输出目录")
    args = parser.parse_args()

    old_root = Path(args.old_root).expanduser().resolve() if str(args.old_root or "").strip() else resolve_default_old_root()
    new_root = Path(args.new_root).expanduser().resolve() if str(args.new_root or "").strip() else resolve_default_new_root()
    benchmark_csv = (
        Path(args.benchmark_csv).expanduser().resolve()
        if str(args.benchmark_csv or "").strip()
        else resolve_default_benchmark_csv()
    )
    real_rate_csv = (
        Path(args.real_rate_csv).expanduser().resolve()
        if str(args.real_rate_csv or "").strip()
        else resolve_default_real_rate_csv(old_root)
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir or "").strip()
        else (REPO_ROOT / f"live_period_eval_{int(args.start_year)}_{int(args.end_year)}_{now_stamp()}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    real_rate_map = load_real_rate_map(real_rate_csv)
    risk_free = average_nominal_rate(real_rate_map, int(args.start_year), int(args.end_year))

    strategy_sources = []
    strategy_sources.extend(load_variant_sources(old_root, "old_transform"))
    strategy_sources.extend(load_variant_sources(new_root, "drawdown_driven"))
    strategy_sources.extend(load_benchmark_sources(benchmark_csv))

    summary_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []

    for source in strategy_sources:
        return_df = extract_yearly_returns(source.detail_csv, source.strategy_col)
        rebased_df = build_rebased_series(
            return_df,
            start_year=int(args.start_year),
            end_year=int(args.end_year),
            initial_capital=float(args.initial_capital),
        )
        metrics = compute_metrics(rebased_df, initial_capital=float(args.initial_capital), risk_free=risk_free)
        first_year_ret = float(rebased_df.iloc[0]["annual_return"])
        last_year_ret = float(rebased_df.iloc[-1]["annual_return"])
        summary_rows.append(
            {
                "family": source.family,
                "slug": source.slug,
                "display_name": source.display_name,
                "period_years": len(rebased_df),
                "start_year": int(args.start_year),
                "end_year": int(args.end_year),
                "first_year_return": first_year_ret,
                "last_year_return": last_year_ret,
                **metrics,
            }
        )
        for _, row in rebased_df.iterrows():
            curve_rows.append(
                {
                    "family": source.family,
                    "slug": source.slug,
                    "display_name": source.display_name,
                    "年份": int(row["年份"]),
                    "annual_return": float(row["annual_return"]),
                    "asset": float(row["asset"]),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["final_asset", "max_drawdown"],
        ascending=[False, False],
    ).reset_index(drop=True)
    curve_df = pd.DataFrame(curve_rows).sort_values(["display_name", "年份"]).reset_index(drop=True)

    summary_df.to_csv(out_dir / "live_period_strategy_comparison.csv", index=False, encoding="utf-8-sig")
    curve_df.to_csv(out_dir / "live_period_rebased_curves.csv", index=False, encoding="utf-8-sig")

    family_best_df = (
        summary_df.sort_values(["family", "final_asset"], ascending=[True, False])
        .groupby("family", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    family_best_df.to_csv(out_dir / "family_best_strategies.csv", index=False, encoding="utf-8-sig")

    make_scatter_plot(summary_df, out_dir / "live_period_scatter.png")
    make_selected_curve_plot(curve_df, summary_df, out_dir / "live_period_selected_curves.png")
    write_report(
        out_dir=out_dir,
        summary_df=summary_df,
        family_best_df=family_best_df,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        initial_capital=float(args.initial_capital),
        risk_free=risk_free,
        old_root=old_root,
        new_root=new_root,
    )

    write_json(
        out_dir / "manifest.json",
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "old_root": str(old_root),
            "new_root": str(new_root),
            "benchmark_csv": str(benchmark_csv),
            "real_rate_csv": None if real_rate_csv is None else str(real_rate_csv),
            "start_year": int(args.start_year),
            "end_year": int(args.end_year),
            "initial_capital": float(args.initial_capital),
            "risk_free": risk_free,
            "strategy_count": len(strategy_sources),
        },
    )

    print(f"[out_dir] {out_dir}")
    print(f"[summary_csv] {out_dir / 'live_period_strategy_comparison.csv'}")
    print(f"[strategy_count] {len(strategy_sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
