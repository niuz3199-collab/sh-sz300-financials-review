#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    path = REPO_ROOT / "backtest_output" / "策略资产明细.csv"
    if path.exists():
        return path
    raise FileNotFoundError("未找到可用的策略资产明细.csv，请先运行任一回测。")


def write_markdown(path: Path, merged: pd.DataFrame, signal_csv: Path, returns_csv: Path) -> None:
    lines = [
        "# 原始因子三维视图",
        "",
        f"- 原始信号表: `{signal_csv}`",
        f"- 年收益率表: `{returns_csv}`",
        "- X 轴: 沪深300 CAPE 原始值",
        "- Y 轴: 沪深300 FCF Yield 原始值（%）",
        "- Z 轴: 该年 4/30 到下一年 4/30 的沪深300区间收益率（%）",
        "- 说明: 该持有区间接近 1 年，因此区间收益率与年化收益率在数值上几乎一致；此处保留原始区间收益率百分比。",
        "",
        "## 数据预览",
        "",
        merged.to_markdown(index=False),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export raw CAPE / FCF Yield / next-year return table and 3D plot.")
    parser.add_argument("--signal-csv", default="", help="原始信号表，默认取最新实验目录下的 base_signal_table.csv")
    parser.add_argument("--returns-csv", default="", help="包含 当年股收益率 的明细表")
    parser.add_argument("--out-dir", default="", help="输出目录；默认按时间戳新建")
    args = parser.parse_args()

    signal_csv = Path(args.signal_csv).expanduser().resolve() if str(args.signal_csv or "").strip() else resolve_default_signal_csv()
    returns_csv = Path(args.returns_csv).expanduser().resolve() if str(args.returns_csv or "").strip() else resolve_default_returns_csv()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir or "").strip()
        else (REPO_ROOT / f"raw_factor_3d_view_{now_stamp()}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_df = pd.read_csv(signal_csv, encoding="utf-8-sig")
    returns_df = pd.read_csv(returns_csv, encoding="utf-8-sig")

    expected_signal_cols = {"year", "cape_index", "fcf_yield_pct"}
    missing_signal_cols = expected_signal_cols - set(signal_df.columns)
    if missing_signal_cols:
        raise ValueError(f"原始信号表缺少字段: {sorted(missing_signal_cols)}")

    if "年份" not in returns_df.columns or "当年股收益率" not in returns_df.columns:
        raise ValueError("收益率表必须包含 `年份` 和 `当年股收益率` 两列")

    signal_df = signal_df[["year", "cape_index", "fcf_yield_pct"]].copy()
    returns_df = returns_df[["年份", "当年股收益率"]].copy()
    returns_df = returns_df.rename(columns={"年份": "year", "当年股收益率": "equity_return_1y"})

    signal_df["year"] = pd.to_numeric(signal_df["year"], errors="coerce")
    signal_df["cape_index"] = pd.to_numeric(signal_df["cape_index"], errors="coerce")
    signal_df["fcf_yield_pct"] = pd.to_numeric(signal_df["fcf_yield_pct"], errors="coerce")
    returns_df["year"] = pd.to_numeric(returns_df["year"], errors="coerce")
    returns_df["equity_return_1y"] = pd.to_numeric(returns_df["equity_return_1y"], errors="coerce")

    merged = signal_df.merge(returns_df, on="year", how="inner")
    merged = merged.dropna(subset=["year", "cape_index", "fcf_yield_pct", "equity_return_1y"]).copy()
    merged["year"] = merged["year"].astype(int)
    merged["equity_return_1y_pct"] = merged["equity_return_1y"] * 100.0
    merged = merged.sort_values("year").reset_index(drop=True)

    output_df = merged[
        ["year", "cape_index", "fcf_yield_pct", "equity_return_1y", "equity_return_1y_pct"]
    ].copy()
    output_df = output_df.rename(
        columns={
            "year": "年份",
            "cape_index": "原始CAPE",
            "fcf_yield_pct": "原始FCFYield_pct",
            "equity_return_1y": "4月30日至次年4月30日收益率",
            "equity_return_1y_pct": "4月30日至次年4月30日收益率_pct",
        }
    )
    output_df.to_csv(out_dir / "raw_factor_3d_dataset.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    scatter = ax.scatter(
        merged["cape_index"],
        merged["fcf_yield_pct"],
        merged["equity_return_1y_pct"],
        c=merged["equity_return_1y_pct"],
        cmap="RdYlGn",
        s=60,
        depthshade=True,
    )
    for _, row in merged.iterrows():
        ax.text(
            float(row["cape_index"]),
            float(row["fcf_yield_pct"]),
            float(row["equity_return_1y_pct"]),
            str(int(row["year"])),
            fontsize=8,
        )
    ax.set_xlabel("CAPE")
    ax.set_ylabel("FCF Yield (%)")
    ax.set_zlabel("Next 1Y Return (%)")
    ax.set_title("Raw CAPE / FCF Yield / Next 1Y Return")
    plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7, label="Next 1Y Return (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "raw_factor_3d_scatter.png", dpi=180)
    plt.close()

    write_markdown(
        out_dir / "说明与数据预览.md",
        output_df,
        signal_csv=signal_csv,
        returns_csv=returns_csv,
    )

    manifest: Dict[str, object] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "signal_csv": str(signal_csv),
        "returns_csv": str(returns_csv),
        "out_dir": str(out_dir),
        "row_count": int(len(output_df)),
    }
    (out_dir / "manifest.json").write_text(pd.Series(manifest).to_json(force_ascii=False, indent=2), encoding="utf-8")

    print(f"[out_dir] {out_dir}")
    print(f"[rows] {len(output_df)}")
    print(f"[csv] {out_dir / 'raw_factor_3d_dataset.csv'}")
    print(f"[plot] {out_dir / 'raw_factor_3d_scatter.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
