#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.step8_backtest import fetch_h00300_series  # noqa: E402


FEATURE_SETS: Dict[str, List[str]] = {
    "cape_only": ["原始CAPE"],
    "cape_fcf": ["原始CAPE", "原始FCFYield_pct"],
    "cape_trend": ["原始CAPE", "trend_gap_pct"],
    "cape_trend_fcf": ["原始CAPE", "trend_gap_pct", "原始FCFYield_pct"],
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_default_factor_csv() -> Path:
    candidates = sorted(
        [p for p in REPO_ROOT.glob("raw_factor_3d_view_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError("未找到 raw_factor_3d_view_* 目录，请先生成原始因子三维数据。")
    path = candidates[-1] / "raw_factor_3d_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到因子数据集: {path}")
    return path


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_window(series_df: pd.DataFrame, d0: date, d1: date) -> pd.DataFrame:
    df = series_df.copy()
    df["_d"] = pd.to_datetime(df["日期"], errors="coerce").dt.date
    df["_v"] = pd.to_numeric(df["收盘"], errors="coerce")
    df = df[df["_d"].notna() & df["_v"].notna()].copy()
    df = df[(df["_d"] >= d0) & (df["_d"] <= d1)].copy()
    return df.sort_values("_d").reset_index(drop=True)


def latest_history_on_or_before(series_df: pd.DataFrame, target: date) -> pd.DataFrame:
    df = series_df.copy()
    df["_d"] = pd.to_datetime(df["日期"], errors="coerce").dt.date
    df["_v"] = pd.to_numeric(df["收盘"], errors="coerce")
    df = df[df["_d"].notna() & df["_v"].notna()].copy()
    df = df[df["_d"] <= target].copy()
    return df.sort_values("_d").reset_index(drop=True)


def compute_window_max_drawdown(window_df: pd.DataFrame) -> Optional[float]:
    if window_df is None or window_df.empty:
        return None
    values = pd.to_numeric(window_df["_v"], errors="coerce").dropna()
    if values.empty:
        return None
    peak = values.cummax()
    dd = (values - peak) / peak
    if dd.empty:
        return None
    return float(dd.min())


def compute_trend_gap_pct(series_df: pd.DataFrame, target: date, ma_window: int = 250) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    hist = latest_history_on_or_before(series_df, target)
    if hist.empty or len(hist) < ma_window:
        return None, None, None
    closes = pd.to_numeric(hist["_v"], errors="coerce").dropna()
    if len(closes) < ma_window:
        return None, None, None
    latest_close = float(closes.iloc[-1])
    ma_value = float(closes.iloc[-ma_window:].mean())
    if ma_value == 0:
        return latest_close, ma_value, None
    trend_gap = latest_close / ma_value - 1.0
    return latest_close, ma_value, float(trend_gap)


def build_dataset(factor_df: pd.DataFrame, eq_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in factor_df.iterrows():
        year = int(row["年份"])
        d0 = date(year, 4, 30)
        d1 = date(year + 1, 4, 30)
        window_df = pick_window(eq_df, d0=d0, d1=d1)
        max_dd = compute_window_max_drawdown(window_df)
        close_0430, ma250_0430, trend_gap = compute_trend_gap_pct(eq_df, target=d0, ma_window=250)
        rows.append(
            {
                "年份": year,
                "原始CAPE": float(row["原始CAPE"]),
                "原始FCFYield_pct": float(row["原始FCFYield_pct"]),
                "4月30日至次年4月30日收益率_pct": float(row["4月30日至次年4月30日收益率_pct"]),
                "下一年最大回撤_pct": None if max_dd is None else float(max_dd) * 100.0,
                "close_0430": close_0430,
                "ma250_0430": ma250_0430,
                "trend_gap_pct": None if trend_gap is None else float(trend_gap) * 100.0,
                "trend_above_ma250": None if trend_gap is None else int(trend_gap > 0),
                "窗口样本数": int(len(window_df)),
            }
        )
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["下一年最大回撤_pct", "trend_gap_pct"]).sort_values("年份").reset_index(drop=True)
    return out


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def fit_model(name: str):
    if name == "linear":
        return LinearRegression()
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        )
    raise ValueError(name)


def fit_loo_predictions(X: np.ndarray, y: np.ndarray, model_name: str) -> np.ndarray:
    preds: List[float] = []
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        model = fit_model(model_name)
        model.fit(X[train_idx], y[train_idx])
        preds.append(float(model.predict(X[test_idx])[0]))
    return np.array(preds, dtype=float)


def make_loo_r2_plot(results_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = results_df[results_df["scope"] == "loo"].copy()
    labels = [f"{fs}\n{model}" for fs, model in zip(plot_df["feature_set"], plot_df["model"])]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, plot_df["r2"])
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    plt.ylabel("LOO R²")
    plt.title("不同特征组合对下一年最大回撤的留一法表现")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def make_feature_scatter(dataset_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        dataset_df["原始CAPE"],
        dataset_df["下一年最大回撤_pct"],
        c=dataset_df["trend_gap_pct"],
        cmap="RdYlGn",
        s=80,
    )
    for _, row in dataset_df.iterrows():
        plt.annotate(str(int(row["年份"])), (row["原始CAPE"], row["下一年最大回撤_pct"]), fontsize=8)
    plt.xlabel("Raw CAPE")
    plt.ylabel("Next-1Y Max Drawdown (%)")
    plt.title("CAPE vs Next-1Y Max Drawdown, colored by Trend Gap")
    plt.colorbar(scatter, label="Trend Gap vs MA250 (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CAPE-only and CAPE+trend feature sets for next-year max drawdown.")
    parser.add_argument("--factor-csv", default="", help="默认使用最新 raw_factor_3d_view 目录下的数据")
    parser.add_argument("--out-dir", default="", help="输出目录；默认按时间戳新建")
    args = parser.parse_args()

    factor_csv = Path(args.factor_csv).expanduser().resolve() if str(args.factor_csv or "").strip() else resolve_default_factor_csv()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir or "").strip()
        else (REPO_ROOT / f"drawdown_feature_compare_{now_stamp()}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_df = pd.read_csv(factor_csv, encoding="utf-8-sig")
    min_year = int(pd.to_numeric(factor_df["年份"], errors="coerce").min())
    max_year = int(pd.to_numeric(factor_df["年份"], errors="coerce").max())
    eq_df = fetch_h00300_series(start=f"{min_year - 1}0101", end=f"{max_year + 1}1231")

    dataset_df = build_dataset(factor_df, eq_df)
    dataset_df.to_csv(out_dir / "feature_drawdown_dataset.csv", index=False, encoding="utf-8-sig")

    corr_df = pd.DataFrame(
        [
            {
                "feature": "原始CAPE",
                "pearson_r": float(pearsonr(dataset_df["原始CAPE"], dataset_df["下一年最大回撤_pct"]).statistic),
                "pearson_pvalue": float(pearsonr(dataset_df["原始CAPE"], dataset_df["下一年最大回撤_pct"]).pvalue),
                "spearman_r": float(spearmanr(dataset_df["原始CAPE"], dataset_df["下一年最大回撤_pct"]).statistic),
                "spearman_pvalue": float(spearmanr(dataset_df["原始CAPE"], dataset_df["下一年最大回撤_pct"]).pvalue),
            },
            {
                "feature": "原始FCFYield_pct",
                "pearson_r": float(pearsonr(dataset_df["原始FCFYield_pct"], dataset_df["下一年最大回撤_pct"]).statistic),
                "pearson_pvalue": float(pearsonr(dataset_df["原始FCFYield_pct"], dataset_df["下一年最大回撤_pct"]).pvalue),
                "spearman_r": float(spearmanr(dataset_df["原始FCFYield_pct"], dataset_df["下一年最大回撤_pct"]).statistic),
                "spearman_pvalue": float(spearmanr(dataset_df["原始FCFYield_pct"], dataset_df["下一年最大回撤_pct"]).pvalue),
            },
            {
                "feature": "trend_gap_pct",
                "pearson_r": float(pearsonr(dataset_df["trend_gap_pct"], dataset_df["下一年最大回撤_pct"]).statistic),
                "pearson_pvalue": float(pearsonr(dataset_df["trend_gap_pct"], dataset_df["下一年最大回撤_pct"]).pvalue),
                "spearman_r": float(spearmanr(dataset_df["trend_gap_pct"], dataset_df["下一年最大回撤_pct"]).statistic),
                "spearman_pvalue": float(spearmanr(dataset_df["trend_gap_pct"], dataset_df["下一年最大回撤_pct"]).pvalue),
            },
        ]
    )
    corr_df.to_csv(out_dir / "feature_correlations.csv", index=False, encoding="utf-8-sig")

    y = dataset_df["下一年最大回撤_pct"].to_numpy(dtype=float)
    result_rows: List[Dict[str, object]] = []
    prediction_tables: List[pd.DataFrame] = []

    for feature_set, columns in FEATURE_SETS.items():
        X = dataset_df[columns].to_numpy(dtype=float)
        for model_name in ("linear", "rf"):
            model = fit_model(model_name)
            model.fit(X, y)
            insample_pred = model.predict(X)
            loo_pred = fit_loo_predictions(X, y, model_name)

            for scope, pred in (("insample", insample_pred), ("loo", loo_pred)):
                result_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "scope": scope,
                        "n_features": len(columns),
                        "features": ",".join(columns),
                        **metrics_dict(y, pred),
                    }
                )

            pred_df = dataset_df[["年份", "下一年最大回撤_pct"] + columns].copy()
            pred_df["feature_set"] = feature_set
            pred_df["model"] = model_name
            pred_df["insample_pred_pct"] = insample_pred
            pred_df["loo_pred_pct"] = loo_pred
            pred_df["loo_error_pct"] = pred_df["loo_pred_pct"] - pred_df["下一年最大回撤_pct"]
            prediction_tables.append(pred_df)

    results_df = pd.DataFrame(result_rows).sort_values(["scope", "r2"], ascending=[True, False]).reset_index(drop=True)
    results_df.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    all_pred_df = pd.concat(prediction_tables, ignore_index=True)
    all_pred_df.to_csv(out_dir / "loo_predictions_all_models.csv", index=False, encoding="utf-8-sig")

    make_loo_r2_plot(results_df, out_dir / "loo_r2_comparison.png")
    make_feature_scatter(dataset_df, out_dir / "cape_trend_drawdown_scatter.png")

    loo_df = results_df[results_df["scope"] == "loo"].copy().sort_values("r2", ascending=False)
    top_loo = loo_df.head(8)

    report_lines = [
        "# CAPE 单因子 vs CAPE+趋势：下一年最大回撤",
        "",
        f"- 因子数据: `{factor_csv}`",
        "- 趋势定义: 4/30 当天收盘价相对 250 日均线的偏离 `close / MA250 - 1`（百分比形式）",
        "- 目标变量: 当年 4/30 到次年 4/30 区间内的最大回撤（%）",
        f"- 样本数: {len(dataset_df)}",
        "",
        "## 特征相关性",
        "",
        corr_df.to_markdown(index=False),
        "",
        "## 模型对比",
        "",
        results_df.to_markdown(index=False),
        "",
        "## 留一法最优组合",
        "",
        top_loo.to_markdown(index=False),
        "",
        "## 结论摘要",
        "",
        "- 若 `cape_trend` 的留一法显著优于 `cape_only`，说明趋势因子确实给避险模型带来了增量信息。",
        "- 若 `cape_fcf` 明显弱于 `cape_trend`，则 FCF 在这一任务中可以降级处理。",
        "- 若随机森林没有超过线性回归，则当前样本下优先使用更简单、更可解释的线性模型。",
    ]
    (out_dir / "分析报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "factor_csv": str(factor_csv),
        "out_dir": str(out_dir),
        "sample_count": int(len(dataset_df)),
        "best_loo_feature_set": str(loo_df.iloc[0]["feature_set"]),
        "best_loo_model": str(loo_df.iloc[0]["model"]),
        "best_loo_r2": float(loo_df.iloc[0]["r2"]),
    }
    write_json(out_dir / "summary.json", summary)

    print(f"[out_dir] {out_dir}")
    print(f"[dataset_csv] {out_dir / 'feature_drawdown_dataset.csv'}")
    print(f"[comparison_csv] {out_dir / 'model_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
