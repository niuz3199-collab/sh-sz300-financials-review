#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def build_drawdown_dataset(
    *,
    factor_df: pd.DataFrame,
    eq_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in factor_df.iterrows():
        year = int(row["年份"])
        d0 = date(year, 4, 30)
        d1 = date(year + 1, 4, 30)
        window_df = pick_window(eq_df, d0=d0, d1=d1)
        max_dd = compute_window_max_drawdown(window_df)
        rows.append(
            {
                "年份": year,
                "原始CAPE": float(row["原始CAPE"]),
                "原始FCFYield_pct": float(row["原始FCFYield_pct"]),
                "4月30日至次年4月30日收益率": float(row["4月30日至次年4月30日收益率"]),
                "4月30日至次年4月30日收益率_pct": float(row["4月30日至次年4月30日收益率_pct"]),
                "下一年最大回撤": max_dd,
                "下一年最大回撤_pct": None if max_dd is None else float(max_dd) * 100.0,
                "窗口样本数": int(len(window_df)),
            }
        )
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["下一年最大回撤"]).sort_values("年份").reset_index(drop=True)
    return out


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def fit_loo_predictions(X: np.ndarray, y: np.ndarray, model_name: str) -> np.ndarray:
    preds: List[float] = []
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        if model_name == "linear":
            model = LinearRegression()
        elif model_name == "rf":
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=3,
                min_samples_leaf=2,
                random_state=42,
            )
        else:
            raise ValueError(model_name)
        model.fit(X_train, y_train)
        preds.append(float(model.predict(X_test)[0]))
    return np.array(preds, dtype=float)


def make_pred_plot(y_true: np.ndarray, y_pred_linear: np.ndarray, y_pred_rf: np.ndarray, out_path: Path) -> None:
    lo = float(min(y_true.min(), y_pred_linear.min(), y_pred_rf.min()))
    hi = float(max(y_true.max(), y_pred_linear.max(), y_pred_rf.max()))
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred_linear, s=60, alpha=0.8, label="Linear")
    plt.scatter(y_true, y_pred_rf, s=60, alpha=0.8, label="RandomForest")
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.5, label="Perfect fit")
    plt.xlabel("Actual Next-1Y Max Drawdown (%)")
    plt.ylabel("Predicted Next-1Y Max Drawdown (%)")
    plt.title("LOO Predicted vs Actual Max Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def make_feature_scatter(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df["原始CAPE"],
        df["下一年最大回撤_pct"],
        c=df["原始FCFYield_pct"],
        cmap="coolwarm",
        s=80,
    )
    for _, row in df.iterrows():
        plt.annotate(str(int(row["年份"])), (row["原始CAPE"], row["下一年最大回撤_pct"]), fontsize=8, alpha=0.9)
    plt.xlabel("Raw CAPE")
    plt.ylabel("Next-1Y Max Drawdown (%)")
    plt.title("CAPE vs Next-1Y Max Drawdown, colored by FCF Yield")
    plt.colorbar(scatter, label="Raw FCF Yield (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze CAPE/FCF against next-year max drawdown.")
    parser.add_argument("--factor-csv", default="", help="默认使用最新 raw_factor_3d_view 目录下的 raw_factor_3d_dataset.csv")
    parser.add_argument("--out-dir", default="", help="输出目录；默认按时间戳新建")
    args = parser.parse_args()

    factor_csv = Path(args.factor_csv).expanduser().resolve() if str(args.factor_csv or "").strip() else resolve_default_factor_csv()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir or "").strip()
        else (REPO_ROOT / f"factor_drawdown_analysis_{now_stamp()}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_df = pd.read_csv(factor_csv, encoding="utf-8-sig")
    eq_df = fetch_h00300_series(
        start=f"{int(pd.to_numeric(factor_df['年份'], errors='coerce').min())}0101",
        end=f"{int(pd.to_numeric(factor_df['年份'], errors='coerce').max()) + 1}1231",
    )
    dataset_df = build_drawdown_dataset(factor_df=factor_df, eq_df=eq_df)
    dataset_df.to_csv(out_dir / "factor_drawdown_dataset.csv", index=False, encoding="utf-8-sig")

    X = dataset_df[["原始CAPE", "原始FCFYield_pct"]].to_numpy(dtype=float)
    y = dataset_df["下一年最大回撤_pct"].to_numpy(dtype=float)

    corr_df = pd.DataFrame(
        [
            {
                "feature": "原始CAPE",
                "pearson_r": float(pearsonr(dataset_df["原始CAPE"], y).statistic),
                "pearson_pvalue": float(pearsonr(dataset_df["原始CAPE"], y).pvalue),
                "spearman_r": float(spearmanr(dataset_df["原始CAPE"], y).statistic),
                "spearman_pvalue": float(spearmanr(dataset_df["原始CAPE"], y).pvalue),
            },
            {
                "feature": "原始FCFYield_pct",
                "pearson_r": float(pearsonr(dataset_df["原始FCFYield_pct"], y).statistic),
                "pearson_pvalue": float(pearsonr(dataset_df["原始FCFYield_pct"], y).pvalue),
                "spearman_r": float(spearmanr(dataset_df["原始FCFYield_pct"], y).statistic),
                "spearman_pvalue": float(spearmanr(dataset_df["原始FCFYield_pct"], y).pvalue),
            },
        ]
    )
    corr_df.to_csv(out_dir / "feature_correlations.csv", index=False, encoding="utf-8-sig")

    linear = LinearRegression().fit(X, y)
    yhat_linear = linear.predict(X)
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42,
    ).fit(X, y)
    yhat_rf = rf.predict(X)

    linear_loo = fit_loo_predictions(X, y, "linear")
    rf_loo = fit_loo_predictions(X, y, "rf")

    metrics_df = pd.DataFrame(
        [
            {"model": "linear_insample", **metrics_dict(y, yhat_linear)},
            {"model": "linear_loo", **metrics_dict(y, linear_loo)},
            {"model": "rf_insample", **metrics_dict(y, yhat_rf)},
            {"model": "rf_loo", **metrics_dict(y, rf_loo)},
        ]
    )
    metrics_df.to_csv(out_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")

    params_df = pd.DataFrame(
        [
            {
                "model": "linear",
                "coef_cape": float(linear.coef_[0]),
                "coef_fcf": float(linear.coef_[1]),
                "intercept": float(linear.intercept_),
            },
            {
                "model": "rf",
                "coef_cape": float(rf.feature_importances_[0]),
                "coef_fcf": float(rf.feature_importances_[1]),
                "intercept": float("nan"),
            },
        ]
    )
    params_df.to_csv(out_dir / "model_parameters.csv", index=False, encoding="utf-8-sig")

    pred_df = dataset_df[
        [
            "年份",
            "原始CAPE",
            "原始FCFYield_pct",
            "4月30日至次年4月30日收益率_pct",
            "下一年最大回撤_pct",
        ]
    ].copy()
    pred_df["linear_insample_pred_pct"] = yhat_linear
    pred_df["linear_loo_pred_pct"] = linear_loo
    pred_df["rf_insample_pred_pct"] = yhat_rf
    pred_df["rf_loo_pred_pct"] = rf_loo
    pred_df["linear_loo_error_pct"] = pred_df["linear_loo_pred_pct"] - pred_df["下一年最大回撤_pct"]
    pred_df["rf_loo_error_pct"] = pred_df["rf_loo_pred_pct"] - pred_df["下一年最大回撤_pct"]
    pred_df.to_csv(out_dir / "loo_predictions.csv", index=False, encoding="utf-8-sig")

    make_pred_plot(y_true=y, y_pred_linear=linear_loo, y_pred_rf=rf_loo, out_path=out_dir / "loo_pred_vs_actual.png")
    make_feature_scatter(dataset_df, out_dir / "cape_drawdown_scatter.png")

    report_lines = [
        "# CAPE / FCF 对下一年最大回撤的拟合分析",
        "",
        f"- 因子数据: `{factor_csv}`",
        "- 目标变量: 当年 4/30 到次年 4/30 区间内的最大回撤（%）",
        f"- 样本数: {len(dataset_df)}",
        "- 特征: 原始 CAPE、原始 FCF Yield",
        "- 评估方式: 样本内 + 留一法（LOOCV）",
        "",
        "## 特征相关性",
        "",
        corr_df.to_markdown(index=False),
        "",
        "## 模型表现",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## 模型参数",
        "",
        params_df.to_markdown(index=False),
        "",
        "## 结论摘要",
        "",
        "- 若留一法 R² 明显改善，说明这两个因子对“避险目标”比对“收益目标”更有解释力。",
        "- 随机森林样本内若显著高于留一法，仍应视为小样本过拟合。",
    ]
    (out_dir / "分析报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "factor_csv": str(factor_csv),
        "out_dir": str(out_dir),
        "sample_count": int(len(dataset_df)),
        "linear_coef_cape": float(linear.coef_[0]),
        "linear_coef_fcf": float(linear.coef_[1]),
        "linear_intercept": float(linear.intercept_),
        "rf_feature_importances": [float(x) for x in rf.feature_importances_.tolist()],
        "linear_loo_r2": float(metrics_df.loc[metrics_df["model"] == "linear_loo", "r2"].iloc[0]),
        "rf_loo_r2": float(metrics_df.loc[metrics_df["model"] == "rf_loo", "r2"].iloc[0]),
    }
    write_json(out_dir / "summary.json", summary)

    print(f"[out_dir] {out_dir}")
    print(f"[dataset_csv] {out_dir / 'factor_drawdown_dataset.csv'}")
    print(f"[metrics_csv] {out_dir / 'model_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
