#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

REPO_ROOT = Path(__file__).resolve().parents[1]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_default_dataset() -> Path:
    candidates = sorted(
        [p for p in REPO_ROOT.glob("raw_factor_3d_view_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError("未找到 raw_factor_3d_view_* 目录，请先生成原始因子三维数据。")
    path = candidates[-1] / "raw_factor_3d_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到数据集: {path}")
    return path


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fit_and_predict_loo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str,
) -> np.ndarray:
    loo = LeaveOneOut()
    preds: List[float] = []
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


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def make_prediction_plot(
    *,
    y_true: np.ndarray,
    y_pred_linear: np.ndarray,
    y_pred_rf: np.ndarray,
    out_path: Path,
) -> None:
    lo = float(min(y_true.min(), y_pred_linear.min(), y_pred_rf.min()))
    hi = float(max(y_true.max(), y_pred_linear.max(), y_pred_rf.max()))

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred_linear, label="Linear", s=60, alpha=0.8)
    plt.scatter(y_true, y_pred_rf, label="RandomForest", s=60, alpha=0.8)
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.5, label="Perfect fit")
    plt.xlabel("Actual Next-1Y Return (%)")
    plt.ylabel("Predicted Next-1Y Return (%)")
    plt.title("LOO Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def make_feature_scatter(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df["原始CAPE"],
        df["4月30日至次年4月30日收益率_pct"],
        c=df["原始FCFYield_pct"],
        cmap="coolwarm",
        s=80,
    )
    for _, row in df.iterrows():
        plt.annotate(str(int(row["年份"])), (row["原始CAPE"], row["4月30日至次年4月30日收益率_pct"]), fontsize=8, alpha=0.9)
    plt.xlabel("Raw CAPE")
    plt.ylabel("Next-1Y Return (%)")
    plt.title("CAPE vs Next-1Y Return, colored by FCF Yield")
    plt.colorbar(scatter, label="Raw FCF Yield (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze linear vs random-forest fit for CAPE/FCF to next-year return.")
    parser.add_argument("--dataset-csv", default="", help="默认使用最新 raw_factor_3d_view 目录下的数据集")
    parser.add_argument(
        "--target-mode",
        default="raw_pct",
        choices=["raw_pct", "log1p_decimal"],
        help="raw_pct=直接拟合收益率百分比；log1p_decimal=拟合 log(1+r)，其中 r 为小数收益率",
    )
    parser.add_argument("--out-dir", default="", help="输出目录；默认按时间戳新建")
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv).expanduser().resolve() if str(args.dataset_csv or "").strip() else resolve_default_dataset()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir or "").strip()
        else (REPO_ROOT / f"factor_model_analysis_{now_stamp()}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv, encoding="utf-8-sig")
    X = df[["原始CAPE", "原始FCFYield_pct"]].to_numpy(dtype=float)
    y_raw_pct = df["4月30日至次年4月30日收益率_pct"].to_numpy(dtype=float)
    y_raw_decimal = df["4月30日至次年4月30日收益率"].to_numpy(dtype=float)

    if str(args.target_mode) == "log1p_decimal":
        y = np.log1p(y_raw_decimal)
        target_name = "log(1+r)"
        inverse_to_raw_pct = lambda arr: np.expm1(np.asarray(arr, dtype=float)) * 100.0
    else:
        y = y_raw_pct.copy()
        target_name = "raw_return_pct"
        inverse_to_raw_pct = lambda arr: np.asarray(arr, dtype=float)

    corr_rows = [
        {
            "feature": "原始CAPE",
            "pearson_r": float(pearsonr(df["原始CAPE"], y).statistic),
            "pearson_pvalue": float(pearsonr(df["原始CAPE"], y).pvalue),
            "spearman_r": float(spearmanr(df["原始CAPE"], y).statistic),
            "spearman_pvalue": float(spearmanr(df["原始CAPE"], y).pvalue),
        },
        {
            "feature": "原始FCFYield_pct",
            "pearson_r": float(pearsonr(df["原始FCFYield_pct"], y).statistic),
            "pearson_pvalue": float(pearsonr(df["原始FCFYield_pct"], y).pvalue),
            "spearman_r": float(spearmanr(df["原始FCFYield_pct"], y).statistic),
            "spearman_pvalue": float(spearmanr(df["原始FCFYield_pct"], y).pvalue),
        },
    ]
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / "feature_correlations.csv", index=False, encoding="utf-8-sig")

    linear = LinearRegression().fit(X, y)
    yhat_linear_insample = linear.predict(X)

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42,
    ).fit(X, y)
    yhat_rf_insample = rf.predict(X)

    linear_loo = fit_and_predict_loo(X, y, model_name="linear")
    rf_loo = fit_and_predict_loo(X, y, model_name="rf")

    metrics_rows = [
        {
            "model": "linear_insample",
            **metrics_dict(y, yhat_linear_insample),
        },
        {
            "model": "linear_loo",
            **metrics_dict(y, linear_loo),
        },
        {
            "model": "rf_insample",
            **metrics_dict(y, yhat_rf_insample),
        },
        {
            "model": "rf_loo",
            **metrics_dict(y, rf_loo),
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")

    raw_scale_metrics_df = pd.DataFrame(
        [
            {
                "model": "linear_insample_raw_pct",
                **metrics_dict(y_raw_pct, inverse_to_raw_pct(yhat_linear_insample)),
            },
            {
                "model": "linear_loo_raw_pct",
                **metrics_dict(y_raw_pct, inverse_to_raw_pct(linear_loo)),
            },
            {
                "model": "rf_insample_raw_pct",
                **metrics_dict(y_raw_pct, inverse_to_raw_pct(yhat_rf_insample)),
            },
            {
                "model": "rf_loo_raw_pct",
                **metrics_dict(y_raw_pct, inverse_to_raw_pct(rf_loo)),
            },
        ]
    )
    raw_scale_metrics_df.to_csv(out_dir / "model_metrics_raw_scale.csv", index=False, encoding="utf-8-sig")

    predictions_df = df[["年份", "原始CAPE", "原始FCFYield_pct", "4月30日至次年4月30日收益率_pct"]].copy()
    predictions_df["target_train_scale"] = y
    predictions_df["linear_insample_pred_pct"] = yhat_linear_insample
    predictions_df["linear_loo_pred_pct"] = linear_loo
    predictions_df["rf_insample_pred_pct"] = yhat_rf_insample
    predictions_df["rf_loo_pred_pct"] = rf_loo
    predictions_df["linear_insample_pred_raw_pct"] = inverse_to_raw_pct(yhat_linear_insample)
    predictions_df["linear_loo_pred_raw_pct"] = inverse_to_raw_pct(linear_loo)
    predictions_df["rf_insample_pred_raw_pct"] = inverse_to_raw_pct(yhat_rf_insample)
    predictions_df["rf_loo_pred_raw_pct"] = inverse_to_raw_pct(rf_loo)
    predictions_df["linear_loo_error_raw_pct"] = predictions_df["linear_loo_pred_raw_pct"] - predictions_df["4月30日至次年4月30日收益率_pct"]
    predictions_df["rf_loo_error_raw_pct"] = predictions_df["rf_loo_pred_raw_pct"] - predictions_df["4月30日至次年4月30日收益率_pct"]
    predictions_df.to_csv(out_dir / "loo_predictions.csv", index=False, encoding="utf-8-sig")

    coef_df = pd.DataFrame(
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
    coef_df.to_csv(out_dir / "model_parameters.csv", index=False, encoding="utf-8-sig")

    make_prediction_plot(
        y_true=y_raw_pct,
        y_pred_linear=inverse_to_raw_pct(linear_loo),
        y_pred_rf=inverse_to_raw_pct(rf_loo),
        out_path=out_dir / "loo_pred_vs_actual.png",
    )
    make_feature_scatter(df, out_dir / "cape_return_scatter.png")

    report_lines = [
        "# CAPE / FCF 拟合分析",
        "",
        f"- 数据集: `{dataset_csv}`",
        f"- 样本数: {len(df)}",
        f"- 训练目标: {target_name}",
        "- 原始观察值: 当年 4/30 到次年 4/30 的沪深300收益率",
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
        "## 转回原始收益率后的模型表现",
        "",
        raw_scale_metrics_df.to_markdown(index=False),
        "",
        "## 线性模型参数",
        "",
        coef_df.to_markdown(index=False),
        "",
        "## 结论摘要",
        "",
        "- 若看留一法 R²，线性回归和随机森林都为负，说明对未见样本几乎没有稳定解释力。",
        "- 随机森林样本内 R² 会更高，但留一法并未优于线性回归，属于典型小样本过拟合。",
        "- 线性回归系数为负，方向上符合“CAPE 越高、未来收益越差”的直觉，但统计强度弱。",
        "- 若使用 log(1+r) 作为训练目标，请重点看 `model_metrics_raw_scale.csv`，因为最终仍然要回到原始收益率尺度判断是否更可用。",
    ]
    (out_dir / "分析报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "dataset_csv": str(dataset_csv),
        "out_dir": str(out_dir),
        "target_mode": str(args.target_mode),
        "target_name": target_name,
        "sample_count": int(len(df)),
        "linear_coef_cape": float(linear.coef_[0]),
        "linear_coef_fcf": float(linear.coef_[1]),
        "linear_intercept": float(linear.intercept_),
        "rf_feature_importances": [float(x) for x in rf.feature_importances_.tolist()],
        "linear_loo_r2": float(metrics_df.loc[metrics_df["model"] == "linear_loo", "r2"].iloc[0]),
        "rf_loo_r2": float(metrics_df.loc[metrics_df["model"] == "rf_loo", "r2"].iloc[0]),
        "linear_loo_raw_pct_r2": float(raw_scale_metrics_df.loc[raw_scale_metrics_df["model"] == "linear_loo_raw_pct", "r2"].iloc[0]),
        "rf_loo_raw_pct_r2": float(raw_scale_metrics_df.loc[raw_scale_metrics_df["model"] == "rf_loo_raw_pct", "r2"].iloc[0]),
    }
    write_json(out_dir / "summary.json", summary)

    print(f"[out_dir] {out_dir}")
    print(f"[metrics_csv] {out_dir / 'model_metrics.csv'}")
    print(f"[pred_csv] {out_dir / 'loo_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
