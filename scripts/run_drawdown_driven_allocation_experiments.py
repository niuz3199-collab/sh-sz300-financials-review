#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    display_name: str
    model_name: str
    feature_cols: Sequence[str]


@dataclass(frozen=True)
class MappingSpec:
    slug: str
    display_name: str
    formula: str
    description: str
    weight_fn: Callable[[float], float]


@dataclass(frozen=True)
class Variant:
    slug: str
    display_name: str
    model_spec: ModelSpec
    mapping_spec: MappingSpec


@dataclass
class StrategyState:
    equity: float = 0.0
    bond: float = 0.0
    gold: float = 0.0
    cash: float = 0.0


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def resolve_default_drawdown_csv() -> Path:
    candidates = sorted(
        [p for p in REPO_ROOT.glob("factor_drawdown_analysis_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError("未找到 factor_drawdown_analysis_* 目录，请先生成回撤分析结果。")
    path = candidates[-1] / "factor_drawdown_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到回撤数据集: {path}")
    return path


def resolve_default_return_csv() -> Path:
    preferred = REPO_ROOT / "backtest_output_initial100k_nodca" / "策略资产明细.csv"
    if preferred.exists():
        return preferred
    fallback = REPO_ROOT / "backtest_output" / "策略资产明细.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("未找到可用的本地年度回测收益明细 CSV。")


def resolve_default_real_rate_csv() -> Optional[Path]:
    candidates = sorted(
        [p for p in REPO_ROOT.glob("allocation_transform_experiments_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    if not candidates:
        return None
    path = candidates[-1] / "base_signal_table.csv"
    return path if path.exists() else None


def descending_linear_band(
    risk_pct: float,
    *,
    low_risk: float,
    high_risk: float,
    high_weight: float,
    low_weight: float,
) -> float:
    risk_pct = float(risk_pct)
    if risk_pct <= low_risk:
        return clamp01(high_weight)
    if risk_pct >= high_risk:
        return clamp01(low_weight)
    t = (risk_pct - low_risk) / (high_risk - low_risk)
    return clamp01(high_weight + (low_weight - high_weight) * t)


def descending_sigmoid_band(
    risk_pct: float,
    *,
    center: float,
    k: float,
    floor: float,
    cap: float,
) -> float:
    z = float(k) * (float(risk_pct) - float(center))
    sig = 1.0 / (1.0 + math.exp(-z))
    return clamp01(float(floor) + (float(cap) - float(floor)) * (1.0 - sig))


def piecewise_risk_map_20_80(risk_pct: float) -> float:
    if risk_pct <= 12.0:
        return 0.80
    if risk_pct <= 18.0:
        return 0.70
    if risk_pct <= 25.0:
        return 0.60
    if risk_pct <= 35.0:
        return 0.45
    if risk_pct <= 45.0:
        return 0.30
    return 0.20


def build_model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            slug="cape_linear",
            display_name="CAPE Linear",
            model_name="linear",
            feature_cols=["原始CAPE"],
        ),
        ModelSpec(
            slug="cape_rf",
            display_name="CAPE RandomForest",
            model_name="rf",
            feature_cols=["原始CAPE"],
        ),
    ]


def build_mapping_specs() -> List[MappingSpec]:
    return [
        MappingSpec(
            slug="linear_20_80",
            display_name="Linear 20-80",
            formula="10%风险->80%仓位, 45%风险->20%仓位, 中间线性下降",
            description="风险越高越降仓，保留 20%-80% 区间，偏平衡。",
            weight_fn=lambda risk: descending_linear_band(
                risk,
                low_risk=10.0,
                high_risk=45.0,
                high_weight=0.80,
                low_weight=0.20,
            ),
        ),
        MappingSpec(
            slug="linear_30_70",
            display_name="Linear 30-70",
            formula="10%风险->70%仓位, 45%风险->30%仓位, 中间线性下降",
            description="抬高下限并压低上限，偏防守。",
            weight_fn=lambda risk: descending_linear_band(
                risk,
                low_risk=10.0,
                high_risk=45.0,
                high_weight=0.70,
                low_weight=0.30,
            ),
        ),
        MappingSpec(
            slug="sigmoid_20_80",
            display_name="Sigmoid 20-80",
            formula="以30%风险为中心的S形降仓，区间20%-80%",
            description="中间区间更敏感，极端区间更平滑。",
            weight_fn=lambda risk: descending_sigmoid_band(
                risk,
                center=30.0,
                k=0.18,
                floor=0.20,
                cap=0.80,
            ),
        ),
        MappingSpec(
            slug="piecewise_20_80",
            display_name="Piecewise 20-80",
            formula="12/18/25/35/45 风险台阶映射到 80/70/60/45/30/20 仓位",
            description="更像人工风控规则，风险抬升时分段降仓。",
            weight_fn=piecewise_risk_map_20_80,
        ),
    ]


def build_variants(model_specs: Sequence[ModelSpec], mapping_specs: Sequence[MappingSpec]) -> List[Variant]:
    variants: List[Variant] = []
    for model_spec in model_specs:
        for mapping_spec in mapping_specs:
            variants.append(
                Variant(
                    slug=f"{model_spec.slug}__{mapping_spec.slug}",
                    display_name=f"{model_spec.display_name} + {mapping_spec.display_name}",
                    model_spec=model_spec,
                    mapping_spec=mapping_spec,
                )
            )
    return variants


def load_drawdown_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["年份", "原始CAPE", "原始FCFYield_pct", "下一年最大回撤_pct"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"回撤数据集缺少列: {missing}")
    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["年份", "原始CAPE", "下一年最大回撤_pct"]).copy()
    out["年份"] = out["年份"].astype(int)
    out = out.sort_values("年份").reset_index(drop=True)
    return out


def load_asset_return_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["年份", "当年股收益率", "当年债收益率", "当年金收益率"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"年度回报明细缺少列: {missing}")
    out = df.copy()
    out["年份"] = pd.to_numeric(out["年份"], errors="coerce")
    out["当年股收益率"] = pd.to_numeric(out["当年股收益率"], errors="coerce")
    out["当年债收益率"] = pd.to_numeric(out["当年债收益率"], errors="coerce")
    out["当年金收益率"] = pd.to_numeric(out["当年金收益率"], errors="coerce")
    out = out.dropna(subset=["年份", "当年股收益率", "当年债收益率"]).copy()
    out["年份"] = out["年份"].astype(int)
    out = out.sort_values("年份").reset_index(drop=True)
    return out[["年份", "当年股收益率", "当年债收益率", "当年金收益率"]]


def load_real_rate_map(path: Optional[Path]) -> Dict[int, float]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "year" not in df.columns or "real_rate" not in df.columns:
        return {}
    out: Dict[int, float] = {}
    tmp = df.copy()
    tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
    tmp["real_rate"] = pd.to_numeric(tmp["real_rate"], errors="coerce")
    for _, row in tmp.iterrows():
        year = safe_float(row["year"])
        rate = safe_float(row["real_rate"])
        if year is None or rate is None:
            continue
        out[int(year)] = float(rate)
    return out


def fit_model(model_name: str):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42,
        )
    raise ValueError(model_name)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def compute_walk_forward_predictions(
    *,
    dataset_df: pd.DataFrame,
    model_spec: ModelSpec,
    min_train_years: int,
    warmup_weight: float,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    live_actual: List[float] = []
    live_pred: List[float] = []

    for _, row in dataset_df.iterrows():
        year = int(row["年份"])
        train_df = dataset_df[dataset_df["年份"] < year].copy()
        pred_dd_pct: Optional[float] = None
        model_status = "warmup"
        train_start_year: Optional[int] = None
        train_end_year: Optional[int] = None

        if len(train_df) >= int(min_train_years):
            X_train = train_df[list(model_spec.feature_cols)].to_numpy(dtype=float)
            y_train = train_df["下一年最大回撤_pct"].to_numpy(dtype=float)
            X_now = row[list(model_spec.feature_cols)].to_numpy(dtype=float).reshape(1, -1)
            model = fit_model(model_spec.model_name)
            model.fit(X_train, y_train)
            pred_dd_pct = float(model.predict(X_now)[0])
            model_status = "live"
            train_start_year = int(train_df["年份"].min())
            train_end_year = int(train_df["年份"].max())
            live_actual.append(float(row["下一年最大回撤_pct"]))
            live_pred.append(float(pred_dd_pct))

        risk_pct = None if pred_dd_pct is None else max(0.0, float(-pred_dd_pct))
        rows.append(
            {
                "年份": year,
                "原始CAPE": float(row["原始CAPE"]),
                "原始FCFYield_pct": safe_float(row.get("原始FCFYield_pct")),
                "下一年最大回撤_pct": float(row["下一年最大回撤_pct"]),
                "预测最大回撤_pct": pred_dd_pct,
                "预测风险幅度_pct": risk_pct,
                "模型状态": model_status,
                "训练样本数": int(len(train_df)),
                "训练起始年": train_start_year,
                "训练结束年": train_end_year,
                "warmup_weight": float(warmup_weight),
            }
        )

    pred_df = pd.DataFrame(rows).sort_values("年份").reset_index(drop=True)
    metrics = None
    if live_actual and live_pred:
        metrics = {
            "model_slug": model_spec.slug,
            "model_name": model_spec.display_name,
            "feature_cols": ",".join(model_spec.feature_cols),
            "prediction_years": len(live_actual),
            "start_live_year": int(pred_df[pred_df["模型状态"] == "live"]["年份"].min()),
            "end_live_year": int(pred_df[pred_df["模型状态"] == "live"]["年份"].max()),
            **metrics_dict(np.array(live_actual, dtype=float), np.array(live_pred, dtype=float)),
        }
    return {"pred_df": pred_df, "metrics": metrics}


def apply_mapping_to_predictions(
    *,
    pred_df: pd.DataFrame,
    mapping_spec: MappingSpec,
    warmup_weight: float,
) -> pd.DataFrame:
    out = pred_df.copy()
    weights: List[float] = []
    for _, row in out.iterrows():
        if row["模型状态"] != "live" or pd.isna(row["预测风险幅度_pct"]):
            weights.append(float(warmup_weight))
            continue
        weights.append(float(mapping_spec.weight_fn(float(row["预测风险幅度_pct"]))))
    out["W"] = weights
    out["W_pct"] = out["W"] * 100.0
    out["映射方案"] = mapping_spec.display_name
    out["映射公式"] = mapping_spec.formula
    out["映射说明"] = mapping_spec.description
    return out


def rebalance(state: StrategyState, weights: Dict[str, float]) -> None:
    total = state.equity + state.bond + state.gold + state.cash
    state.equity = total * weights.get("equity", 0.0)
    state.bond = total * weights.get("bond", 0.0)
    state.gold = total * weights.get("gold", 0.0)
    state.cash = total * weights.get("cash", 0.0)


def apply_returns(state: StrategyState, returns_map: Dict[str, float]) -> None:
    state.equity *= 1.0 + returns_map.get("equity", 0.0)
    state.bond *= 1.0 + returns_map.get("bond", 0.0)
    state.gold *= 1.0 + returns_map.get("gold", 0.0)
    state.cash *= 1.0 + returns_map.get("cash", 0.0)


def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    dd = (series - peak) / peak
    return dd.fillna(0.0)


def write_variant_year_configs(variant_dir: Path, pred_df: pd.DataFrame) -> None:
    rows_to_save = pred_df[
        [
            "年份",
            "原始CAPE",
            "原始FCFYield_pct",
            "下一年最大回撤_pct",
            "预测最大回撤_pct",
            "预测风险幅度_pct",
            "模型状态",
            "训练样本数",
            "训练起始年",
            "训练结束年",
            "W",
            "W_pct",
            "映射方案",
            "映射公式",
        ]
    ].copy()
    rows_to_save.to_csv(variant_dir / "variant_weights_by_year.csv", index=False, encoding="utf-8-sig")

    for _, row in rows_to_save.iterrows():
        year = int(row["年份"])
        year_dir = variant_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(year_dir / f"{year}_配置比例.csv", index=False, encoding="utf-8-sig")


def run_backtest_for_variant(
    *,
    variant: Variant,
    variant_dir: Path,
    pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    real_rate_map: Dict[int, float],
    initial_capital: float,
    annual_contribution: float,
) -> Dict[str, object]:
    merged = pred_df.merge(returns_df, on="年份", how="inner")
    merged = merged.sort_values("年份").reset_index(drop=True)
    if merged.empty:
        raise RuntimeError(f"{variant.slug}: 未能匹配到任何年度收益率数据。")

    out_dir = variant_dir / "backtest_output"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    for _, row in merged.iterrows():
        year = int(row["年份"])
        r_eq = float(row["当年股收益率"])
        r_bond = float(row["当年债收益率"])
        r_gold = safe_float(row.get("当年金收益率"))
        w = float(row["W"])

        principal += float(annual_contribution)

        a_state.cash += float(annual_contribution)
        rebalance(a_state, {"equity": w, "bond": 1.0 - w, "gold": 0.0, "cash": 0.0})
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
            "W": w,
            "W_pct": w * 100.0,
            "模型状态": row["模型状态"],
            "训练样本数": int(row["训练样本数"]),
            "预测最大回撤_pct": safe_float(row["预测最大回撤_pct"]),
            "预测风险幅度_pct": safe_float(row["预测风险幅度_pct"]),
            "实际下一年最大回撤_pct": float(row["下一年最大回撤_pct"]),
            "原始CAPE": float(row["原始CAPE"]),
        }
        rows.append(out_row)

        year_dir = variant_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([out_row]).to_csv(year_dir / f"{year}_策略资产明细.csv", index=False, encoding="utf-8-sig")

    detail_df = pd.DataFrame(rows).sort_values("年份").reset_index(drop=True)
    detail_df.to_csv(out_dir / "策略资产明细.csv", index=False, encoding="utf-8-sig")

    nominal_candidates: List[float] = []
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
        dd = compute_drawdown(pd.to_numeric(detail_df[col], errors="coerce").ffill())
        plt.plot(detail_df["年份"], dd, label=name)
    plt.title(f"回撤对比（{variant.display_name}）")
    plt.xlabel("年份")
    plt.ylabel("回撤")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "回撤对比.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(detail_df["年份"], detail_df["W_pct"], marker="o", linewidth=2.0)
    plt.title(f"权益仓位（{variant.display_name}）")
    plt.xlabel("年份")
    plt.ylabel("W (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "权益仓位走势.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    actual = pd.to_numeric(detail_df["实际下一年最大回撤_pct"], errors="coerce")
    pred = pd.to_numeric(detail_df["预测最大回撤_pct"], errors="coerce")
    plt.plot(detail_df["年份"], actual, marker="o", label="Actual Max DD", linewidth=2.0)
    plt.plot(detail_df["年份"], pred, marker="s", label="Predicted Max DD", linewidth=2.0)
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    plt.title(f"预测回撤 vs 实际回撤（{variant.display_name}）")
    plt.xlabel("年份")
    plt.ylabel("Max Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "预测回撤_vs_实际回撤.png", dpi=150)
    plt.close()

    report_lines = [
        "# 回撤驱动仓位策略报告",
        "",
        f"- 变体: {variant.display_name}",
        f"- 模型: {variant.model_spec.display_name}",
        f"- 特征: {','.join(variant.model_spec.feature_cols)}",
        f"- 映射公式: {variant.mapping_spec.formula}",
        f"- 映射说明: {variant.mapping_spec.description}",
        f"- 回测区间: {int(detail_df['年份'].min())} ~ {int(detail_df['年份'].max())}",
        f"- 初始资金: {float(initial_capital):.2f} 元",
        f"- 年度投入: {float(annual_contribution):.2f} 元",
        "",
        "## 回测结果对比",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## 关键说明",
        "",
        "- 策略A 为本次“预测下一年最大回撤 -> 映射权益仓位”的新方案。",
        "- 预测采用严格按年份扩展训练的 walk-forward 方式；warmup 年份使用固定中性仓位。",
        "- 策略D 早期黄金收益率缺失部分仍按 0 近似，口径与旧回测保持一致。",
    ]
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
    pred_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    strategy_a_metrics: Dict[str, object],
    model_metrics: Optional[Dict[str, object]],
) -> Dict[str, object]:
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
    row_2020 = detail_df[detail_df["年份"] == 2020]
    row_2021 = detail_df[detail_df["年份"] == 2021]

    ret_2008 = None
    if not row_2007.empty and not row_2008.empty:
        v0 = float(row_2007.iloc[0]["策略A_资产"])
        v1 = float(row_2008.iloc[0]["策略A_资产"])
        if v0:
            ret_2008 = v1 / v0 - 1.0

    ret_2021 = None
    if not row_2020.empty and not row_2021.empty:
        v0 = float(row_2020.iloc[0]["策略A_资产"])
        v1 = float(row_2021.iloc[0]["策略A_资产"])
        if v0:
            ret_2021 = v1 / v0 - 1.0

    live_rows = pred_df[pred_df["模型状态"] == "live"].copy()
    last_pred_dd = None
    if not live_rows.empty:
        last_pred_dd = safe_float(live_rows.iloc[-1]["预测最大回撤_pct"])

    summary = {
        "variant_slug": variant.slug,
        "variant_name": variant.display_name,
        "model_name": variant.model_spec.display_name,
        "mapping_name": variant.mapping_spec.display_name,
        "features": ",".join(variant.model_spec.feature_cols),
        "formula": variant.mapping_spec.formula,
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
        "predicted_2024_dd_pct": last_pred_dd,
        "live_year_count": int(len(live_rows)),
        "warmup_year_count": int((pred_df["模型状态"] != "live").sum()),
    }
    if model_metrics:
        summary["walk_forward_r2"] = model_metrics.get("r2")
        summary["walk_forward_mae"] = model_metrics.get("mae")
        summary["walk_forward_rmse"] = model_metrics.get("rmse")
        summary["walk_forward_start_year"] = model_metrics.get("start_live_year")
        summary["walk_forward_end_year"] = model_metrics.get("end_live_year")
    return summary


def make_root_strategy_plot(root_dir: Path, experiment_results: Sequence[Dict[str, object]]) -> None:
    plt.figure(figsize=(12, 6))
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
                linewidth=2.1,
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
    plt.title("回撤驱动仓位策略A横向对比")
    plt.xlabel("年份")
    plt.ylabel("资产（元）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_dir / "A策略横向对比.png", dpi=160)
    plt.close()


def make_root_weight_plot(root_dir: Path, experiment_results: Sequence[Dict[str, object]]) -> None:
    plt.figure(figsize=(12, 6))
    for result in experiment_results:
        variant = result["variant"]
        detail_df = result["detail_df"]
        plt.plot(detail_df["年份"], detail_df["W_pct"], label=variant.display_name, linewidth=2.0)
    plt.title("回撤驱动仓位年度权益配置")
    plt.xlabel("年份")
    plt.ylabel("W (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_dir / "W权重横向对比.png", dpi=160)
    plt.close()


def make_root_scatter_plot(root_dir: Path, summary_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    x = pd.to_numeric(summary_df["max_drawdown_A"], errors="coerce")
    y = pd.to_numeric(summary_df["final_asset_A"], errors="coerce")
    labels = summary_df["variant_name"].astype(str).tolist()
    plt.scatter(x, y, s=90)
    for px, py, label in zip(x, y, labels):
        plt.annotate(label, (px, py), textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.title("策略A：终值 vs 最大回撤")
    plt.xlabel("最大回撤")
    plt.ylabel("期末总资产（元）")
    plt.tight_layout()
    plt.savefig(root_dir / "A策略终值_vs_回撤.png", dpi=160)
    plt.close()


def make_model_prediction_plot(root_dir: Path, combined_pred_df: pd.DataFrame) -> None:
    if combined_pred_df.empty:
        return
    plt.figure(figsize=(12, 6))
    actual_plotted = False
    for model_name, grp in combined_pred_df.groupby("model_display_name"):
        grp = grp.sort_values("年份")
        plt.plot(grp["年份"], grp["预测最大回撤_pct"], marker="o", linewidth=2.0, label=f"Pred - {model_name}")
        if not actual_plotted:
            plt.plot(
                grp["年份"],
                grp["下一年最大回撤_pct"],
                marker="s",
                linewidth=2.2,
                linestyle="--",
                color="black",
                label="Actual Max DD",
            )
            actual_plotted = True
    plt.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)
    plt.title("Walk-forward 预测回撤对比")
    plt.xlabel("年份")
    plt.ylabel("Max Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_dir / "模型预测回撤对比.png", dpi=160)
    plt.close()


def write_root_report(
    *,
    root_dir: Path,
    summary_df: pd.DataFrame,
    model_metrics_df: pd.DataFrame,
    source_df: pd.DataFrame,
    variants: Sequence[Variant],
    initial_capital: float,
    annual_contribution: float,
    min_train_years: int,
    warmup_weight: float,
) -> None:
    lines = [
        "# 回撤驱动仓位实验汇总",
        "",
        f"- 初始资金: {initial_capital:.2f} 元",
        f"- 年度投入: {annual_contribution:.2f} 元",
        f"- walk-forward 最小训练样本数: {min_train_years}",
        f"- warmup 仓位: {warmup_weight * 100.0:.1f}%",
        f"- 变体数量: {len(list(variants))}",
        "",
        "## 模型设定",
        "",
    ]
    seen_models = set()
    for variant in variants:
        if variant.model_spec.slug in seen_models:
            continue
        seen_models.add(variant.model_spec.slug)
        lines.append(
            f"- `{variant.model_spec.slug}`: {variant.model_spec.display_name}，特征=`{','.join(variant.model_spec.feature_cols)}`"
        )
    lines.extend(
        [
            "",
            "## 仓位映射",
            "",
        ]
    )
    seen_mappings = set()
    for variant in variants:
        if variant.mapping_spec.slug in seen_mappings:
            continue
        seen_mappings.add(variant.mapping_spec.slug)
        lines.append(
            f"- `{variant.mapping_spec.slug}`: {variant.mapping_spec.display_name}，{variant.mapping_spec.formula}。{variant.mapping_spec.description}"
        )
    lines.extend(
        [
            "",
            "## Walk-forward 模型表现",
            "",
            model_metrics_df.to_markdown(index=False),
            "",
            "## 策略A汇总对比",
            "",
            summary_df.to_markdown(index=False),
            "",
            "## 源数据快照",
            "",
            source_df.to_markdown(index=False),
            "",
            "## 文件说明",
            "",
            "- 每个变体目录下包含年度配置比例、年度策略资产明细、回测 CSV、报告和图表。",
            "- 顶层额外提供策略A横向对比、权重横向对比、模型预测回撤对比和终值/回撤散点图。",
        ]
    )
    (root_dir / "实验汇总报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run walk-forward drawdown-driven allocation experiments.")
    parser.add_argument("--drawdown-csv", default="", help="默认使用最新 factor_drawdown_analysis 目录下的 factor_drawdown_dataset.csv")
    parser.add_argument("--return-csv", default="", help="默认使用本地已有回测明细中的年度收益率")
    parser.add_argument("--real-rate-csv", default="", help="默认使用最新 allocation_transform_experiments 的 base_signal_table.csv")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--annual-contribution", type=float, default=0.0)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--warmup-weight", type=float, default=0.50)
    parser.add_argument("--out-root", default="", help="实验输出目录；默认按时间戳新建")
    args = parser.parse_args()

    drawdown_csv = (
        Path(args.drawdown_csv).expanduser().resolve()
        if str(args.drawdown_csv or "").strip()
        else resolve_default_drawdown_csv()
    )
    return_csv = (
        Path(args.return_csv).expanduser().resolve()
        if str(args.return_csv or "").strip()
        else resolve_default_return_csv()
    )
    real_rate_csv = (
        Path(args.real_rate_csv).expanduser().resolve()
        if str(args.real_rate_csv or "").strip()
        else resolve_default_real_rate_csv()
    )
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if str(args.out_root or "").strip()
        else (REPO_ROOT / f"drawdown_driven_allocation_{now_stamp()}").resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)

    drawdown_df = load_drawdown_dataset(drawdown_csv)
    returns_df = load_asset_return_df(return_csv)
    real_rate_map = load_real_rate_map(real_rate_csv)
    source_df = drawdown_df.merge(returns_df, on="年份", how="left").sort_values("年份").reset_index(drop=True)
    source_df.to_csv(out_root / "source_dataset_with_returns.csv", index=False, encoding="utf-8-sig")

    model_specs = build_model_specs()
    mapping_specs = build_mapping_specs()
    variants = build_variants(model_specs=model_specs, mapping_specs=mapping_specs)

    walk_forward_tables: Dict[str, pd.DataFrame] = {}
    model_metrics_rows: List[Dict[str, object]] = []

    for model_spec in model_specs:
        wf_result = compute_walk_forward_predictions(
            dataset_df=drawdown_df,
            model_spec=model_spec,
            min_train_years=int(args.min_train_years),
            warmup_weight=float(args.warmup_weight),
        )
        pred_df = wf_result["pred_df"]
        pred_df["model_slug"] = model_spec.slug
        pred_df["model_display_name"] = model_spec.display_name
        walk_forward_tables[model_spec.slug] = pred_df
        pred_df.to_csv(out_root / f"{model_spec.slug}_walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
        if wf_result["metrics"] is not None:
            model_metrics_rows.append(dict(wf_result["metrics"]))

    model_metrics_df = pd.DataFrame(model_metrics_rows).sort_values("r2", ascending=False).reset_index(drop=True)
    model_metrics_df.to_csv(out_root / "walk_forward_model_comparison.csv", index=False, encoding="utf-8-sig")

    combined_pred_df = pd.concat(list(walk_forward_tables.values()), ignore_index=True)
    combined_pred_df.to_csv(out_root / "walk_forward_predictions_all_models.csv", index=False, encoding="utf-8-sig")
    make_model_prediction_plot(out_root, combined_pred_df[combined_pred_df["模型状态"] == "live"].copy())

    experiment_results: List[Dict[str, object]] = []

    for variant in variants:
        print(f"[variant] {variant.slug}")
        variant_dir = out_root / variant.slug
        variant_dir.mkdir(parents=True, exist_ok=True)
        base_pred_df = walk_forward_tables[variant.model_spec.slug]
        mapped_df = apply_mapping_to_predictions(
            pred_df=base_pred_df,
            mapping_spec=variant.mapping_spec,
            warmup_weight=float(args.warmup_weight),
        )
        write_variant_year_configs(variant_dir=variant_dir, pred_df=mapped_df)
        write_json(
            variant_dir / "variant_meta.json",
            {
                "slug": variant.slug,
                "display_name": variant.display_name,
                "model_slug": variant.model_spec.slug,
                "model_name": variant.model_spec.display_name,
                "mapping_slug": variant.mapping_spec.slug,
                "mapping_name": variant.mapping_spec.display_name,
                "features": list(variant.model_spec.feature_cols),
                "formula": variant.mapping_spec.formula,
                "description": variant.mapping_spec.description,
                "initial_capital": float(args.initial_capital),
                "annual_contribution": float(args.annual_contribution),
                "min_train_years": int(args.min_train_years),
                "warmup_weight": float(args.warmup_weight),
            },
        )

        backtest_result = run_backtest_for_variant(
            variant=variant,
            variant_dir=variant_dir,
            pred_df=mapped_df,
            returns_df=returns_df,
            real_rate_map=real_rate_map,
            initial_capital=float(args.initial_capital),
            annual_contribution=float(args.annual_contribution),
        )
        model_metrics = None
        if not model_metrics_df.empty:
            matched = model_metrics_df[model_metrics_df["model_slug"] == variant.model_spec.slug]
            if not matched.empty:
                model_metrics = matched.iloc[0].to_dict()
        summary_row = summarize_variant(
            variant=variant,
            pred_df=mapped_df,
            detail_df=backtest_result["detail_df"],
            strategy_a_metrics=backtest_result["strategy_a_metrics"],
            model_metrics=model_metrics,
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

    make_root_strategy_plot(out_root, experiment_results)
    make_root_weight_plot(out_root, experiment_results)
    make_root_scatter_plot(out_root, summary_df)
    write_root_report(
        root_dir=out_root,
        summary_df=summary_df,
        model_metrics_df=model_metrics_df,
        source_df=source_df[
            [
                "年份",
                "原始CAPE",
                "原始FCFYield_pct",
                "下一年最大回撤_pct",
                "当年股收益率",
                "当年债收益率",
                "当年金收益率",
            ]
        ],
        variants=variants,
        initial_capital=float(args.initial_capital),
        annual_contribution=float(args.annual_contribution),
        min_train_years=int(args.min_train_years),
        warmup_weight=float(args.warmup_weight),
    )

    write_json(
        out_root / "experiment_manifest.json",
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "drawdown_csv": str(drawdown_csv),
            "return_csv": str(return_csv),
            "real_rate_csv": None if real_rate_csv is None else str(real_rate_csv),
            "out_root": str(out_root),
            "initial_capital": float(args.initial_capital),
            "annual_contribution": float(args.annual_contribution),
            "min_train_years": int(args.min_train_years),
            "warmup_weight": float(args.warmup_weight),
            "variants": [
                {
                    "slug": variant.slug,
                    "display_name": variant.display_name,
                    "model": variant.model_spec.display_name,
                    "mapping": variant.mapping_spec.display_name,
                    "features": list(variant.model_spec.feature_cols),
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
