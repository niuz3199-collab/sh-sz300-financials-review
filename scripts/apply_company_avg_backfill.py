#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repair_gemma_markdown_financials import (  # noqa: E402
    LOG_FIELDS,
    YEAR_FIELDS,
    append_csv_row,
    compute_missing_fields,
    load_run_config,
    resolve_csv_name,
    upsert_csv_row,
)
from scripts.step6_extract_financials_qwen_pdf import (  # noqa: E402
    infer_code_name,
    normalize_stock_code,
)


FIELD_SPECS = {
    "parent_netprofit": {
        "history_key": "parent_netprofit_yuan",
        "page_key": "income",
        "raw_unit": "元",
        "raw_value": lambda amount: float(amount),
        "normalized": lambda amount: {"parent_netprofit_yuan": float(amount)},
    },
    "total_shares": {
        "history_key": "total_shares_shares",
        "page_key": "shares",
        "raw_unit": "万股",
        "raw_value": lambda amount: float(amount) / 10000.0,
        "normalized": lambda amount: {
            "total_shares_shares": float(amount),
            "total_shares_wan": float(amount) / 10000.0,
        },
    },
    "operating_cashflow": {
        "history_key": "operating_cashflow_yuan",
        "page_key": "cfo",
        "raw_unit": "元",
        "raw_value": lambda amount: float(amount),
        "normalized": lambda amount: {"operating_cashflow_yuan": float(amount)},
    },
    "capex": {
        "history_key": "capex_yuan",
        "page_key": "capex",
        "raw_unit": "元",
        "raw_value": lambda amount: float(amount),
        "normalized": lambda amount: {"capex_yuan": float(amount)},
    },
}
FIELD_ORDER = list(FIELD_SPECS.keys())


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_path(value: str, *, default: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        return default.resolve()
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_company_history(raw_root: Path) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    history: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for year_dir in raw_root.iterdir():
        if not year_dir.is_dir():
            continue
        for path in year_dir.glob("*.json"):
            data = read_json(path)
            if not isinstance(data, dict):
                continue
            normalized = dict(data.get("normalized") or {})
            code = normalize_stock_code(path.stem)
            if not code:
                continue
            try:
                year = int(path.parent.name)
            except Exception:
                continue
            bucket = history.setdefault(code, {field_name: [] for field_name in FIELD_ORDER})
            for field_name, spec in FIELD_SPECS.items():
                raw_field = dict((data.get("raw") or {}).get(field_name) or {})
                if str(raw_field.get("source") or "").strip() == "company_average":
                    continue
                value = normalized.get(str(spec["history_key"]))
                if value is None:
                    continue
                try:
                    bucket[field_name].append((year, float(value)))
                except Exception:
                    continue
    return history


def filter_history_rows(field_name: str, history_rows: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
    finite_rows: List[Tuple[int, float]] = []
    for src_year, value in history_rows:
        try:
            amount = float(value)
        except Exception:
            continue
        if not math.isfinite(amount):
            continue
        if field_name == "total_shares" and amount <= 0:
            continue
        finite_rows.append((int(src_year), amount))

    if len(finite_rows) < 3:
        return finite_rows

    if field_name == "total_shares":
        positives = [value for _, value in finite_rows if value > 0]
        if len(positives) < 3:
            return finite_rows
        median_value = statistics.median(positives)
        lower = median_value / 100.0
        upper = median_value * 100.0
        filtered = [(src_year, value) for src_year, value in finite_rows if lower <= value <= upper]
        return filtered or finite_rows

    non_zero_magnitudes = [abs(value) for _, value in finite_rows if abs(value) > 0]
    if len(non_zero_magnitudes) < 3:
        return finite_rows
    median_log = statistics.median(math.log10(value) for value in non_zero_magnitudes)
    filtered: List[Tuple[int, float]] = []
    for src_year, value in finite_rows:
        if value == 0:
            filtered.append((src_year, value))
            continue
        if abs(math.log10(abs(value)) - median_log) <= 3.0:
            filtered.append((src_year, value))
    return filtered or finite_rows


def apply_company_average(
    extracted: Dict[str, object],
    *,
    field_name: str,
    average_value: float,
    source_years: Sequence[int],
) -> None:
    spec = FIELD_SPECS[field_name]
    raw = extracted.setdefault("raw", {})
    normalized = extracted.setdefault("normalized", {})
    pages = extracted.setdefault("pages", {})

    sorted_years = sorted({int(year) for year in source_years})
    raw[field_name] = {
        "value": spec["raw_value"](average_value),
        "unit": spec["raw_unit"],
        "evidence": "company_avg_imputed",
        "snippet_ids": [],
        "page": "company_avg_imputed",
        "source": "company_average",
        "source_years": sorted_years,
        "source_count": len(sorted_years),
    }
    normalized.update(spec["normalized"](average_value))
    pages[str(spec["page_key"])] = "company_avg_imputed"


def build_year_csv_row(extracted: Dict[str, object]) -> Dict[str, object]:
    task = dict(extracted.get("task") or {})
    raw = dict(extracted.get("raw") or {})
    normalized = dict(extracted.get("normalized") or {})
    stock_code = normalize_stock_code(task.get("stock_code") or raw.get("code") or "")
    pdf_path = str(task.get("pdf_path") or task.get("markdown_path") or raw.get("source_markdown_path") or "")
    return {
        "year": task.get("year") or raw.get("year") or "",
        "stock_code": stock_code,
        "code_name": infer_code_name(stock_code),
        "stock_name": "",
        "parent_netprofit": normalized.get("parent_netprofit_yuan"),
        "share_capital": normalized.get("total_shares_shares"),
        "share_capital_wan": normalized.get("total_shares_wan"),
        "netcash_operate": normalized.get("operating_cashflow_yuan"),
        "construct_long_asset": normalized.get("capex_yuan"),
        "pdf_path": pdf_path,
        "parent_netprofit_page": (raw.get("parent_netprofit") or {}).get("page"),
        "share_capital_page": (raw.get("total_shares") or {}).get("page"),
        "netcash_operate_page": (raw.get("operating_cashflow") or {}).get("page"),
        "construct_long_asset_page": (raw.get("capex") or {}).get("page"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply company-average backfill into main Gemma markdown results.")
    parser.add_argument(
        "--imputable-csv",
        default=str(REPO_ROOT / ".tmp_gemma_markdown_repair_runner" / "partial_tasks_manual_pdf_gt30_company_avg_imputable.csv"),
    )
    parser.add_argument("--out-dir", default=str(REPO_ROOT / ".tmp_gemma_markdown_financials_full"))
    parser.add_argument("--target-raw-root", default="")
    parser.add_argument("--year-csv-root", default="")
    parser.add_argument("--csv-name", default="")
    parser.add_argument("--runner-dir", default=str(REPO_ROOT / ".tmp_gemma_markdown_repair_runner"))
    args = parser.parse_args()

    imputable_csv = resolve_path(args.imputable_csv, default=REPO_ROOT / ".tmp_gemma_markdown_repair_runner" / "partial_tasks_manual_pdf_gt30_company_avg_imputable.csv")
    out_dir = resolve_path(args.out_dir, default=REPO_ROOT / ".tmp_gemma_markdown_financials_full")
    run_config = load_run_config(out_dir)
    target_raw_root = resolve_path(
        args.target_raw_root or str((out_dir / "raw_json")),
        default=out_dir / "raw_json",
    )
    year_csv_root = resolve_path(
        args.year_csv_root or str(run_config.get("year_csv_root") or (REPO_ROOT / ".tmp_gemma_year_csvs_full")),
        default=REPO_ROOT / ".tmp_gemma_year_csvs_full",
    )
    csv_name = resolve_csv_name(str(args.csv_name or ""), run_config)
    runner_dir = resolve_path(args.runner_dir, default=REPO_ROOT / ".tmp_gemma_markdown_repair_runner")
    log_path = out_dir / "extract_log.csv"

    rows = read_csv_rows(imputable_csv)
    history = build_company_history(target_raw_root)
    summary_counter = Counter()
    field_counter = Counter()
    manifest_rows: List[Dict[str, object]] = []

    for row in rows:
        try:
            year = int(str(row.get("year") or "").strip())
        except Exception:
            summary_counter["invalid_year_rows"] += 1
            continue
        stock_code = normalize_stock_code(row.get("stock_code") or "")
        if not stock_code:
            summary_counter["invalid_code_rows"] += 1
            continue

        raw_json_path = str(row.get("raw_json_path") or "").strip()
        target_path = Path(raw_json_path).resolve() if raw_json_path else (target_raw_root / str(year) / f"{stock_code}.json")
        extracted = read_json(target_path)
        if not isinstance(extracted, dict):
            summary_counter["missing_target_json"] += 1
            manifest_rows.append(
                {
                    "year": year,
                    "stock_code": stock_code,
                    "status": "missing_target_json",
                    "applied_fields": "",
                    "missing_before": "",
                    "missing_after": "",
                    "target_path": str(target_path),
                }
            )
            continue

        missing_before = compute_missing_fields(extracted)
        applied_fields: List[str] = []
        source_years_by_field: Dict[str, str] = {}

        for field_name in missing_before:
            history_rows = [
                (src_year, value)
                for src_year, value in history.get(stock_code, {}).get(field_name, [])
                if int(src_year) != year
            ]
            history_rows = filter_history_rows(field_name, history_rows)
            if not history_rows:
                continue
            average_value = sum(value for _, value in history_rows) / len(history_rows)
            source_years = sorted({src_year for src_year, _ in history_rows})
            apply_company_average(
                extracted,
                field_name=field_name,
                average_value=average_value,
                source_years=source_years,
            )
            applied_fields.append(field_name)
            field_counter[field_name] += 1
            source_years_by_field[field_name] = ",".join(str(item) for item in source_years)

        missing_after = compute_missing_fields(extracted)
        if applied_fields:
            write_json(target_path, extracted)
            year_dir = year_csv_root / str(year)
            year_csv = year_dir / csv_name.format(year=year)
            upsert_csv_row(year_csv, build_year_csv_row(extracted), fieldnames=YEAR_FIELDS, key_field="stock_code")

            append_csv_row(
                log_path,
                {
                    "ts": now_iso(),
                    "year": year,
                    "stock_code": stock_code,
                    "code_name": infer_code_name(stock_code),
                    "pdf_path": str((extracted.get("task") or {}).get("pdf_path") or (extracted.get("task") or {}).get("markdown_path") or ""),
                    "status": "ok" if not missing_after else "partial",
                    "message": "company_avg_backfill="
                    + ",".join(applied_fields)
                    + "; source_years="
                    + "|".join(f"{name}:{source_years_by_field.get(name, '')}" for name in applied_fields),
                    "raw_json_path": str(target_path),
                },
                fieldnames=LOG_FIELDS,
            )
            summary_counter["docs_updated"] += 1
        else:
            summary_counter["docs_unchanged"] += 1

        if missing_before and not missing_after:
            summary_counter["docs_fully_fixed"] += 1
        elif len(missing_after) < len(missing_before):
            summary_counter["docs_partially_improved"] += 1
        elif not missing_before:
            summary_counter["docs_already_ok"] += 1

        manifest_row = {
            "year": year,
            "stock_code": stock_code,
            "status": "updated" if applied_fields else "unchanged",
            "applied_fields": ",".join(applied_fields),
            "missing_before": ",".join(missing_before),
            "missing_after": ",".join(missing_after),
            "target_path": str(target_path),
        }
        for field_name in FIELD_ORDER:
            manifest_row[f"{field_name}_source_years"] = source_years_by_field.get(field_name, "")
        manifest_rows.append(manifest_row)

    summary_payload = {
        "ts": now_iso(),
        "imputable_csv": str(imputable_csv),
        "out_dir": str(out_dir),
        "target_raw_root": str(target_raw_root),
        "year_csv_root": str(year_csv_root),
        "counts": {
            "rows_total": len(rows),
            "docs_updated": int(summary_counter["docs_updated"]),
            "docs_unchanged": int(summary_counter["docs_unchanged"]),
            "docs_fully_fixed": int(summary_counter["docs_fully_fixed"]),
            "docs_partially_improved": int(summary_counter["docs_partially_improved"]),
            "docs_already_ok": int(summary_counter["docs_already_ok"]),
            "missing_target_json": int(summary_counter["missing_target_json"]),
            "invalid_year_rows": int(summary_counter["invalid_year_rows"]),
            "invalid_code_rows": int(summary_counter["invalid_code_rows"]),
        },
        "field_backfill_counts": dict(field_counter),
    }

    summary_json = runner_dir / "company_avg_backfill_summary.json"
    manifest_csv = runner_dir / "company_avg_backfill_manifest.csv"
    manifest_fields = [
        "year",
        "stock_code",
        "status",
        "applied_fields",
        "missing_before",
        "missing_after",
        "target_path",
    ] + [f"{field_name}_source_years" for field_name in FIELD_ORDER]
    write_json(summary_json, summary_payload)
    write_csv(manifest_csv, manifest_rows, manifest_fields)

    print(f"[summary_json] {summary_json}")
    print(f"[manifest_csv] {manifest_csv}")
    print(json.dumps(summary_payload["counts"], ensure_ascii=False))
    print(json.dumps(summary_payload["field_backfill_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
