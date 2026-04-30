#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repair_gemma_markdown_financials import compute_missing_fields  # noqa: E402
from scripts.step6_extract_financials_qwen_pdf import normalize_stock_code  # noqa: E402


FIELD_SPECS = {
    "net_profit_yuan": {
        "field_name": "parent_netprofit",
        "normalized_keys": [("parent_netprofit_yuan", lambda value: float(value))],
        "raw_unit": "元",
        "raw_value": lambda value: value,
    },
    "total_shares_wan": {
        "field_name": "total_shares",
        "normalized_keys": [
            ("total_shares_wan", lambda value: float(value)),
            ("total_shares_shares", lambda value: float(value) * 10000.0),
        ],
        "raw_unit": "万股",
        "raw_value": lambda value: value,
    },
    "operating_cashflow_yuan": {
        "field_name": "operating_cashflow",
        "normalized_keys": [("operating_cashflow_yuan", lambda value: float(value))],
        "raw_unit": "元",
        "raw_value": lambda value: value,
    },
    "capex_yuan": {
        "field_name": "capex",
        "normalized_keys": [("capex_yuan", lambda value: float(value))],
        "raw_unit": "元",
        "raw_value": lambda value: value,
    },
}


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


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def import_record(
    *,
    extracted: Dict[str, object],
    record: Dict[str, object],
    source_tag: str,
) -> List[str]:
    normalized = extracted.setdefault("normalized", {})
    raw = extracted.setdefault("raw", {})
    imported_fields: List[str] = []

    for source_key, spec in FIELD_SPECS.items():
        incoming_value = record.get(source_key)
        if incoming_value is None:
            continue
        field_name = str(spec["field_name"])
        final_norm_key = str(spec["normalized_keys"][-1][0])
        if normalized.get(final_norm_key) is not None:
            continue

        raw[field_name] = {
            "value": spec["raw_value"](incoming_value),
            "unit": spec["raw_unit"],
            "evidence": f"imported_from_{source_tag}",
            "snippet_ids": [],
            "page": None,
            "source": "financial_data_json",
        }
        for norm_key, transform in spec["normalized_keys"]:
            normalized[str(norm_key)] = transform(incoming_value)
        imported_fields.append(field_name)

    return imported_fields


def main() -> int:
    parser = argparse.ArgumentParser(description="Import flat financial_data.json results into main raw_json files.")
    parser.add_argument("--financial-data-json", default=str(REPO_ROOT / "financial_data.json"))
    parser.add_argument("--target-raw-root", default=str(REPO_ROOT / ".tmp_gemma_markdown_financials_full" / "raw_json"))
    parser.add_argument("--runner-dir", default=str(REPO_ROOT / ".tmp_gemma_markdown_repair_runner"))
    args = parser.parse_args()

    source_path = Path(args.financial_data_json).resolve()
    target_raw_root = Path(args.target_raw_root).resolve()
    runner_dir = Path(args.runner_dir).resolve()

    records = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError("financial_data.json must be a list")

    summary_counter = Counter()
    field_counter = Counter()
    manifest_rows: List[Dict[str, object]] = []
    source_tag = source_path.name

    for record in records:
        try:
            year = int(record.get("year"))
        except Exception:
            summary_counter["invalid_year_rows"] += 1
            continue
        stock_code = normalize_stock_code(record.get("code") or "")
        if not stock_code:
            summary_counter["invalid_code_rows"] += 1
            continue

        target_path = target_raw_root / str(year) / f"{stock_code}.json"
        extracted = read_json(target_path)
        if not isinstance(extracted, dict):
            summary_counter["missing_target_json"] += 1
            manifest_rows.append(
                {
                    "year": year,
                    "stock_code": stock_code,
                    "status": "missing_target_json",
                    "imported_fields": "",
                    "missing_before": "",
                    "missing_after": "",
                    "target_path": str(target_path),
                }
            )
            continue

        missing_before = compute_missing_fields(extracted)
        imported_fields = import_record(extracted=extracted, record=record, source_tag=source_tag)
        missing_after = compute_missing_fields(extracted)

        for field_name in imported_fields:
            field_counter[field_name] += 1

        status = "unchanged"
        if imported_fields:
            write_json(target_path, extracted)
            status = "updated"
            summary_counter["docs_updated"] += 1
        else:
            summary_counter["docs_unchanged"] += 1

        if len(missing_after) < len(missing_before):
            summary_counter["docs_improved"] += 1
        if missing_before and not missing_after:
            summary_counter["docs_fully_fixed"] += 1

        manifest_rows.append(
            {
                "year": year,
                "stock_code": stock_code,
                "status": status,
                "imported_fields": ",".join(imported_fields),
                "missing_before": ",".join(missing_before),
                "missing_after": ",".join(missing_after),
                "target_path": str(target_path),
            }
        )

    summary_payload = {
        "ts": now_iso(),
        "financial_data_json": str(source_path),
        "target_raw_root": str(target_raw_root),
        "records_total": len(records),
        "counts": {
            "docs_updated": int(summary_counter["docs_updated"]),
            "docs_unchanged": int(summary_counter["docs_unchanged"]),
            "docs_improved": int(summary_counter["docs_improved"]),
            "docs_fully_fixed": int(summary_counter["docs_fully_fixed"]),
            "missing_target_json": int(summary_counter["missing_target_json"]),
            "invalid_year_rows": int(summary_counter["invalid_year_rows"]),
            "invalid_code_rows": int(summary_counter["invalid_code_rows"]),
        },
        "field_import_counts": dict(field_counter),
    }

    summary_json = runner_dir / "financial_data_import_summary.json"
    summary_csv = runner_dir / "financial_data_import_manifest.csv"
    write_json(summary_json, summary_payload)
    write_csv(
        summary_csv,
        manifest_rows,
        fieldnames=["year", "stock_code", "status", "imported_fields", "missing_before", "missing_after", "target_path"],
    )

    print(f"[summary_json] {summary_json}")
    print(f"[summary_csv] {summary_csv}")
    print(json.dumps(summary_payload["counts"], ensure_ascii=False))
    print(json.dumps(summary_payload["field_import_counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
