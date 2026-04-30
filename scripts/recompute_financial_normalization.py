#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.apply_company_avg_backfill import (  # noqa: E402
    apply_company_average,
    build_company_history,
    build_year_csv_row,
    filter_history_rows,
    resolve_path,
    write_csv,
    write_json,
)
from scripts.repair_gemma_markdown_financials import (  # noqa: E402
    FIELD_ORDER,
    YEAR_FIELDS,
    compute_missing_fields,
    load_run_config,
    normalize_field_value,
    resolve_csv_name,
)


FIELD_NORMALIZED_KEYS = {
    "parent_netprofit": ["parent_netprofit_yuan"],
    "total_shares": ["total_shares_wan", "total_shares_shares"],
    "operating_cashflow": ["operating_cashflow_yuan"],
    "capex": ["capex_yuan"],
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def iter_raw_json_paths(raw_root: Path, *, start_year: int, end_year: int) -> Iterable[Path]:
    for year_dir in sorted(raw_root.iterdir(), key=lambda p: p.name):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except Exception:
            continue
        if year < start_year or year > end_year:
            continue
        for path in sorted(year_dir.glob("*.json"), key=lambda p: p.name):
            yield path


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_field_snapshot(extracted: Dict[str, object], field_name: str) -> Dict[str, object]:
    normalized = dict(extracted.get("normalized") or {})
    out: Dict[str, object] = {}
    for key in FIELD_NORMALIZED_KEYS[field_name]:
        out[key] = normalized.get(key)
    return out


def clear_field_normalized(extracted: Dict[str, object], field_name: str) -> None:
    normalized = extracted.setdefault("normalized", {})
    for key in FIELD_NORMALIZED_KEYS[field_name]:
        normalized[key] = None


def write_year_csvs(
    *,
    raw_root: Path,
    year_csv_root: Path,
    csv_name: str,
    start_year: int,
    end_year: int,
) -> Dict[int, int]:
    rows_by_year: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for path in iter_raw_json_paths(raw_root, start_year=start_year, end_year=end_year):
        extracted = read_json(path)
        try:
            year = int(path.parent.name)
        except Exception:
            continue
        rows_by_year[year].append(build_year_csv_row(extracted))

    counts: Dict[int, int] = {}
    for year, rows in sorted(rows_by_year.items()):
        ordered_rows = sorted(rows, key=lambda row: str(row.get("stock_code") or ""))
        counts[year] = len(ordered_rows)

        year_tmp_path = year_csv_root / str(year) / csv_name.format(year=year)
        write_csv(year_tmp_path, ordered_rows, YEAR_FIELDS)

        root_year_dir = REPO_ROOT / str(year)
        root_year_dir.mkdir(parents=True, exist_ok=True)
        root_year_csv = root_year_dir / f"{year}_财报数据.csv"
        write_csv(root_year_csv, ordered_rows, YEAR_FIELDS)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute normalized financial values from raw_json and rebuild year CSVs.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / ".tmp_gemma_markdown_financials_full"))
    parser.add_argument("--target-raw-root", default="")
    parser.add_argument("--year-csv-root", default="")
    parser.add_argument("--csv-name", default="")
    parser.add_argument("--runner-dir", default=str(REPO_ROOT / ".tmp_gemma_markdown_repair_runner"))
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    out_dir = resolve_path(args.out_dir, default=REPO_ROOT / ".tmp_gemma_markdown_financials_full")
    run_config = load_run_config(out_dir)
    raw_root = resolve_path(args.target_raw_root or str(out_dir / "raw_json"), default=out_dir / "raw_json")
    year_csv_root = resolve_path(
        args.year_csv_root or str(run_config.get("year_csv_root") or (REPO_ROOT / ".tmp_gemma_year_csvs_full")),
        default=REPO_ROOT / ".tmp_gemma_year_csvs_full",
    )
    csv_name = resolve_csv_name(str(args.csv_name or ""), run_config)
    runner_dir = resolve_path(args.runner_dir, default=REPO_ROOT / ".tmp_gemma_markdown_repair_runner")
    runner_dir.mkdir(parents=True, exist_ok=True)

    summary = Counter()
    manifest_rows: List[Dict[str, object]] = []
    imputed_backlog: List[Tuple[Path, int, str, List[str]]] = []

    for path in iter_raw_json_paths(raw_root, start_year=int(args.start_year), end_year=int(args.end_year)):
        extracted = read_json(path)
        raw = extracted.setdefault("raw", {})
        extracted.setdefault("normalized", {})
        task = dict(extracted.get("task") or {})
        stock_code = str(task.get("stock_code") or path.stem).strip()
        year = int(task.get("year") or path.parent.name)
        changed_fields: List[str] = []
        deferred_imputed_fields: List[str] = []

        for field_name in FIELD_ORDER:
            raw_field = dict(raw.get(field_name) or {})
            before = get_field_snapshot(extracted, field_name)
            clear_field_normalized(extracted, field_name)

            if not raw_field:
                after = get_field_snapshot(extracted, field_name)
                if before != after:
                    changed_fields.append(field_name)
                continue

            if str(raw_field.get("source") or "").strip() == "company_average":
                raw[field_name] = raw_field
                deferred_imputed_fields.append(field_name)
                after = get_field_snapshot(extracted, field_name)
                if before != after:
                    changed_fields.append(field_name)
                continue

            normalized_updates = normalize_field_value(field_name, raw_field, [])
            raw[field_name] = raw_field
            extracted["normalized"].update(normalized_updates)
            after = get_field_snapshot(extracted, field_name)
            if before != after:
                changed_fields.append(field_name)
                summary["fields_recomputed"] += 1

        write_json(path, extracted)
        summary["docs_rewritten_first_pass"] += 1
        if deferred_imputed_fields:
            imputed_backlog.append((path, year, stock_code, deferred_imputed_fields))
            summary["docs_with_company_average"] += 1
        if changed_fields:
            summary["docs_changed_first_pass"] += 1
        manifest_rows.append(
            {
                "year": year,
                "stock_code": stock_code,
                "path": str(path),
                "changed_fields_first_pass": ",".join(changed_fields),
                "deferred_company_average_fields": ",".join(deferred_imputed_fields),
                "missing_after_first_pass": ",".join(compute_missing_fields(extracted)),
            }
        )

    company_history = build_company_history(raw_root)

    for path, year, stock_code, field_names in imputed_backlog:
        extracted = read_json(path)
        changed_fields: List[str] = []
        for field_name in field_names:
            before = get_field_snapshot(extracted, field_name)
            history_rows = [
                (src_year, value)
                for src_year, value in company_history.get(stock_code, {}).get(field_name, [])
                if int(src_year) != int(year)
            ]
            history_rows = filter_history_rows(field_name, history_rows)
            clear_field_normalized(extracted, field_name)
            if history_rows:
                average_value = sum(value for _, value in history_rows) / len(history_rows)
                source_years = sorted({src_year for src_year, _ in history_rows})
                apply_company_average(
                    extracted,
                    field_name=field_name,
                    average_value=average_value,
                    source_years=source_years,
                )
                summary["company_average_fields_refreshed"] += 1
            else:
                summary["company_average_fields_without_clean_history"] += 1

            after = get_field_snapshot(extracted, field_name)
            if before != after:
                changed_fields.append(field_name)

        write_json(path, extracted)
        if changed_fields:
            summary["docs_changed_second_pass"] += 1
        summary["docs_rewritten_second_pass"] += 1

    year_row_counts = write_year_csvs(
        raw_root=raw_root,
        year_csv_root=year_csv_root,
        csv_name=csv_name,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )
    summary["years_written"] = len(year_row_counts)
    summary["year_rows_written"] = sum(year_row_counts.values())

    final_missing = Counter()
    for path in iter_raw_json_paths(raw_root, start_year=int(args.start_year), end_year=int(args.end_year)):
        missing = compute_missing_fields(read_json(path))
        if not missing:
            summary["docs_ok_after_cleanup"] += 1
        else:
            summary["docs_partial_after_cleanup"] += 1
            for field_name in missing:
                final_missing[field_name] += 1

    summary_payload = {
        "ts": now_iso(),
        "out_dir": str(out_dir),
        "raw_root": str(raw_root),
        "year_csv_root": str(year_csv_root),
        "csv_name": csv_name,
        "counts": dict(summary),
        "missing_fields_after_cleanup": dict(final_missing),
        "year_row_counts": year_row_counts,
    }

    summary_json = runner_dir / "recompute_financial_normalization_summary.json"
    manifest_csv = runner_dir / "recompute_financial_normalization_manifest.csv"
    write_json(summary_json, summary_payload)
    write_csv(
        manifest_csv,
        manifest_rows,
        ["year", "stock_code", "path", "changed_fields_first_pass", "deferred_company_average_fields", "missing_after_first_pass"],
    )

    print(f"[summary_json] {summary_json}")
    print(f"[manifest_csv] {manifest_csv}")
    print(json.dumps(summary_payload["counts"], ensure_ascii=False))
    print(json.dumps(summary_payload["missing_fields_after_cleanup"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
