#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_csv_list(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    return conn


def parse_message(message: str) -> Tuple[List[str], List[str]]:
    missing_pages: List[str] = []
    missing_fields: List[str] = []
    for part in str(message or "").split(";"):
        part = part.strip()
        if part.startswith("missing_pages="):
            missing_pages = [item.strip() for item in part.split("=", 1)[1].split(",") if item.strip()]
        elif part.startswith("missing_fields="):
            missing_fields = [item.strip() for item in part.split("=", 1)[1].split(",") if item.strip()]
    return missing_pages, missing_fields


def classify_issue(status: str, message: str) -> Tuple[str, str]:
    status = str(status or "").strip().lower()
    message = str(message or "")
    missing_pages, missing_fields = parse_message(message)

    if status == "error":
        if "429 Client Error" in message:
            return "error_api_rate_limit", "API rate limit"
        if "127.0.0.1:11434" in message or "malloc" in message or "realloc" in message:
            return "error_local_memory_or_ollama", "Local memory/Ollama failure"
        if "naapi.cc" in message and "401" in message:
            return "error_naapi_unauthorized", "NAAPI unauthorized"
        if "naapi.cc" in message and "500" in message:
            return "error_naapi_server", "NAAPI server error"
        if "naapi.cc" in message:
            return "error_naapi_network", "NAAPI network/proxy error"
        if "Expecting value" in message or "Unterminated string" in message or "delimiter" in message:
            return "error_model_non_json", "Model returned invalid/truncated JSON"
        if "empty_response_content" in message:
            return "error_empty_response", "Empty model response"
        if "Connection aborted" in message or "ConnectionResetError" in message:
            return "error_network_disconnect", "Network disconnect"
        if "float" in message or "int" in message:
            return "error_code_scalar_payload", "Model returned scalar payload shape"
        return "error_other", "Other runtime/model error"

    if missing_pages:
        if len(missing_pages) == 4:
            return "partial_no_key_pages", "No key pages located"
        return "partial_page_locator_failed", "Some key pages not located"
    if missing_fields:
        if len(missing_fields) == 1:
            return "partial_field_extraction_failed", f"Field extraction failed: {missing_fields[0]}"
        return "partial_multi_field_extraction_failed", "Multiple fields still missing"
    return "other", "Other"


def recommended_handoff_bucket(status: str, message: str) -> str:
    missing_pages, _ = parse_message(message)
    if str(status or "").strip().lower() == "error":
        return "hard"
    if missing_pages:
        return "hard"
    return "soft"


def rows_to_csv(path: Path, rows: Iterable[Dict[str, object]], *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Export non-ok Qwen extraction tasks for manual web-AI handoff.")
    parser.add_argument("--source-out-dir", required=True, help="Run output directory containing task_queue.sqlite")
    parser.add_argument("--target-dir", default="", help="Directory to write handoff manifests; defaults to <source>/manual_handoff")
    parser.add_argument("--statuses", default="partial,error", help="Comma-separated final statuses to include")
    args = parser.parse_args()

    source_out_dir = Path(args.source_out_dir).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve() if str(args.target_dir or "").strip() else (source_out_dir / "manual_handoff")
    db_path = source_out_dir / "task_queue.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing queue DB: {db_path}")

    statuses = parse_csv_list(args.statuses)
    if not statuses:
        raise RuntimeError("No statuses selected.")

    conn = connect_db(db_path)
    try:
        placeholders = ",".join("?" for _ in statuses)
        rows = list(
            conn.execute(
                f"""
                SELECT
                    year,
                    stock_code,
                    code_name,
                    pdf_path,
                    claimed_by,
                    last_backend,
                    last_status,
                    last_message,
                    raw_json_path
                FROM tasks
                WHERE last_status IN ({placeholders})
                ORDER BY year, stock_code
                """,
                statuses,
            )
        )
    finally:
        conn.close()

    fieldnames = [
        "year",
        "stock_code",
        "code_name",
        "pdf_path",
        "final_status",
        "issue_code",
        "issue_label",
        "handoff_bucket",
        "missing_pages",
        "missing_fields",
        "source_worker",
        "source_backend",
        "raw_json_path",
        "last_message",
    ]

    export_rows: List[Dict[str, object]] = []
    hard_rows: List[Dict[str, object]] = []
    by_status = Counter()
    by_bucket = Counter()
    by_issue = Counter()
    by_year = Counter()
    by_worker = Counter()
    by_missing_page = Counter()
    by_missing_field = Counter()
    bucket_by_year: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        message = str(row["last_message"] or "")
        missing_pages, missing_fields = parse_message(message)
        issue_code, issue_label = classify_issue(str(row["last_status"] or ""), message)
        handoff_bucket = recommended_handoff_bucket(str(row["last_status"] or ""), message)

        export_row = {
            "year": int(row["year"]),
            "stock_code": str(row["stock_code"] or ""),
            "code_name": str(row["code_name"] or ""),
            "pdf_path": str(row["pdf_path"] or ""),
            "final_status": str(row["last_status"] or ""),
            "issue_code": issue_code,
            "issue_label": issue_label,
            "handoff_bucket": handoff_bucket,
            "missing_pages": ",".join(missing_pages),
            "missing_fields": ",".join(missing_fields),
            "source_worker": str(row["claimed_by"] or ""),
            "source_backend": str(row["last_backend"] or ""),
            "raw_json_path": str(row["raw_json_path"] or ""),
            "last_message": message,
        }
        export_rows.append(export_row)
        if handoff_bucket == "hard":
            hard_rows.append(export_row)

        by_status[export_row["final_status"]] += 1
        by_bucket[handoff_bucket] += 1
        by_issue[issue_code] += 1
        by_year[int(row["year"])] += 1
        by_worker[export_row["source_worker"]] += 1
        bucket_by_year[handoff_bucket][int(row["year"])] += 1
        for page in missing_pages:
            by_missing_page[page] += 1
        for field in missing_fields:
            by_missing_field[field] += 1

    all_path = target_dir / "handoff_all_non_ok.csv"
    hard_path = target_dir / "handoff_hard_failures.csv"
    summary_json_path = target_dir / "handoff_summary.json"
    summary_txt_path = target_dir / "handoff_summary.txt"

    rows_to_csv(all_path, export_rows, fieldnames=fieldnames)
    rows_to_csv(hard_path, hard_rows, fieldnames=fieldnames)

    summary = {
        "source_out_dir": str(source_out_dir),
        "db_path": str(db_path),
        "total_handoff_rows": len(export_rows),
        "hard_handoff_rows": len(hard_rows),
        "soft_handoff_rows": len(export_rows) - len(hard_rows),
        "by_status": dict(by_status),
        "by_bucket": dict(by_bucket),
        "by_issue": dict(by_issue.most_common()),
        "by_worker": dict(by_worker),
        "top_years": by_year.most_common(15),
        "top_missing_pages": by_missing_page.most_common(),
        "top_missing_fields": by_missing_field.most_common(),
        "hard_top_years": bucket_by_year["hard"].most_common(15),
        "soft_top_years": bucket_by_year["soft"].most_common(15),
        "all_manifest_path": str(all_path),
        "hard_manifest_path": str(hard_path),
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        f"source_out_dir: {source_out_dir}",
        f"total_handoff_rows: {len(export_rows)}",
        f"hard_handoff_rows: {len(hard_rows)}",
        f"soft_handoff_rows: {len(export_rows) - len(hard_rows)}",
        f"all_manifest_path: {all_path}",
        f"hard_manifest_path: {hard_path}",
        "by_status:",
    ]
    for key, value in by_status.items():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.append("by_issue:")
    for key, value in by_issue.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.append("top_missing_fields:")
    for key, value in by_missing_field.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.append("top_missing_pages:")
    for key, value in by_missing_page.most_common():
        summary_lines.append(f"  {key}: {value}")
    summary_txt_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
