#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


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


def classify_issue(status: str, message: str) -> str:
    status = str(status or "").strip().lower()
    message = str(message or "")
    missing_pages, missing_fields = parse_message(message)

    if status == "error":
        if "429 Client Error" in message:
            return "error_api_rate_limit"
        if "127.0.0.1:11434" in message or "malloc" in message or "realloc" in message:
            return "error_local_memory_or_ollama"
        if "naapi.cc" in message and "401" in message:
            return "error_naapi_unauthorized"
        if "naapi.cc" in message and "500" in message:
            return "error_naapi_server"
        if "naapi.cc" in message:
            return "error_naapi_network"
        if "Expecting value" in message or "Unterminated string" in message or "delimiter" in message:
            return "error_model_non_json"
        if "empty_response_content" in message:
            return "error_empty_response"
        if "Connection aborted" in message or "ConnectionResetError" in message:
            return "error_network_disconnect"
        if "float" in message or "int" in message:
            return "error_code_scalar_payload"
        return "error_other"

    if missing_pages:
        if len(missing_pages) >= 4:
            return "partial_no_key_pages"
        return "partial_page_locator_failed"
    if len(missing_fields) >= 2:
        return "partial_multi_field_extraction_failed"
    if len(missing_fields) == 1:
        return "partial_field_extraction_failed"
    return "other"


def difficulty_sort_key(row: sqlite3.Row) -> Tuple[int, int, int, int, int, str]:
    status = str(row["last_status"] or "")
    message = str(row["last_message"] or "")
    issue = classify_issue(status, message)
    missing_pages, missing_fields = parse_message(message)
    year = int(row["year"])
    code = str(row["stock_code"] or "")

    priority_map = {
        "partial_no_key_pages": 0,
        "partial_page_locator_failed": 1,
        "partial_multi_field_extraction_failed": 2,
        "partial_field_extraction_failed": 3,
        "error_model_non_json": 4,
        "error_local_memory_or_ollama": 5,
        "error_code_scalar_payload": 6,
        "error_empty_response": 7,
        "error_other": 8,
        "error_network_disconnect": 9,
        "error_api_rate_limit": 10,
        "error_naapi_network": 11,
        "error_naapi_server": 12,
        "error_naapi_unauthorized": 13,
        "other": 99,
    }

    field_weight = 0
    if "parent_netprofit" in missing_fields:
        field_weight += 2
    if "operating_cashflow" in missing_fields:
        field_weight += 2
    if "capex" in missing_fields:
        field_weight += 2
    if "total_shares" in missing_fields:
        field_weight += 1

    return (
        int(priority_map.get(issue, 99)),
        -len(missing_pages),
        -len(missing_fields),
        -field_weight,
        year,
        code,
    )


def safe_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except Exception:
        shutil.copy2(str(src), str(dst))
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a folder of the hardest annual-report PDFs for manual web-AI processing.")
    parser.add_argument("--source-out-dir", required=True, help="Run output directory containing task_queue.sqlite")
    parser.add_argument("--target-dir", required=True, help="Target folder to populate with selected PDFs")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of PDFs to export")
    args = parser.parse_args()

    source_out_dir = Path(args.source_out_dir).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    db_path = source_out_dir / "task_queue.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing queue DB: {db_path}")
    if int(args.limit) <= 0:
        raise RuntimeError("--limit must be positive")

    conn = sqlite3.connect(str(db_path), timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(
            conn.execute(
                """
                SELECT
                    year,
                    stock_code,
                    code_name,
                    pdf_path,
                    claimed_by,
                    last_backend,
                    last_status,
                    last_message
                FROM tasks
                WHERE last_status IN ('partial', 'error')
                ORDER BY year, stock_code
                """
            )
        )
    finally:
        conn.close()

    ranked = sorted(rows, key=difficulty_sort_key)
    selected = ranked[: int(args.limit)]

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    link_mode_counts = Counter()
    issue_counts = Counter()

    for row in selected:
        pdf_path = Path(str(row["pdf_path"]))
        issue = classify_issue(str(row["last_status"] or ""), str(row["last_message"] or ""))
        missing_pages, missing_fields = parse_message(str(row["last_message"] or ""))
        file_name = pdf_path.name
        dst_name = file_name
        dst_path = target_dir / dst_name
        if dst_path.exists():
            dst_name = f"{int(row['year'])}_{str(row['stock_code'])}_{file_name}"
            dst_path = target_dir / dst_name
        link_mode = safe_link_or_copy(pdf_path, dst_path)
        link_mode_counts[link_mode] += 1
        issue_counts[issue] += 1
        manifest_rows.append(
            {
                "year": int(row["year"]),
                "stock_code": str(row["stock_code"] or ""),
                "code_name": str(row["code_name"] or ""),
                "pdf_path": str(pdf_path),
                "exported_path": str(dst_path),
                "issue_code": issue,
                "missing_pages": ",".join(missing_pages),
                "missing_fields": ",".join(missing_fields),
                "source_worker": str(row["claimed_by"] or ""),
                "source_backend": str(row["last_backend"] or ""),
                "final_status": str(row["last_status"] or ""),
                "last_message": str(row["last_message"] or ""),
                "link_mode": link_mode,
            }
        )

    manifest_path = target_dir / "_selection_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "stock_code",
                "code_name",
                "pdf_path",
                "exported_path",
                "issue_code",
                "missing_pages",
                "missing_fields",
                "source_worker",
                "source_backend",
                "final_status",
                "last_message",
                "link_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source_out_dir": str(source_out_dir),
        "target_dir": str(target_dir),
        "selected_count": len(selected),
        "limit": int(args.limit),
        "issue_counts": dict(issue_counts),
        "link_mode_counts": dict(link_mode_counts),
        "manifest_path": str(manifest_path),
        "selection_rule": [
            "1) partial_no_key_pages",
            "2) partial_page_locator_failed",
            "3) partial_multi_field_extraction_failed",
            "4) partial_field_extraction_failed",
            "5) error_model_non_json",
            "6) other errors",
            "Within the same class, more missing pages/fields rank earlier; older years rank earlier.",
        ],
    }
    (target_dir / "_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
