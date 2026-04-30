#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repair_gemma_markdown_financials import compute_missing_fields  # noqa: E402
from scripts.step6_extract_financials_qwen_pdf import normalize_stock_code  # noqa: E402


FIELD_ORDER = ["parent_netprofit", "total_shares", "operating_cashflow", "capex"]
FIELD_PAGE_KEYS = {
    "parent_netprofit": "income",
    "total_shares": "shares",
    "operating_cashflow": "cfo",
    "capex": "capex",
}
FIELD_NORMALIZED_KEYS = {
    "parent_netprofit": ["parent_netprofit_yuan"],
    "total_shares": ["total_shares_wan", "total_shares_shares"],
    "operating_cashflow": ["operating_cashflow_yuan"],
    "capex": ["capex_yuan"],
}


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def choose_preferred_pdf(current: Optional[Path], candidate: Path) -> Path:
    if current is None:
        return candidate
    current_score = (1 if "fulltext" in str(current).lower() else 0, current.stat().st_size if current.exists() else 0)
    candidate_score = (1 if "fulltext" in str(candidate).lower() else 0, candidate.stat().st_size if candidate.exists() else 0)
    return candidate if candidate_score > current_score else current


def build_pdf_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in root.rglob("*.pdf"):
        index[path.name] = choose_preferred_pdf(index.get(path.name), path)
    return index


def ensure_target_json(target_path: Path, row: Dict[str, str]) -> Dict[str, object]:
    current = read_json(target_path)
    if isinstance(current, dict):
        return current
    year = int(str(row.get("year") or "0").strip())
    stock_code = normalize_stock_code(row.get("stock_code") or "")
    report_name = str(row.get("report_name") or "").strip()
    md_path = str(row.get("markdown_path") or "").strip()
    return {
        "task": {
            "stock_code": stock_code,
            "year": year,
            "report_name": report_name,
            "pdf_path": md_path,
            "markdown_path": md_path,
            "code_name": "",
        },
        "raw": {
            "code": stock_code,
            "year": year,
            "source_markdown_path": md_path,
            "parent_netprofit": {"value": None, "unit": None, "evidence": "missing_response", "snippet_ids": [], "page": None},
            "total_shares": {"value": None, "unit": None, "evidence": "missing_response", "snippet_ids": [], "page": None},
            "operating_cashflow": {"value": None, "unit": None, "evidence": "missing_response", "snippet_ids": [], "page": None},
            "capex": {"value": None, "unit": None, "evidence": "missing_response", "snippet_ids": [], "page": None},
        },
        "normalized": {
            "parent_netprofit_yuan": None,
            "total_shares_wan": None,
            "total_shares_shares": None,
            "operating_cashflow_yuan": None,
            "capex_yuan": None,
        },
        "pages": {"income": None, "shares": None, "cfo": None, "capex": None},
    }


def overlay_missing_fields(target: Dict[str, object], source: Dict[str, object]) -> List[str]:
    merged_fields: List[str] = []
    target_task = target.setdefault("task", {})
    source_task = dict(source.get("task") or {})
    for key in ("pdf_path", "markdown_path", "report_name", "code_name"):
        if source_task.get(key):
            target_task[key] = source_task.get(key)

    target_raw = target.setdefault("raw", {})
    source_raw = dict(source.get("raw") or {})
    target_norm = target.setdefault("normalized", {})
    source_norm = dict(source.get("normalized") or {})
    target_pages = target.setdefault("pages", {})
    source_pages = dict(source.get("pages") or {})

    for field_name in FIELD_ORDER:
        normalized_keys = FIELD_NORMALIZED_KEYS[field_name]
        target_present = target_norm.get(normalized_keys[-1]) is not None
        source_present = source_norm.get(normalized_keys[-1]) is not None
        if target_present or not source_present:
            continue
        if field_name in source_raw:
            target_raw[field_name] = source_raw[field_name]
        for key in normalized_keys:
            if key in source_norm:
                target_norm[key] = source_norm.get(key)
        page_key = FIELD_PAGE_KEYS[field_name]
        if page_key in source_pages and source_pages.get(page_key) is not None:
            target_pages[page_key] = source_pages.get(page_key)
        merged_fields.append(field_name)
    return merged_fields


def resolve_pdf_pages(pdf_index: Dict[str, Path], report_name: str) -> Tuple[str, Optional[int]]:
    pdf_path = pdf_index.get(f"{report_name}.pdf")
    if pdf_path is None:
        return "", None
    try:
        import fitz  # local import to keep startup cheap

        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
    except Exception:
        page_count = None
    return str(pdf_path), page_count


def build_company_present_map(raw_root: Path) -> Dict[str, Dict[str, List[float]]]:
    company_present: Dict[str, Dict[str, List[float]]] = {}
    for year_dir in raw_root.iterdir():
        if not year_dir.is_dir():
            continue
        for path in year_dir.glob("*.json"):
            data = read_json(path)
            if not isinstance(data, dict):
                continue
            code = path.stem
            normalized = dict(data.get("normalized") or {})
            bucket = company_present.setdefault(code, {field: [] for field in FIELD_ORDER})
            for field_name in FIELD_ORDER:
                key = FIELD_NORMALIZED_KEYS[field_name][-1]
                value = normalized.get(key)
                if value is not None:
                    bucket[field_name].append(value)
    return company_present


def count_md_lines(path_text: str) -> int:
    try:
        return len(Path(path_text).read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        return 0


def build_message(original: str, merged_fields: Sequence[str], missing_fields: Sequence[str]) -> str:
    parts: List[str] = []
    raw_original = str(original or "").strip()
    if raw_original:
        parts.append(raw_original)
    if merged_fields:
        parts.append(f"hybrid_backfill_merged={','.join(merged_fields)}")
    parts.append(f"missing_after_merge={','.join(missing_fields)}")
    return "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge prior Gemma hybrid successes back into the main raw_json set.")
    parser.add_argument(
        "--partial-csv",
        default=str(REPO_ROOT / ".tmp_gemma_markdown_repair_runner" / "partial_tasks_latest.csv"),
    )
    parser.add_argument(
        "--target-raw-root",
        default=str(REPO_ROOT / ".tmp_gemma_markdown_financials_full" / "raw_json"),
    )
    parser.add_argument(
        "--source-raw-roots",
        nargs="+",
        default=[
            str(REPO_ROOT / ".tmp_gemma_pdf_hybrid_repair_v4_smoke" / "raw_json"),
            str(REPO_ROOT / ".tmp_gemma_pdf_hybrid_repair_repro" / "raw_json"),
            str(REPO_ROOT / ".tmp_gemma_pdf_hybrid_repair_smoke2" / "raw_json"),
            str(REPO_ROOT / ".tmp_gemma_pdf_hybrid_repair_v3" / "raw_json"),
        ],
    )
    parser.add_argument(
        "--runner-dir",
        default=str(REPO_ROOT / ".tmp_gemma_markdown_repair_runner"),
    )
    parser.add_argument("--replace-latest", action="store_true")
    parser.add_argument("--short-pdf-threshold", type=int, default=15)
    parser.add_argument("--manual-pdf-threshold", type=int, default=30)
    args = parser.parse_args()

    partial_csv = Path(args.partial_csv).resolve()
    target_raw_root = Path(args.target_raw_root).resolve()
    runner_dir = Path(args.runner_dir).resolve()
    source_raw_roots = [Path(item).resolve() for item in args.source_raw_roots if str(item).strip()]
    source_raw_roots = [path for path in source_raw_roots if path.exists()]

    if not partial_csv.exists():
        raise FileNotFoundError(f"Missing partial csv: {partial_csv}")

    tag = now_tag()
    rows = read_csv_rows(partial_csv)
    pdf_index = build_pdf_index(REPO_ROOT)

    updated_rows: List[Dict[str, object]] = []
    short_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []
    long_single_field_rows: List[Dict[str, object]] = []
    summary_counter = Counter()
    missing_counter = Counter()

    for row in rows:
        year = int(str(row.get("year") or "0").strip())
        stock_code = normalize_stock_code(row.get("stock_code") or "")
        if not year or not stock_code:
            continue
        report_name = str(row.get("report_name") or "").strip()
        target_path = target_raw_root / str(year) / f"{stock_code}.json"
        target = ensure_target_json(target_path, row)

        merged_fields: List[str] = []
        for source_root in source_raw_roots:
            source_path = source_root / str(year) / f"{stock_code}.json"
            source = read_json(source_path)
            if not isinstance(source, dict):
                continue
            merged_fields.extend(overlay_missing_fields(target, source))

        merged_fields = [field for field in FIELD_ORDER if field in set(merged_fields)]
        if merged_fields:
            write_json(target_path, target)
            summary_counter["docs_with_any_merge"] += 1

        missing_fields = compute_missing_fields(target)
        if not missing_fields:
            summary_counter["docs_fully_fixed"] += 1
            continue

        pdf_path, pdf_pages = resolve_pdf_pages(pdf_index, report_name)
        md_lines = count_md_lines(str(row.get("markdown_path") or ""))
        missing_key = "|".join(sorted(missing_fields))
        missing_counter[missing_key] += 1

        out_row: Dict[str, object] = dict(row)
        out_row["year"] = year
        out_row["stock_code"] = stock_code
        out_row["raw_json_path"] = str(target_path)
        out_row["missing_fields"] = ",".join(missing_fields)
        out_row["message"] = build_message(row.get("message") or "", merged_fields, missing_fields)
        out_row["pdf_path"] = pdf_path
        out_row["pdf_pages"] = pdf_pages if pdf_pages is not None else ""
        out_row["md_lines"] = md_lines
        out_row["missing_count"] = len(missing_fields)
        out_row["single_missing_field"] = missing_fields[0] if len(missing_fields) == 1 else ""
        out_row["short_pdf_flag"] = int(pdf_pages is not None and pdf_pages <= int(args.short_pdf_threshold))
        out_row["manual_pdf_flag"] = int(pdf_pages is not None and pdf_pages > int(args.manual_pdf_threshold))
        updated_rows.append(out_row)
        summary_counter["remaining_after_merge"] += 1

        if pdf_pages is not None and pdf_pages <= int(args.short_pdf_threshold):
            short_rows.append(out_row)
        if pdf_pages is not None and pdf_pages > int(args.manual_pdf_threshold):
            long_rows.append(out_row)
            if len(missing_fields) == 1:
                long_single_field_rows.append(out_row)

    base_fieldnames = list(rows[0].keys()) if rows else [
        "year",
        "stock_code",
        "report_name",
        "folder_name",
        "missing_fields",
        "message",
        "markdown_path",
        "raw_json_path",
    ]
    extra_fieldnames = [
        "pdf_path",
        "pdf_pages",
        "md_lines",
        "missing_count",
        "single_missing_field",
        "short_pdf_flag",
        "manual_pdf_flag",
    ]
    fieldnames = list(dict.fromkeys(base_fieldnames + extra_fieldnames))

    merged_csv = runner_dir / "partial_tasks_after_hybrid_merge.csv"
    short_csv = runner_dir / f"partial_tasks_short_pdf_le{int(args.short_pdf_threshold)}.csv"
    long_csv = runner_dir / f"partial_tasks_manual_pdf_gt{int(args.manual_pdf_threshold)}.csv"
    long_single_csv = runner_dir / f"partial_tasks_manual_pdf_gt{int(args.manual_pdf_threshold)}_single_field.csv"
    summary_json = runner_dir / "partial_summary_after_hybrid_merge.json"
    summary_txt = runner_dir / "partial_summary_after_hybrid_merge.txt"

    write_csv_rows(merged_csv, updated_rows, fieldnames)
    write_csv_rows(short_csv, short_rows, fieldnames)
    write_csv_rows(long_csv, long_rows, fieldnames)
    write_csv_rows(long_single_csv, long_single_field_rows, fieldnames)

    company_present = build_company_present_map(target_raw_root)
    imputable_rows: List[Dict[str, object]] = []
    manual_after_avg_rows: List[Dict[str, object]] = []
    for row in long_rows:
        code = str(row.get("stock_code") or "").strip()
        missing_fields = [item.strip() for item in str(row.get("missing_fields") or "").split(",") if item.strip()]
        no_history = [field for field in missing_fields if not company_present.get(code, {}).get(field)]
        enriched = dict(row)
        enriched["avg_imputable_flag"] = int(len(no_history) == 0)
        enriched["avg_rule_no_history_fields"] = ",".join(no_history)
        if no_history:
            manual_after_avg_rows.append(enriched)
        else:
            imputable_rows.append(enriched)

    imputable_csv = runner_dir / f"partial_tasks_manual_pdf_gt{int(args.manual_pdf_threshold)}_company_avg_imputable.csv"
    manual_after_avg_csv = runner_dir / f"partial_tasks_manual_pdf_gt{int(args.manual_pdf_threshold)}_after_company_avg_rule.csv"
    avg_fieldnames = list(dict.fromkeys(fieldnames + ["avg_imputable_flag", "avg_rule_no_history_fields"]))
    write_csv_rows(imputable_csv, imputable_rows, avg_fieldnames)
    write_csv_rows(manual_after_avg_csv, manual_after_avg_rows, avg_fieldnames)

    summary_payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "source_partial_csv": str(partial_csv),
        "target_raw_root": str(target_raw_root),
        "source_raw_roots": [str(path) for path in source_raw_roots],
        "counts": {
            "input_partial_docs": len(rows),
            "docs_with_any_merge": int(summary_counter["docs_with_any_merge"]),
            "docs_fully_fixed": int(summary_counter["docs_fully_fixed"]),
            "remaining_after_merge": int(summary_counter["remaining_after_merge"]),
            f"short_pdf_le_{int(args.short_pdf_threshold)}": len(short_rows),
            f"manual_pdf_gt_{int(args.manual_pdf_threshold)}": len(long_rows),
            f"manual_pdf_gt_{int(args.manual_pdf_threshold)}_single_field": len(long_single_field_rows),
            f"manual_pdf_gt_{int(args.manual_pdf_threshold)}_company_avg_imputable": len(imputable_rows),
            f"manual_pdf_gt_{int(args.manual_pdf_threshold)}_after_company_avg_rule": len(manual_after_avg_rows),
        },
        "remaining_by_missing_fields": dict(sorted(missing_counter.items(), key=lambda item: (-item[1], item[0]))),
        "paths": {
            "merged_csv": str(merged_csv),
            "short_csv": str(short_csv),
            "long_csv": str(long_csv),
            "long_single_csv": str(long_single_csv),
            "imputable_csv": str(imputable_csv),
            "manual_after_avg_csv": str(manual_after_avg_csv),
        },
    }
    write_json(summary_json, summary_payload)

    summary_lines = [
        "gemma hybrid backfill summary",
        f"ts: {summary_payload['ts']}",
        f"input_partial_docs: {len(rows)}",
        f"docs_with_any_merge: {int(summary_counter['docs_with_any_merge'])}",
        f"docs_fully_fixed: {int(summary_counter['docs_fully_fixed'])}",
        f"remaining_after_merge: {int(summary_counter['remaining_after_merge'])}",
        f"short_pdf_le_{int(args.short_pdf_threshold)}: {len(short_rows)}",
        f"manual_pdf_gt_{int(args.manual_pdf_threshold)}: {len(long_rows)}",
        f"manual_pdf_gt_{int(args.manual_pdf_threshold)}_single_field: {len(long_single_field_rows)}",
        f"manual_pdf_gt_{int(args.manual_pdf_threshold)}_company_avg_imputable: {len(imputable_rows)}",
        f"manual_pdf_gt_{int(args.manual_pdf_threshold)}_after_company_avg_rule: {len(manual_after_avg_rows)}",
        f"merged_csv: {merged_csv}",
        f"long_single_csv: {long_single_csv}",
        f"imputable_csv: {imputable_csv}",
        f"manual_after_avg_csv: {manual_after_avg_csv}",
    ]
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if args.replace_latest:
        latest_backup = runner_dir / f"partial_tasks_latest.backup_{tag}.csv"
        shutil.copy2(partial_csv, latest_backup)
        shutil.copy2(merged_csv, partial_csv)

    print(f"[merged_csv] {merged_csv}")
    print(f"[short_csv] {short_csv}")
    print(f"[long_csv] {long_csv}")
    print(f"[long_single_csv] {long_single_csv}")
    print(json.dumps(summary_payload["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
