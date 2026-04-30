#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-repair partial Markdown extraction results with the hybrid Markdown + PDF-vision path.

Current supported fields
- parent_netprofit
- total_shares
- operating_cashflow
- capex
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.monitor_gemma_pdf_hybrid_progress import build_snapshot, write_status_files  # noqa: E402
from scripts.repair_gemma_markdown_financials import (  # noqa: E402
    FIELD_META,
    LOG_FIELDS,
    YEAR_FIELDS,
    append_csv_row,
    build_year_csv_row,
    compute_missing_fields,
    ensure_extracted_skeleton,
    read_json_file,
    upsert_csv_row,
)
from scripts.step6_extract_financials_from_markdown import MarkdownTask, now_iso  # noqa: E402
from scripts.step6_extract_financials_qwen_pdf import infer_code_name, normalize_stock_code  # noqa: E402
from scripts import smoke_gemma_pdf_hybrid_fields as hybrid_smoke  # noqa: E402


SUMMARY_FIELDS = [
    "task_key",
    "completed_at",
    "year",
    "stock_code",
    "report_name",
    "status",
    "planned_fields",
    "rerun_fields",
    "recovered_fields",
    "failed_fields",
    "skipped_fields",
    "missing_fields_after",
    "missing_supported_after",
    "parent_netprofit",
    "share_capital",
    "netcash_operate",
    "construct_long_asset",
    "parent_netprofit_page",
    "share_capital_page",
    "netcash_operate_page",
    "construct_long_asset_page",
    "elapsed_sec",
    "source_raw_json_path",
    "out_raw_json_path",
    "pdf_path",
    "markdown_path",
    "message",
]

FIELD_RESULT_FIELDS = [
    "field_task_key",
    "task_key",
    "completed_at",
    "year",
    "stock_code",
    "report_name",
    "field_name",
    "status",
    "ok_match",
    "page_number",
    "value",
    "unit",
    "normalized_value",
    "source_value",
    "source_unit",
    "salvage_value",
    "elapsed_sec",
    "error",
    "task_dir",
]

MANIFEST_FIELDS = [
    "task_key",
    "year",
    "stock_code",
    "report_name",
    "planned_fields",
    "source_missing_fields",
    "source_raw_json_path",
    "markdown_path",
    "pdf_path",
]


@dataclass(frozen=True)
class RepairTask:
    year: int
    stock_code: str
    report_name: str
    md_path: Path
    pdf_path: Path
    source_raw_json_path: Optional[Path]
    source_missing_fields: Tuple[str, ...]


def split_fields(text: object) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def join_fields(values: Sequence[str]) -> str:
    return ",".join([str(x).strip() for x in values if str(x).strip()])


def resolve_path(base_dir: Path, raw: str) -> Path:
    path = Path(str(raw or "").strip())
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def task_key(year: int, stock_code: str) -> str:
    return f"{int(year)}:{normalize_stock_code(stock_code)}"


def field_task_key(year: int, stock_code: str, field_name: str) -> str:
    return f"{task_key(year, stock_code)}:{str(field_name).strip()}"


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_existing_pdf_path(
    *,
    base_dir: Path,
    configured_fulltext_root: Path,
    year: int,
    report_name: str,
    stock_code: str,
) -> Path:
    filename = f"{report_name}.pdf"
    candidate_roots: List[Path] = []
    for root in [
        configured_fulltext_root,
        base_dir / "年报" / "下载年报_fulltext",
    ]:
        try:
            resolved_root = root.resolve()
        except Exception:
            resolved_root = root
        if resolved_root not in candidate_roots:
            candidate_roots.append(resolved_root)

    for root in candidate_roots:
        path = root / str(int(year)) / filename
        if path.exists():
            return path.resolve()

    code = normalize_stock_code(stock_code)
    for root in candidate_roots:
        year_dir = root / str(int(year))
        if not year_dir.exists():
            continue
        pattern = f"{code}_{int(year)}*.pdf"
        matches = sorted(year_dir.glob(pattern))
        if matches:
            return matches[0].resolve()

    return (configured_fulltext_root / str(int(year)) / filename).resolve()


def read_partial_tasks(
    partial_csv: Path,
    *,
    fulltext_root: Path,
    base_dir: Path,
    start_year: int,
    end_year: int,
    requested_fields: Sequence[str],
    force_rerun: bool,
) -> List[Tuple[RepairTask, List[str]]]:
    if not partial_csv.exists():
        raise FileNotFoundError(f"Partial CSV not found: {partial_csv}")

    requested_set = {str(field).strip() for field in requested_fields if str(field).strip()}
    dedup: Dict[Tuple[int, str], Tuple[RepairTask, List[str]]] = {}
    with partial_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                year = int(str(row.get("year") or "").strip())
            except Exception:
                continue
            if int(year) < int(start_year) or int(year) > int(end_year):
                continue
            stock_code = normalize_stock_code(row.get("stock_code") or "")
            if not stock_code:
                continue
            md_text = str(row.get("markdown_path") or "").strip()
            if not md_text:
                continue
            md_path = resolve_path(base_dir, md_text)
            report_name = str(row.get("report_name") or "").strip() or md_path.stem
            pdf_path = resolve_existing_pdf_path(
                base_dir=base_dir,
                configured_fulltext_root=fulltext_root,
                year=int(year),
                report_name=report_name,
                stock_code=stock_code,
            )
            raw_text = str(row.get("raw_json_path") or "").strip()
            source_raw_json_path = resolve_path(base_dir, raw_text) if raw_text else None
            source_missing_fields = split_fields(row.get("missing_fields"))
            if force_rerun:
                planned_fields = [field for field in requested_fields if str(field).strip()]
            else:
                planned_fields = [field for field in source_missing_fields if field in requested_set]
            if not planned_fields:
                continue
            task = RepairTask(
                year=int(year),
                stock_code=stock_code,
                report_name=report_name,
                md_path=md_path,
                pdf_path=pdf_path,
                source_raw_json_path=source_raw_json_path,
                source_missing_fields=tuple(source_missing_fields),
            )
            dedup[(task.year, task.stock_code)] = (task, planned_fields)

    selected = list(dedup.values())
    selected.sort(key=lambda item: (item[0].year, item[0].stock_code))
    return selected


def build_markdown_task(task: RepairTask) -> MarkdownTask:
    return MarkdownTask(
        year=int(task.year),
        stock_code=str(task.stock_code),
        report_name=str(task.report_name),
        md_path=task.md_path,
        pdf_path=task.pdf_path,
    )


def load_or_init_extracted(task: RepairTask, *, seed_path: Path) -> Dict[str, object]:
    extracted = read_json_file(seed_path) if seed_path.exists() else None
    md_task = build_markdown_task(task)
    if not isinstance(extracted, dict):
        extracted = ensure_extracted_skeleton(md_task)

    skeleton = ensure_extracted_skeleton(md_task)
    for section_name in ("task", "raw", "normalized", "pages"):
        current = extracted.setdefault(section_name, {})
        default = skeleton.get(section_name) or {}
        if not isinstance(current, dict):
            current = {}
            extracted[section_name] = current
        for key, value in default.items():
            current.setdefault(key, value)

    task_meta = extracted.setdefault("task", {})
    task_meta.update(
        {
            "stock_code": task.stock_code,
            "year": task.year,
            "report_name": task.report_name,
            "pdf_path": str(task.pdf_path),
            "markdown_path": str(task.md_path),
            "code_name": infer_code_name(task.stock_code),
        }
    )

    raw = extracted.setdefault("raw", {})
    raw["code"] = task.stock_code
    raw["year"] = task.year
    raw["source_markdown_path"] = str(task.md_path)
    return extracted


def apply_hybrid_field_result(extracted: Dict[str, object], *, field_name: str, result: Dict[str, object]) -> None:
    normalized = dict(result.get("normalized") or {})
    field_payload = dict(normalized.get("field") or {})
    chosen_page = dict(result.get("chosen_page") or {})
    page_number = chosen_page.get("page_number")

    raw = extracted.setdefault("raw", {})
    raw[field_name] = {
        "value": field_payload.get("value"),
        "unit": field_payload.get("unit"),
        "evidence": field_payload.get("evidence"),
        "snippet_ids": list(field_payload.get("snippet_ids") or []),
        "page": page_number,
        "source": "gemma_pdf_hybrid",
    }

    pages = extracted.setdefault("pages", {})
    page_key = str(FIELD_META[field_name]["page_key"])
    pages[page_key] = page_number

    normalized_section = extracted.setdefault("normalized", {})
    normalized_key = str(FIELD_META[field_name]["normalized_key"])
    normalized_section[normalized_key] = normalized.get("normalized_value")


def upsert_csv_row_exact(path: Path, row: Dict[str, object], *, fieldnames: Sequence[str], key_field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    row_key = str(row.get(key_field) or "").strip()
    replaced = False

    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for old in reader:
                old_key = str(old.get(key_field) or "").strip()
                if old_key == row_key:
                    if not replaced:
                        rows.append({name: row.get(name, "") for name in fieldnames})
                        replaced = True
                    continue
                rows.append({name: old.get(name, "") for name in fieldnames})

    if not replaced:
        rows.append({name: row.get(name, "") for name in fieldnames})

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for item in rows:
            writer.writerow({name: item.get(name, "") for name in fieldnames})


def write_manifest(path: Path, selected: Sequence[Tuple[RepairTask, List[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for task, planned_fields in selected:
            writer.writerow(
                {
                    "task_key": task_key(task.year, task.stock_code),
                    "year": task.year,
                    "stock_code": task.stock_code,
                    "report_name": task.report_name,
                    "planned_fields": join_fields(planned_fields),
                    "source_missing_fields": join_fields(task.source_missing_fields),
                    "source_raw_json_path": str(task.source_raw_json_path) if task.source_raw_json_path else "",
                    "markdown_path": str(task.md_path),
                    "pdf_path": str(task.pdf_path),
                }
            )


def refresh_progress(out_dir: Path, *, pid_file: Path, status_file: Path, json_file: Path) -> None:
    try:
        snapshot = build_snapshot(out_dir=out_dir, pid_file=pid_file)
        write_status_files(snapshot, status_file=status_file, json_file=json_file)
    except Exception:
        pass


def build_field_result_row(
    task: RepairTask,
    *,
    field_name: str,
    status: str,
    result: Optional[Dict[str, object]],
    error_text: str,
) -> Dict[str, object]:
    normalized = dict((result or {}).get("normalized") or {})
    field_payload = dict(normalized.get("field") or {})
    chosen_page = dict((result or {}).get("chosen_page") or {})
    current_raw_field = dict((result or {}).get("current_raw_field") or {})
    salvage = dict((result or {}).get("page_text_salvage") or {})
    task_dir = REPO_ROOT
    if result:
        crop_path = str((result.get("images") or {}).get("crop_path") or "")
        if crop_path:
            task_dir = Path(crop_path).resolve().parent
    return {
        "field_task_key": field_task_key(task.year, task.stock_code, field_name),
        "task_key": task_key(task.year, task.stock_code),
        "completed_at": now_iso(),
        "year": task.year,
        "stock_code": task.stock_code,
        "report_name": task.report_name,
        "field_name": field_name,
        "status": status,
        "ok_match": status == "ok",
        "page_number": chosen_page.get("page_number"),
        "value": field_payload.get("value"),
        "unit": field_payload.get("unit"),
        "normalized_value": normalized.get("normalized_value"),
        "source_value": current_raw_field.get("value"),
        "source_unit": current_raw_field.get("unit"),
        "salvage_value": salvage.get("current_value"),
        "elapsed_sec": (result or {}).get("elapsed_sec"),
        "error": error_text,
        "task_dir": str(task_dir),
    }


def build_summary_row(
    task: RepairTask,
    *,
    extracted: Dict[str, object],
    status: str,
    planned_fields: Sequence[str],
    rerun_fields: Sequence[str],
    recovered_fields: Sequence[str],
    failed_fields: Sequence[str],
    skipped_fields: Sequence[str],
    message: str,
    elapsed_sec: float,
    out_raw_json_path: Path,
) -> Dict[str, object]:
    normalized = dict(extracted.get("normalized") or {})
    raw = dict(extracted.get("raw") or {})
    missing_all = compute_missing_fields(extracted)
    missing_supported = [field for field in planned_fields if field in set(missing_all)]
    return {
        "task_key": task_key(task.year, task.stock_code),
        "completed_at": now_iso(),
        "year": task.year,
        "stock_code": task.stock_code,
        "report_name": task.report_name,
        "status": status,
        "planned_fields": join_fields(planned_fields),
        "rerun_fields": join_fields(rerun_fields),
        "recovered_fields": join_fields(recovered_fields),
        "failed_fields": join_fields(failed_fields),
        "skipped_fields": join_fields(skipped_fields),
        "missing_fields_after": join_fields(missing_all),
        "missing_supported_after": join_fields(missing_supported),
        "parent_netprofit": normalized.get("parent_netprofit_yuan"),
        "share_capital": normalized.get("total_shares_shares"),
        "netcash_operate": normalized.get("operating_cashflow_yuan"),
        "construct_long_asset": normalized.get("capex_yuan"),
        "parent_netprofit_page": (raw.get("parent_netprofit") or {}).get("page"),
        "share_capital_page": (raw.get("total_shares") or {}).get("page"),
        "netcash_operate_page": (raw.get("operating_cashflow") or {}).get("page"),
        "construct_long_asset_page": (raw.get("capex") or {}).get("page"),
        "elapsed_sec": round(float(elapsed_sec), 2),
        "source_raw_json_path": str(task.source_raw_json_path) if task.source_raw_json_path else "",
        "out_raw_json_path": str(out_raw_json_path),
        "pdf_path": str(task.pdf_path),
        "markdown_path": str(task.md_path),
        "message": message,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch repair partial results with the Gemma PDF hybrid pipeline")
    parser.add_argument("--base-dir", default=".", help="项目根目录")
    parser.add_argument(
        "--partial-csv",
        default=".tmp_gemma_markdown_repair_runner/partial_tasks_latest.csv",
        help="待修复 partial 列表 CSV",
    )
    parser.add_argument("--fulltext-root", default="年报/下载年报_fulltext", help="恢复后的 PDF 根目录")
    parser.add_argument("--out-dir", default=".tmp_gemma_pdf_hybrid_repair", help="输出目录")
    parser.add_argument("--year-csv-root", default=".tmp_gemma_year_csvs_pdf_hybrid", help="年度 CSV 根目录")
    parser.add_argument("--csv-name", default="{year}_财报数据_gemma_pdf_hybrid.csv", help="年度 CSV 文件名模板")
    parser.add_argument("--model", default="google/gemma-4-26b-a4b", help="LM Studio 模型名")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:1234/v1", help="OpenAI 兼容 API base URL")
    parser.add_argument("--api-key-env", default="LM_STUDIO_API_KEY", help="API key 环境变量")
    parser.add_argument("--timeout", type=int, default=240, help="单字段请求超时")
    parser.add_argument("--fields", nargs="+", default=["parent_netprofit", "total_shares", "operating_cashflow", "capex"])
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-snippets", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--dpi-crop", type=int, default=220)
    parser.add_argument("--dpi-page", type=int, default=150)
    parser.add_argument("--pid-file", default="", help="PID 文件路径")
    parser.add_argument("--status-file", default="", help="进度文本文件路径")
    parser.add_argument("--json-file", default="", help="进度 JSON 文件路径")
    parser.add_argument("--force-rerun", action="store_true", help="忽略旧缺失列表，强制重跑所选字段")
    args = parser.parse_args()

    requested_fields = [str(field).strip() for field in args.fields if str(field).strip()]
    unsupported = [field for field in requested_fields if field not in hybrid_smoke.FIELD_CONFIGS]
    if unsupported:
        raise RuntimeError(f"Unsupported fields for hybrid runner: {unsupported}")

    base_dir = Path(args.base_dir).expanduser().resolve()
    partial_csv = resolve_path(base_dir, str(args.partial_csv))
    fulltext_root = resolve_path(base_dir, str(args.fulltext_root))
    out_dir = resolve_path(base_dir, str(args.out_dir))
    year_csv_root = resolve_path(base_dir, str(args.year_csv_root))
    pid_file = (
        resolve_path(base_dir, str(args.pid_file))
        if str(args.pid_file or "").strip()
        else (out_dir / "run.pid.txt").resolve()
    )
    status_file = (
        resolve_path(base_dir, str(args.status_file))
        if str(args.status_file or "").strip()
        else (out_dir / "progress_status.txt").resolve()
    )
    json_file = (
        resolve_path(base_dir, str(args.json_file))
        if str(args.json_file or "").strip()
        else (out_dir / "progress_status.json").resolve()
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    year_csv_root.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="ascii")

    selected = read_partial_tasks(
        partial_csv,
        fulltext_root=fulltext_root,
        base_dir=base_dir,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        requested_fields=requested_fields,
        force_rerun=bool(args.force_rerun),
    )
    start = max(0, int(args.start))
    end = len(selected) if int(args.limit) <= 0 else min(len(selected), start + int(args.limit))
    selected = selected[start:end]
    if not selected:
        raise RuntimeError("No tasks matched the requested slice/fields.")

    total_field_tasks = sum(len(fields) for _, fields in selected)
    write_manifest(out_dir / "task_manifest.csv", selected)
    run_config = {
        "started_at": now_iso(),
        "base_dir": str(base_dir),
        "partial_csv": str(partial_csv),
        "fulltext_root": str(fulltext_root),
        "out_dir": str(out_dir),
        "year_csv_root": str(year_csv_root),
        "csv_name": str(args.csv_name),
        "fields": requested_fields,
        "selected_tasks": len(selected),
        "total_field_tasks": total_field_tasks,
        "start": int(args.start),
        "limit": int(args.limit),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "force_rerun": bool(args.force_rerun),
        "model": str(args.model),
        "api_base_url": str(args.api_base_url),
        "timeout": int(args.timeout),
        "max_snippets": int(args.max_snippets),
        "max_chars": int(args.max_chars),
        "dpi_crop": int(args.dpi_crop),
        "dpi_page": int(args.dpi_page),
        "pid_file": str(pid_file),
        "status_file": str(status_file),
        "json_file": str(json_file),
    }
    write_json(out_dir / "run_config.json", run_config)

    runner_state: Dict[str, object] = {
        "pid": os.getpid(),
        "started_at": run_config["started_at"],
        "last_heartbeat_at": now_iso(),
        "current_task_key": None,
        "current_field": None,
        "docs_total": len(selected),
        "docs_completed": 0,
        "fields_total": total_field_tasks,
        "fields_completed": 0,
    }
    write_json(out_dir / "runner_state.json", runner_state)
    refresh_progress(out_dir, pid_file=pid_file, status_file=status_file, json_file=json_file)

    summary_path = out_dir / "summary.csv"
    field_results_path = out_dir / "field_results.csv"
    log_path = out_dir / "extract_log.csv"
    raw_root = out_dir / "raw_json"

    docs_ok = 0
    docs_partial = 0
    docs_error = 0
    fields_completed = 0

    print(f"[hybrid-repair] selected={len(selected)} total_field_tasks={total_field_tasks}", flush=True)
    print(f"[partial_csv] {partial_csv}", flush=True)
    print(f"[fulltext_root] {fulltext_root}", flush=True)
    print(f"[out_dir] {out_dir}", flush=True)
    print(f"[year_csv_root] {year_csv_root}", flush=True)
    print(
        f"[backend] api_base_url={args.api_base_url} model={args.model} timeout={int(args.timeout)}",
        flush=True,
    )
    print(
        f"[render] dpi_crop={int(args.dpi_crop)} dpi_page={int(args.dpi_page)} "
        f"max_snippets={int(args.max_snippets)} max_chars={int(args.max_chars)}",
        flush=True,
    )

    for doc_index, (task, planned_fields) in enumerate(selected, start=1):
        started_at = time.time()
        out_raw_json_path = raw_root / str(task.year) / f"{task.stock_code}.json"
        seed_path = out_raw_json_path if out_raw_json_path.exists() else (task.source_raw_json_path or out_raw_json_path)
        extracted: Optional[Dict[str, object]] = None
        rerun_fields: List[str] = []
        recovered_fields: List[str] = []
        failed_fields: List[str] = []
        skipped_fields: List[str] = []
        doc_status = "error"
        doc_error_text = ""

        runner_state.update(
            {
                "last_heartbeat_at": now_iso(),
                "current_task_key": task_key(task.year, task.stock_code),
                "current_field": None,
                "docs_completed": doc_index - 1,
                "fields_completed": fields_completed,
            }
        )
        write_json(out_dir / "runner_state.json", runner_state)
        refresh_progress(out_dir, pid_file=pid_file, status_file=status_file, json_file=json_file)

        print(
            f"[doc {doc_index}/{len(selected)}] {task.year} {task.stock_code} "
            f"planned={join_fields(planned_fields)}",
            flush=True,
        )

        try:
            extracted = load_or_init_extracted(task, seed_path=seed_path)
            missing_before = set(compute_missing_fields(extracted))
            if bool(args.force_rerun):
                rerun_fields = list(planned_fields)
            else:
                rerun_fields = [field for field in planned_fields if field in missing_before]
            skipped_fields = [field for field in planned_fields if field not in rerun_fields]

            for field_name in skipped_fields:
                field_row = build_field_result_row(
                    task,
                    field_name=field_name,
                    status="skipped",
                    result=None,
                    error_text="already_present_in_current_output",
                )
                upsert_csv_row_exact(field_results_path, field_row, fieldnames=FIELD_RESULT_FIELDS, key_field="field_task_key")
                fields_completed += 1

            seed_raw_for_run = out_raw_json_path if out_raw_json_path.exists() else (task.source_raw_json_path or out_raw_json_path)
            doc_meta = hybrid_smoke.SmokeDoc(
                year=int(task.year),
                stock_code=str(task.stock_code),
                report_name=str(task.report_name),
                md_path=task.md_path,
                raw_json_path=seed_raw_for_run,
                pdf_path=task.pdf_path,
            )

            for field_name in rerun_fields:
                runner_state.update(
                    {
                        "last_heartbeat_at": now_iso(),
                        "current_task_key": task_key(task.year, task.stock_code),
                        "current_field": field_name,
                        "docs_completed": doc_index - 1,
                        "fields_completed": fields_completed,
                    }
                )
                write_json(out_dir / "runner_state.json", runner_state)
                refresh_progress(out_dir, pid_file=pid_file, status_file=status_file, json_file=json_file)

                result: Optional[Dict[str, object]] = None
                error_text = ""
                try:
                    result = hybrid_smoke.run_one_field(
                        doc_meta,
                        field_name=field_name,
                        out_dir=out_dir,
                        model=str(args.model),
                        api_base_url=str(args.api_base_url),
                        api_key=str(os.environ.get(str(args.api_key_env), "") or ""),
                        timeout=int(args.timeout),
                        dpi_crop=int(args.dpi_crop),
                        dpi_page=int(args.dpi_page),
                        max_snippets=int(args.max_snippets),
                        max_chars=int(args.max_chars),
                    )
                    normalized = dict(result.get("normalized") or {})
                    current_raw_field = dict(result.get("current_raw_field") or {})
                    page_text_salvage = dict(result.get("page_text_salvage") or {})
                    normalized_value = normalized.get("normalized_value")
                    ok_match = hybrid_smoke.evaluate_result(
                        field_name=field_name,
                        current_raw_field=current_raw_field,
                        page_text_salvage=page_text_salvage,
                        normalized_value=normalized_value,
                    )
                    if ok_match:
                        apply_hybrid_field_result(extracted, field_name=field_name, result=result)
                        write_json(out_raw_json_path, extracted)
                        recovered_fields.append(field_name)
                        field_status = "ok"
                    else:
                        field_status = "failed"
                        failed_fields.append(field_name)
                        error_text = str(result.get("error") or "normalized_value_missing_or_mismatch")
                except Exception as exc:
                    field_status = "failed"
                    error_text = str(exc)
                    failed_fields.append(field_name)

                field_row = build_field_result_row(
                    task,
                    field_name=field_name,
                    status=field_status,
                    result=result,
                    error_text=error_text,
                )
                upsert_csv_row_exact(field_results_path, field_row, fieldnames=FIELD_RESULT_FIELDS, key_field="field_task_key")
                fields_completed += 1
                runner_state.update({"last_heartbeat_at": now_iso(), "fields_completed": fields_completed})
                write_json(out_dir / "runner_state.json", runner_state)
                refresh_progress(out_dir, pid_file=pid_file, status_file=status_file, json_file=json_file)

            if extracted is None:
                raise RuntimeError("failed_to_initialize_extracted")
            write_json(out_raw_json_path, extracted)

            missing_after = compute_missing_fields(extracted)
            if not missing_after:
                doc_status = "ok"
                docs_ok += 1
            else:
                doc_status = "partial"
                docs_partial += 1
        except Exception as exc:
            doc_status = "error"
            docs_error += 1
            doc_error_text = str(exc)
            if extracted is None:
                extracted = load_or_init_extracted(task, seed_path=seed_path)
            write_json(out_raw_json_path, extracted)

        elapsed_sec = time.time() - started_at
        message_parts = [
            f"planned_fields={join_fields(planned_fields)}",
            f"rerun_fields={join_fields(rerun_fields)}",
            f"recovered_fields={join_fields(recovered_fields)}",
            f"failed_fields={join_fields(failed_fields)}",
            f"skipped_fields={join_fields(skipped_fields)}",
        ]
        if extracted is not None:
            message_parts.append(f"missing_after={join_fields(compute_missing_fields(extracted))}")
        if doc_error_text:
            message_parts.append(f"error={doc_error_text}")
        message = "; ".join(message_parts)

        final_extracted = extracted or load_or_init_extracted(task, seed_path=seed_path)
        summary_row = build_summary_row(
            task,
            extracted=final_extracted,
            status=doc_status,
            planned_fields=planned_fields,
            rerun_fields=rerun_fields,
            recovered_fields=recovered_fields,
            failed_fields=failed_fields,
            skipped_fields=skipped_fields,
            message=message,
            elapsed_sec=elapsed_sec,
            out_raw_json_path=out_raw_json_path,
        )
        upsert_csv_row_exact(summary_path, summary_row, fieldnames=SUMMARY_FIELDS, key_field="task_key")

        md_task = build_markdown_task(task)
        year_csv_row = build_year_csv_row(md_task, final_extracted)
        year_csv_name = str(args.csv_name).format(year=task.year)
        year_csv_path = year_csv_root / str(task.year) / year_csv_name
        upsert_csv_row(year_csv_path, year_csv_row, fieldnames=YEAR_FIELDS, key_field="stock_code")

        log_row = {
            "ts": now_iso(),
            "year": task.year,
            "stock_code": task.stock_code,
            "code_name": infer_code_name(task.stock_code),
            "pdf_path": str(task.pdf_path),
            "status": doc_status,
            "message": message,
            "raw_json_path": str(out_raw_json_path),
        }
        append_csv_row(log_path, log_row, fieldnames=LOG_FIELDS)

        runner_state.update(
            {
                "last_heartbeat_at": now_iso(),
                "current_task_key": task_key(task.year, task.stock_code),
                "current_field": None,
                "docs_completed": doc_index,
                "fields_completed": fields_completed,
                "latest_status": doc_status,
            }
        )
        write_json(out_dir / "runner_state.json", runner_state)
        refresh_progress(out_dir, pid_file=pid_file, status_file=status_file, json_file=json_file)

        print(
            f"[done] {task.year} {task.stock_code} status={doc_status} "
            f"recovered={join_fields(recovered_fields) or '-'} "
            f"failed={join_fields(failed_fields) or '-'} "
            f"missing_after={join_fields(compute_missing_fields(final_extracted)) or '-'}",
            flush=True,
        )

    runner_state.update(
        {
            "last_heartbeat_at": now_iso(),
            "current_task_key": None,
            "current_field": None,
            "docs_completed": len(selected),
            "fields_completed": fields_completed,
        }
    )
    write_json(out_dir / "runner_state.json", runner_state)
    refresh_progress(out_dir, pid_file=pid_file, status_file=status_file, json_file=json_file)

    print(
        f"[finished] docs_ok={docs_ok} docs_partial={docs_partial} docs_error={docs_error} "
        f"fields_completed={fields_completed}/{total_field_tasks}",
        flush=True,
    )
    print(f"[summary] {summary_path}", flush=True)
    print(f"[progress] {status_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
