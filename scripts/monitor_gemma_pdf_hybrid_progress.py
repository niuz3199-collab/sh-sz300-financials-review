#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_iso_timestamp(text: object) -> Optional[datetime]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def split_fields(text: object) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        text = pid_file.read_text(encoding="ascii", errors="ignore").strip().splitlines()[0].strip()
        return int(text)
    except Exception:
        return None


def process_alive(pid: Optional[int]) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def build_snapshot(*, out_dir: Path, pid_file: Path) -> Dict[str, object]:
    run_config = read_json(out_dir / "run_config.json")
    runner_state = read_json(out_dir / "runner_state.json")

    manifest_rows = read_csv_rows(out_dir / "task_manifest.csv")
    manifest_task_keys = set()
    total_field_tasks = 0
    per_field_totals: Dict[str, int] = {}
    for row in manifest_rows:
        task_key = str(row.get("task_key") or "").strip()
        if not task_key:
            continue
        manifest_task_keys.add(task_key)
        planned_fields = split_fields(row.get("planned_fields"))
        total_field_tasks += len(planned_fields)
        for field_name in planned_fields:
            per_field_totals[field_name] = int(per_field_totals.get(field_name, 0)) + 1

    summary_rows = read_csv_rows(out_dir / "summary.csv")
    summary_latest: Dict[str, Dict[str, str]] = {}
    for row in summary_rows:
        task_key = str(row.get("task_key") or "").strip()
        if not task_key:
            continue
        if manifest_task_keys and task_key not in manifest_task_keys:
            continue
        summary_latest[task_key] = row

    field_rows = read_csv_rows(out_dir / "field_results.csv")
    field_latest: Dict[str, Dict[str, str]] = {}
    for row in field_rows:
        field_task_key = str(row.get("field_task_key") or "").strip()
        task_key = str(row.get("task_key") or "").strip()
        if not field_task_key or not task_key:
            continue
        if manifest_task_keys and task_key not in manifest_task_keys:
            continue
        field_latest[field_task_key] = row

    total_docs = len(manifest_task_keys) or int(run_config.get("selected_tasks") or 0)
    ok_docs = 0
    partial_docs = 0
    error_docs = 0
    skipped_docs = 0
    latest_doc_row: Optional[Dict[str, str]] = None
    latest_doc_ts: Optional[datetime] = None
    for row in summary_latest.values():
        status = str(row.get("status") or "").strip().lower()
        if status == "ok":
            ok_docs += 1
        elif status == "partial":
            partial_docs += 1
        elif status == "error":
            error_docs += 1
        elif status == "skipped":
            skipped_docs += 1
        row_ts = parse_iso_timestamp(row.get("completed_at"))
        if row_ts is not None and (latest_doc_ts is None or row_ts >= latest_doc_ts):
            latest_doc_ts = row_ts
            latest_doc_row = row

    field_done = len(field_latest)
    field_ok = 0
    field_failed = 0
    field_skipped = 0
    per_field_counts: Dict[str, Dict[str, int]] = {}
    latest_field_row: Optional[Dict[str, str]] = None
    latest_field_ts: Optional[datetime] = None
    for field_name, total in per_field_totals.items():
        per_field_counts[field_name] = {
            "total": int(total),
            "done": 0,
            "ok": 0,
            "failed": 0,
            "skipped": 0,
        }
    for row in field_latest.values():
        field_name = str(row.get("field_name") or "").strip()
        status = str(row.get("status") or "").strip().lower()
        bucket = per_field_counts.setdefault(
            field_name,
            {"total": 0, "done": 0, "ok": 0, "failed": 0, "skipped": 0},
        )
        bucket["done"] += 1
        if status == "ok":
            field_ok += 1
            bucket["ok"] += 1
        elif status == "failed":
            field_failed += 1
            bucket["failed"] += 1
        elif status == "skipped":
            field_skipped += 1
            bucket["skipped"] += 1
        row_ts = parse_iso_timestamp(row.get("completed_at"))
        if row_ts is not None and (latest_field_ts is None or row_ts >= latest_field_ts):
            latest_field_ts = row_ts
            latest_field_row = row

    processed_docs = ok_docs + partial_docs + error_docs + skipped_docs
    remaining_docs = max(0, total_docs - processed_docs)
    remaining_field_tasks = max(0, total_field_tasks - field_done)
    doc_percent = round((processed_docs / total_docs * 100.0), 2) if total_docs > 0 else 0.0
    field_percent = round((field_done / total_field_tasks * 100.0), 2) if total_field_tasks > 0 else 0.0

    pid = read_pid(pid_file)
    if pid is None:
        try:
            pid = int(runner_state.get("pid")) if runner_state.get("pid") is not None else None
        except Exception:
            pid = None
    alive = process_alive(pid)

    started_at = parse_iso_timestamp(run_config.get("started_at"))
    elapsed_hours: Optional[float] = None
    docs_per_hour: Optional[float] = None
    eta_hours: Optional[float] = None
    if started_at is not None:
        elapsed_seconds = max(0.0, (datetime.now() - started_at).total_seconds())
        elapsed_hours = elapsed_seconds / 3600.0
        if elapsed_hours > 0:
            docs_per_hour = processed_docs / elapsed_hours
            if docs_per_hour > 0:
                eta_hours = remaining_docs / docs_per_hour

    latest_task = None
    latest_status = None
    latest_completed_at = None
    if latest_doc_row is not None:
        latest_task = f"{latest_doc_row.get('year')} {latest_doc_row.get('stock_code')}"
        latest_status = str(latest_doc_row.get("status") or "").strip()
        latest_completed_at = latest_doc_row.get("completed_at")

    latest_field_task = None
    latest_field_status = None
    if latest_field_row is not None:
        latest_field_task = (
            f"{latest_field_row.get('year')} {latest_field_row.get('stock_code')} {latest_field_row.get('field_name')}"
        )
        latest_field_status = str(latest_field_row.get("status") or "").strip()

    return {
        "ts": now_iso(),
        "out_dir": str(out_dir),
        "pid_file": str(pid_file),
        "pid": pid,
        "run_alive": alive,
        "started_at": run_config.get("started_at"),
        "last_heartbeat_at": runner_state.get("last_heartbeat_at"),
        "current_task_key": runner_state.get("current_task_key"),
        "current_field": runner_state.get("current_field"),
        "selected_fields": run_config.get("fields") or [],
        "docs": {
            "total": total_docs,
            "processed": processed_docs,
            "remaining": remaining_docs,
            "ok": ok_docs,
            "partial": partial_docs,
            "error": error_docs,
            "skipped": skipped_docs,
            "percent": doc_percent,
        },
        "fields": {
            "total": total_field_tasks,
            "done": field_done,
            "remaining": remaining_field_tasks,
            "ok": field_ok,
            "failed": field_failed,
            "skipped": field_skipped,
            "percent": field_percent,
            "per_field": per_field_counts,
        },
        "throughput": {
            "elapsed_hours": round(elapsed_hours, 4) if elapsed_hours is not None else None,
            "docs_per_hour": round(docs_per_hour, 4) if docs_per_hour is not None else None,
            "eta_hours": round(eta_hours, 4) if eta_hours is not None else None,
        },
        "latest_doc": {
            "task": latest_task,
            "status": latest_status,
            "completed_at": latest_completed_at,
        },
        "latest_field": {
            "task": latest_field_task,
            "status": latest_field_status,
            "completed_at": latest_field_row.get("completed_at") if latest_field_row else None,
        },
        "summary_path": str(out_dir / "summary.csv"),
        "field_results_path": str(out_dir / "field_results.csv"),
        "log_path": str(out_dir / "extract_log.csv"),
        "manifest_path": str(out_dir / "task_manifest.csv"),
    }


def write_status_files(snapshot: Dict[str, object], *, status_file: Path, json_file: Path) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    docs = dict(snapshot.get("docs") or {})
    fields = dict(snapshot.get("fields") or {})
    throughput = dict(snapshot.get("throughput") or {})
    per_field = dict(fields.get("per_field") or {})

    lines = [
        "gemma pdf hybrid repair progress",
        f"ts: {snapshot.get('ts')}",
        f"run_alive: {snapshot.get('run_alive')}",
        f"pid: {snapshot.get('pid')}",
        f"started_at: {snapshot.get('started_at') or ''}",
        f"last_heartbeat_at: {snapshot.get('last_heartbeat_at') or ''}",
        f"current_task_key: {snapshot.get('current_task_key') or ''}",
        f"current_field: {snapshot.get('current_field') or ''}",
        (
            "docs: "
            f"{docs.get('processed', 0)}/{docs.get('total', 0)} "
            f"({docs.get('percent', 0)}%) "
            f"ok={docs.get('ok', 0)} partial={docs.get('partial', 0)} "
            f"error={docs.get('error', 0)} skipped={docs.get('skipped', 0)} "
            f"remaining={docs.get('remaining', 0)}"
        ),
        (
            "fields: "
            f"{fields.get('done', 0)}/{fields.get('total', 0)} "
            f"({fields.get('percent', 0)}%) "
            f"ok={fields.get('ok', 0)} failed={fields.get('failed', 0)} "
            f"skipped={fields.get('skipped', 0)} remaining={fields.get('remaining', 0)}"
        ),
        (
            "throughput: "
            f"docs_per_hour={format_float(throughput.get('docs_per_hour'))} "
            f"elapsed_hours={format_float(throughput.get('elapsed_hours'))} "
            f"eta_hours={format_float(throughput.get('eta_hours'))}"
        ),
        f"latest_doc: {(snapshot.get('latest_doc') or {}).get('task') or ''}",
        f"latest_doc_status: {(snapshot.get('latest_doc') or {}).get('status') or ''}",
        f"latest_doc_completed_at: {(snapshot.get('latest_doc') or {}).get('completed_at') or ''}",
        f"latest_field: {(snapshot.get('latest_field') or {}).get('task') or ''}",
        f"latest_field_status: {(snapshot.get('latest_field') or {}).get('status') or ''}",
        f"latest_field_completed_at: {(snapshot.get('latest_field') or {}).get('completed_at') or ''}",
    ]
    for field_name in sorted(per_field):
        bucket = per_field[field_name]
        lines.append(
            f"field.{field_name}: total={bucket.get('total', 0)} done={bucket.get('done', 0)} "
            f"ok={bucket.get('ok', 0)} failed={bucket.get('failed', 0)} skipped={bucket.get('skipped', 0)}"
        )
    lines.extend(
        [
            f"summary_path: {snapshot.get('summary_path')}",
            f"field_results_path: {snapshot.get('field_results_path')}",
            f"log_path: {snapshot.get('log_path')}",
            f"manifest_path: {snapshot.get('manifest_path')}",
            f"out_dir: {snapshot.get('out_dir')}",
        ]
    )

    status_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor progress for Gemma PDF hybrid repair runs")
    parser.add_argument("--out-dir", required=True, help="Runner output directory")
    parser.add_argument("--pid-file", default="", help="PID file written by the runner")
    parser.add_argument("--status-file", default="", help="Human-readable status output path")
    parser.add_argument("--json-file", default="", help="JSON status output path")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    pid_file = (
        Path(args.pid_file).expanduser().resolve()
        if str(args.pid_file or "").strip()
        else (out_dir / "run.pid.txt").resolve()
    )
    status_file = (
        Path(args.status_file).expanduser().resolve()
        if str(args.status_file or "").strip()
        else (out_dir / "progress_status.txt").resolve()
    )
    json_file = (
        Path(args.json_file).expanduser().resolve()
        if str(args.json_file or "").strip()
        else (out_dir / "progress_status.json").resolve()
    )

    while True:
        snapshot = build_snapshot(out_dir=out_dir, pid_file=pid_file)
        write_status_files(snapshot, status_file=status_file, json_file=json_file)
        if bool(args.once):
            break
        if not bool(snapshot.get("run_alive")):
            break
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    main()
