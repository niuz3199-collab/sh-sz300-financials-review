#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple


def normalize_stock_code(raw: str) -> str:
    text = str(raw or "").strip()
    match = re.search(r"\d+", text)
    if not match:
        return ""
    return match.group(0).zfill(6)


def extract_code_from_filename(pdf_path: Path) -> str:
    match = re.search(r"(\d{6})", pdf_path.name)
    return match.group(1) if match else ""


def autodetect_pdf_root(base_dir: Path) -> Path:
    candidates = []
    for top_dir in sorted([path for path in base_dir.iterdir() if path.is_dir()]):
        for child_dir in sorted([path for path in top_dir.iterdir() if path.is_dir()]):
            if child_dir.name.endswith("_fulltext") and (child_dir / "batch_download_log.csv").exists():
                candidates.append(child_dir)
    if not candidates:
        raise FileNotFoundError(f"Could not locate a *_fulltext directory under {base_dir}")
    return candidates[0]


def collect_task_total(pdf_root: Path, *, start_year: Optional[int] = None, end_year: Optional[int] = None) -> int:
    tasks: Dict[Tuple[int, str], Path] = {}
    for year_dir in sorted([path for path in pdf_root.iterdir() if path.is_dir()], key=lambda path: path.name):
        if not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if start_year is not None and year < int(start_year):
            continue
        if end_year is not None and year > int(end_year):
            continue
        for pdf_path in year_dir.glob("*.pdf"):
            code = normalize_stock_code(extract_code_from_filename(pdf_path))
            if not code:
                continue
            key = (year, code)
            current = tasks.get(key)
            if current is None:
                tasks[key] = pdf_path
                continue
            try:
                if pdf_path.stat().st_size > current.stat().st_size:
                    tasks[key] = pdf_path
            except OSError:
                pass
    return len(tasks)


def parse_iso_timestamp(raw: str) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def load_latest_statuses(log_path: Path) -> Tuple[Dict[Tuple[int, str], Dict[str, str]], Optional[Dict[str, str]], Optional[datetime], Optional[datetime]]:
    latest_by_task: Dict[Tuple[int, str], Dict[str, str]] = {}
    latest_row: Optional[Dict[str, str]] = None
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None

    if not log_path.exists():
        return latest_by_task, latest_row, first_ts, last_ts

    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                year = int(str(row.get("year") or "").strip())
            except ValueError:
                continue
            code = normalize_stock_code(str(row.get("stock_code") or ""))
            if not code:
                continue

            ts = parse_iso_timestamp(str(row.get("ts") or ""))
            if ts is not None and (first_ts is None or ts < first_ts):
                first_ts = ts
            if ts is not None and (last_ts is None or ts >= last_ts):
                last_ts = ts
                latest_row = row

            latest_by_task[(year, code)] = row

    return latest_by_task, latest_row, first_ts, last_ts


def format_duration(delta: timedelta) -> str:
    total_seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def safe_float(value: float) -> str:
    return f"{value:.2f}"


def probe_ollama() -> Dict[str, Optional[str]]:
    result = {
        "model": None,
        "processor": None,
        "context": None,
        "until": None,
        "raw": None,
    }
    try:
        completed = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception:
        return result

    lines = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return result

    first_data_line = lines[1]
    parts = re.split(r"\s{2,}", first_data_line.strip())
    if parts:
        result["model"] = parts[0]
    if len(parts) > 3:
        result["processor"] = parts[3]
    if len(parts) > 4:
        result["context"] = parts[4]
    if len(parts) > 5:
        result["until"] = parts[5]
    result["raw"] = first_data_line.strip()
    return result


def detect_current_run_start(out_dir: Path) -> Optional[datetime]:
    latest_log = None
    candidates = sorted(out_dir.glob("run_full_resume_*.out.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        latest_log = candidates[0]
    if latest_log is None:
        return None

    match = re.search(r"run_full_resume_(\d{8}_\d{6})\.out\.log$", latest_log.name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass

    try:
        return datetime.fromtimestamp(latest_log.stat().st_mtime)
    except OSError:
        return None


def load_run_config(out_dir: Path) -> Dict[str, object]:
    path = out_dir / "run_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_worker_states(out_dir: Path) -> Dict[str, Dict[str, object]]:
    workers_dir = out_dir / "workers"
    if not workers_dir.exists():
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for path in sorted(workers_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        worker_id = str(payload.get("worker_id") or path.stem).strip() or path.stem
        out[worker_id] = payload
    return out


def load_requested_ollama_context(run_config: Dict[str, object]) -> Optional[int]:
    direct_value = run_config.get("ollama_num_ctx")
    try:
        if direct_value is not None and int(direct_value) > 0:
            return int(direct_value)
    except Exception:
        pass

    for worker in run_config.get("workers") or []:
        worker_payload = dict(worker or {})
        if str(worker_payload.get("backend") or "").strip().lower() != "ollama":
            continue
        try:
            value = int(worker_payload.get("ollama_num_ctx") or 0)
            if value > 0:
                return value
        except Exception:
            continue
    return None


def load_issue_source_distribution(
    out_dir: Path,
    run_config: Dict[str, object],
    *,
    since_ts: Optional[datetime] = None,
) -> Dict[str, Dict[str, int]]:
    backend = str(run_config.get("backend") or "").strip().lower()
    if backend != "dynamic_queue":
        return {"partial": {}, "error": {}}

    queue_db_raw = str(run_config.get("queue_db") or "").strip()
    if not queue_db_raw:
        return {"partial": {}, "error": {}}

    queue_db = Path(queue_db_raw)
    if not queue_db.is_absolute():
        queue_db = (out_dir / queue_db).resolve()
    if not queue_db.exists():
        return {"partial": {}, "error": {}}

    worker_map: Dict[str, Dict[str, object]] = {}
    for worker in run_config.get("workers") or []:
        worker_id = str((worker or {}).get("worker_id") or "").strip()
        if worker_id:
            worker_map[worker_id] = dict(worker or {})

    def resolve_label(source_id: str, source_backend: str) -> str:
        source_id = str(source_id or "").strip()
        source_backend = str(source_backend or "").strip()
        worker = worker_map.get(source_id) or {}
        backend_name = str(worker.get("backend") or source_backend or "").strip()
        if source_id and backend_name and source_id != backend_name:
            return f"{source_id} ({backend_name})"
        if source_id:
            return source_id
        if backend_name:
            return backend_name
        return "unknown"

    out: Dict[str, Dict[str, int]] = {"partial": {}, "error": {}}
    for worker_id, worker in worker_map.items():
        label = resolve_label(worker_id, str((worker or {}).get("backend") or ""))
        out["partial"][label] = 0
        out["error"][label] = 0
    where_clauses = [
        "queue_state = 'done'",
        "last_status IN ('partial', 'error')",
    ]
    params: List[object] = []
    if since_ts is not None:
        where_clauses.append("done_ts >= ?")
        params.append(since_ts.isoformat(timespec="seconds"))

    try:
        conn = sqlite3.connect(str(queue_db))
        rows = conn.execute(
            f"""
            SELECT
                last_status,
                COALESCE(NULLIF(claimed_by, ''), NULLIF(last_backend, ''), 'unknown') AS source_id,
                COALESCE(NULLIF(last_backend, ''), '') AS source_backend,
                COUNT(*) AS n
            FROM tasks
            WHERE {' AND '.join(where_clauses)}
            GROUP BY last_status, source_id, source_backend
            ORDER BY last_status, n DESC, source_id
            """
        , params).fetchall()
    except Exception:
        return out
    finally:
        try:
            conn.close()
        except Exception:
            pass

    for last_status, source_id, source_backend, count in rows:
        status = str(last_status or "").strip().lower()
        if status not in out:
            continue
        label = resolve_label(str(source_id or ""), str(source_backend or ""))
        out[status][label] = int(count or 0)
    return out


def load_dynamic_queue_counts(run_config: Dict[str, object], out_dir: Path) -> Optional[Tuple[int, Dict[str, int]]]:
    backend = str(run_config.get("backend") or "").strip().lower()
    if backend != "dynamic_queue":
        return None

    queue_db_raw = str(run_config.get("queue_db") or "").strip()
    if not queue_db_raw:
        return None

    queue_db = Path(queue_db_raw)
    if not queue_db.is_absolute():
        queue_db = (out_dir / queue_db).resolve()
    if not queue_db.exists():
        return None

    counts = {"ok": 0, "partial": 0, "error": 0, "other": 0}
    conn = None
    try:
        conn = sqlite3.connect(str(queue_db))
        total_tasks = int(conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
        for last_status, count in conn.execute(
            """
            SELECT COALESCE(last_status, ''), COUNT(*)
            FROM tasks
            WHERE queue_state = 'done'
            GROUP BY COALESCE(last_status, '')
            """
        ):
            status = str(last_status or "").strip().lower()
            if status in counts:
                counts[status] = int(count or 0)
            else:
                counts["other"] += int(count or 0)
        return total_tasks, counts
    except Exception:
        return None
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def build_snapshot(base_dir: Path, out_dir: Path, pdf_root: Optional[Path]) -> Dict[str, object]:
    resolved_pdf_root = pdf_root or autodetect_pdf_root(base_dir)
    run_config = load_run_config(out_dir)
    start_year = run_config.get("start_year")
    end_year = run_config.get("end_year")
    total_tasks = collect_task_total(
        resolved_pdf_root,
        start_year=int(start_year) if start_year is not None else None,
        end_year=int(end_year) if end_year is not None else None,
    )
    current_run_start = detect_current_run_start(out_dir)
    config_written_at = parse_iso_timestamp(run_config.get("written_at"))
    if config_written_at is not None and (current_run_start is None or config_written_at < current_run_start):
        current_run_start = config_written_at
    worker_states = load_worker_states(out_dir)
    requested_ollama_context = load_requested_ollama_context(run_config)
    issue_source_distribution = load_issue_source_distribution(out_dir, run_config)
    current_run_issue_source_distribution = load_issue_source_distribution(
        out_dir,
        run_config,
        since_ts=current_run_start,
    )

    log_path = out_dir / "extract_log.csv"
    latest_by_task, latest_row, first_ts, last_ts = load_latest_statuses(log_path)
    now = datetime.now()

    counts = {
        "ok": 0,
        "partial": 0,
        "error": 0,
        "other": 0,
    }
    for row in latest_by_task.values():
        status = str(row.get("status") or "").strip().lower()
        if status in counts:
            counts[status] += 1
        else:
            counts["other"] += 1

    queue_counts_snapshot = load_dynamic_queue_counts(run_config, out_dir)
    if queue_counts_snapshot is not None:
        total_tasks, counts = queue_counts_snapshot

    completed = sum(int(counts.get(key, 0)) for key in counts)
    remaining = max(0, total_tasks - completed)
    percent = (completed / total_tasks * 100.0) if total_tasks else 0.0

    elapsed_seconds = None
    tasks_per_minute = None
    eta_seconds = None
    processed_since_run_start = None
    latest_row_since_run_start = None
    latest_row_since_run_start_ts = None
    if current_run_start is not None:
        processed_since_run_start = 0
        for row in latest_by_task.values():
            row_ts = parse_iso_timestamp(str(row.get("ts") or ""))
            if row_ts is not None and row_ts >= current_run_start:
                processed_since_run_start += 1
                if latest_row_since_run_start_ts is None or row_ts >= latest_row_since_run_start_ts:
                    latest_row_since_run_start_ts = row_ts
                    latest_row_since_run_start = row
        reference_ts = last_ts if last_ts is not None and last_ts >= current_run_start else now
        elapsed_seconds = max(1.0, (reference_ts - current_run_start).total_seconds())
        if processed_since_run_start and elapsed_seconds > 0:
            tasks_per_minute = processed_since_run_start / elapsed_seconds * 60.0
            if tasks_per_minute > 0 and remaining > 0:
                eta_seconds = remaining / tasks_per_minute * 60.0
    elif first_ts is not None and last_ts is not None and last_ts > first_ts and completed > 0:
        elapsed_seconds = max(1.0, (last_ts - first_ts).total_seconds())
        tasks_per_minute = completed / elapsed_seconds * 60.0
        if tasks_per_minute > 0 and remaining > 0:
            eta_seconds = remaining / tasks_per_minute * 60.0

    backend = str(run_config.get("backend") or "").strip().lower()
    workers = run_config.get("workers") or []
    should_probe_ollama = backend in {"", "ollama"} or any(
        str((worker or {}).get("backend") or "").strip().lower() == "ollama" for worker in workers
    )
    ollama_info = probe_ollama() if should_probe_ollama else {
        "model": None,
        "processor": None,
        "context": None,
        "until": None,
        "raw": None,
    }

    last_task_row = latest_row_since_run_start if latest_row_since_run_start is not None else latest_row
    last_task = None
    if last_task_row is not None:
        last_task = {
            "ts": str(last_task_row.get("ts") or ""),
            "year": str(last_task_row.get("year") or ""),
            "stock_code": str(last_task_row.get("stock_code") or ""),
            "status": str(last_task_row.get("status") or ""),
            "message": str(last_task_row.get("message") or ""),
        }
    done = total_tasks > 0 and completed >= total_tasks

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "base_dir": str(base_dir),
        "pdf_root": str(resolved_pdf_root),
        "out_dir": str(out_dir),
        "log_path": str(log_path),
        "total_tasks": total_tasks,
        "completed_tasks": completed,
        "remaining_tasks": remaining,
        "percent_complete": round(percent, 2),
        "counts": counts,
        "current_run_start": current_run_start.isoformat(timespec="seconds") if current_run_start else None,
        "processed_since_run_start": processed_since_run_start,
        "first_log_ts": first_ts.isoformat(timespec="seconds") if first_ts else None,
        "last_log_ts": last_ts.isoformat(timespec="seconds") if last_ts else None,
        "elapsed_seconds": round(elapsed_seconds, 2) if elapsed_seconds is not None else None,
        "tasks_per_minute": round(tasks_per_minute, 2) if tasks_per_minute is not None else None,
        "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
        "eta_wall_clock": (now + timedelta(seconds=eta_seconds)).isoformat(timespec="seconds") if eta_seconds is not None else None,
        "last_task": last_task,
        "run_config": run_config,
        "worker_states": worker_states,
        "issue_source_distribution": issue_source_distribution,
        "ollama_requested_context": requested_ollama_context,
        "current_run_issue_source_distribution": current_run_issue_source_distribution,
        "ollama": ollama_info,
        "done": done,
    }


def render_text(snapshot: Dict[str, object]) -> str:
    counts = snapshot["counts"]
    last_task = snapshot.get("last_task") or {}
    ollama_info = snapshot.get("ollama") or {}
    requested_ollama_context = snapshot.get("ollama_requested_context")
    run_config = snapshot.get("run_config") or {}
    worker_states = snapshot.get("worker_states") or {}
    issue_source_distribution = snapshot.get("issue_source_distribution") or {}
    current_run_issue_source_distribution = snapshot.get("current_run_issue_source_distribution") or {}

    lines = [
        f"generated_at: {snapshot['generated_at']}",
        f"pdf_root: {snapshot['pdf_root']}",
        f"out_dir: {snapshot['out_dir']}",
        f"log_path: {snapshot['log_path']}",
        f"backend: {run_config.get('backend')}",
        f"model: {run_config.get('model')}",
        f"input_mode: {run_config.get('input_mode')}",
        f"api_base_url: {run_config.get('api_base_url')}",
        f"api_key_env: {run_config.get('api_key_env')}",
        f"progress: {snapshot['completed_tasks']}/{snapshot['total_tasks']} ({safe_float(float(snapshot['percent_complete']))}%)",
        f"remaining: {snapshot['remaining_tasks']}",
        f"status_counts: ok={counts['ok']} partial={counts['partial']} error={counts['error']} other={counts['other']}",
        f"current_run_start: {snapshot['current_run_start']}",
        f"processed_since_run_start: {snapshot['processed_since_run_start']}",
        f"first_log_ts: {snapshot['first_log_ts']}",
        f"last_log_ts: {snapshot['last_log_ts']}",
        f"tasks_per_minute: {snapshot['tasks_per_minute']}",
        f"eta_wall_clock: {snapshot['eta_wall_clock']}",
        f"done: {snapshot['done']}",
        f"ollama_model: {ollama_info.get('model')}",
        f"ollama_processor: {ollama_info.get('processor')}",
        f"ollama_requested_context: {requested_ollama_context}",
        f"ollama_context: {ollama_info.get('context')}",
        f"ollama_until: {ollama_info.get('until')}",
        f"last_task_ts: {last_task.get('ts')}",
        f"last_task_year: {last_task.get('year')}",
        f"last_task_code: {last_task.get('stock_code')}",
        f"last_task_status: {last_task.get('status')}",
        f"last_task_message: {last_task.get('message')}",
    ]

    if worker_states:
        lines.append("worker_states:")
        for worker_id in sorted(worker_states):
            worker = worker_states.get(worker_id) or {}
            task = worker.get("task") or worker.get("last_task") or {}
            task_desc = ""
            if task:
                task_desc = f" {task.get('year')}/{task.get('stock_code')}"
            lines.append(
                f"  {worker_id}: status={worker.get('status')} backend={worker.get('backend')} "
                f"updated_at={worker.get('updated_at')}{task_desc}"
            )

    partial_sources = issue_source_distribution.get("partial") or {}
    error_sources = issue_source_distribution.get("error") or {}
    lines.append("issue_source_distribution:")
    if partial_sources:
        lines.append("  partial:")
        for source_label, count in sorted(partial_sources.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"    {source_label}: {count}")
    else:
        lines.append("  partial: none")
    if error_sources:
        lines.append("  error:")
        for source_label, count in sorted(error_sources.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"    {source_label}: {count}")
    else:
        lines.append("  error: none")

    current_run_partial_sources = current_run_issue_source_distribution.get("partial") or {}
    current_run_error_sources = current_run_issue_source_distribution.get("error") or {}
    lines.append("current_run_issue_source_distribution:")
    if current_run_partial_sources:
        lines.append("  partial:")
        for source_label, count in sorted(current_run_partial_sources.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"    {source_label}: {count}")
    else:
        lines.append("  partial: none")
    if current_run_error_sources:
        lines.append("  error:")
        for source_label, count in sorted(current_run_error_sources.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append(f"    {source_label}: {count}")
    else:
        lines.append("  error: none")

    if snapshot.get("elapsed_seconds") is not None:
        lines.insert(10, f"elapsed: {format_duration(timedelta(seconds=float(snapshot['elapsed_seconds'])))}")
    if snapshot.get("eta_seconds") is not None:
        lines.insert(12, f"eta: {format_duration(timedelta(seconds=float(snapshot['eta_seconds'])))}")

    return "\n".join(lines) + "\n"


def write_outputs(snapshot: Dict[str, object], status_path: Path, json_path: Path) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_text = render_text(snapshot)
    status_path.write_text(status_text, encoding="utf-8")
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Qwen PDF extraction progress snapshots.")
    parser.add_argument("--base-dir", default=".", help="Repository root directory.")
    parser.add_argument("--pdf-root", default="", help="Override PDF root directory.")
    parser.add_argument("--out-dir", default=".cache/qwen_pdf_financials_v2", help="Extraction output directory.")
    parser.add_argument("--status-file", default=".cache/qwen_pdf_financials_v2/progress_status.txt", help="Human-readable status file.")
    parser.add_argument("--json-file", default=".cache/qwen_pdf_financials_v2/progress_status.json", help="JSON status file.")
    parser.add_argument("--watch", action="store_true", help="Refresh snapshots continuously.")
    parser.add_argument("--interval", type=float, default=30.0, help="Watch refresh interval in seconds.")
    parser.add_argument("--quiet", action="store_true", help="Do not print snapshots to stdout.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = (Path(args.out_dir) if Path(args.out_dir).is_absolute() else (base_dir / args.out_dir)).resolve()
    status_path = (Path(args.status_file) if Path(args.status_file).is_absolute() else (base_dir / args.status_file)).resolve()
    json_path = (Path(args.json_file) if Path(args.json_file).is_absolute() else (base_dir / args.json_file)).resolve()
    pdf_root = None
    if args.pdf_root:
        pdf_root = (Path(args.pdf_root) if Path(args.pdf_root).is_absolute() else (base_dir / args.pdf_root)).resolve()

    while True:
        snapshot = build_snapshot(base_dir=base_dir, out_dir=out_dir, pdf_root=pdf_root)
        write_outputs(snapshot, status_path=status_path, json_path=json_path)
        if not args.quiet:
            sys.stdout.write(render_text(snapshot))
            sys.stdout.flush()
        if not args.watch or snapshot.get("done"):
            return 0
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
