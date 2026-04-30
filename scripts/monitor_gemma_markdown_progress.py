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
from typing import Dict, Optional, Tuple


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_iso_timestamp(text: str) -> Optional[datetime]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
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


def read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        text = pid_file.read_text(encoding="ascii", errors="ignore").strip().splitlines()[0].strip()
        return int(text)
    except Exception:
        return None


def count_total_tasks(markdown_root: Path) -> int:
    total = 0
    if not markdown_root.exists():
        return 0
    for child in markdown_root.iterdir():
        if not child.is_dir():
            continue
        has_md = any(p.is_file() for p in list(child.glob("*.md")) + list(child.glob("*.markdown")))
        if has_md:
            total += 1
    return total


def count_raw_json(raw_root: Path) -> int:
    if not raw_root.exists():
        return 0
    return sum(1 for _ in raw_root.rglob("*.json"))


def parse_log(log_path: Path) -> Dict[str, object]:
    latest_by_task: Dict[Tuple[int, str], Dict[str, str]] = {}
    latest_row: Optional[Dict[str, str]] = None
    latest_ts: Optional[datetime] = None

    if not log_path.exists():
        return {
            "ok": 0,
            "partial": 0,
            "error": 0,
            "processed_total": 0,
            "latest_status": None,
            "latest_task": None,
            "latest_log_at": None,
        }

    try:
        with log_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    year = int(str(row.get("year") or "").strip())
                except Exception:
                    continue
                stock_code = str(row.get("stock_code") or "").strip()
                if not stock_code:
                    continue
                latest_by_task[(year, stock_code)] = dict(row)
                ts = parse_iso_timestamp(str(row.get("ts") or ""))
                if ts is not None and (latest_ts is None or ts >= latest_ts):
                    latest_ts = ts
                    latest_row = dict(row)
    except Exception:
        return {
            "ok": 0,
            "partial": 0,
            "error": 0,
            "processed_total": 0,
            "latest_status": None,
            "latest_task": None,
            "latest_log_at": None,
        }

    ok = 0
    partial = 0
    error = 0
    for row in latest_by_task.values():
        status = str(row.get("status") or "").strip().lower()
        if status == "ok":
            ok += 1
        elif status == "partial":
            partial += 1
        elif status == "error":
            error += 1

    latest_task = None
    latest_status = None
    if latest_row is not None:
        latest_task = f"{latest_row.get('year')} {latest_row.get('stock_code')}"
        latest_status = str(latest_row.get("status") or "").strip()

    return {
        "ok": ok,
        "partial": partial,
        "error": error,
        "processed_total": ok + partial + error,
        "latest_status": latest_status,
        "latest_task": latest_task,
        "latest_log_at": latest_ts.isoformat(timespec="seconds") if latest_ts else None,
    }


def build_snapshot(
    *,
    markdown_root: Path,
    out_dir: Path,
    pid_file: Path,
) -> Dict[str, object]:
    total_tasks = count_total_tasks(markdown_root)
    raw_json_root = out_dir / "raw_json"
    log_path = out_dir / "extract_log.csv"
    pid = read_pid(pid_file)
    alive = process_alive(pid)
    log_stats = parse_log(log_path)
    raw_json_count = count_raw_json(raw_json_root)

    ok = int(log_stats["ok"])
    partial = int(log_stats["partial"])
    error = int(log_stats["error"])
    success_done = ok + partial
    percent_complete = (success_done / total_tasks * 100.0) if total_tasks > 0 else 0.0
    remaining_est = max(0, total_tasks - success_done - error)

    snapshot = {
        "ts": now_iso(),
        "markdown_root": str(markdown_root),
        "out_dir": str(out_dir),
        "pid_file": str(pid_file),
        "pid": pid,
        "run_alive": alive,
        "total_tasks": total_tasks,
        "ok": ok,
        "partial": partial,
        "error": error,
        "success_done": success_done,
        "processed_total": int(log_stats["processed_total"]),
        "remaining_est": remaining_est,
        "percent_complete": round(percent_complete, 2),
        "raw_json_count": raw_json_count,
        "latest_task": log_stats["latest_task"],
        "latest_status": log_stats["latest_status"],
        "latest_log_at": log_stats["latest_log_at"],
        "log_path": str(log_path),
    }
    return snapshot


def write_status_files(snapshot: Dict[str, object], *, status_file: Path, json_file: Path) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "gemma markdown extraction progress",
        f"ts: {snapshot.get('ts')}",
        f"run_alive: {snapshot.get('run_alive')}",
        f"pid: {snapshot.get('pid')}",
        f"progress_success: {snapshot.get('success_done')}/{snapshot.get('total_tasks')} ({snapshot.get('percent_complete')}%)",
        f"processed_total: {snapshot.get('processed_total')}",
        f"ok: {snapshot.get('ok')}",
        f"partial: {snapshot.get('partial')}",
        f"error: {snapshot.get('error')}",
        f"remaining_est: {snapshot.get('remaining_est')}",
        f"raw_json_count: {snapshot.get('raw_json_count')}",
        f"latest_task: {snapshot.get('latest_task') or ''}",
        f"latest_status: {snapshot.get('latest_status') or ''}",
        f"latest_log_at: {snapshot.get('latest_log_at') or ''}",
        f"log_path: {snapshot.get('log_path')}",
        f"out_dir: {snapshot.get('out_dir')}",
    ]
    status_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor progress for Gemma markdown extraction runs")
    parser.add_argument("--markdown-root", required=True, help="Markdown task root directory")
    parser.add_argument("--out-dir", required=True, help="Extraction output directory")
    parser.add_argument("--pid-file", required=True, help="PID file of the extraction runner")
    parser.add_argument("--status-file", required=True, help="Human-readable status output path")
    parser.add_argument("--json-file", required=True, help="JSON status output path")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit")
    args = parser.parse_args()

    markdown_root = Path(args.markdown_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    pid_file = Path(args.pid_file).expanduser().resolve()
    status_file = Path(args.status_file).expanduser().resolve()
    json_file = Path(args.json_file).expanduser().resolve()

    while True:
        snapshot = build_snapshot(markdown_root=markdown_root, out_dir=out_dir, pid_file=pid_file)
        write_status_files(snapshot, status_file=status_file, json_file=json_file)
        if bool(args.once):
            break
        if not bool(snapshot.get("run_alive")):
            break
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    main()
