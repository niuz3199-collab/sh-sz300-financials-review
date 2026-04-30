#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.monitor_qwen_progress import build_snapshot, format_duration, safe_float  # noqa: E402


def parse_shard(raw: str) -> Tuple[str, str]:
    text = str(raw or "").strip()
    if "=" not in text:
        raise ValueError(f"Invalid --shard value: {raw!r}; expected label=path")
    label, path = text.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --shard value: {raw!r}; expected label=path")
    return label, path


def resolve_path(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def parse_iso(raw: Optional[str]) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def build_aggregate_snapshot(base_dir: Path, shard_specs: List[Tuple[str, Path]], pdf_root: Optional[Path]) -> Dict[str, object]:
    now = datetime.now()
    shard_snapshots: List[Dict[str, object]] = []
    for label, out_dir in shard_specs:
        snapshot = build_snapshot(base_dir=base_dir, out_dir=out_dir, pdf_root=pdf_root)
        shard_snapshots.append({"label": label, "snapshot": snapshot})

    total_tasks = sum(int((item["snapshot"] or {}).get("total_tasks") or 0) for item in shard_snapshots)
    completed_tasks = sum(int((item["snapshot"] or {}).get("completed_tasks") or 0) for item in shard_snapshots)
    remaining_tasks = max(0, total_tasks - completed_tasks)
    percent_complete = round((completed_tasks / total_tasks * 100.0), 2) if total_tasks else 0.0

    counts = {"ok": 0, "partial": 0, "error": 0, "other": 0}
    processed_since_run_start = 0
    tasks_per_minute = 0.0
    last_log_ts: Optional[datetime] = None
    current_run_start: Optional[datetime] = None
    last_task: Optional[Dict[str, object]] = None

    shard_rows: List[Dict[str, object]] = []
    for item in shard_snapshots:
        label = str(item["label"])
        snapshot = item["snapshot"] or {}
        shard_counts = snapshot.get("counts") or {}
        for key in counts:
            counts[key] += int(shard_counts.get(key) or 0)
        processed_since_run_start += int(snapshot.get("processed_since_run_start") or 0)
        tasks_per_minute += float(snapshot.get("tasks_per_minute") or 0.0)

        shard_last_ts = parse_iso(snapshot.get("last_log_ts"))
        if shard_last_ts is not None and (last_log_ts is None or shard_last_ts > last_log_ts):
            last_log_ts = shard_last_ts
            last_task = dict(snapshot.get("last_task") or {})
            if last_task:
                last_task["shard"] = label

        shard_run_start = parse_iso(snapshot.get("current_run_start"))
        if shard_run_start is not None and (current_run_start is None or shard_run_start < current_run_start):
            current_run_start = shard_run_start

        shard_rows.append(
            {
                "label": label,
                "out_dir": str(snapshot.get("out_dir") or ""),
                "backend": ((snapshot.get("run_config") or {}).get("backend")),
                "model": ((snapshot.get("run_config") or {}).get("model")),
                "progress": {
                    "completed": int(snapshot.get("completed_tasks") or 0),
                    "total": int(snapshot.get("total_tasks") or 0),
                    "percent": float(snapshot.get("percent_complete") or 0.0),
                    "remaining": int(snapshot.get("remaining_tasks") or 0),
                },
                "counts": shard_counts,
                "tasks_per_minute": snapshot.get("tasks_per_minute"),
                "last_log_ts": snapshot.get("last_log_ts"),
                "last_task": snapshot.get("last_task"),
                "done": bool(snapshot.get("done")),
            }
        )

    eta_seconds = None
    if tasks_per_minute > 0 and remaining_tasks > 0:
        eta_seconds = remaining_tasks / tasks_per_minute * 60.0

    done = total_tasks > 0 and completed_tasks >= total_tasks

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "base_dir": str(base_dir),
        "pdf_root": str(pdf_root) if pdf_root else "",
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "remaining_tasks": remaining_tasks,
        "percent_complete": percent_complete,
        "counts": counts,
        "current_run_start": current_run_start.isoformat(timespec="seconds") if current_run_start else None,
        "processed_since_run_start": processed_since_run_start,
        "last_log_ts": last_log_ts.isoformat(timespec="seconds") if last_log_ts else None,
        "tasks_per_minute": round(tasks_per_minute, 2) if tasks_per_minute > 0 else None,
        "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
        "eta_wall_clock": (now + timedelta(seconds=eta_seconds)).isoformat(timespec="seconds") if eta_seconds is not None else None,
        "last_task": last_task,
        "shards": shard_rows,
        "done": done,
    }


def render_text(snapshot: Dict[str, object]) -> str:
    counts = snapshot.get("counts") or {}
    last_task = snapshot.get("last_task") or {}
    lines = [
        f"generated_at: {snapshot.get('generated_at')}",
        f"progress: {snapshot.get('completed_tasks')}/{snapshot.get('total_tasks')} ({safe_float(float(snapshot.get('percent_complete') or 0.0))}%)",
        f"remaining: {snapshot.get('remaining_tasks')}",
        f"status_counts: ok={counts.get('ok', 0)} partial={counts.get('partial', 0)} error={counts.get('error', 0)} other={counts.get('other', 0)}",
        f"current_run_start: {snapshot.get('current_run_start')}",
        f"processed_since_run_start: {snapshot.get('processed_since_run_start')}",
        f"last_log_ts: {snapshot.get('last_log_ts')}",
        f"tasks_per_minute: {snapshot.get('tasks_per_minute')}",
        f"eta_wall_clock: {snapshot.get('eta_wall_clock')}",
        f"done: {snapshot.get('done')}",
        "",
        "[shards]",
    ]

    if snapshot.get("eta_seconds") is not None:
        lines.insert(8, f"eta: {format_duration(timedelta(seconds=float(snapshot['eta_seconds'])))}")

    for shard in snapshot.get("shards") or []:
        progress = shard.get("progress") or {}
        shard_counts = shard.get("counts") or {}
        lines.append(
            f"{shard.get('label')}: {progress.get('completed')}/{progress.get('total')} "
            f"({safe_float(float(progress.get('percent') or 0.0))}%) "
            f"backend={shard.get('backend')} model={shard.get('model')} "
            f"ok={shard_counts.get('ok', 0)} partial={shard_counts.get('partial', 0)} error={shard_counts.get('error', 0)} "
            f"tpm={shard.get('tasks_per_minute')} last_log_ts={shard.get('last_log_ts')}"
        )

    lines.extend(
        [
            "",
            f"last_task_shard: {last_task.get('shard')}",
            f"last_task_ts: {last_task.get('ts')}",
            f"last_task_year: {last_task.get('year')}",
            f"last_task_code: {last_task.get('stock_code')}",
            f"last_task_status: {last_task.get('status')}",
            f"last_task_message: {last_task.get('message')}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(snapshot: Dict[str, object], status_path: Path, json_path: Path) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(render_text(snapshot), encoding="utf-8")
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor multiple Qwen extraction shards and write an aggregate snapshot.")
    parser.add_argument("--base-dir", default=".", help="Repository root directory.")
    parser.add_argument("--pdf-root", default="", help="Optional explicit PDF root directory.")
    parser.add_argument("--shard", action="append", default=[], help="Shard spec in label=out_dir format; repeatable.")
    parser.add_argument("--status-file", default=".cache/qwen_pdf_financials_v5_sharded/progress_status.txt", help="Aggregate status text path.")
    parser.add_argument("--json-file", default=".cache/qwen_pdf_financials_v5_sharded/progress_status.json", help="Aggregate status JSON path.")
    parser.add_argument("--watch", action="store_true", help="Refresh snapshots continuously.")
    parser.add_argument("--interval", type=float, default=30.0, help="Watch refresh interval in seconds.")
    parser.add_argument("--quiet", action="store_true", help="Do not print snapshots to stdout.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    pdf_root = resolve_path(base_dir, args.pdf_root) if args.pdf_root else None
    shard_specs = [parse_shard(item) for item in (args.shard or [])]
    if not shard_specs:
        raise RuntimeError("At least one --shard label=out_dir must be provided")
    resolved_shards = [(label, resolve_path(base_dir, raw_path)) for label, raw_path in shard_specs]
    status_path = resolve_path(base_dir, args.status_file)
    json_path = resolve_path(base_dir, args.json_file)

    while True:
        snapshot = build_aggregate_snapshot(base_dir=base_dir, shard_specs=resolved_shards, pdf_root=pdf_root)
        write_outputs(snapshot, status_path=status_path, json_path=json_path)
        if not args.quiet:
            sys.stdout.write(render_text(snapshot))
            sys.stdout.flush()
        if not args.watch or snapshot.get("done"):
            return 0
        time.sleep(max(1.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
