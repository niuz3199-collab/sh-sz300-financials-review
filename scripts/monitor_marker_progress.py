#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOCAL_TZ = datetime.now().astimezone().tzinfo


def now_local() -> datetime:
    return datetime.now().astimezone()


def parse_iso_timestamp(raw: object) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None and LOCAL_TZ is not None:
        value = value.replace(tzinfo=LOCAL_TZ)
    return value


def safe_float(value: float) -> str:
    return f"{value:.2f}"


def format_duration(delta: timedelta) -> str:
    total_seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_ts(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=LOCAL_TZ)
    return value.isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8-sig")
    tmp_path.replace(path)


def run_command(args: List[str], timeout: int = 10) -> Tuple[int, str, str]:
    completed = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def load_latest_launcher(output_dir: Path) -> Tuple[Optional[Path], Dict[str, object]]:
    candidates = list(output_dir.glob("marker_gpu_launcher_*.json"))
    candidates.extend(output_dir.glob("marker_launcher_*.json"))
    marker_launcher = output_dir / "marker_launcher.json"
    if marker_launcher.exists():
        candidates.append(marker_launcher)
    if not candidates:
        return None, {}

    unique_candidates: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve()).lower()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(path.resolve())

    def sort_key(path: Path) -> Tuple[float, int, str]:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        is_gpu = 1 if "marker_gpu_launcher_" in path.name.lower() else 0
        return (mtime, is_gpu, path.name.lower())

    for path in sorted(unique_candidates, key=sort_key, reverse=True):
        try:
            return path, json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
    return None, {}


def detect_latest_log(output_dir: Path) -> Optional[Path]:
    candidates = sorted(
        output_dir.glob("marker_batch_*.err.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def read_tail_text(path: Path, max_bytes: int = 1_000_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_log_progress(log_path: Optional[Path]) -> Dict[str, object]:
    if log_path is None or not log_path.exists():
        return {}

    text = read_tail_text(log_path)
    if not text:
        return {}

    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    progress_lines = [line for line in lines if "Processing PDFs:" in line]
    if not progress_lines:
        return {}

    last_line = progress_lines[-1]
    processed = None
    total = None
    percent = None

    match = re.search(r"Processing PDFs:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)", last_line)
    if match:
        try:
            percent = float(match.group(1))
        except Exception:
            percent = None
        try:
            processed = int(match.group(2))
            total = int(match.group(3))
        except Exception:
            processed = None
            total = None

    result: Dict[str, object] = {
        "line": last_line,
        "processed": processed,
        "total": total,
        "percent": percent,
    }
    return result


def detect_output_markdown(output_subdir: Path) -> Optional[Path]:
    expected = output_subdir / f"{output_subdir.name}.md"
    if expected.exists():
        return expected

    matches = sorted(output_subdir.glob("*.md"))
    if matches:
        return matches[0]

    matches = sorted(output_subdir.glob("*.markdown"))
    if matches:
        return matches[0]
    return None


def collect_output_stats(output_dir: Path) -> Dict[str, object]:
    started_dirs = 0
    completed_markdown = 0
    meta_json_count = 0
    latest_markdown_path: Optional[Path] = None
    latest_markdown_at: Optional[datetime] = None
    latest_dir_path: Optional[Path] = None
    latest_dir_at: Optional[datetime] = None

    for child in sorted(output_dir.iterdir()) if output_dir.exists() else []:
        if not child.is_dir():
            continue
        started_dirs += 1

        try:
            child_mtime = datetime.fromtimestamp(child.stat().st_mtime).astimezone()
            if latest_dir_at is None or child_mtime >= latest_dir_at:
                latest_dir_at = child_mtime
                latest_dir_path = child
        except OSError:
            pass

        md_path = detect_output_markdown(child)
        if md_path is not None:
            completed_markdown += 1
            try:
                md_mtime = datetime.fromtimestamp(md_path.stat().st_mtime).astimezone()
                if latest_markdown_at is None or md_mtime >= latest_markdown_at:
                    latest_markdown_at = md_mtime
                    latest_markdown_path = md_path
            except OSError:
                pass

        expected_meta = child / f"{child.name}_meta.json"
        if expected_meta.exists():
            meta_json_count += 1

    return {
        "started_dirs": started_dirs,
        "completed_markdown": completed_markdown,
        "meta_json_count": meta_json_count,
        "latest_markdown_path": str(latest_markdown_path) if latest_markdown_path else None,
        "latest_markdown_name": latest_markdown_path.stem if latest_markdown_path else None,
        "latest_markdown_at": latest_markdown_at,
        "latest_output_dir": latest_dir_path.name if latest_dir_path else None,
        "latest_output_dir_at": latest_dir_at,
    }


def count_input_pdfs(input_dir: Path) -> int:
    if not input_dir.exists():
        return 0
    return sum(1 for _ in input_dir.glob("*.pdf"))


def detect_gpu_status() -> Dict[str, object]:
    result: Dict[str, object] = {
        "available": False,
        "timestamp": None,
        "utilization_gpu": None,
        "memory_used_mb": None,
        "pstate": None,
        "power_draw_w": None,
        "compute_apps": [],
        "active": False,
    }

    code, stdout, _ = run_command(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,memory.used,pstate,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if code != 0:
        return result

    first_line = next((line.strip() for line in stdout.splitlines() if line.strip()), "")
    if not first_line:
        return result

    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) >= 5:
        result["available"] = True
        result["timestamp"] = parts[0]
        try:
            result["utilization_gpu"] = int(float(parts[1]))
        except Exception:
            result["utilization_gpu"] = None
        try:
            result["memory_used_mb"] = int(float(parts[2]))
        except Exception:
            result["memory_used_mb"] = None
        result["pstate"] = parts[3]
        try:
            result["power_draw_w"] = float(parts[4])
        except Exception:
            result["power_draw_w"] = None

    code, stdout, _ = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        timeout=10,
    )
    apps: List[Dict[str, object]] = []
    if code == 0:
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) < 2:
                continue
            memory_text = parts[2] if len(parts) > 2 else ""
            try:
                pid = int(parts[0])
            except Exception:
                pid = None
            apps.append(
                {
                    "pid": pid,
                    "process_name": parts[1],
                    "used_gpu_memory": memory_text,
                }
            )

    result["compute_apps"] = apps
    result["active"] = bool(apps) or bool((result.get("utilization_gpu") or 0) > 0)
    return result


def is_pid_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    code, stdout, _ = run_command(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ 'RUNNING' }}",
        ],
        timeout=10,
    )
    if code == 0 and "RUNNING" in stdout:
        return True

    code, stdout, _ = run_command(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        timeout=10,
    )
    if code != 0:
        return False
    rows = [row.strip() for row in stdout.splitlines() if row.strip()]
    if not rows:
        return False
    first = rows[0]
    return not first.startswith("INFO:")


def compute_rate(processed: Optional[int], started_at: Optional[datetime], observed_at: datetime) -> Dict[str, object]:
    if processed is None or processed <= 0 or started_at is None:
        return {}
    elapsed_seconds = max(1.0, (observed_at - started_at).total_seconds())
    items_per_hour = processed * 3600.0 / elapsed_seconds
    return {
        "items_per_hour": items_per_hour,
        "elapsed_seconds": elapsed_seconds,
    }


def compute_eta(
    completed: Optional[int],
    total: Optional[int],
    items_per_hour: Optional[float],
    observed_at: datetime,
) -> Dict[str, object]:
    if completed is None or total is None or items_per_hour is None or items_per_hour <= 0:
        return {}
    remaining = max(0, total - completed)
    if remaining <= 0:
        return {"remaining_items": 0, "eta_seconds": 0.0, "eta_at": observed_at}
    eta_seconds = remaining / items_per_hour * 3600.0
    return {
        "remaining_items": remaining,
        "eta_seconds": eta_seconds,
        "eta_at": observed_at + timedelta(seconds=eta_seconds),
    }


def determine_status(
    *,
    total_inputs: int,
    completed_markdown: int,
    wrapper_running: bool,
    gpu_active: bool,
    last_activity_at: Optional[datetime],
    observed_at: datetime,
) -> str:
    if total_inputs > 0 and completed_markdown >= total_inputs:
        return "completed"
    if wrapper_running or gpu_active:
        return "processing"
    if last_activity_at is not None and (observed_at - last_activity_at) <= timedelta(minutes=10):
        return "processing"
    if completed_markdown > 0:
        return "idle_or_stalled"
    return "starting"


def build_snapshot(input_dir: Path, output_dir: Path) -> Dict[str, object]:
    observed_at = now_local()
    launcher_path, launcher = load_latest_launcher(output_dir)
    launcher_started_at = parse_iso_timestamp(launcher.get("StartedAt"))
    wrapper_pid = None
    try:
        wrapper_pid = int(launcher.get("WrapperPid") or launcher.get("MarkerPid") or 0) or None
    except Exception:
        wrapper_pid = None
    wrapper_running = is_pid_running(wrapper_pid)

    total_inputs = count_input_pdfs(input_dir)
    output_stats = collect_output_stats(output_dir)
    completed_markdown = int(output_stats["completed_markdown"])
    started_dirs = int(output_stats["started_dirs"])
    remaining = max(0, total_inputs - completed_markdown)
    percent_complete = (completed_markdown / total_inputs * 100.0) if total_inputs else 0.0

    latest_log = detect_latest_log(output_dir)
    log_progress = parse_log_progress(latest_log)
    latest_log_at = None
    if latest_log is not None:
        try:
            latest_log_at = datetime.fromtimestamp(latest_log.stat().st_mtime).astimezone()
        except OSError:
            latest_log_at = None

    latest_markdown_at = output_stats.get("latest_markdown_at")
    if isinstance(latest_markdown_at, str):
        latest_markdown_at = parse_iso_timestamp(latest_markdown_at)

    last_activity_candidates = [value for value in [latest_markdown_at, latest_log_at] if isinstance(value, datetime)]
    last_activity_at = max(last_activity_candidates) if last_activity_candidates else None

    gpu = detect_gpu_status()
    status = determine_status(
        total_inputs=total_inputs,
        completed_markdown=completed_markdown,
        wrapper_running=wrapper_running,
        gpu_active=bool(gpu.get("active")),
        last_activity_at=last_activity_at,
        observed_at=observed_at,
    )

    rate = compute_rate(
        processed=log_progress.get("processed") if log_progress else None,
        started_at=launcher_started_at,
        observed_at=observed_at,
    )
    eta = compute_eta(
        completed=log_progress.get("processed") if log_progress else None,
        total=log_progress.get("total") if log_progress else None,
        items_per_hour=rate.get("items_per_hour") if rate else None,
        observed_at=observed_at,
    )

    snapshot: Dict[str, object] = {
        "status": status,
        "observed_at": format_ts(observed_at),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_input_pdfs": total_inputs,
        "started_output_dirs": started_dirs,
        "completed_markdown": completed_markdown,
        "remaining_pdfs": remaining,
        "percent_complete": round(percent_complete, 4),
        "latest_output_dir": output_stats.get("latest_output_dir"),
        "latest_output_dir_at": format_ts(output_stats.get("latest_output_dir_at")),
        "latest_markdown_name": output_stats.get("latest_markdown_name"),
        "latest_markdown_path": output_stats.get("latest_markdown_path"),
        "latest_markdown_at": format_ts(latest_markdown_at),
        "meta_json_count": output_stats.get("meta_json_count"),
        "latest_log_path": str(latest_log) if latest_log else None,
        "latest_log_at": format_ts(latest_log_at),
        "log_progress": log_progress or {},
        "launcher": {
            "path": str(launcher_path) if launcher_path else None,
            "started_at": format_ts(launcher_started_at),
            "wrapper_pid": wrapper_pid,
            "wrapper_running": wrapper_running,
            "marker_exe": launcher.get("MarkerExe"),
            "model_cache_dir": launcher.get("ModelCacheDir"),
            "temp_dir": launcher.get("TempDir"),
        },
        "gpu": gpu,
        "throughput": {
            "log_items_per_hour": round(rate["items_per_hour"], 4) if rate else None,
            "elapsed_since_launcher_seconds": round(rate["elapsed_seconds"], 2) if rate else None,
        },
        "eta": {
            "remaining_log_items": eta.get("remaining_items") if eta else None,
            "eta_seconds": round(eta["eta_seconds"], 2) if eta else None,
            "eta_at": format_ts(eta.get("eta_at")) if eta else None,
        },
    }
    return snapshot


def render_text(snapshot: Dict[str, object]) -> str:
    launcher = dict(snapshot.get("launcher") or {})
    gpu = dict(snapshot.get("gpu") or {})
    log_progress = dict(snapshot.get("log_progress") or {})
    throughput = dict(snapshot.get("throughput") or {})
    eta = dict(snapshot.get("eta") or {})

    lines = [
        "marker markdown progress",
        f"status: {snapshot.get('status')}",
        f"observed_at: {snapshot.get('observed_at')}",
        f"progress: {snapshot.get('completed_markdown')}/{snapshot.get('total_input_pdfs')} ({safe_float(float(snapshot.get('percent_complete') or 0.0))}%)",
        f"started_output_dirs: {snapshot.get('started_output_dirs')}",
        f"remaining_pdfs: {snapshot.get('remaining_pdfs')}",
        f"latest_markdown: {snapshot.get('latest_markdown_name') or ''}",
        f"latest_markdown_at: {snapshot.get('latest_markdown_at') or ''}",
        f"latest_output_dir: {snapshot.get('latest_output_dir') or ''}",
        f"latest_log_at: {snapshot.get('latest_log_at') or ''}",
    ]

    if launcher.get("started_at"):
        started_at = parse_iso_timestamp(launcher.get("started_at"))
        observed_at = parse_iso_timestamp(snapshot.get("observed_at"))
        if started_at is not None and observed_at is not None:
            lines.append(f"elapsed_since_launcher: {format_duration(observed_at - started_at)}")
        lines.append(f"launcher_started_at: {launcher.get('started_at')}")

    if log_progress:
        percent = log_progress.get("percent")
        percent_text = safe_float(float(percent)) if percent is not None else ""
        lines.append(
            f"log_progress: {log_progress.get('processed')}/{log_progress.get('total')} ({percent_text}%)"
        )
        if throughput.get("log_items_per_hour") is not None:
            lines.append(f"log_rate: {safe_float(float(throughput['log_items_per_hour']))} items/hour")
        if eta.get("eta_at"):
            lines.append(f"log_eta: {eta.get('eta_at')}")
        if log_progress.get("line"):
            lines.append(f"log_line: {log_progress.get('line')}")

    gpu_summary = "unavailable"
    if gpu.get("available"):
        gpu_summary = (
            f"active={'yes' if gpu.get('active') else 'no'} "
            f"util={gpu.get('utilization_gpu')}% "
            f"mem={gpu.get('memory_used_mb')}MiB "
            f"power={gpu.get('power_draw_w')}W "
            f"pstate={gpu.get('pstate')}"
        )
    lines.append(f"gpu: {gpu_summary}")

    compute_apps = list(gpu.get("compute_apps") or [])
    if compute_apps:
        lines.append("gpu_apps:")
        for app in compute_apps:
            lines.append(
                f"  pid={app.get('pid')} process={app.get('process_name')} mem={app.get('used_gpu_memory')}"
            )

    lines.append(
        f"wrapper: pid={launcher.get('wrapper_pid')} running={'yes' if launcher.get('wrapper_running') else 'no'}"
    )
    if launcher.get("marker_exe"):
        lines.append(f"marker_exe: {launcher.get('marker_exe')}")
    if snapshot.get("latest_log_path"):
        lines.append(f"latest_log_path: {snapshot.get('latest_log_path')}")
    return "\n".join(lines) + "\n"


def write_outputs(snapshot: Dict[str, object], status_path: Path, json_path: Path) -> None:
    json_payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
    atomic_write_text(status_path, render_text(snapshot))
    atomic_write_text(json_path, json_payload + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write marker PDF→Markdown progress snapshots.")
    parser.add_argument("--input-dir", default=".cache/qwen_pdf_markdown_remaining/input_pdfs", help="Input PDF directory.")
    parser.add_argument("--output-dir", default=".cache/qwen_pdf_markdown_remaining/output_markdown", help="Marker output directory.")
    parser.add_argument("--status-file", default=".cache/qwen_pdf_markdown_remaining/output_markdown/progress_status.txt", help="Human-readable status file.")
    parser.add_argument("--json-file", default=".cache/qwen_pdf_markdown_remaining/output_markdown/progress_status.json", help="JSON status file.")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh snapshots.")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds when watching.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def resolve_path(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def main() -> int:
    args = parse_args()
    base_dir = Path.cwd()
    input_dir = resolve_path(base_dir, args.input_dir)
    output_dir = resolve_path(base_dir, args.output_dir)
    status_path = resolve_path(base_dir, args.status_file)
    json_path = resolve_path(base_dir, args.json_file)

    while True:
        snapshot = build_snapshot(input_dir=input_dir, output_dir=output_dir)
        write_outputs(snapshot=snapshot, status_path=status_path, json_path=json_path)
        if not args.quiet:
            sys.stdout.write(render_text(snapshot))
            sys.stdout.flush()
        if not args.watch:
            return 0
        time.sleep(max(1, int(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
