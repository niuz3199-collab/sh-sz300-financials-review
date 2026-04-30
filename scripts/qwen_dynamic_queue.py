#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import msvcrt
import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.step6_extract_financials_qwen_pdf import (  # noqa: E402
    Task,
    append_csv_row,
    collect_tasks,
    extract_from_pdf,
    infer_code_name,
)


TASK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    stock_code TEXT NOT NULL,
    pdf_path TEXT NOT NULL,
    code_name TEXT NOT NULL,
    queue_state TEXT NOT NULL DEFAULT 'pending',
    claimed_by TEXT,
    claim_ts TEXT,
    claim_expires_ts TEXT,
    done_ts TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_status TEXT,
    last_message TEXT,
    last_backend TEXT,
    raw_json_path TEXT,
    UNIQUE(year, stock_code)
);
CREATE INDEX IF NOT EXISTS idx_tasks_queue_state ON tasks(queue_state, claim_expires_ts, year, stock_code);
"""


LOG_FIELDS = [
    "ts",
    "year",
    "stock_code",
    "code_name",
    "pdf_path",
    "status",
    "message",
    "raw_json_path",
]


YEAR_FIELDS = [
    "year",
    "stock_code",
    "code_name",
    "stock_name",
    "parent_netprofit",
    "share_capital",
    "share_capital_wan",
    "netcash_operate",
    "construct_long_asset",
    "pdf_path",
    "parent_netprofit_page",
    "share_capital_page",
    "netcash_operate_page",
    "construct_long_asset_page",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def serialize_task(task: Task) -> Dict[str, object]:
    payload = asdict(task)
    payload["pdf_path"] = str(task.pdf_path)
    return payload


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(TASK_TABLE_SQL)
    return conn


def init_queue(
    *,
    db_path: Path,
    pdf_root: Path,
    start_year: int,
    end_year: int,
) -> int:
    tasks = collect_tasks(pdf_root, start_year=int(start_year), end_year=int(end_year))
    conn = connect_db(db_path)
    try:
        existing = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        if int(existing) > 0:
            return int(existing)
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            """
            INSERT OR IGNORE INTO tasks (year, stock_code, pdf_path, code_name, queue_state)
            VALUES (?, ?, ?, ?, 'pending')
            """,
            [
                (int(task.year), str(task.stock_code), str(task.pdf_path), infer_code_name(task.stock_code))
                for task in tasks
            ],
        )
        conn.commit()
        return len(tasks)
    finally:
        conn.close()


def write_dynamic_run_config(
    *,
    out_dir: Path,
    queue_db: Path,
    pdf_root: Path,
    csv_name: str,
    start_year: int,
    end_year: int,
    workers: List[Dict[str, object]],
) -> None:
    payload = {
        "backend": "dynamic_queue",
        "queue_db": str(queue_db),
        "pdf_root": str(pdf_root),
        "csv_name": str(csv_name),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "workers": workers,
        "written_at": now_iso(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_iso(raw: Optional[str]) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def claim_next_task(
    *,
    db_path: Path,
    worker_id: str,
    backend: str,
    lease_seconds: int,
) -> Optional[Dict[str, object]]:
    claim_ts = datetime.now()
    claim_expires_ts = claim_ts + timedelta(seconds=int(lease_seconds))
    conn = connect_db(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT id, year, stock_code, pdf_path
            FROM tasks
            WHERE queue_state = 'pending'
               OR (queue_state = 'claimed' AND (claim_expires_ts IS NULL OR claim_expires_ts < ?))
            ORDER BY year, stock_code
            LIMIT 1
            """,
            (claim_ts.isoformat(timespec="seconds"),),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            """
            UPDATE tasks
            SET queue_state = 'claimed',
                claimed_by = ?,
                claim_ts = ?,
                claim_expires_ts = ?,
                attempts = attempts + 1,
                last_backend = ?,
                last_status = 'claimed',
                last_message = ''
            WHERE id = ?
            """,
            (
                str(worker_id),
                claim_ts.isoformat(timespec="seconds"),
                claim_expires_ts.isoformat(timespec="seconds"),
                str(backend),
                int(row["id"]),
            ),
        )
        conn.commit()
        return {
            "task": Task(year=int(row["year"]), stock_code=str(row["stock_code"]), pdf_path=Path(str(row["pdf_path"]))),
            "claim_ts": claim_ts.isoformat(timespec="seconds"),
            "claim_expires_ts": claim_expires_ts.isoformat(timespec="seconds"),
        }
    finally:
        conn.close()


def unfinished_counts(db_path: Path) -> Dict[str, int]:
    conn = connect_db(db_path)
    try:
        rows = conn.execute(
            "SELECT queue_state, COUNT(*) AS n FROM tasks GROUP BY queue_state"
        ).fetchall()
        out = {"pending": 0, "claimed": 0, "done": 0}
        for row in rows:
            out[str(row["queue_state"])] = int(row["n"])
        return out
    finally:
        conn.close()


def mark_task_done(
    *,
    db_path: Path,
    task: Task,
    worker_id: str,
    claim_ts: str,
    backend: str,
    status: str,
    message: str,
    raw_json_path: str,
) -> bool:
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = connect_db(db_path)
            cur = conn.execute(
                """
                UPDATE tasks
                SET queue_state = 'done',
                    done_ts = ?,
                    claim_expires_ts = NULL,
                    last_backend = ?,
                    last_status = ?,
                    last_message = ?,
                    raw_json_path = ?
                WHERE year = ? AND stock_code = ? AND queue_state = 'claimed' AND claimed_by = ? AND claim_ts = ?
                """,
                (
                    now_iso(),
                    str(backend),
                    str(status),
                    str(message or ""),
                    str(raw_json_path or ""),
                    int(task.year),
                    str(task.stock_code),
                    str(worker_id),
                    str(claim_ts),
                ),
            )
            return int(cur.rowcount or 0) > 0
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if attempt >= 3:
                raise
            time.sleep(0.5 * (attempt + 1))
        finally:
            if conn is not None:
                conn.close()
    if last_exc is not None:
        raise last_exc
    return False


def still_owns_claim(
    *,
    db_path: Path,
    task: Task,
    worker_id: str,
    claim_ts: str,
) -> bool:
    conn = connect_db(db_path)
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM tasks
            WHERE year = ? AND stock_code = ? AND queue_state = 'claimed' AND claimed_by = ? AND claim_ts = ?
            """,
            (int(task.year), str(task.stock_code), str(worker_id), str(claim_ts)),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def write_worker_state(
    out_dir: Path,
    *,
    worker_id: str,
    backend: str,
    model: str,
    status: str,
    task: Optional[Task],
    timeout: int,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    payload: Dict[str, object] = {
        "worker_id": str(worker_id),
        "backend": str(backend),
        "model": str(model),
        "status": str(status),
        "pid": os.getpid(),
        "updated_at": now_iso(),
        "timeout_seconds": int(timeout),
    }
    if task is not None:
        payload["task"] = serialize_task(task)
    if extra:
        payload.update(extra)
    workers_dir = out_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    (workers_dir / f"{worker_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def file_mutex(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    existed = lock_path.exists()
    handle = lock_path.open("a+b")
    try:
        if not existed or lock_path.stat().st_size <= 0:
            handle.seek(0, os.SEEK_END)
            handle.write(b"0")
            handle.flush()
        locked = False
        while not locked:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                time.sleep(0.2)
        yield
    finally:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        handle.close()


def append_csv_row_locked(path: Path, row: Dict[str, object], *, fieldnames: List[str]) -> None:
    lock_path = Path(f"{path}.lock")
    with file_mutex(lock_path):
        append_csv_row(path, row, fieldnames=fieldnames)


def build_year_row(task: Task, extracted: Dict) -> Dict[str, object]:
    norm = extracted.get("normalized") or {}
    raw = extracted.get("raw") or {}
    total_shares_wan = norm.get("total_shares_wan")
    share_capital_shares = norm.get("total_shares_shares")
    return {
        "year": task.year,
        "stock_code": task.stock_code,
        "code_name": infer_code_name(task.stock_code),
        "stock_name": "",
        "parent_netprofit": norm.get("parent_netprofit_yuan"),
        "share_capital": share_capital_shares,
        "share_capital_wan": total_shares_wan,
        "netcash_operate": norm.get("operating_cashflow_yuan"),
        "construct_long_asset": norm.get("capex_yuan"),
        "pdf_path": str(task.pdf_path),
        "parent_netprofit_page": (raw.get("parent_netprofit") or {}).get("page"),
        "share_capital_page": (raw.get("total_shares") or {}).get("page"),
        "netcash_operate_page": (raw.get("operating_cashflow") or {}).get("page"),
        "construct_long_asset_page": (raw.get("capex") or {}).get("page"),
    }


def determine_status(extracted: Dict) -> Dict[str, object]:
    norm = extracted.get("normalized") or {}
    pages_info = extracted.get("pages") or {}
    missing_fields: List[str] = []
    if norm.get("parent_netprofit_yuan") is None:
        missing_fields.append("parent_netprofit")
    if norm.get("total_shares_shares") is None:
        missing_fields.append("total_shares")
    if norm.get("operating_cashflow_yuan") is None:
        missing_fields.append("operating_cashflow")
    if norm.get("capex_yuan") is None:
        missing_fields.append("capex")

    missing_pages = [k for k in ("income", "shares", "cfo", "capex") if not pages_info.get(k)]
    status = "ok" if not missing_fields else "partial"
    message_parts: List[str] = []
    if missing_pages:
        message_parts.append(f"missing_pages={','.join(missing_pages)}")
    if missing_fields:
        message_parts.append(f"missing_fields={','.join(missing_fields)}")
    return {"status": status, "message": "; ".join(message_parts)}


def extract_task_result(
    *,
    task: Task,
    backend: str,
    model: str,
    dpi: int,
    timeout: int,
    api_base_url: str,
    api_key_env: str,
    debug: bool,
) -> Dict[str, object]:
    api_key = ""
    if str(backend).strip().lower() == "openai_text":
        api_key = str(os.environ.get(api_key_env, "") or "").strip()
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is empty; cannot call external API")

    extracted = extract_from_pdf(
        task,
        backend=backend,
        model=model,
        dpi=int(dpi),
        timeout=int(timeout),
        api_base_url=api_base_url,
        api_key=api_key,
        debug=bool(debug),
    )
    return extracted


def write_extracted_result(
    *,
    task: Task,
    extracted: Dict,
    base_dir: Path,
    out_dir: Path,
    csv_name: str,
) -> Dict[str, object]:
    raw_json_path = out_dir / "raw_json" / str(task.year) / f"{task.stock_code}.json"
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    raw_json_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    row = build_year_row(task, extracted)
    year_dir = base_dir / str(task.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    year_csv = year_dir / str(csv_name).format(year=task.year)
    append_csv_row_locked(year_csv, row, fieldnames=YEAR_FIELDS)

    status_info = determine_status(extracted)
    append_csv_row_locked(
        out_dir / "extract_log.csv",
        {
            "ts": now_iso(),
            "year": task.year,
            "stock_code": task.stock_code,
            "code_name": infer_code_name(task.stock_code),
            "pdf_path": str(task.pdf_path),
            "status": status_info["status"],
            "message": status_info["message"],
            "raw_json_path": str(raw_json_path),
        },
        fieldnames=LOG_FIELDS,
    )
    status_info["raw_json_path"] = str(raw_json_path)
    return status_info


def worker_loop(
    *,
    base_dir: Path,
    out_dir: Path,
    queue_db: Path,
    worker_id: str,
    backend: str,
    model: str,
    dpi: int,
    timeout: int,
    lease_seconds: int,
    poll_seconds: float,
    api_base_url: str,
    api_key_env: str,
    csv_name: str,
    debug: bool,
    max_tasks: int,
) -> int:
    processed = 0
    while True:
        claim = claim_next_task(db_path=queue_db, worker_id=worker_id, backend=backend, lease_seconds=lease_seconds)
        if claim is None:
            counts = unfinished_counts(queue_db)
            if int(counts.get("pending", 0)) == 0 and int(counts.get("claimed", 0)) == 0:
                write_worker_state(
                    out_dir,
                    worker_id=worker_id,
                    backend=backend,
                    model=model,
                    status="done",
                    task=None,
                    timeout=timeout,
                    extra={"queue_counts": counts},
                )
                return 0
            write_worker_state(
                out_dir,
                worker_id=worker_id,
                backend=backend,
                model=model,
                status="waiting",
                task=None,
                timeout=timeout,
                extra={"queue_counts": counts},
            )
            time.sleep(max(0.5, float(poll_seconds)))
            continue
        task = claim["task"]
        claim_ts = str(claim["claim_ts"])

        write_worker_state(
            out_dir,
            worker_id=worker_id,
            backend=backend,
            model=model,
            status="processing",
            task=task,
            timeout=timeout,
            extra={"lease_seconds": int(lease_seconds)},
        )

        try:
            extracted = extract_task_result(
                task=task,
                backend=backend,
                model=model,
                dpi=dpi,
                timeout=timeout,
                api_base_url=api_base_url,
                api_key_env=api_key_env,
                debug=debug,
            )
            if not still_owns_claim(db_path=queue_db, task=task, worker_id=worker_id, claim_ts=claim_ts):
                write_worker_state(
                    out_dir,
                    worker_id=worker_id,
                    backend=backend,
                    model=model,
                    status="stale_result_discarded",
                    task=None,
                    timeout=timeout,
                    extra={
                        "last_task": serialize_task(task),
                        "processed_tasks": processed,
                    },
                )
                continue
            status_info = write_extracted_result(
                task=task,
                extracted=extracted,
                base_dir=base_dir,
                out_dir=out_dir,
                csv_name=csv_name,
            )
            finalize_error = ""
            try:
                mark_task_done(
                    db_path=queue_db,
                    task=task,
                    worker_id=worker_id,
                    claim_ts=claim_ts,
                    backend=backend,
                    status=str(status_info["status"]),
                    message=str(status_info["message"]),
                    raw_json_path=str(status_info["raw_json_path"]),
                )
            except Exception as finalize_exc:
                finalize_error = str(finalize_exc)
            processed += 1
            write_worker_state(
                out_dir,
                worker_id=worker_id,
                backend=backend,
                model=model,
                status="idle",
                task=None,
                timeout=timeout,
                extra={
                    "last_task": serialize_task(task),
                    "last_status": status_info["status"],
                    "last_message": (
                        str(status_info["message"])
                        if not finalize_error
                        else f"{status_info['message']}; finalize_failed={finalize_error[:200]}"
                    ),
                    "processed_tasks": processed,
                },
            )
        except Exception as exc:
            message = str(exc)
            finalize_error = ""
            if still_owns_claim(db_path=queue_db, task=task, worker_id=worker_id, claim_ts=claim_ts):
                append_csv_row_locked(
                    out_dir / "extract_log.csv",
                    {
                        "ts": now_iso(),
                        "year": task.year,
                        "stock_code": task.stock_code,
                        "code_name": infer_code_name(task.stock_code),
                        "pdf_path": str(task.pdf_path),
                        "status": "error",
                        "message": message,
                        "raw_json_path": "",
                    },
                    fieldnames=LOG_FIELDS,
                )
                try:
                    mark_task_done(
                        db_path=queue_db,
                        task=task,
                        worker_id=worker_id,
                        claim_ts=claim_ts,
                        backend=backend,
                        status="error",
                        message=message,
                        raw_json_path="",
                    )
                except Exception as finalize_exc:
                    finalize_error = str(finalize_exc)
            processed += 1
            write_worker_state(
                out_dir,
                worker_id=worker_id,
                backend=backend,
                model=model,
                status="idle",
                task=None,
                timeout=timeout,
                extra={
                    "last_task": serialize_task(task),
                    "last_status": "error",
                    "last_message": message if not finalize_error else f"{message}; finalize_failed={finalize_error[:200]}",
                    "processed_tasks": processed,
                },
            )

        if int(max_tasks) > 0 and processed >= int(max_tasks):
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic shared-queue runner for annual-report extraction.")
    parser.add_argument("--base-dir", default=".", help="Repository root directory.")
    parser.add_argument("--pdf-root", default="年报/下载年报_fulltext", help="PDF annual report root directory.")
    parser.add_argument("--out-dir", default=".cache/qwen_pdf_financials_v6_dynamic", help="Shared output directory.")
    parser.add_argument("--queue-db", default=".cache/qwen_pdf_financials_v6_dynamic/task_queue.sqlite", help="SQLite queue database path.")
    parser.add_argument("--start-year", type=int, default=2001, help="Start year inclusive.")
    parser.add_argument("--end-year", type=int, default=2025, help="End year inclusive.")
    parser.add_argument("--csv-name", default="{year}_qwen_v6_dynamic.csv", help="Year CSV filename template.")
    parser.add_argument("--init-only", action="store_true", help="Only initialize queue DB and run_config, then exit.")
    parser.add_argument("--worker-id", default="", help="Worker identifier for claim ownership.")
    parser.add_argument("--backend", choices=["ollama", "openai_text"], default="ollama", help="Backend used by this worker.")
    parser.add_argument("--model", default="qwen3.5:9b", help="Model name.")
    parser.add_argument("--api-base-url", default="", help="OpenAI-compatible API base URL for backend=openai_text.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var name holding the API key for backend=openai_text.")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI for backend=ollama.")
    parser.add_argument("--ollama-num-ctx", type=int, default=8192, help="num_ctx for backend=ollama.")
    parser.add_argument("--timeout", type=int, default=1200, help="Single model request timeout in seconds.")
    parser.add_argument("--lease-seconds", type=int, default=1800, help="Task claim lease duration in seconds.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Sleep time when queue is temporarily empty but not finished.")
    parser.add_argument("--debug", action="store_true", help="Print selected key pages for each claimed task.")
    parser.add_argument("--max-tasks", type=int, default=0, help="Stop after processing N tasks; 0 means unlimited.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    pdf_root = (Path(args.pdf_root) if Path(args.pdf_root).is_absolute() else (base_dir / args.pdf_root)).resolve()
    out_dir = (Path(args.out_dir) if Path(args.out_dir).is_absolute() else (base_dir / args.out_dir)).resolve()
    queue_db = (Path(args.queue_db) if Path(args.queue_db).is_absolute() else (base_dir / args.queue_db)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = init_queue(
        db_path=queue_db,
        pdf_root=pdf_root,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )
    print(f"[queue] db={queue_db} total={total}", flush=True)

    run_config_path = out_dir / "run_config.json"
    workers: List[Dict[str, object]] = []
    if run_config_path.exists():
        try:
            workers = list((json.loads(run_config_path.read_text(encoding="utf-8")) or {}).get("workers") or [])
        except Exception:
            workers = []
    worker_entry = {
        "worker_id": str(args.worker_id or ""),
        "backend": str(args.backend),
        "model": str(args.model),
        "timeout_seconds": int(args.timeout),
        "lease_seconds": int(args.lease_seconds),
        "api_base_url": str(args.api_base_url or ""),
        "api_key_env": str(args.api_key_env or ""),
        "ollama_num_ctx": int(args.ollama_num_ctx) if str(args.backend).strip().lower() == "ollama" else None,
    }
    if worker_entry["worker_id"]:
        workers = [w for w in workers if str((w or {}).get("worker_id") or "") != worker_entry["worker_id"]]
        workers.append(worker_entry)
    write_dynamic_run_config(
        out_dir=out_dir,
        queue_db=queue_db,
        pdf_root=pdf_root,
        csv_name=str(args.csv_name),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        workers=workers,
    )

    if args.init_only:
        return 0

    if not str(args.worker_id or "").strip():
        raise RuntimeError("--worker-id is required unless --init-only is used")
    if str(args.backend).strip().lower() == "openai_text" and not str(args.api_base_url or "").strip():
        raise RuntimeError("--api-base-url is required for backend=openai_text")
    if str(args.backend).strip().lower() == "ollama":
        os.environ["OLLAMA_NUM_CTX"] = str(int(args.ollama_num_ctx))

    return worker_loop(
        base_dir=base_dir,
        out_dir=out_dir,
        queue_db=queue_db,
        worker_id=str(args.worker_id).strip(),
        backend=str(args.backend).strip(),
        model=str(args.model).strip(),
        dpi=int(args.dpi),
        timeout=int(args.timeout),
        lease_seconds=int(args.lease_seconds),
        poll_seconds=float(args.poll_seconds),
        api_base_url=str(args.api_base_url or "").strip(),
        api_key_env=str(args.api_key_env or "").strip(),
        csv_name=str(args.csv_name),
        debug=bool(args.debug),
        max_tasks=int(args.max_tasks),
    )


if __name__ == "__main__":
    raise SystemExit(main())
