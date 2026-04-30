#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple


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


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=60, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(TASK_TABLE_SQL)
    return conn


def parse_csv_list(raw: str) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def load_manifest_keys(path: Path) -> Set[Tuple[int, str]]:
    keys: Set[Tuple[int, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year_raw = str(row.get("year") or "").strip()
            code_raw = str(row.get("stock_code") or "").strip()
            if not year_raw or not code_raw:
                continue
            try:
                keys.add((int(year_raw), code_raw))
            except ValueError:
                continue
    return keys


def build_query(
    *,
    statuses: Sequence[str],
    include_workers: Sequence[str],
    exclude_workers: Sequence[str],
    start_year: int | None,
    end_year: int | None,
    limit: int,
) -> tuple[str, List[object]]:
    where = [f"last_status IN ({','.join('?' for _ in statuses)})"]
    params: List[object] = list(statuses)

    if include_workers:
        where.append(f"claimed_by IN ({','.join('?' for _ in include_workers)})")
        params.extend(include_workers)
    if exclude_workers:
        where.append(f"COALESCE(claimed_by, '') NOT IN ({','.join('?' for _ in exclude_workers)})")
        params.extend(exclude_workers)
    if start_year is not None:
        where.append("year >= ?")
        params.append(int(start_year))
    if end_year is not None:
        where.append("year <= ?")
        params.append(int(end_year))

    sql = f"""
        SELECT
            year,
            stock_code,
            pdf_path,
            code_name,
            claimed_by,
            last_status,
            last_message,
            last_backend,
            raw_json_path
        FROM tasks
        WHERE {' AND '.join(where)}
        ORDER BY year, stock_code
    """
    if int(limit) > 0:
        sql += " LIMIT ?"
        params.append(int(limit))
    return sql, params


def write_manifest(path: Path, rows: Iterable[sqlite3.Row]) -> None:
    fieldnames = [
        "year",
        "stock_code",
        "code_name",
        "pdf_path",
        "source_worker",
        "source_status",
        "source_message",
        "source_backend",
        "source_raw_json_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "year": row["year"],
                    "stock_code": row["stock_code"],
                    "code_name": row["code_name"],
                    "pdf_path": row["pdf_path"],
                    "source_worker": row["claimed_by"] or "",
                    "source_status": row["last_status"] or "",
                    "source_message": row["last_message"] or "",
                    "source_backend": row["last_backend"] or "",
                    "source_raw_json_path": row["raw_json_path"] or "",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize a retry queue from non-ok tasks of a previous Qwen run.")
    parser.add_argument("--source-out-dir", required=True, help="Existing run output directory containing task_queue.sqlite.")
    parser.add_argument("--target-out-dir", required=True, help="New retry output directory to create.")
    parser.add_argument("--statuses", default="error,partial", help="Comma-separated final statuses to include.")
    parser.add_argument("--workers", default="", help="Comma-separated worker ids to include.")
    parser.add_argument("--exclude-workers", default="", help="Comma-separated worker ids to exclude.")
    parser.add_argument("--exclude-manifest", default="", help="CSV manifest whose year/stock_code rows will be excluded from the retry queue.")
    parser.add_argument("--start-year", type=int, default=None, help="Optional lower bound for report year.")
    parser.add_argument("--end-year", type=int, default=None, help="Optional upper bound for report year.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of tasks to include.")
    args = parser.parse_args()

    source_out_dir = Path(args.source_out_dir).expanduser().resolve()
    target_out_dir = Path(args.target_out_dir).expanduser().resolve()
    source_db = source_out_dir / "task_queue.sqlite"
    target_db = target_out_dir / "task_queue.sqlite"

    if not source_db.exists():
        raise FileNotFoundError(f"Missing source queue DB: {source_db}")
    if target_db.exists():
        raise FileExistsError(f"Target queue DB already exists: {target_db}")

    statuses = parse_csv_list(args.statuses)
    include_workers = parse_csv_list(args.workers)
    exclude_workers = parse_csv_list(args.exclude_workers)
    exclude_manifest_path = Path(args.exclude_manifest).expanduser().resolve() if str(args.exclude_manifest or "").strip() else None
    if not statuses:
        raise RuntimeError("No statuses selected; use --statuses error,partial or a subset.")
    exclude_manifest_keys: Set[Tuple[int, str]] = set()
    if exclude_manifest_path is not None:
        if not exclude_manifest_path.exists():
            raise FileNotFoundError(f"Exclude manifest not found: {exclude_manifest_path}")
        exclude_manifest_keys = load_manifest_keys(exclude_manifest_path)

    source_conn = connect_db(source_db)
    try:
        sql, params = build_query(
            statuses=statuses,
            include_workers=include_workers,
            exclude_workers=exclude_workers,
            start_year=args.start_year,
            end_year=args.end_year,
            limit=args.limit,
        )
        rows = list(source_conn.execute(sql, params))
        if exclude_manifest_keys:
            rows = [
                row
                for row in rows
                if (int(row["year"]), str(row["stock_code"])) not in exclude_manifest_keys
            ]
    finally:
        source_conn.close()

    if not rows:
        raise RuntimeError("No retry tasks matched the requested filters.")

    target_conn = connect_db(target_db)
    try:
        target_conn.execute("BEGIN IMMEDIATE")
        target_conn.executemany(
            """
            INSERT INTO tasks (
                year,
                stock_code,
                pdf_path,
                code_name,
                queue_state,
                claimed_by,
                claim_ts,
                claim_expires_ts,
                done_ts,
                attempts,
                last_status,
                last_message,
                last_backend,
                raw_json_path
            )
            VALUES (?, ?, ?, ?, 'pending', NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, NULL)
            """,
            [
                (
                    int(row["year"]),
                    str(row["stock_code"]),
                    str(row["pdf_path"]),
                    str(row["code_name"]),
                )
                for row in rows
            ],
        )
        target_conn.commit()
    finally:
        target_conn.close()

    manifest_path = target_out_dir / "retry_manifest.csv"
    write_manifest(manifest_path, rows)

    source_run_config_path = source_out_dir / "run_config.json"
    source_run_config = {}
    if source_run_config_path.exists():
        try:
            source_run_config = json.loads(source_run_config_path.read_text(encoding="utf-8"))
        except Exception:
            source_run_config = {}

    years = [int(row["year"]) for row in rows]
    run_config = {
        "mode": "retry_queue",
        "source_out_dir": str(source_out_dir),
        "source_queue_db": str(source_db),
        "target_out_dir": str(target_out_dir),
        "target_queue_db": str(target_db),
        "created_at": now_iso(),
        "statuses": statuses,
        "workers": [],
        "source_workers_filter": include_workers,
        "source_exclude_workers": exclude_workers,
        "source_exclude_manifest": str(exclude_manifest_path) if exclude_manifest_path is not None else "",
        "retry_count": len(rows),
        "start_year": min(years),
        "end_year": max(years),
        "pdf_root": source_run_config.get("pdf_root", ""),
        "csv_name": source_run_config.get("csv_name", "{year}_qwen_retry.csv"),
        "source_run_config_path": str(source_run_config_path) if source_run_config_path.exists() else "",
    }
    (target_out_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "total": len(rows),
        "by_status": {},
        "by_worker": {},
    }
    for row in rows:
        status = str(row["last_status"] or "")
        worker = str(row["claimed_by"] or "")
        summary["by_status"][status] = int(summary["by_status"].get(status, 0)) + 1
        summary["by_worker"][worker] = int(summary["by_worker"].get(worker, 0)) + 1
    (target_out_dir / "retry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "target_out_dir": str(target_out_dir),
                "retry_count": len(rows),
                "statuses": statuses,
                "workers": include_workers,
                "exclude_workers": exclude_workers,
                "exclude_manifest": str(exclude_manifest_path) if exclude_manifest_path is not None else "",
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
