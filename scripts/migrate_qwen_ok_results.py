#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen_dynamic_queue import YEAR_FIELDS, build_year_row, connect_db, now_iso  # noqa: E402
from scripts.step6_extract_financials_qwen_pdf import Task  # noqa: E402


def normalize_stock_code(raw: object) -> str:
    digits = "".join(ch for ch in str(raw or "") if ch.isdigit())
    return digits.zfill(6) if digits else ""


def parse_source_ts(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).isoformat(timespec="seconds")
    except ValueError:
        return text


@dataclass(frozen=True)
class Candidate:
    source_dir: Path
    row: Dict[str, str]
    priority: int

    @property
    def year(self) -> int:
        return int(str(self.row.get("year") or "").strip())

    @property
    def stock_code(self) -> str:
        return normalize_stock_code(self.row.get("stock_code") or "")

    @property
    def source_ts(self) -> str:
        return parse_source_ts(self.row.get("ts") or "")

    @property
    def source_status(self) -> str:
        return str(self.row.get("status") or "").strip().lower()

    @property
    def source_message(self) -> str:
        return str(self.row.get("message") or "").strip()

    @property
    def source_raw_json(self) -> Path:
        raw = str(self.row.get("raw_json_path") or "").strip()
        if raw:
            return Path(raw)
        return self.source_dir / "raw_json" / str(self.year) / f"{self.stock_code}.json"


def load_latest_rows(log_path: Path) -> Dict[Tuple[int, str], Dict[str, str]]:
    latest: Dict[Tuple[int, str], Dict[str, str]] = {}
    if not log_path.exists():
        return latest
    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                year = int(str(row.get("year") or "").strip())
            except ValueError:
                continue
            stock_code = normalize_stock_code(row.get("stock_code") or "")
            if not stock_code:
                continue
            latest[(year, stock_code)] = {str(k): str(v or "") for k, v in row.items()}
    return latest


def choose_candidates(source_dirs: List[Path]) -> Dict[Tuple[int, str], Candidate]:
    chosen: Dict[Tuple[int, str], Candidate] = {}
    total_ok = 0
    for priority, source_dir in enumerate(source_dirs):
        latest = load_latest_rows(source_dir / "extract_log.csv")
        for key, row in latest.items():
            candidate = Candidate(source_dir=source_dir, row=row, priority=priority)
            if candidate.source_status != "ok":
                continue
            if not candidate.source_raw_json.exists():
                continue
            total_ok += 1
            existing = chosen.get(key)
            if existing is None:
                chosen[key] = candidate
                continue
            if candidate.priority < existing.priority:
                chosen[key] = candidate
                continue
            if candidate.priority == existing.priority and candidate.source_ts > existing.source_ts:
                chosen[key] = candidate
    print(f"[migrate] ok candidates after priority merge: {len(chosen)} (scanned ok rows={total_ok})")
    return chosen


def load_run_config(target_out_dir: Path) -> Dict[str, object]:
    path = target_out_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run_config.json under {target_out_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def reset_claimed_tasks(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        """
        UPDATE tasks
        SET queue_state = 'pending',
            claimed_by = NULL,
            claim_ts = NULL,
            claim_expires_ts = NULL,
            last_status = CASE WHEN last_status = 'claimed' THEN NULL ELSE last_status END,
            last_message = CASE WHEN last_status = 'claimed' THEN '' ELSE last_message END
        WHERE queue_state = 'claimed'
        """
    )
    return int(cur.rowcount or 0)


def append_log_row(log_path: Path, row: Dict[str, object]) -> None:
    exists = log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ts", "year", "stock_code", "code_name", "pdf_path", "status", "message", "raw_json_path"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def rebuild_year_csvs(
    *,
    base_dir: Path,
    target_out_dir: Path,
    csv_name: str,
    years: Iterable[int],
) -> Dict[int, int]:
    conn = connect_db(target_out_dir / "task_queue.sqlite")
    try:
        counts: Dict[int, int] = {}
        for year in sorted({int(y) for y in years}):
            rows = conn.execute(
                """
                SELECT year, stock_code, pdf_path, raw_json_path, last_status
                FROM tasks
                WHERE year = ?
                  AND queue_state = 'done'
                  AND raw_json_path IS NOT NULL
                  AND raw_json_path <> ''
                  AND last_status IN ('ok', 'partial')
                ORDER BY stock_code
                """,
                (int(year),),
            ).fetchall()
            year_csv = base_dir / str(year) / str(csv_name).format(year=year)
            year_csv.parent.mkdir(parents=True, exist_ok=True)
            with year_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=YEAR_FIELDS)
                writer.writeheader()
                written = 0
                for row in rows:
                    raw_json_path = Path(str(row["raw_json_path"]))
                    if not raw_json_path.exists():
                        continue
                    try:
                        extracted = json.loads(raw_json_path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    task = Task(
                        year=int(row["year"]),
                        stock_code=str(row["stock_code"]),
                        pdf_path=Path(str(row["pdf_path"])),
                    )
                    writer.writerow(build_year_row(task, extracted))
                    written += 1
                counts[year] = written
        return counts
    finally:
        conn.close()


def migrate_into_target(
    *,
    base_dir: Path,
    target_out_dir: Path,
    candidates: Dict[Tuple[int, str], Candidate],
    reset_claimed: bool,
) -> Dict[str, object]:
    run_config = load_run_config(target_out_dir)
    csv_name = str(run_config.get("csv_name") or "{year}_qwen_v6_dynamic_hybrid.csv")
    queue_db = target_out_dir / "task_queue.sqlite"
    log_path = target_out_dir / "extract_log.csv"
    raw_root = target_out_dir / "raw_json"
    raw_root.mkdir(parents=True, exist_ok=True)

    conn = connect_db(queue_db)
    touched_years: set[int] = set()
    migrated = 0
    upgraded = 0
    skipped_missing_target = 0
    skipped_already_ok = 0
    skipped_missing_json = 0
    reset_count = 0

    try:
        conn.execute("BEGIN IMMEDIATE")
        if reset_claimed:
            reset_count = reset_claimed_tasks(conn)

        for (year, stock_code), candidate in sorted(candidates.items()):
            row = conn.execute(
                """
                SELECT year, stock_code, pdf_path, queue_state, last_status, raw_json_path
                FROM tasks
                WHERE year = ? AND stock_code = ?
                """,
                (int(year), str(stock_code)),
            ).fetchone()
            if row is None:
                skipped_missing_target += 1
                continue

            current_status = str(row["last_status"] or "").strip().lower()
            current_raw_json_path = Path(str(row["raw_json_path"] or "").strip()) if str(row["raw_json_path"] or "").strip() else None
            if current_status == "ok" and current_raw_json_path and current_raw_json_path.exists():
                skipped_already_ok += 1
                continue

            source_json_path = candidate.source_raw_json
            if not source_json_path.exists():
                skipped_missing_json += 1
                continue
            extracted = json.loads(source_json_path.read_text(encoding="utf-8"))

            target_raw_json_path = raw_root / str(year) / f"{stock_code}.json"
            target_raw_json_path.parent.mkdir(parents=True, exist_ok=True)
            target_raw_json_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

            migration_ts = candidate.source_ts or now_iso()
            message = candidate.source_message
            backend_tag = f"migrated:{candidate.source_dir.name}"
            conn.execute(
                """
                UPDATE tasks
                SET queue_state = 'done',
                    claimed_by = 'migrated',
                    claim_ts = ?,
                    claim_expires_ts = NULL,
                    done_ts = ?,
                    last_backend = ?,
                    last_status = 'ok',
                    last_message = ?,
                    raw_json_path = ?
                WHERE year = ? AND stock_code = ?
                """,
                (
                    migration_ts,
                    now_iso(),
                    backend_tag,
                    message,
                    str(target_raw_json_path),
                    int(year),
                    str(stock_code),
                ),
            )

            append_log_row(
                log_path,
                {
                    "ts": migration_ts,
                    "year": int(year),
                    "stock_code": str(stock_code),
                    "code_name": str(candidate.row.get("code_name") or ""),
                    "pdf_path": str(row["pdf_path"]),
                    "status": "ok",
                    "message": message,
                    "raw_json_path": str(target_raw_json_path),
                },
            )

            touched_years.add(int(year))
            migrated += 1
            if current_status and current_status != "ok":
                upgraded += 1

        conn.commit()
    finally:
        conn.close()

    year_counts = rebuild_year_csvs(
        base_dir=base_dir,
        target_out_dir=target_out_dir,
        csv_name=csv_name,
        years=touched_years,
    )

    summary = {
        "target_out_dir": str(target_out_dir),
        "migrated_ok_rows": migrated,
        "upgraded_from_non_ok": upgraded,
        "skipped_missing_target": skipped_missing_target,
        "skipped_already_ok": skipped_already_ok,
        "skipped_missing_json": skipped_missing_json,
        "reset_claimed_rows": reset_count,
        "touched_years": sorted(touched_years),
        "rebuilt_year_csv_counts": year_counts,
        "written_at": now_iso(),
    }
    summary_path = target_out_dir / "migration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate prior OK extraction results into a dynamic queue output directory.")
    parser.add_argument("--base-dir", default=".", help="Repository root.")
    parser.add_argument("--target-out-dir", required=True, help="Target dynamic queue output directory.")
    parser.add_argument(
        "--source-out-dir",
        action="append",
        default=[],
        help="Source output directory to scan for OK rows. May be repeated. Earlier entries have higher priority.",
    )
    parser.add_argument("--reset-claimed", action="store_true", help="Reset claimed rows back to pending before migration.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    target_out_dir = (Path(args.target_out_dir) if Path(args.target_out_dir).is_absolute() else (base_dir / args.target_out_dir)).resolve()
    source_dirs = [
        (Path(value) if Path(value).is_absolute() else (base_dir / value)).resolve()
        for value in (args.source_out_dir or [])
    ]
    if not source_dirs:
        raise RuntimeError("At least one --source-out-dir is required")

    candidates = choose_candidates(source_dirs)
    migrate_into_target(
        base_dir=base_dir,
        target_out_dir=target_out_dir,
        candidates=candidates,
        reset_claimed=bool(args.reset_claimed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
