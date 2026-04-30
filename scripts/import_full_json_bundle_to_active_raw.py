#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_bundle_name(path: Path) -> Tuple[int, str]:
    stem = path.stem
    year_text, stock_code = stem.split("_", 1)
    return int(year_text), stock_code.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Import flat JSON bundle into active raw_json tree without overwriting existing files.")
    parser.add_argument("--bundle-dir", default=str(REPO_ROOT / "全量json汇总_20260428_220018"))
    parser.add_argument("--target-raw-root", default=str(REPO_ROOT / ".tmp_gemma_markdown_financials_full" / "raw_json"))
    parser.add_argument("--runner-dir", default=str(REPO_ROOT / ".tmp_gemma_markdown_repair_runner"))
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    target_raw_root = Path(args.target_raw_root).resolve()
    runner_dir = Path(args.runner_dir).resolve()
    runner_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    manifest_rows: List[Dict[str, object]] = []

    for source_path in sorted(bundle_dir.glob("*.json"), key=lambda p: p.name):
        counts["bundle_total"] += 1

        try:
            year, stock_code = parse_bundle_name(source_path)
        except Exception:
            counts["invalid_filename"] += 1
            manifest_rows.append(
                {
                    "status": "invalid_filename",
                    "year": "",
                    "stock_code": "",
                    "source_path": str(source_path),
                    "target_path": "",
                    "message": "filename must be <year>_<stock_code>.json",
                }
            )
            continue

        target_path = target_raw_root / str(year) / f"{stock_code}.json"
        if target_path.exists():
            counts["skipped_existing"] += 1
            manifest_rows.append(
                {
                    "status": "skipped_existing",
                    "year": year,
                    "stock_code": stock_code,
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "message": "",
                }
            )
            continue

        try:
            extracted = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception as exc:
            counts["invalid_json"] += 1
            manifest_rows.append(
                {
                    "status": "invalid_json",
                    "year": year,
                    "stock_code": stock_code,
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "message": str(exc),
                }
            )
            continue

        write_json(target_path, extracted)
        counts["copied"] += 1
        manifest_rows.append(
            {
                "status": "copied",
                "year": year,
                "stock_code": stock_code,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "message": "",
            }
        )

    summary_payload = {
        "ts": now_iso(),
        "bundle_dir": str(bundle_dir),
        "target_raw_root": str(target_raw_root),
        "counts": dict(counts),
    }

    summary_json = runner_dir / "import_full_json_bundle_summary.json"
    manifest_csv = runner_dir / "import_full_json_bundle_manifest.csv"
    write_json(summary_json, summary_payload)
    write_csv(
        manifest_csv,
        manifest_rows,
        fieldnames=["status", "year", "stock_code", "source_path", "target_path", "message"],
    )

    print(f"[summary_json] {summary_json}")
    print(f"[manifest_csv] {manifest_csv}")
    print(json.dumps(summary_payload["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
