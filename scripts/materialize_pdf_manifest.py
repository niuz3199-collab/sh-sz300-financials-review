#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def safe_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except Exception:
        shutil.copy2(str(src), str(dst))
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a PDF manifest into a flat input folder.")
    parser.add_argument("--manifest", required=True, help="CSV manifest path with at least pdf_path/year/stock_code columns.")
    parser.add_argument("--target-dir", required=True, help="Directory to place hardlinked/copied PDFs.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    seen_names: Dict[str, int] = {}
    manifest_rows: List[Dict[str, object]] = []
    mode_counts: Dict[str, int] = {}

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_path = Path(str(row.get("pdf_path") or "")).expanduser()
            if not pdf_path.exists():
                continue
            year = str(row.get("year") or "").strip()
            stock_code = str(row.get("stock_code") or "").strip()
            base_name = pdf_path.name
            if base_name in seen_names:
                seen_names[base_name] += 1
                file_name = f"{year}_{stock_code}_{seen_names[base_name]}_{base_name}"
            else:
                seen_names[base_name] = 1
                file_name = base_name
            out_path = target_dir / file_name
            mode = safe_link_or_copy(pdf_path, out_path)
            mode_counts[mode] = int(mode_counts.get(mode, 0)) + 1
            manifest_rows.append(
                {
                    "year": year,
                    "stock_code": stock_code,
                    "pdf_path": str(pdf_path),
                    "materialized_path": str(out_path),
                    "source": str(row.get("source") or ""),
                    "status": str(row.get("status") or ""),
                    "message": str(row.get("message") or ""),
                    "link_mode": mode,
                }
            )

    out_manifest = target_dir / "_materialized_manifest.csv"
    with out_manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "stock_code",
                "pdf_path",
                "materialized_path",
                "source",
                "status",
                "message",
                "link_mode",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "manifest": str(manifest_path),
        "target_dir": str(target_dir),
        "materialized_count": len(manifest_rows),
        "mode_counts": mode_counts,
        "materialized_manifest": str(out_manifest),
    }
    (target_dir / "_materialized_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
