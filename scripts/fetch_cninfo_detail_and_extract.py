#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.repair_gemma_markdown_financials import (
    call_model_once,
    normalize_field_value,
    parse_model_response_content,
)
from scripts.step6_extract_financials_from_markdown import Snippet, build_prompt_for_field, retrieve_snippets
from scripts.step6_extract_financials_qwen_pdf import infer_code_name


FIELD_ORDER = ["parent_netprofit", "total_shares", "operating_cashflow", "capex"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch CNINFO detail content and send it to local Gemma")
    parser.add_argument("--url", required=True, help="CNINFO detail URL")
    parser.add_argument("--out-dir", default=".tmp_cninfo_detail_extract", help="Output directory")
    parser.add_argument("--model", default="google/gemma-4-26b-a4b", help="LM Studio model name")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:1234/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key-env", default="LM_STUDIO_API_KEY", help="API key env name; LM Studio can be empty")
    parser.add_argument("--timeout", type=int, default=180, help="Single model request timeout")
    parser.add_argument("--max-snippets", type=int, default=6, help="Max snippets per field")
    parser.add_argument("--max-chars-per-field", type=int, default=12000, help="Max chars per field")
    return parser.parse_args()


def parse_cninfo_detail_url(url: str) -> Dict[str, str]:
    parsed = urlparse(str(url or "").strip())
    qs = parse_qs(parsed.query)
    stock_code = str((qs.get("stockCode") or [""])[0]).strip()
    announcement_id = str((qs.get("announcementId") or [""])[0]).strip()
    org_id = str((qs.get("orgId") or [""])[0]).strip()
    announcement_time = str((qs.get("announcementTime") or [""])[0]).strip()
    if not announcement_id:
        raise RuntimeError(f"Missing announcementId in URL: {url}")
    return {
        "stock_code": stock_code,
        "announcement_id": announcement_id,
        "org_id": org_id,
        "announcement_time": announcement_time,
    }


def fetch_detail_page(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.cninfo.com.cn/",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def infer_plate(html_text: str, org_id: str) -> str:
    match = re.search(r'var\s+plate\s*=\s*"([^"]+)"', html_text)
    if match:
        return str(match.group(1) or "").strip()
    org_id = str(org_id or "").lower()
    if org_id.startswith("gssz"):
        return "szse"
    if org_id.startswith("gssh"):
        return "sse"
    return "szse"


def fetch_bulletin_detail(*, announcement_id: str, announcement_time: str, plate: str) -> Dict[str, object]:
    flag = "true" if str(plate or "").strip().lower() == "szse" else "false"
    url = (
        "https://www.cninfo.com.cn/new/announcement/bulletin_detail"
        f"?announceId={announcement_id}&flag={flag}&announceTime={announcement_time}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.cninfo.com.cn/",
    }
    response = requests.post(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    announcement = data.get("announcement") or {}
    if not announcement:
        raise RuntimeError(f"Missing announcement payload for announceId={announcement_id}")
    return announcement


def clean_announcement_content(content: str) -> str:
    text = str(content or "")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def build_single_snippet(text: str) -> List[Snippet]:
    lines = text.splitlines()
    end_line = max(1, len(lines))
    snippet = Snippet(
        snippet_id="SNIPPET_1",
        start_line=1,
        end_line=end_line,
        score=100.0,
        text=text,
        matched_terms=("cninfo_bulletin_detail",),
    )
    return [snippet]


def retrieve_summary_snippets(*, text: str, year: int, field_name: str, max_snippets: int, max_chars: int) -> List[Snippet]:
    snippets = retrieve_snippets(
        markdown_text=text,
        field_name=field_name,
        year=year,
        max_snippets=max_snippets,
        max_chars=max_chars,
    )
    if snippets:
        return snippets
    return build_single_snippet(text[:max_chars])


def extract_with_gemma(
    *,
    year: int,
    text: str,
    model: str,
    api_base_url: str,
    api_key: str,
    timeout: int,
    max_snippets: int,
    max_chars_per_field: int,
) -> Dict[str, object]:
    raw: Dict[str, Dict[str, object]] = {}
    normalized: Dict[str, object] = {
        "parent_netprofit_yuan": None,
        "total_shares_wan": None,
        "total_shares_shares": None,
        "operating_cashflow_yuan": None,
        "capex_yuan": None,
    }

    for field_name in FIELD_ORDER:
        snippets = retrieve_summary_snippets(
            text=text,
            year=year,
            field_name=field_name,
            max_snippets=max_snippets,
            max_chars=max_chars_per_field,
        )
        system_prompt, user_prompt, schema = build_prompt_for_field(year=year, field_name=field_name, snippets=snippets)
        content = None
        last_error: Optional[str] = None
        for mode in ("json_schema", "plain_text"):
            try:
                content = call_model_once(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_schema=schema,
                    api_base_url=api_base_url,
                    api_key=api_key,
                    timeout=timeout,
                    schema_name=f"{field_name}_extract",
                    mode=mode,
                )
                payload = parse_model_response_content(content=content, field_name=field_name, snippets=snippets)
                payload["page"] = f"cninfo_lines:1-{snippets[0].end_line}"
                raw[field_name] = payload
                normalized.update(normalize_field_value(field_name, payload, snippets))
                break
            except Exception as exc:
                last_error = str(exc)
        else:
            raw[field_name] = {
                "value": None,
                "unit": None,
                "evidence": f"model_error:{last_error or 'unknown'}",
                "snippet_ids": [],
                "page": None,
            }

    return {
        "raw": raw,
        "normalized": normalized,
    }


def main() -> None:
    args = parse_args()
    detail = parse_cninfo_detail_url(args.url)
    out_dir = (Path(args.out_dir) if Path(args.out_dir).is_absolute() else (REPO_ROOT / args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    html_text = fetch_detail_page(args.url)
    plate = infer_plate(html_text, detail["org_id"])
    announcement = fetch_bulletin_detail(
        announcement_id=detail["announcement_id"],
        announcement_time=detail["announcement_time"],
        plate=plate,
    )
    title = str(announcement.get("announcementTitle") or "").strip()
    content_html = str(announcement.get("announcementContent") or "")
    cleaned_text = clean_announcement_content(content_html)

    announce_year = 0
    title_year = re.search(r"(\d{4})年", title)
    if title_year:
        announce_year = int(title_year.group(1))
    elif detail["announcement_time"][:4].isdigit():
        announce_year = int(detail["announcement_time"][:4]) - 1

    stock_code = detail["stock_code"]
    payload = {
        "source_url": args.url,
        "stock_code": stock_code,
        "code_name": infer_code_name(stock_code),
        "org_id": detail["org_id"],
        "plate": plate,
        "announcement_id": detail["announcement_id"],
        "announcement_time": detail["announcement_time"],
        "announcement_title": title,
        "adjunct_url": announcement.get("adjunctUrl"),
        "clean_text_path": str((out_dir / f"{stock_code}_{detail['announcement_id']}.txt").resolve()),
    }

    html_path = out_dir / f"{stock_code}_{detail['announcement_id']}.detail.html"
    json_path = out_dir / f"{stock_code}_{detail['announcement_id']}.announcement.json"
    text_path = out_dir / f"{stock_code}_{detail['announcement_id']}.txt"
    result_path = out_dir / f"{stock_code}_{detail['announcement_id']}.gemma_extract.json"

    html_path.write_text(html_text, encoding="utf-8")
    json_path.write_text(json.dumps(announcement, ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(cleaned_text, encoding="utf-8")

    api_key = str(os.environ.get(str(args.api_key_env or "").strip(), "") or "").strip()
    gemma_result = extract_with_gemma(
        year=int(announce_year),
        text=cleaned_text,
        model=str(args.model),
        api_base_url=str(args.api_base_url),
        api_key=api_key,
        timeout=int(args.timeout),
        max_snippets=int(args.max_snippets),
        max_chars_per_field=int(args.max_chars_per_field),
    )
    output = {
        "meta": payload,
        "year": announce_year,
        "gemma_result": gemma_result,
    }
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "title": title,
        "year": announce_year,
        "stock_code": stock_code,
        "text_path": str(text_path.resolve()),
        "result_path": str(result_path.resolve()),
        "normalized": gemma_result.get("normalized"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
