#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair partial/error results for Gemma Markdown financial extraction.

- partial: rerun only the missing fields
- error: rerun field-by-field so one malformed model response does not kill the whole task
- write repaired raw_json back into the existing output directory
- upsert yearly CSV rows by stock_code instead of appending duplicates
- append repair results to the same extract_log.csv so the existing monitor keeps working
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.step6_extract_financials_from_markdown import (
    MarkdownTask,
    Snippet,
    _coerce_field_payload,
    _null_field,
    _resolve_field_location,
    build_prompt_for_field,
    collect_markdown_tasks,
    now_iso,
    retrieve_snippets,
)
from scripts.step6_extract_financials_qwen_pdf import (
    infer_code_name,
    normalize_money_to_yuan,
    normalize_stock_code,
    normalize_total_shares_to_wan,
)


OPENAI_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
FIELD_ORDER = ["parent_netprofit", "total_shares", "operating_cashflow", "capex"]
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
FIELD_META = {
    "parent_netprofit": {"kind": "money", "normalized_key": "parent_netprofit_yuan", "page_key": "income"},
    "total_shares": {
        "kind": "shares",
        "normalized_key": "total_shares_shares",
        "normalized_aux_key": "total_shares_wan",
        "page_key": "shares",
    },
    "operating_cashflow": {"kind": "money", "normalized_key": "operating_cashflow_yuan", "page_key": "cfo"},
    "capex": {"kind": "money", "normalized_key": "capex_yuan", "page_key": "capex"},
}
MONEY_UNIT_PATTERNS = [
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币百万元", "百万元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币千元", "千元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币亿元", "亿元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币万元", "万元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币元", "人民币元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*百万元", "百万元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*千元", "千元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*亿元", "亿元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*万元", "万元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*元", "元"),
    (r"\bRMB['\s]*000\b", "RMB'000"),
    (r"\bthousand\s+RMB\b", "RMB'000"),
    (r"\bRMB\s+million\b", "RMB million"),
    (r"\bCNY\s+million\b", "RMB million"),
    (r"[（(]\s*亿元\s*[)）]", "亿元"),
    (r"[（(]\s*万元\s*[)）]", "万元"),
    (r"[（(]\s*元\s*[)）]", "元"),
]
SHARE_UNIT_PATTERNS = [
    (r"(?:单位|数量单位)\s*[:：]?\s*亿股", "亿股"),
    (r"(?:单位|数量单位)\s*[:：]?\s*万股", "万股"),
    (r"(?:单位|数量单位)\s*[:：]?\s*股", "股"),
    (r"\bmillion\s+shares?\b", "million shares"),
    (r"\bthousand\s+shares?\b", "thousand shares"),
    (r"\bshares?\b", "shares"),
]


def _resolve_openai_chat_url(api_base_url: str) -> str:
    base = str(api_base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("Missing --api-base-url")
    if base.endswith(OPENAI_CHAT_COMPLETIONS_SUFFIX):
        return base
    return f"{base}{OPENAI_CHAT_COMPLETIONS_SUFFIX}"


def load_run_config(out_dir: Path) -> Dict[str, object]:
    run_config = out_dir / "run_config.json"
    if not run_config.exists():
        return {}
    try:
        return json.loads(run_config.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_latest_rows(log_path: Path) -> Dict[Tuple[int, str], Dict[str, str]]:
    latest: Dict[Tuple[int, str], Dict[str, str]] = {}
    if not log_path.exists():
        return latest
    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                year = int(str(row.get("year") or "").strip())
            except Exception:
                continue
            code = normalize_stock_code(row.get("stock_code") or "")
            if not code:
                continue
            latest[(year, code)] = dict(row)
    return latest


def read_json_file(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ensure_extracted_skeleton(task: MarkdownTask) -> Dict[str, object]:
    pdf_path = str(task.pdf_path if task.pdf_path.exists() else task.md_path)
    return {
        "task": {
            "stock_code": task.stock_code,
            "year": task.year,
            "pdf_path": pdf_path,
            "markdown_path": str(task.md_path),
            "code_name": infer_code_name(task.stock_code),
        },
        "raw": {
            "code": task.stock_code,
            "year": task.year,
            "source_markdown_path": str(task.md_path),
            "parent_netprofit": dict(_null_field("missing_response"), page=None),
            "total_shares": dict(_null_field("missing_response"), page=None),
            "operating_cashflow": dict(_null_field("missing_response"), page=None),
            "capex": dict(_null_field("missing_response"), page=None),
        },
        "normalized": {
            "parent_netprofit_yuan": None,
            "total_shares_wan": None,
            "total_shares_shares": None,
            "operating_cashflow_yuan": None,
            "capex_yuan": None,
        },
        "pages": {
            "income": None,
            "shares": None,
            "cfo": None,
            "capex": None,
        },
    }


def extract_content_text(content: object) -> str:
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if str(part.get("type") or "").strip() == "text":
                    chunks.append(str(part.get("text") or ""))
                elif "text" in part:
                    chunks.append(str(part.get("text") or ""))
            else:
                chunks.append(str(part))
        return "".join(chunks).strip()
    return str(content or "").strip()


def parse_jsonish_token(token: str) -> Optional[str]:
    raw = str(token or "").strip()
    if not raw:
        return None
    if raw == "null":
        return None
    if raw.startswith('"') and raw.endswith('"'):
        try:
            return json.loads(raw)
        except Exception:
            return raw[1:-1]
    return raw


def extract_scalar_token(text: str, key: str) -> Optional[str]:
    quoted = re.search(rf'"{re.escape(key)}"\s*:\s*(null|"(?:[^"\\]|\\.)*")', text, flags=re.S)
    if quoted:
        return parse_jsonish_token(quoted.group(1))
    numeric = re.search(
        rf'"{re.escape(key)}"\s*:\s*([-+]?\d(?:[\d,\s\u00a0\u202f]*\d)?(?:\.\d+)?)',
        text,
        flags=re.S,
    )
    if numeric:
        return numeric.group(1).strip()
    return None


def extract_evidence_token(text: str) -> Optional[str]:
    match = re.search(r'"evidence"\s*:\s*"([^"]*)"', text, flags=re.S)
    if not match:
        return None
    raw = match.group(1)
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        return raw


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def canonicalize_unit(unit: Optional[str], *, field_kind: str) -> Optional[str]:
    raw = compact_whitespace(unit)
    if not raw:
        return None
    lower = raw.lower()
    if field_kind == "money":
        if raw in {"元", "人民币元", "万元", "亿元", "百万元", "千元"}:
            return raw
        if raw in {"rmb'000", "RMB'000"}:
            return "RMB'000"
        if lower in {"rmb million", "cny million"}:
            return "RMB million"
        if raw in {"Ԫ", "円"}:
            return "元"
        if "rmb" in lower or "cny" in lower or "yuan" in lower or "renminbi" in lower:
            if "000" in lower or "thousand" in lower:
                return "RMB'000"
            if "million" in lower:
                return "RMB million"
            return "元"
        if "亿元" in raw:
            return "亿元"
        if "万元" in raw:
            return "万元"
        if "百万元" in raw:
            return "百万元"
        if "千元" in raw:
            return "千元"
        if "元" in raw:
            return "元"
        return None

    if raw in {"股", "万股", "亿股"}:
        return raw
    if raw in {"shares", "share", "Shares", "Share"}:
        return "shares"
    if lower in {"million shares", "million share"}:
        return "million shares"
    if lower in {"thousand shares", "thousand share"}:
        return "thousand shares"
    if "亿股" in raw:
        return "亿股"
    if "万股" in raw:
        return "万股"
    if "股" in raw:
        return "股"
    if "share" in lower:
        if "million" in lower:
            return "million shares"
        if "thousand" in lower or "000" in lower:
            return "thousand shares"
        return "shares"
    return None


def parse_number(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    text = (
        text.replace(",", "")
        .replace("，", "")
        .replace("\u00a0", "")
        .replace("\u202f", "")
        .replace(" ", "")
    )
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3}){2,}", text):
        text = text.replace(".", "")
    if not text:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    return -out if negative else out


def infer_unit_from_snippets(field_name: str, snippets: Sequence[Snippet], value: Optional[str]) -> Optional[str]:
    field_kind = str(FIELD_META[field_name]["kind"])
    text = "\n".join(str(snippet.text or "") for snippet in snippets)
    if not text:
        return None
    patterns = SHARE_UNIT_PATTERNS if field_kind == "shares" else MONEY_UNIT_PATTERNS
    for pattern, unit in patterns:
        if re.search(pattern, text, flags=re.I):
            return unit
    if field_kind == "shares":
        num = parse_number(value)
        if num is not None and abs(num) > 1000000:
            return "股"
    return None


def sanitize_unit(unit: Optional[str], *, field_name: str, snippets: Sequence[Snippet], value: Optional[str]) -> Optional[str]:
    canonical = canonicalize_unit(unit, field_kind=str(FIELD_META[field_name]["kind"]))
    if canonical is not None:
        return canonical
    return infer_unit_from_snippets(field_name, snippets, value)


def normalize_field_value(field_name: str, payload: Dict[str, object], snippets: Sequence[Snippet]) -> Dict[str, object]:
    value = payload.get("value")
    unit = sanitize_unit(
        payload.get("unit"),
        field_name=field_name,
        snippets=snippets,
        value=value if value is None else str(value),
    )
    payload["unit"] = unit

    if field_name == "total_shares":
        wan = normalize_total_shares_to_wan(value, str(unit or ""))
        shares = (float(wan) * 10000.0) if wan is not None else None
        return {"total_shares_wan": wan, "total_shares_shares": shares}

    amount = normalize_money_to_yuan(value, str(unit or ""))
    if field_name == "capex" and amount is not None and amount < 0:
        amount = -amount
    return {str(FIELD_META[field_name]["normalized_key"]): amount}


def parse_model_response_content(*, content: object, field_name: str, snippets: Sequence[Snippet]) -> Dict[str, object]:
    text = extract_content_text(content)
    if not text:
        raise RuntimeError("empty_response_content")

    parse_candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.S | re.I)
    if fence:
        parse_candidates.append(fence.group(1))
    object_match = re.search(r"(\{.*\})", text, flags=re.S)
    if object_match:
        parse_candidates.append(object_match.group(1))

    for candidate in parse_candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        payload = data.get(field_name) if isinstance(data, dict) else None
        if payload is None and isinstance(data, dict) and {"value", "unit", "evidence", "snippet_ids"} & set(data.keys()):
            payload = data
        if payload is None:
            continue
        out = _coerce_field_payload(payload)
        out["unit"] = sanitize_unit(out.get("unit"), field_name=field_name, snippets=snippets, value=out.get("value"))
        snippet_ids = [str(x).upper() for x in (out.get("snippet_ids") or []) if str(x).strip()]
        out["snippet_ids"] = list(dict.fromkeys(snippet_ids))
        return out

    value = extract_scalar_token(text, "value")
    unit = extract_scalar_token(text, "unit")
    evidence = extract_evidence_token(text)
    snippet_ids = list(dict.fromkeys([m.upper() for m in re.findall(r"SNIPPET_\d+", text, flags=re.I)]))

    if value is None and '"value": null' not in text and "'value': null" not in text:
        raise RuntimeError("unable_to_salvage_value_from_non_json_response")

    payload = _coerce_field_payload({"value": value, "unit": unit, "evidence": evidence, "snippet_ids": snippet_ids})
    payload["unit"] = sanitize_unit(
        payload.get("unit"),
        field_name=field_name,
        snippets=snippets,
        value=payload.get("value"),
    )
    if payload.get("evidence") in (None, ""):
        if payload["snippet_ids"]:
            payload["evidence"] = f"[{payload['snippet_ids'][0]}] salvaged_non_json_response"
        else:
            payload["evidence"] = "salvaged_non_json_response"
    return payload


def call_model_once(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: Dict[str, object],
    api_base_url: str,
    api_key: str,
    timeout: int,
    schema_name: str,
    mode: str,
) -> object:
    url = _resolve_openai_chat_url(api_base_url)
    headers = {"Content-Type": "application/json"}
    if str(api_key or "").strip():
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 900,
        "stream": False,
    }
    if mode == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": response_schema},
        }

    response = requests.post(url, headers=headers, json=payload, timeout=(30, int(timeout)))
    if int(response.status_code) >= 400:
        body = str(response.text or "")
        if mode == "json_schema" and ("response_format" in body.lower() or "json_schema" in body.lower()):
            raise RuntimeError("response_format_not_supported")
        response.raise_for_status()

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"missing_choices: {json.dumps(data, ensure_ascii=False)[:300]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not str(content or "").strip() and mode == "plain_text":
        reasoning = message.get("reasoning_content")
        if str(reasoning or "").strip():
            content = reasoning
    if not str(content or "").strip():
        raise RuntimeError("empty_message_content")
    return content


def build_retry_prompt(base_user_prompt: str, attempt: int) -> str:
    if int(attempt) <= 1:
        return base_user_prompt
    suffix = [
        "重要补充要求：",
        "- 只输出 JSON 对象，不要输出任何解释、前缀或代码块。",
        "- evidence 必须尽量短，只写 snippet id 和极短行名，不要长句。",
        "- 如果 evidence 容易导致输出损坏，写成类似 \"[SNIPPET_2] 命中目标行\" 即可。",
    ]
    return f"{base_user_prompt}\n\n" + "\n".join(suffix)


def extract_one_field(
    *,
    markdown_text: str,
    task: MarkdownTask,
    field_name: str,
    model: str,
    api_base_url: str,
    api_key: str,
    timeout: int,
    max_snippets: int,
    max_chars_per_field: int,
    max_attempts: int,
) -> Tuple[Dict[str, object], List[Snippet], Optional[str]]:
    snippets = retrieve_snippets(
        markdown_text=markdown_text,
        field_name=field_name,
        year=task.year,
        max_snippets=max_snippets,
        max_chars=max_chars_per_field,
    )
    if not snippets:
        return _null_field("no_candidate_snippets"), [], "no_candidate_snippets"

    system_prompt, user_prompt, schema = build_prompt_for_field(year=task.year, field_name=field_name, snippets=snippets)

    best_payload: Optional[Dict[str, object]] = None
    last_error: Optional[str] = None
    for attempt in range(1, int(max_attempts) + 1):
        prompt = build_retry_prompt(user_prompt, attempt)
        for mode in ("json_schema", "plain_text"):
            try:
                content = call_model_once(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    response_schema=schema,
                    api_base_url=api_base_url,
                    api_key=api_key,
                    timeout=timeout,
                    schema_name=f"{field_name}_extract",
                    mode=mode,
                )
                payload = parse_model_response_content(content=content, field_name=field_name, snippets=snippets)
                normalized = normalize_field_value(field_name, payload, snippets)
                best_payload = payload
                if field_name == "total_shares":
                    if normalized.get("total_shares_shares") is not None:
                        return payload, snippets, None
                else:
                    if normalized.get(str(FIELD_META[field_name]["normalized_key"])) is not None:
                        return payload, snippets, None
            except Exception as exc:
                last_error = str(exc)
                if mode == "json_schema" and "response_format_not_supported" in str(exc):
                    continue
        if attempt < int(max_attempts):
            time.sleep(min(3.0, 0.8 * attempt))

    if best_payload is not None:
        if best_payload.get("evidence") in (None, ""):
            best_payload["evidence"] = "model_returned_null_or_incomplete_result"
        return best_payload, snippets, last_error

    return _null_field(f"field_error:{last_error or 'unknown'}"), snippets, last_error


def compute_missing_fields(extracted: Dict[str, object]) -> List[str]:
    normalized = extracted.get("normalized") or {}
    missing: List[str] = []
    if normalized.get("parent_netprofit_yuan") is None:
        missing.append("parent_netprofit")
    if normalized.get("total_shares_shares") is None:
        missing.append("total_shares")
    if normalized.get("operating_cashflow_yuan") is None:
        missing.append("operating_cashflow")
    if normalized.get("capex_yuan") is None:
        missing.append("capex")
    return missing


def apply_field_result(extracted: Dict[str, object], *, field_name: str, payload: Dict[str, object], snippets: Sequence[Snippet]) -> None:
    raw = extracted.setdefault("raw", {})
    raw_field = dict(_coerce_field_payload(payload))
    raw_field["page"] = _resolve_field_location(raw_field, snippets)
    raw[field_name] = raw_field

    pages = extracted.setdefault("pages", {})
    pages[str(FIELD_META[field_name]["page_key"])] = raw_field.get("page")

    normalized = extracted.setdefault("normalized", {})
    normalized.update(normalize_field_value(field_name, raw_field, snippets))


def build_year_csv_row(task: MarkdownTask, extracted: Dict[str, object]) -> Dict[str, object]:
    normalized = extracted.get("normalized") or {}
    raw = extracted.get("raw") or {}
    pdf_path = str(task.pdf_path if task.pdf_path.exists() else task.md_path)
    return {
        "year": task.year,
        "stock_code": task.stock_code,
        "code_name": infer_code_name(task.stock_code),
        "stock_name": "",
        "parent_netprofit": normalized.get("parent_netprofit_yuan"),
        "share_capital": normalized.get("total_shares_shares"),
        "share_capital_wan": normalized.get("total_shares_wan"),
        "netcash_operate": normalized.get("operating_cashflow_yuan"),
        "construct_long_asset": normalized.get("capex_yuan"),
        "pdf_path": pdf_path,
        "parent_netprofit_page": (raw.get("parent_netprofit") or {}).get("page"),
        "share_capital_page": (raw.get("total_shares") or {}).get("page"),
        "netcash_operate_page": (raw.get("operating_cashflow") or {}).get("page"),
        "construct_long_asset_page": (raw.get("capex") or {}).get("page"),
    }


def upsert_csv_row(path: Path, row: Dict[str, object], *, fieldnames: List[str], key_field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    row_key = normalize_stock_code(row.get(key_field) or "")
    replaced = False

    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for old in reader:
                old_key = normalize_stock_code(old.get(key_field) or "")
                if old_key == row_key:
                    if not replaced:
                        rows.append({k: row.get(k, "") for k in fieldnames})
                        replaced = True
                    continue
                rows.append({k: old.get(k, "") for k in fieldnames})

    if not replaced:
        rows.append({k: row.get(k, "") for k in fieldnames})

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow({k: item.get(k, "") for k in fieldnames})


def append_csv_row(path: Path, row: Dict[str, object], *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def resolve_csv_name(requested: str, run_config: Dict[str, object]) -> str:
    explicit = str(requested or "").strip()
    if explicit:
        return explicit
    cfg_value = str(run_config.get("csv_name") or "").strip()
    if cfg_value:
        return cfg_value
    return "{year}_财报数据_gemma_md.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair partial/error results for Gemma Markdown extraction")
    parser.add_argument("--base-dir", default=".", help="项目根目录")
    parser.add_argument("--markdown-root", default="", help="Marker Markdown 根目录；留空则从 run_config 读取")
    parser.add_argument("--out-dir", default=".tmp_gemma_markdown_financials_full", help="输出目录（raw_json + log）")
    parser.add_argument("--year-csv-root", default="", help="年度 CSV 根目录；留空则从 run_config 读取")
    parser.add_argument("--csv-name", default="", help="年度 CSV 文件名模板；留空则从 run_config 读取")
    parser.add_argument("--model", default="", help="模型名；留空则从 run_config 读取")
    parser.add_argument("--api-base-url", default="", help="OpenAI 兼容 API base URL；留空则从 run_config 读取")
    parser.add_argument("--api-key-env", default="", help="API key 环境变量；留空则从 run_config 读取")
    parser.add_argument("--timeout", type=int, default=0, help="单次请求超时；0 表示从 run_config 读取")
    parser.add_argument("--start-year", type=int, default=0, help="起始年份；0 表示从 run_config 读取")
    parser.add_argument("--end-year", type=int, default=0, help="结束年份；0 表示从 run_config 读取")
    parser.add_argument("--statuses", default="partial,error", help="要修复的最新状态，逗号分隔")
    parser.add_argument("--start", type=int, default=0, help="从第 N 个待修复任务开始")
    parser.add_argument("--limit", type=int, default=0, help="最多修复多少个任务（0=全部）")
    parser.add_argument("--max-snippets", type=int, default=0, help="每字段最多送模片段数；0 表示从 run_config 读取")
    parser.add_argument("--max-chars-per-field", type=int, default=0, help="每字段最大字符数；0 表示从 run_config 读取")
    parser.add_argument("--max-attempts", type=int, default=3, help="单字段最大修复尝试次数")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = (Path(args.out_dir) if Path(args.out_dir).is_absolute() else base_dir / args.out_dir).resolve()
    run_config = load_run_config(out_dir)

    markdown_root = (
        Path(args.markdown_root)
        if str(args.markdown_root or "").strip() and Path(args.markdown_root).is_absolute()
        else (base_dir / str(args.markdown_root or run_config.get("markdown_root") or ".cache/qwen_pdf_markdown_remaining/output_markdown"))
    ).resolve()
    year_csv_root = (
        Path(args.year_csv_root)
        if str(args.year_csv_root or "").strip() and Path(args.year_csv_root).is_absolute()
        else (base_dir / str(args.year_csv_root or run_config.get("year_csv_root") or ".tmp_gemma_year_csvs_full"))
    ).resolve()
    csv_name = resolve_csv_name(str(args.csv_name or ""), run_config)
    model = str(args.model or run_config.get("model") or "google/gemma-4-26b-a4b").strip()
    api_base_url = str(args.api_base_url or run_config.get("api_base_url") or "http://127.0.0.1:1234/v1").strip()
    api_key_env = str(args.api_key_env or run_config.get("api_key_env") or "LM_STUDIO_API_KEY").strip()
    api_key = str(os.environ.get(api_key_env, "") or "").strip() if api_key_env else ""
    timeout = int(args.timeout or run_config.get("timeout_seconds") or 180)
    start_year = int(args.start_year or run_config.get("start_year") or 2001)
    end_year = int(args.end_year or run_config.get("end_year") or 2025)
    max_snippets = int(args.max_snippets or run_config.get("max_snippets") or 6)
    max_chars_per_field = int(args.max_chars_per_field or run_config.get("max_chars_per_field") or 28000)
    target_statuses = {x.strip().lower() for x in str(args.statuses or "").split(",") if x.strip()}

    out_dir.mkdir(parents=True, exist_ok=True)
    year_csv_root.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "extract_log.csv"
    raw_root = out_dir / "raw_json"

    latest_rows = load_latest_rows(log_path)
    all_tasks = collect_markdown_tasks(markdown_root, base_dir=base_dir, start_year=start_year, end_year=end_year)
    task_map: Dict[Tuple[int, str], MarkdownTask] = {(task.year, task.stock_code): task for task in all_tasks}

    selected: List[Tuple[MarkdownTask, Dict[str, str]]] = []
    for key, row in sorted(latest_rows.items(), key=lambda item: item[0]):
        status = str(row.get("status") or "").strip().lower()
        if status not in target_statuses:
            continue
        task = task_map.get(key)
        if task is None:
            continue
        selected.append((task, row))

    start = max(0, int(args.start))
    end = len(selected) if int(args.limit) <= 0 else min(len(selected), start + int(args.limit))
    selected = selected[start:end]
    if not selected:
        raise RuntimeError(f"No tasks matched statuses={sorted(target_statuses)} under {markdown_root}")

    print(f"[repair] selected={len(selected)} statuses={','.join(sorted(target_statuses))}", flush=True)
    print(f"[markdown_root] {markdown_root}", flush=True)
    print(f"[out_dir] {out_dir}", flush=True)
    print(f"[year_csv_root] {year_csv_root}", flush=True)
    print(f"[backend] api_base_url={api_base_url} model={model} timeout={timeout}", flush=True)
    print(f"[retrieval] max_snippets={max_snippets} max_chars_per_field={max_chars_per_field} max_attempts={int(args.max_attempts)}", flush=True)

    ok = 0
    partial = 0
    error = 0

    for idx, (task, latest_row) in enumerate(selected, start=1):
        current_status = str(latest_row.get("status") or "").strip().lower()
        raw_json_path = raw_root / str(task.year) / f"{task.stock_code}.json"
        try:
            extracted = read_json_file(raw_json_path) if current_status == "partial" else None
            if not isinstance(extracted, dict):
                extracted = ensure_extracted_skeleton(task)

            task_meta = extracted.setdefault("task", {})
            task_meta.update(
                {
                    "stock_code": task.stock_code,
                    "year": task.year,
                    "pdf_path": str(task.pdf_path if task.pdf_path.exists() else task.md_path),
                    "markdown_path": str(task.md_path),
                    "code_name": infer_code_name(task.stock_code),
                }
            )
            extracted.setdefault("raw", {})
            extracted.setdefault("normalized", {})
            extracted.setdefault("pages", {})
            extracted["raw"]["code"] = task.stock_code
            extracted["raw"]["year"] = task.year
            extracted["raw"]["source_markdown_path"] = str(task.md_path)

            fields_to_rerun = compute_missing_fields(extracted) if current_status == "partial" else list(FIELD_ORDER)
            if not fields_to_rerun:
                fields_to_rerun = list(FIELD_ORDER)

            markdown_text = task.md_path.read_text(encoding="utf-8", errors="ignore")
            field_notes: List[str] = []

            for field_name in fields_to_rerun:
                payload, snippets, note = extract_one_field(
                    markdown_text=markdown_text,
                    task=task,
                    field_name=field_name,
                    model=model,
                    api_base_url=api_base_url,
                    api_key=api_key,
                    timeout=timeout,
                    max_snippets=max_snippets,
                    max_chars_per_field=max_chars_per_field,
                    max_attempts=int(args.max_attempts),
                )
                apply_field_result(extracted, field_name=field_name, payload=payload, snippets=snippets)
                if note:
                    field_notes.append(f"{field_name}:{note}")

            missing_fields = compute_missing_fields(extracted)
            status = "ok" if not missing_fields else "partial"
            message_parts = [
                f"repaired_from={current_status}",
                f"rerun_fields={','.join(fields_to_rerun)}",
            ]
            if missing_fields:
                message_parts.append(f"missing_fields={','.join(missing_fields)}")
            if field_notes:
                message_parts.append("notes=" + " | ".join(field_notes))
            message = "; ".join(message_parts)

            raw_json_path.parent.mkdir(parents=True, exist_ok=True)
            raw_json_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

            year_dir = year_csv_root / str(task.year)
            year_csv = year_dir / csv_name.format(year=task.year)
            upsert_csv_row(year_csv, build_year_csv_row(task, extracted), fieldnames=YEAR_FIELDS, key_field="stock_code")

            append_csv_row(
                log_path,
                {
                    "ts": now_iso(),
                    "year": task.year,
                    "stock_code": task.stock_code,
                    "code_name": infer_code_name(task.stock_code),
                    "pdf_path": str(task.pdf_path if task.pdf_path.exists() else task.md_path),
                    "status": status,
                    "message": message,
                    "raw_json_path": str(raw_json_path),
                },
                fieldnames=LOG_FIELDS,
            )

            if status == "ok":
                ok += 1
            else:
                partial += 1

            print(
                f"[{idx}/{len(selected)}] {task.year} {task.stock_code} {status} repaired_from={current_status} rerun={','.join(fields_to_rerun)}".rstrip(),
                flush=True,
            )
        except Exception as exc:
            error += 1
            append_csv_row(
                log_path,
                {
                    "ts": now_iso(),
                    "year": task.year,
                    "stock_code": task.stock_code,
                    "code_name": infer_code_name(task.stock_code),
                    "pdf_path": str(task.pdf_path if task.pdf_path.exists() else task.md_path),
                    "status": "error",
                    "message": f"repair_failed:{str(exc)}",
                    "raw_json_path": str(raw_json_path) if raw_json_path.exists() else "",
                },
                fieldnames=LOG_FIELDS,
            )
            print(f"[{idx}/{len(selected)}] {task.year} {task.stock_code} error {exc}", flush=True)

    print(f"[done] ok={ok} partial={partial} error={error} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
