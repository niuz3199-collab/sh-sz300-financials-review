#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step6 (Markdown/Local-LLM): extract annual-report key fields from marker Markdown.

Why this exists
- Marker Markdown is often too large to fit into a local model context window in one shot.
- The four target fields are fixed, so it is more robust to do:
  1) deterministic keyword-based snippet retrieval
  2) small structured LLM extraction on the retrieved snippets only

What it does
- Traverse marker output folders under `output_markdown/*`
- For each annual-report Markdown:
  - retrieve a handful of small candidate snippets for each field
  - call an OpenAI-compatible local text model (for example LM Studio)
  - extract:
      1) parent net profit
      2) total shares
      3) operating cashflow
      4) CapEx
  - normalize units to the same schema used by the PDF step6 pipeline
- Save per-year CSV to `<base_dir>/<year>/{year}_财报数据_*.csv`
- Save raw JSON evidence to `<out-dir>/raw_json/<year>/<code>.json`
- Maintain `<out-dir>/extract_log.csv` for resume

Notes
- This script intentionally avoids sending the full Markdown into the model.
- It uses `response_format.type=json_schema`, which works well with LM Studio's local server.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.step6_extract_financials_qwen_pdf import (
    append_csv_row,
    infer_code_name,
    normalize_money_to_yuan,
    normalize_stock_code,
    normalize_total_shares_to_wan,
)


OPENAI_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


@dataclass(frozen=True)
class MarkdownTask:
    year: int
    stock_code: str
    report_name: str
    md_path: Path
    pdf_path: Path


@dataclass(frozen=True)
class Snippet:
    snippet_id: str
    start_line: int
    end_line: int
    score: float
    text: str
    matched_terms: Tuple[str, ...]


FIELD_SPECS: Dict[str, Dict[str, object]] = {
    "parent_netprofit": {
        "field_kind": "money",
        "specific_terms": {
            "归属于母公司股东的净利润": 12.0,
            "归属于母公司所有者的净利润": 12.0,
            "归属于本公司股东的净利润": 12.0,
            "归属于本行股东的净利润": 12.0,
            "归属于本集团股东的净利润": 12.0,
            "netprofitattributabletoshareholdersoftheparentcompany": 12.0,
            "profitattributabletoshareholdersoftheparentcompany": 12.0,
        },
        "fallback_terms": {
            "净利润": 3.5,
            "profitfortheyear": 4.0,
            "netprofit": 4.0,
            "利润表": 2.5,
            "合并利润表": 3.0,
            "利润及利润分配表": 3.0,
            "利润分配表": 2.0,
            "statementofincome": 2.0,
            "incomestatement": 2.0,
            "主要会计数据": 1.5,
            "财务摘要": 1.5,
        },
        "before_lines": 18,
        "after_lines": 28,
    },
    "total_shares": {
        "field_kind": "shares",
        "specific_terms": {
            "期末普通股股份总数": 12.0,
            "期末总股本": 12.0,
            "股份总数": 8.0,
            "总股本": 7.0,
            "实收资本": 5.0,
            "totalnumberofsharesatendofperiod": 12.0,
            "totalnumberofshares": 10.0,
            "totalsharecapital": 10.0,
            "sharecapital": 4.0,
            "amountattheyear-end": 8.0,
        },
        "fallback_terms": {
            "股本变动及股东情况": 5.0,
            "股份变动情况": 5.0,
            "股东情况": 2.5,
            "changesinsharecapital": 5.0,
            "statementofchangesinsharecapital": 5.0,
            "particularsabouttheshareholders": 4.0,
            "利润分配预案": 1.5,
        },
        "before_lines": 18,
        "after_lines": 30,
    },
    "operating_cashflow": {
        "field_kind": "money",
        "specific_terms": {
            "经营活动产生的现金流量净额": 12.0,
            "经营活动现金流量净额": 10.0,
            "netcashgeneratedfromoperatingactivities": 12.0,
            "netcashflowsfromoperatingactivities": 12.0,
            "netcashprovidedbyoperatingactivities": 12.0,
            "netcashinflowfromoperatingactivities": 12.0,
            "netcashoutflowfromoperatingactivities": 12.0,
            "netcash(outflow)/inflowfromoperatingactivities": 12.0,
            "netcash(inflow)/outflowfromoperatingactivities": 12.0,
        },
        "fallback_terms": {
            "现金流量表": 3.0,
            "合并现金流量表": 4.0,
            "statementofcashflows": 3.5,
            "cashflowstatement": 3.5,
        },
        "before_lines": 20,
        "after_lines": 32,
    },
    "capex": {
        "field_kind": "money",
        "specific_terms": {
            "购建固定资产、无形资产及其他长期资产支付的现金": 12.0,
            "购建固定资产、无形资产和其他长期资产支付的现金": 12.0,
            "paymentsfortheacquisitionandconstructionoffixedassets,intangibleassetsandotherlong-termassets": 12.0,
            "paymentsfortheacquisitionandconstructionoffixedassets,intangibleassetsandotherlongtermassets": 12.0,
            "purchasesofproperty,plantandequipment": 10.0,
            "purchaseofproperty,plantandequipment": 10.0,
            "paymentsforpurchaseoffixedassets": 10.0,
            "acquisitionoffixedassets": 10.0,
            "additionofconstructioninprogress": 8.0,
            "acquisitionofproperty,plantandequipment": 10.0,
        },
        "fallback_terms": {
            "投资活动产生的现金流量": 2.5,
            "现金流量表": 2.5,
            "合并现金流量表": 3.0,
            "statementofcashflows": 2.5,
            "cashflowstatement": 2.5,
        },
        "before_lines": 20,
        "after_lines": 36,
    },
}


def normalize_search_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<br\s*/?>", "", cleaned, flags=re.I)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", cleaned)
    return cleaned


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _resolve_openai_chat_url(api_base_url: str) -> str:
    base = str(api_base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("Missing --api-base-url")
    if base.endswith(OPENAI_CHAT_COMPLETIONS_SUFFIX):
        return base
    return f"{base}{OPENAI_CHAT_COMPLETIONS_SUFFIX}"


def _extract_json_from_content(content) -> Dict:
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
        content = "".join(chunks)

    text = str(content or "").strip()
    if not text:
        raise RuntimeError("empty_response_content")

    try:
        return json.loads(text)
    except Exception:
        fence = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.S | re.I)
        if fence:
            return json.loads(fence.group(1))
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if m:
            return json.loads(m.group(1))
        raise


def build_field_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "snippet_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["value", "unit", "evidence", "snippet_ids"],
        "additionalProperties": False,
    }


def call_openai_json_schema(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: Dict[str, object],
    api_base_url: str,
    api_key: str,
    timeout: int,
    schema_name: str,
) -> Dict:
    url = _resolve_openai_chat_url(api_base_url)
    headers = {"Content-Type": "application/json"}
    if str(api_key or "").strip():
        headers["Authorization"] = f"Bearer {api_key}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    attempts: List[Tuple[str, Optional[Dict[str, object]]]] = [
        (
            "json_schema",
            {
                "type": "json_schema",
                "json_schema": {
                    "name": str(schema_name or "extract_result"),
                    "schema": response_schema,
                },
            },
        ),
        ("plain_text", None),
    ]

    last_exc: Optional[BaseException] = None
    for attempt_index, (mode_name, response_format) in enumerate(attempts):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 900,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=(30, int(timeout)))
            if int(r.status_code) >= 400 and response_format is not None:
                response_text = str(r.text or "")
                if "response_format" in response_text.lower() or "json_schema" in response_text.lower():
                    time.sleep(1.5 + attempt_index)
                    continue
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"missing_choices: {json.dumps(data, ensure_ascii=False)[:300]}")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if not str(content or "").strip():
                reasoning = message.get("reasoning_content")
                if str(reasoning or "").strip() and mode_name == "plain_text":
                    content = reasoning
            return _extract_json_from_content(content)
        except Exception as exc:
            last_exc = exc
            if attempt_index < len(attempts) - 1:
                time.sleep(2.0 + attempt_index)
                continue
            break

    raise RuntimeError(f"Model returned non-JSON content: {str(last_exc)[:300]!r}") from last_exc


def _parse_code_year_from_name(name: str) -> Tuple[str, Optional[int]]:
    m = re.search(r"(?P<code>\d{6}).*?(?P<year>\d{4})", str(name or ""))
    if not m:
        return "", None
    return normalize_stock_code(m.group("code")), int(m.group("year"))


def _guess_original_pdf_path(base_dir: Path, report_name: str, year: int) -> Path:
    pdf_root = base_dir / "年报" / "下载年报_fulltext" / str(int(year))
    return pdf_root / f"{report_name}.pdf"


def collect_markdown_tasks(markdown_root: Path, *, base_dir: Path, start_year: int, end_year: int) -> List[MarkdownTask]:
    if not markdown_root.exists():
        raise FileNotFoundError(f"Markdown root not found: {markdown_root}")

    tasks: Dict[Tuple[int, str], MarkdownTask] = {}
    for child in sorted([p for p in markdown_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        md_files = sorted(list(child.glob("*.md")) + list(child.glob("*.markdown")))
        if not md_files:
            continue
        md_path = md_files[0]
        stock_code, year = _parse_code_year_from_name(md_path.stem)
        if not stock_code or year is None:
            stock_code, year = _parse_code_year_from_name(child.name)
        if not stock_code or year is None:
            continue
        if int(year) < int(start_year) or int(year) > int(end_year):
            continue
        report_name = md_path.stem
        task = MarkdownTask(
            year=int(year),
            stock_code=stock_code,
            report_name=report_name,
            md_path=md_path,
            pdf_path=_guess_original_pdf_path(base_dir, report_name, year),
        )
        key = (task.year, task.stock_code)
        cur = tasks.get(key)
        if cur is None:
            tasks[key] = task
            continue
        try:
            if task.md_path.stat().st_size > cur.md_path.stat().st_size:
                tasks[key] = task
        except Exception:
            pass

    out = list(tasks.values())
    out.sort(key=lambda t: (t.year, t.stock_code))
    return out


def _find_previous_heading_line(lines: Sequence[str], start_line: int, *, max_backtrack: int = 20) -> int:
    lo = max(0, int(start_line) - int(max_backtrack))
    for idx in range(int(start_line), lo - 1, -1):
        line = str(lines[idx] or "").lstrip()
        if line.startswith("#"):
            return idx
    return int(start_line)


def _collect_raw_spans(
    lines: Sequence[str],
    term_weights: Dict[str, float],
    *,
    before_lines: int,
    after_lines: int,
    max_hits_per_term: int = 16,
) -> List[Tuple[int, int, float, Tuple[str, ...]]]:
    normalized_lines = [normalize_search_text(line) for line in lines]
    spans: List[Tuple[int, int, float, Tuple[str, ...]]] = []
    hit_counts: Dict[str, int] = {}
    for idx, norm_line in enumerate(normalized_lines):
        matched_terms: List[str] = []
        score = 0.0
        for term, weight in term_weights.items():
            norm_term = normalize_search_text(term)
            if not norm_term:
                continue
            if norm_term in norm_line:
                current = hit_counts.get(term, 0)
                if current >= int(max_hits_per_term):
                    continue
                hit_counts[term] = current + 1
                matched_terms.append(term)
                score += float(weight)
        if not matched_terms:
            continue
        start = max(0, idx - int(before_lines))
        end = min(len(lines), idx + int(after_lines) + 1)
        start = _find_previous_heading_line(lines, start)
        spans.append((start, end, score, tuple(sorted(set(matched_terms)))))
    return spans


def _merge_spans(
    lines: Sequence[str],
    spans: Sequence[Tuple[int, int, float, Tuple[str, ...]]],
    *,
    merge_gap_lines: int = 8,
) -> List[Snippet]:
    if not spans:
        return []
    spans_sorted = sorted(spans, key=lambda item: (item[0], item[1]))
    merged: List[Tuple[int, int, float, set[str]]] = []
    for start, end, score, matched_terms in spans_sorted:
        if not merged:
            merged.append((start, end, float(score), set(matched_terms)))
            continue
        cur_start, cur_end, cur_score, cur_terms = merged[-1]
        if int(start) <= int(cur_end) + int(merge_gap_lines):
            merged[-1] = (
                min(cur_start, start),
                max(cur_end, end),
                float(cur_score) + float(score),
                set(cur_terms).union(set(matched_terms)),
            )
        else:
            merged.append((start, end, float(score), set(matched_terms)))

    out: List[Snippet] = []
    for idx, (start, end, score, matched_terms) in enumerate(merged, start=1):
        snippet_lines = lines[int(start) : int(end)]
        text = "\n".join(snippet_lines).strip()
        if not text:
            continue
        numeric_bonus = 0.0
        if re.search(r"\d", text):
            numeric_bonus += 1.0
        if re.search(r"(元|万元|亿元|股|万股|亿股|rmb|cny|shares?)", text, flags=re.I):
            numeric_bonus += 1.0
        out.append(
            Snippet(
                snippet_id=f"SNIPPET_{idx}",
                start_line=int(start) + 1,
                end_line=int(end),
                score=float(score) + numeric_bonus,
                text=text,
                matched_terms=tuple(sorted(matched_terms)),
            )
        )
    return out


def retrieve_snippets(
    *,
    markdown_text: str,
    field_name: str,
    year: int,
    max_snippets: int,
    max_chars: int,
) -> List[Snippet]:
    spec = FIELD_SPECS[field_name]
    lines = markdown_text.splitlines()
    before_lines = int(spec.get("before_lines") or 16)
    after_lines = int(spec.get("after_lines") or 24)

    specific_terms = dict(spec.get("specific_terms") or {})
    fallback_terms = dict(spec.get("fallback_terms") or {})

    spans = _collect_raw_spans(
        lines,
        specific_terms,
        before_lines=before_lines,
        after_lines=after_lines,
    )

    if int(year) <= 2006 and field_name == "parent_netprofit":
        legacy_terms = {
            "净利润": 4.0,
            "利润及利润分配表": 4.0,
            "利润分配表": 3.0,
            "主要会计数据": 2.0,
            "majoraccountingdata": 2.0,
            "financialhighlights": 2.0,
        }
        spans.extend(
            _collect_raw_spans(
                lines,
                legacy_terms,
                before_lines=before_lines + 4,
                after_lines=after_lines + 6,
            )
        )
    elif int(year) <= 2006 and field_name == "total_shares":
        legacy_terms = {
            "股本变动及股东情况": 6.0,
            "股份变动情况": 6.0,
            "amountattheyear-end": 8.0,
            "totalsharecapital": 8.0,
            "sharecapital": 3.0,
        }
        spans.extend(
            _collect_raw_spans(
                lines,
                legacy_terms,
                before_lines=before_lines + 4,
                after_lines=after_lines + 6,
            )
        )
    elif int(year) <= 2006 and field_name == "capex":
        legacy_terms = {
            "purchasesofproperty,plantandequipment": 10.0,
            "purchaseofproperty,plantandequipment": 10.0,
            "paymentsforpurchaseoffixedassets": 10.0,
        }
        spans.extend(
            _collect_raw_spans(
                lines,
                legacy_terms,
                before_lines=before_lines,
                after_lines=after_lines,
            )
        )

    if not spans:
        spans.extend(
            _collect_raw_spans(
                lines,
                fallback_terms,
                before_lines=before_lines,
                after_lines=after_lines,
            )
        )
    else:
        spans.extend(
            _collect_raw_spans(
                lines,
                fallback_terms,
                before_lines=max(8, before_lines // 2),
                after_lines=max(12, after_lines // 2),
                max_hits_per_term=6,
            )
        )

    snippets = _merge_spans(lines, spans)
    snippets = sorted(
        snippets,
        key=lambda s: (-float(s.score), len(s.text), s.start_line, s.end_line),
    )

    selected: List[Snippet] = []
    total_chars = 0
    for snippet in snippets:
        text_len = len(str(snippet.text or ""))
        if text_len <= 0:
            continue
        if selected and total_chars + text_len > int(max_chars):
            continue
        selected.append(snippet)
        total_chars += text_len
        if len(selected) >= int(max_snippets):
            break

    if selected:
        return selected

    fallback_text = "\n".join(lines[: min(len(lines), 200)]).strip()
    if not fallback_text:
        return []
    return [
        Snippet(
            snippet_id="SNIPPET_1",
            start_line=1,
            end_line=min(len(lines), 200),
            score=0.0,
            text=fallback_text,
            matched_terms=tuple(),
        )
    ]


def build_prompt_for_field(
    *,
    year: int,
    field_name: str,
    snippets: Sequence[Snippet],
) -> Tuple[str, str, Dict[str, object]]:
    field_schema = build_field_schema()
    top_schema = {
        "type": "object",
        "properties": {
            field_name: field_schema,
        },
        "required": [field_name],
        "additionalProperties": False,
    }

    system_prompt = (
        "你是一名严格的财务报表信息抽取助手。"
        "你将看到从上市公司年报 Markdown 中召回的若干文本片段。"
        "你只能依据这些片段抽取字段，不要猜测，不要补充。"
        "如果片段没有明确给出，就返回 null。"
        "evidence 中必须引用 snippet id。"
    )

    if field_name == "parent_netprofit":
        task_prompt = f"""
你现在只提取一个字段：parent_netprofit。
目标是本报告年度（{year}年）的“归属于…股东/所有者的净利润”。

可接受的口径：
- 归属于母公司股东的净利润
- 归属于母公司所有者的净利润
- 归属于本公司股东的净利润
- 归属于本行股东的净利润
- 归属于本集团股东的净利润
- 英文报表中的 Net profit attributable to shareholders of the parent company / Profit attributable to shareholders of the parent company

若年份较早（尤其 2006 年及以前）且不存在归母口径：
- 允许使用利润表、利润分配表、利润及利润分配表或主要会计数据中的“净利润”作为 fallback；
- 但不能把“利润总额”“未分配利润”“扣除非经常性损益后的净利润”当成结果。

只取 {year} 年这一列，不要取上年对比列。
返回字段：
- value: 原文数值字符串，保留负号、括号或逗号
- unit: 原文单位；若片段未明确给出可为 null
- evidence: 必须引用 snippet id，并简要说明命中的行名
- snippet_ids: 使用到的 snippet id 列表
""".strip()
    elif field_name == "total_shares":
        task_prompt = f"""
你现在只提取一个字段：total_shares。
目标是截至 {year} 年 12 月 31 日的期末总股本。

优先口径：
- 总股本
- 期末总股本
- 期末普通股股份总数
- 股份总数
- 实收资本（仅当明确表示期末股本时）
- 英文中的 Total share capital / Total number of shares / Amount at the year-end

排除项：
- 历史发行情况中的旧年度总股本
- 股东持股数、高管持股、内部职工股
- 每股收益公式里的“期末股份总数”
- 期初数、加权平均股数

返回字段：
- value: 原文数值字符串
- unit: 原文单位；尽量给出 股 / 万股 / 亿股 / shares 等
- evidence: 必须引用 snippet id，并简要说明命中的行名
- snippet_ids: 使用到的 snippet id 列表
""".strip()
    elif field_name == "operating_cashflow":
        task_prompt = f"""
你现在只提取一个字段：operating_cashflow。
目标是本报告年度（{year}年）的“经营活动产生的现金流量净额”。

英文可接受写法包括：
- Net cash generated from operating activities
- Net cash flows from operating activities
- Net cash provided by operating activities

只取 {year} 年这一列，不要取上年对比列。
必须来自现金流量表主表，不要使用补充资料或调节表。
返回字段：
- value: 原文数值字符串
- unit: 原文单位
- evidence: 必须引用 snippet id，并简要说明命中的行名
- snippet_ids: 使用到的 snippet id 列表
""".strip()
    elif field_name == "capex":
        task_prompt = f"""
你现在只提取一个字段：capex。
目标是本报告年度（{year}年）的资本支出现金流，优先口径为：
- 购建固定资产、无形资产及其他长期资产支付的现金
- 购建固定资产、无形资产和其他长期资产支付的现金

旧版英文报表可接受 fallback：
- Payments for the acquisition and construction of fixed assets, intangible assets and other long-term assets
- Purchases of property, plant and equipment
- Purchase of property, plant and equipment
- Payments for purchase of fixed assets
- Acquisition of fixed assets
- Acquisition of property, plant and equipment
- Addition of construction in progress（若明显代表当年资本开支现金流）

只取 {year} 年这一列，不要取上年对比列。
必须来自现金流量表主表，不要使用补充资料。
返回字段：
- value: 原文数值字符串
- unit: 原文单位
- evidence: 必须引用 snippet id，并简要说明命中的行名
- snippet_ids: 使用到的 snippet id 列表
""".strip()
    else:
        raise RuntimeError(f"Unsupported field_name: {field_name}")

    snippet_parts: List[str] = [task_prompt, "", "以下是召回片段：", ""]
    for snippet in snippets:
        snippet_parts.append(
            f"[{snippet.snippet_id}] lines {snippet.start_line}-{snippet.end_line} matched={','.join(snippet.matched_terms) or '-'}"
        )
        snippet_parts.append(snippet.text)
        snippet_parts.append("")

    return system_prompt, "\n".join(snippet_parts).strip(), top_schema


def _coerce_field_payload(payload: object) -> Dict[str, object]:
    if isinstance(payload, dict):
        out = dict(payload)
    elif payload in (None, ""):
        out = {}
    else:
        out = {"value": payload}
    out.setdefault("value", None)
    out.setdefault("unit", None)
    out.setdefault("evidence", None)
    out.setdefault("snippet_ids", [])
    if not isinstance(out.get("snippet_ids"), list):
        out["snippet_ids"] = [str(out.get("snippet_ids"))] if out.get("snippet_ids") else []
    return out


def _null_field(reason: str) -> Dict[str, object]:
    return {"value": None, "unit": None, "evidence": reason, "snippet_ids": []}


def _resolve_field_location(field_payload: Dict[str, object], snippets: Sequence[Snippet]) -> Optional[str]:
    snippet_ids = {str(x).upper() for x in (field_payload.get("snippet_ids") or []) if str(x).strip()}
    resolved = None
    if snippet_ids:
        for snippet in snippets:
            if str(snippet.snippet_id).upper() in snippet_ids:
                resolved = snippet
                break
    if resolved is None:
        evidence = str(field_payload.get("evidence") or "")
        matches = {x.upper() for x in re.findall(r"SNIPPET_\d+", evidence, flags=re.I)}
        for snippet in snippets:
            if str(snippet.snippet_id).upper() in matches:
                resolved = snippet
                break
    if resolved is None and snippets and field_payload.get("value") not in (None, ""):
        resolved = snippets[0]
    if resolved is not None:
        return f"md_lines:{resolved.start_line}-{resolved.end_line}"
    return None


def extract_from_markdown(
    task: MarkdownTask,
    *,
    model: str,
    api_base_url: str,
    api_key: str,
    timeout: int,
    max_snippets: int,
    max_chars_per_field: int,
    debug: bool,
) -> Dict[str, object]:
    markdown_text = task.md_path.read_text(encoding="utf-8", errors="ignore")

    snippets_by_field: Dict[str, List[Snippet]] = {}
    raw_fields: Dict[str, Dict[str, object]] = {}

    for field_name in ("parent_netprofit", "total_shares", "operating_cashflow", "capex"):
        snippets = retrieve_snippets(
            markdown_text=markdown_text,
            field_name=field_name,
            year=task.year,
            max_snippets=max_snippets,
            max_chars=max_chars_per_field,
        )
        snippets_by_field[field_name] = snippets
        if not snippets:
            raw_fields[field_name] = _null_field("no_candidate_snippets")
            continue

        system_prompt, user_prompt, schema = build_prompt_for_field(
            year=task.year,
            field_name=field_name,
            snippets=snippets,
        )
        response = call_openai_json_schema(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=schema,
            api_base_url=api_base_url,
            api_key=api_key,
            timeout=timeout,
            schema_name=f"{field_name}_extract",
        )
        raw_fields[field_name] = _coerce_field_payload(response.get(field_name))

    parent_netprofit = raw_fields.get("parent_netprofit") or _null_field("missing_response")
    total_shares = raw_fields.get("total_shares") or _null_field("missing_response")
    operating_cashflow = raw_fields.get("operating_cashflow") or _null_field("missing_response")
    capex = raw_fields.get("capex") or _null_field("missing_response")

    net_profit_yuan = normalize_money_to_yuan(parent_netprofit.get("value"), str(parent_netprofit.get("unit") or ""))
    operating_cashflow_yuan = normalize_money_to_yuan(
        operating_cashflow.get("value"),
        str(operating_cashflow.get("unit") or ""),
    )
    capex_yuan = normalize_money_to_yuan(capex.get("value"), str(capex.get("unit") or ""))
    if capex_yuan is not None and capex_yuan < 0:
        capex_yuan = -capex_yuan

    total_shares_wan = normalize_total_shares_to_wan(total_shares.get("value"), str(total_shares.get("unit") or ""))
    total_shares_shares = (float(total_shares_wan) * 10000.0) if total_shares_wan is not None else None

    raw = {
        "code": task.stock_code,
        "year": task.year,
        "source_markdown_path": str(task.md_path),
        "parent_netprofit": dict(
            parent_netprofit,
            page=_resolve_field_location(parent_netprofit, snippets_by_field.get("parent_netprofit") or []),
        ),
        "total_shares": dict(
            total_shares,
            page=_resolve_field_location(total_shares, snippets_by_field.get("total_shares") or []),
        ),
        "operating_cashflow": dict(
            operating_cashflow,
            page=_resolve_field_location(operating_cashflow, snippets_by_field.get("operating_cashflow") or []),
        ),
        "capex": dict(
            capex,
            page=_resolve_field_location(capex, snippets_by_field.get("capex") or []),
        ),
    }
    if bool(debug):
        raw["retrieval_debug"] = {
            field_name: [
                {
                    "snippet_id": s.snippet_id,
                    "lines": [s.start_line, s.end_line],
                    "score": s.score,
                    "matched_terms": list(s.matched_terms),
                }
                for s in snippets
            ]
            for field_name, snippets in snippets_by_field.items()
        }

    return {
        "task": {
            "stock_code": task.stock_code,
            "year": task.year,
            "pdf_path": str(task.pdf_path if task.pdf_path.exists() else task.md_path),
            "markdown_path": str(task.md_path),
            "code_name": infer_code_name(task.stock_code),
        },
        "raw": raw,
        "normalized": {
            "parent_netprofit_yuan": net_profit_yuan,
            "total_shares_wan": total_shares_wan,
            "total_shares_shares": total_shares_shares,
            "operating_cashflow_yuan": operating_cashflow_yuan,
            "capex_yuan": capex_yuan,
        },
        "pages": {
            "income": raw["parent_netprofit"].get("page"),
            "shares": raw["total_shares"].get("page"),
            "cfo": raw["operating_cashflow"].get("page"),
            "capex": raw["capex"].get("page"),
        },
    }


def write_run_config(
    out_dir: Path,
    *,
    model: str,
    api_base_url: str,
    markdown_root: Path,
    year_csv_root: Path,
    csv_name: str,
    start_year: int,
    end_year: int,
    timeout: int,
    max_snippets: int,
    max_chars_per_field: int,
    api_key_env: str,
) -> None:
    payload = {
        "backend": "openai_text_markdown_chunks",
        "model": str(model or "").strip(),
        "api_base_url": str(api_base_url or "").strip(),
        "markdown_root": str(markdown_root),
        "year_csv_root": str(year_csv_root),
        "csv_name": str(csv_name),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "timeout_seconds": int(timeout),
        "max_snippets": int(max_snippets),
        "max_chars_per_field": int(max_chars_per_field),
        "api_key_env": str(api_key_env or "").strip(),
        "written_at": now_iso(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_done_set_from_log(log_path: Path) -> set[Tuple[int, str]]:
    if not log_path.exists():
        return set()
    done: set[Tuple[int, str]] = set()
    try:
        import csv

        with log_path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    year = int(str(row.get("year") or "").strip())
                except Exception:
                    continue
                code = normalize_stock_code(row.get("stock_code") or "")
                status = str(row.get("status") or "").strip().lower()
                if code and status in {"ok", "partial"}:
                    done.add((year, code))
    except Exception:
        return set()
    return done


def load_done_codes_from_year_csv(year_csv: Path) -> set[str]:
    if not year_csv.exists():
        return set()
    done: set[str] = set()
    try:
        import csv

        with year_csv.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            if "stock_code" not in (r.fieldnames or []):
                return set()
            for row in r:
                code = normalize_stock_code(row.get("stock_code") or "")
                if code:
                    done.add(code)
    except Exception:
        return set()
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Step6 (Markdown/Local-LLM): 从 marker Markdown 中提取年报关键字段")
    parser.add_argument("--base-dir", default=".", help="项目根目录")
    parser.add_argument(
        "--markdown-root",
        default=".cache/qwen_pdf_markdown_remaining/output_markdown",
        help="Marker Markdown 输出目录",
    )
    parser.add_argument("--out-dir", default=".cache/gemma_markdown_financials", help="输出目录（raw_json + log）")
    parser.add_argument("--model", default="google/gemma-4-26b-a4b", help="OpenAI 兼容接口中的模型名")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:1234/v1", help="OpenAI 兼容 API base URL")
    parser.add_argument("--api-key-env", default="LM_STUDIO_API_KEY", help="API key 环境变量名；本地 LM Studio 可留空")
    parser.add_argument("--timeout", type=int, default=180, help="单次模型请求超时（秒）")
    parser.add_argument("--start-year", type=int, default=2001, help="起始年份（含）")
    parser.add_argument("--end-year", type=int, default=2025, help="结束年份（含）")
    parser.add_argument("--year-csv-root", default="", help="年度 CSV 根目录；默认写回 base-dir/<year>/")
    parser.add_argument(
        "--csv-name",
        default="{year}_财报数据_gemma_md.csv",
        help="写入年度目录的 CSV 文件名模板",
    )
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过日志/CSV中已完成任务")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在 raw_json 和 CSV 记录")
    parser.add_argument("--start", type=int, default=0, help="从第 N 个任务开始")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个任务（0=全部）")
    parser.add_argument("--sleep", type=float, default=0.0, help="每个任务完成后额外 sleep 秒数")
    parser.add_argument("--jitter", type=float, default=0.0, help="sleep 随机抖动秒数")
    parser.add_argument("--max-snippets", type=int, default=6, help="每个字段最多送模多少个片段")
    parser.add_argument("--max-chars-per-field", type=int, default=28000, help="每个字段送模的最大字符数")
    parser.add_argument("--debug", action="store_true", help="在 raw_json 中记录召回片段信息")
    args = parser.parse_args()

    import random

    base_dir = Path(args.base_dir).expanduser().resolve()
    markdown_root = (
        Path(args.markdown_root) if Path(args.markdown_root).is_absolute() else (base_dir / args.markdown_root)
    ).resolve()
    out_dir = (Path(args.out_dir) if Path(args.out_dir).is_absolute() else (base_dir / args.out_dir)).resolve()
    year_csv_root = (
        (Path(args.year_csv_root) if Path(args.year_csv_root).is_absolute() else (base_dir / args.year_csv_root)).resolve()
        if str(args.year_csv_root or "").strip()
        else base_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key_env = str(args.api_key_env or "").strip()
    api_key = str(os.environ.get(api_key_env, "") or "").strip() if api_key_env else ""

    log_path = out_dir / "extract_log.csv"
    raw_root = out_dir / "raw_json"

    write_run_config(
        out_dir,
        model=str(args.model),
        api_base_url=str(args.api_base_url),
        markdown_root=markdown_root,
        year_csv_root=year_csv_root,
        csv_name=str(args.csv_name),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        timeout=int(args.timeout),
        max_snippets=int(args.max_snippets),
        max_chars_per_field=int(args.max_chars_per_field),
        api_key_env=api_key_env,
    )

    tasks = collect_markdown_tasks(
        markdown_root,
        base_dir=base_dir,
        start_year=int(args.start_year),
        end_year=int(args.end_year),
    )
    if not tasks:
        raise RuntimeError(f"No markdown tasks found under {markdown_root}")

    done_from_log: set[Tuple[int, str]] = set()
    if bool(args.resume) and not bool(args.overwrite):
        done_from_log = load_done_set_from_log(log_path)

    start = max(0, int(args.start))
    end = len(tasks) if int(args.limit) <= 0 else min(len(tasks), start + int(args.limit))
    tasks = tasks[start:end]

    print(f"[tasks] total={len(tasks)} markdown_root={markdown_root}", flush=True)
    print(f"[out] out_dir={out_dir}", flush=True)
    print(f"[log] {log_path}", flush=True)
    print(f"[year_csv_root] {year_csv_root}", flush=True)
    print(f"[backend] api_base_url={args.api_base_url} model={args.model} timeout={int(args.timeout)}", flush=True)
    print(
        f"[retrieval] max_snippets={int(args.max_snippets)} max_chars_per_field={int(args.max_chars_per_field)}",
        flush=True,
    )

    log_fields = [
        "ts",
        "year",
        "stock_code",
        "code_name",
        "pdf_path",
        "status",
        "message",
        "raw_json_path",
    ]

    year_done_cache: Dict[int, set[str]] = {}
    ok = 0
    partial = 0
    skip = 0
    fail = 0

    for idx, task in enumerate(tasks, start=1):
        if (task.year, task.stock_code) in done_from_log:
            skip += 1
            continue

        year_dir = year_csv_root / str(task.year)
        year_dir.mkdir(parents=True, exist_ok=True)
        year_csv = year_dir / str(args.csv_name).format(year=task.year)

        if bool(args.resume) and not bool(args.overwrite):
            if task.year not in year_done_cache:
                year_done_cache[task.year] = load_done_codes_from_year_csv(year_csv)
            if task.stock_code in year_done_cache[task.year]:
                skip += 1
                continue

        raw_json_path = raw_root / str(task.year) / f"{task.stock_code}.json"
        if raw_json_path.exists() and not bool(args.overwrite):
            skip += 1
            continue

        try:
            extracted = extract_from_markdown(
                task,
                model=str(args.model),
                api_base_url=str(args.api_base_url),
                api_key=api_key,
                timeout=int(args.timeout),
                max_snippets=int(args.max_snippets),
                max_chars_per_field=int(args.max_chars_per_field),
                debug=bool(args.debug),
            )

            raw_json_path.parent.mkdir(parents=True, exist_ok=True)
            raw_json_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

            norm = extracted.get("normalized") or {}
            raw = extracted.get("raw") or {}
            pdf_path = str(task.pdf_path if task.pdf_path.exists() else task.md_path)

            row = {
                "year": task.year,
                "stock_code": task.stock_code,
                "code_name": infer_code_name(task.stock_code),
                "stock_name": "",
                "parent_netprofit": norm.get("parent_netprofit_yuan"),
                "share_capital": norm.get("total_shares_shares"),
                "share_capital_wan": norm.get("total_shares_wan"),
                "netcash_operate": norm.get("operating_cashflow_yuan"),
                "construct_long_asset": norm.get("capex_yuan"),
                "pdf_path": pdf_path,
                "parent_netprofit_page": (raw.get("parent_netprofit") or {}).get("page"),
                "share_capital_page": (raw.get("total_shares") or {}).get("page"),
                "netcash_operate_page": (raw.get("operating_cashflow") or {}).get("page"),
                "construct_long_asset_page": (raw.get("capex") or {}).get("page"),
            }
            year_fields = [
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
            append_csv_row(year_csv, row, fieldnames=year_fields)

            missing_fields: List[str] = []
            if norm.get("parent_netprofit_yuan") is None:
                missing_fields.append("parent_netprofit")
            if norm.get("total_shares_shares") is None:
                missing_fields.append("total_shares")
            if norm.get("operating_cashflow_yuan") is None:
                missing_fields.append("operating_cashflow")
            if norm.get("capex_yuan") is None:
                missing_fields.append("capex")

            status = "ok" if not missing_fields else "partial"
            message = ""
            if missing_fields:
                message = f"missing_fields={','.join(missing_fields)}"

            append_csv_row(
                log_path,
                {
                    "ts": now_iso(),
                    "year": task.year,
                    "stock_code": task.stock_code,
                    "code_name": infer_code_name(task.stock_code),
                    "pdf_path": pdf_path,
                    "status": status,
                    "message": message,
                    "raw_json_path": str(raw_json_path),
                },
                fieldnames=log_fields,
            )

            if status == "ok":
                ok += 1
            else:
                partial += 1

            print(
                f"[{idx}/{len(tasks)}] {task.year} {task.stock_code} {status} {message}".rstrip(),
                flush=True,
            )
        except Exception as exc:
            fail += 1
            append_csv_row(
                log_path,
                {
                    "ts": now_iso(),
                    "year": task.year,
                    "stock_code": task.stock_code,
                    "code_name": infer_code_name(task.stock_code),
                    "pdf_path": str(task.pdf_path if task.pdf_path.exists() else task.md_path),
                    "status": "error",
                    "message": str(exc),
                    "raw_json_path": "",
                },
                fieldnames=log_fields,
            )
            print(f"[{idx}/{len(tasks)}] {task.year} {task.stock_code} error {exc}", flush=True)

        sleep_seconds = float(args.sleep or 0.0)
        jitter_seconds = float(args.jitter or 0.0)
        if sleep_seconds > 0 or jitter_seconds > 0:
            time.sleep(max(0.0, sleep_seconds + random.uniform(0.0, max(0.0, jitter_seconds))))

    print(f"[done] ok={ok} partial={partial} skip={skip} fail={fail} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
