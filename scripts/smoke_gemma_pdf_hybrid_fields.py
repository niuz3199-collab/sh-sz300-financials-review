#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test hybrid Markdown + PDF-vision extraction for multiple fields.

Current scope
- parent_netprofit
- total_shares
- operating_cashflow
- capex
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fitz
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.step6_extract_financials_from_markdown import (  # noqa: E402
    Snippet,
    _coerce_field_payload,
    _extract_json_from_content,
    build_field_schema,
    call_openai_json_schema,
    retrieve_snippets,
)
from scripts import smoke_gemma_pdf_hybrid_capex as capex_smoke  # noqa: E402
from scripts.step6_extract_financials_qwen_pdf import (  # noqa: E402
    OPENAI_CHAT_COMPLETIONS_SUFFIX,
    normalize_money_to_yuan,
    normalize_total_shares_to_wan,
)


NUMBER_PATTERN = re.compile(r"-?\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?")
PAGE_MARKER_PATTERNS = [
    re.compile(r"_page_(\d+)_", re.I),
    re.compile(r"page-(\d+)-", re.I),
]


FIELD_CONFIGS: Dict[str, Dict[str, object]] = {
    "parent_netprofit": {
        "kind": "money",
        "primary_terms": [
            "归属于母公司股东的净利润",
            "归属于母公司所有者的净利润",
            "归属于上市公司股东的净利润",
            "归属于本公司股东的净利润",
            "归属于本行股东的净利润",
            "归属于本集团股东的净利润",
            "net profits attributable to shareholders of the listed companies",
            "net profit attributable to shareholders of the parent company",
            "profit attributable to shareholders of the parent company",
        ],
        "fragment_terms": [
            "归属于",
            "股东的净利润",
            "所有者的净利润",
            "净利润",
            "net profits attributable",
            "profit attributable",
            "net profit",
            "profit for the year",
        ],
        "continuation_terms": [
            "股东的净利润",
            "所有者的净利润",
            "listed companies",
            "shareholders of the parent company",
        ],
        "page_hint_terms": [
            "主要会计数据",
            "财务指标",
            "主要会计数据和财务指标",
            "合并利润表",
            "利润表",
            "main accounting data",
            "financial indicators",
            "consolidated income statement",
            "statement of profit",
        ],
        "prompt_target": "本报告年度的归属于母公司/上市公司/本集团/本行股东(或所有者)的净利润。",
        "prompt_accept": [
            "归属于母公司股东的净利润",
            "归属于母公司所有者的净利润",
            "归属于上市公司股东的净利润",
            "归属于本公司/本行/本集团股东的净利润",
            "英文等价行 Net profit(s) attributable to shareholders of the parent company / listed companies",
        ],
        "prompt_reject": [
            "扣除非经常性损益后的净利润",
            "利润总额",
            "未分配利润",
            "每股收益",
        ],
    },
    "total_shares": {
        "kind": "shares",
        "primary_terms": [
            "\u671f\u672b\u666e\u901a\u80a1\u80a1\u4efd\u603b\u6570",
            "\u671f\u672b\u603b\u80a1\u672c",
            "\u80a1\u4efd\u603b\u6570",
            "\u603b\u80a1\u672c",
            "\u5b9e\u6536\u8d44\u672c",
            "total number of shares at end of period",
            "total number of shares",
            "total share capital",
            "amount at the year-end",
        ],
        "fragment_terms": [
            "\u80a1\u672c",
            "\u80a1\u4efd",
            "\u603b\u80a1",
            "share capital",
            "number of shares",
            "shares in issue",
        ],
        "continuation_terms": [
            "\u80a1",
            "\u4e07\u80a1",
            "\u4ebf\u80a1",
            "shares",
            "million shares",
            "thousand shares",
        ],
        "page_hint_terms": [
            "\u80a1\u672c\u53d8\u52a8",
            "\u80a1\u4efd\u53d8\u52a8\u60c5\u51b5",
            "\u80a1\u4e1c\u60c5\u51b5",
            "changes in share capital",
            "statement of changes in share capital",
            "share capital",
        ],
        "prompt_target": "year-end total shares / share capital for the report year",
        "prompt_accept": [
            "\u603b\u80a1\u672c / \u671f\u672b\u603b\u80a1\u672c / \u671f\u672b\u666e\u901a\u80a1\u80a1\u4efd\u603b\u6570",
            "\u80a1\u4efd\u603b\u6570",
            "Total share capital / Total number of shares / Amount at the year-end",
        ],
        "prompt_reject": [
            "\u80a1\u4e1c\u6301\u80a1\u6570",
            "\u671f\u521d\u6570",
            "\u52a0\u6743\u5e73\u5747\u80a1\u6570",
            "weighted average shares",
            "shareholders' holdings",
        ],
        "prefer_next_line_when_split": True,
    },
    "operating_cashflow": {
        "kind": "money",
        "primary_terms": [
            "经营活动产生的现金流量净额",
            "经营活动现金流量净额",
            "net cash generated from operating activities",
            "net cash inflow from operating activities",
            "net cash flows from operating activities",
            "net cash provided by operating activities",
            "net cash flow from operating activities",
        ],
        "fragment_terms": [
            "经营活动",
            "现金流量净额",
            "net cash",
            "operating activities",
        ],
        "continuation_terms": [
            "现金流量净额",
            "operating activities",
        ],
        "page_hint_terms": [
            "合并现金流量表",
            "现金流量表",
            "statement of cash flows",
            "cash flow statement",
            "consolidated cash flow statement",
            "cashflow from operating activities",
        ],
        "prompt_target": "本报告年度的经营活动产生的现金流量净额。",
        "prompt_accept": [
            "经营活动产生的现金流量净额",
            "经营活动现金流量净额",
            "英文等价行 Net cash generated/flows/provided by operating activities",
        ],
        "prompt_reject": [
            "经营活动现金流入/流出小计",
            "补充资料",
            "调节表",
        ],
    },
    "capex": {
        "kind": "money",
        "primary_terms": [
            "购建固定资产、无形资产及其他长期资产支付的现金",
            "购建固定资产、无形资产和其他长期资产支付的现金",
            "购建固定资产、无形资产及其他长期资产所支付的现金",
            "购建固定资产、无形资产和其他长期资产所支付的现金",
            "cash paid for the purchase and construction of fixed assets, intangible assets, and other long-term assets",
            "payments for the acquisition and construction of fixed assets, intangible assets and other long-term assets",
            "cash paid to acquire property, plant and equipment",
            "cash paid for acquisition of property, plant and equipment",
            "payments for purchase of property, plant and equipment",
        ],
        "fragment_terms": [
            "购建固定资产",
            "长期资产所支付的现金",
            "长期资产支付的现金",
            "fixed assets, intangible assets",
            "other long-term assets",
            "cash paid to acquire",
            "acquire property, plant and equipment",
        ],
        "continuation_terms": [
            "长期资产",
            "other long-term assets",
            "and other long-term assets",
            "construction in progress",
            "and construction in progress",
        ],
        "page_hint_terms": [
            "合并现金流量表",
            "现金流量表",
            "投资活动产生的现金流量",
            "投资活动现金流出小计",
            "consolidated cash flow statement",
            "cash flow statement",
            "investment activities",
            "cashflow from investing activities",
            "net cashflow from investing activities",
            "property, plant and equipment",
        ],
        "prompt_target": "本报告年度的资本支出现金流(capex)。",
        "prompt_accept": [
            "购建固定资产、无形资产及其他长期资产支付的现金",
            "购建固定资产、无形资产和其他长期资产支付的现金",
            "购建固定资产、无形资产及其他长期资产所支付的现金",
            "英文等价行 Cash paid / Payments for the acquisition and construction of fixed assets, intangible assets and other long-term assets",
        ],
        "prompt_reject": [
            "投资支付的现金 / Cash paid for investments",
            "投资活动现金流出小计",
            "其他与投资活动有关的现金",
        ],
        "prefer_next_line_when_split": True,
    },
}


@dataclass(frozen=True)
class SmokeDoc:
    year: int
    stock_code: str
    report_name: str
    md_path: Path
    raw_json_path: Path
    pdf_path: Path


@dataclass
class AnchorWindow:
    start_line: int
    end_line: int
    text: str
    matched_terms: List[str]
    numbers: List[str]
    score: float


@dataclass
class TargetNumberHint:
    current_value: Optional[str]
    prior_value: Optional[str]
    source: str
    evidence_line: str
    unit: Optional[str] = None


@dataclass(frozen=True)
class MarkdownJudgeCandidate:
    snippet: Snippet
    confidence: int
    matched_label: str
    evidence_excerpt: str
    reason: str


def _resolve_openai_chat_url(api_base_url: str) -> str:
    base = str(api_base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("Missing --api-base-url")
    if base.endswith(OPENAI_CHAT_COMPLETIONS_SUFFIX):
        return base
    return f"{base}{OPENAI_CHAT_COMPLETIONS_SUFFIX}"


def _clean(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _extract_numbers(text: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for match in NUMBER_PATTERN.findall(str(text or "")):
        if match not in seen:
            seen.add(match)
            out.append(match)
    return out


MONEY_UNIT_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币百万元", "百万元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币千元", "千元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币亿元", "亿元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币万元", "万元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*人民币元", "元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*百万元", "百万元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*千元", "千元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*亿元", "亿元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*万元", "万元"),
    (r"(?:单位|币种|货币单位)\s*[:：]?\s*元", "元"),
    (r"\bRMB['\s]*000\b", "千元"),
    (r"\bthousand\s+RMB\b", "千元"),
    (r"\bRMB\s+million\b", "百万元"),
    (r"\bCNY\s+million\b", "百万元"),
    (r"[（(]\s*亿元\s*[)）]", "亿元"),
    (r"[（(]\s*万元\s*[)）]", "万元"),
    (r"[（(]\s*元\s*[)）]", "元"),
]

TABLE_LIKE_TOKENS = [
    "本年数",
    "上年数",
    "本期数",
    "上期数",
    "本年金额",
    "上年金额",
    "本期金额",
    "上期金额",
    "附注",
    "单位",
    "币种",
    "项目",
]

NARRATIVE_PENALTY_TOKENS = [
    "主要是",
    "所致",
    "由于",
    "原因",
    "增加主要",
    "减少主要",
    "借款",
    "汇兑",
    "折合",
    "会计政策",
    "现金等价物的确定标准",
    "现金等价物",
    "外币业务",
    "capital expenditure",
    "capital commitments",
    "not recognised in the financial statements",
    "contracted",
    "authorized",
    "authorised",
]

AUDIT_REPORT_PENALTY_TOKENS = [
    "审计报告",
    "我们接受委托",
    "会计报表发表审计意见",
    "audit report",
    "we have audited",
    "our responsibility is to express an opinion",
]

CAPEX_FALSE_POSITIVE_TOKENS = [
    "capital expenditure",
    "capital commitments",
    "not recognised in the financial statements",
    "contracted",
    "authorized",
    "authorised",
]

MD_LLM_WINDOW_LINES = 110
MD_LLM_OVERLAP_LINES = 28
MD_LLM_BATCH_SIZE = 4
MD_LLM_TOP_K = 2
MD_LLM_MAX_WINDOWS = 48
ENABLE_MARKDOWN_LLM_FALLBACK = str(os.environ.get("GEMMA_ENABLE_MARKDOWN_LLM_FALLBACK", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _parse_page_markers(text: str) -> List[int]:
    out: List[int] = []
    for pattern in PAGE_MARKER_PATTERNS:
        for raw in pattern.findall(str(text or "")):
            try:
                out.append(int(raw))
            except Exception:
                continue
    return sorted(set(out))


def infer_money_unit_from_text(text: str, fallback_unit: Optional[str] = None) -> Optional[str]:
    page_text = str(text or "")
    for pattern, unit in MONEY_UNIT_PATTERNS:
        if re.search(pattern, page_text, flags=re.I):
            return unit
    raw = str(fallback_unit or "").strip()
    if raw in {"元", "人民币元", "万元", "亿元", "百万元", "千元"}:
        return "元" if raw == "人民币元" else raw
    if raw.lower() in {"rmb", "cny"}:
        return "元"
    return raw or None


SHARE_UNIT_PATTERNS: List[Tuple[str, str]] = [
    ("(?:\u5355\u4f4d|\u6570\u91cf\u5355\u4f4d)\\s*[:\uff1a]?\\s*\u4ebf\u80a1", "\u4ebf\u80a1"),
    ("(?:\u5355\u4f4d|\u6570\u91cf\u5355\u4f4d)\\s*[:\uff1a]?\\s*\u4e07\u80a1", "\u4e07\u80a1"),
    ("(?:\u5355\u4f4d|\u6570\u91cf\u5355\u4f4d)\\s*[:\uff1a]?\\s*\u80a1", "\u80a1"),
    (r"\bbillion\s+shares?\b", "billion shares"),
    (r"\bmillion\s+shares?\b", "million shares"),
    (r"\bthousand\s+shares?\b", "thousand shares"),
    (r"\bshares?\b", "shares"),
]


def infer_share_unit_from_text(text: str, fallback_unit: Optional[str] = None) -> Optional[str]:
    page_text = str(text or "")
    for pattern, unit in SHARE_UNIT_PATTERNS:
        if re.search(pattern, page_text, flags=re.I):
            return unit
    raw = str(fallback_unit or "").strip()
    valid_units = {"\u80a1", "\u4e07\u80a1", "\u4ebf\u80a1", "shares", "million shares", "thousand shares", "billion shares"}
    return raw if raw in valid_units else (raw or None)


def build_overlap_markdown_snippets(
    markdown_text: str,
    *,
    lines_per_window: int = MD_LLM_WINDOW_LINES,
    overlap_lines: int = MD_LLM_OVERLAP_LINES,
    max_windows: int = MD_LLM_MAX_WINDOWS,
) -> List[Snippet]:
    lines = [str(line or "") for line in str(markdown_text or "").splitlines()]
    if not lines:
        return []
    step = max(1, int(lines_per_window) - int(overlap_lines))
    out: List[Snippet] = []
    for idx, start in enumerate(range(0, len(lines), step), start=1):
        if len(out) >= int(max_windows):
            break
        end = min(len(lines), start + int(lines_per_window))
        text = "\n".join(lines[start:end]).strip()
        if not text:
            if end >= len(lines):
                break
            continue
        score = 0.0
        if re.search(r"\d", text):
            score += 1.0
        number_count = len(_extract_numbers(text))
        if re.search(r"(元|万元|亿元|百万元|千元|rmb|cny)", text, flags=re.I):
            score += 1.0
        if any(_clean(token) in _clean(text) for token in TABLE_LIKE_TOKENS):
            score += 1.0
        if number_count < 4 and score < 2.0:
            if end >= len(lines):
                break
            continue
        out.append(
            Snippet(
                snippet_id=f"MDWIN_{idx:03d}",
                start_line=start + 1,
                end_line=end,
                score=score,
                text=text,
                matched_terms=tuple(),
            )
        )
        if end >= len(lines):
            break
    return out


def build_markdown_judge_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "snippet_id": {"type": "string"},
                        "found": {"type": "boolean"},
                        "confidence": {"type": "integer"},
                        "matched_label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "evidence_excerpt": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "reason": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    },
                    "required": ["snippet_id", "found", "confidence", "matched_label", "evidence_excerpt", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def build_markdown_judge_prompt(
    *,
    field_name: str,
    year: int,
    snippets: Sequence[Snippet],
) -> Tuple[str, str, Dict[str, object]]:
    cfg = FIELD_CONFIGS[field_name]
    prompt_accept = list(cfg.get("prompt_accept") or [])
    prompt_reject = list(cfg.get("prompt_reject") or [])
    target = str(cfg.get("prompt_target") or field_name)
    schema = build_markdown_judge_schema()

    system_prompt = (
        "你是一名严格的年报文本定位助手。"
        "你将看到若干来自同一份年报 Markdown 的重叠窗口。"
        "你的任务不是抽取最终值，而是判断每个窗口是否真正包含目标字段所在的正文/表格行。"
        "必须严格排除同名但错误口径的上下文。"
        "只输出 JSON。"
    )

    chunk_blocks: List[str] = []
    for snippet in snippets:
        snippet_text = str(snippet.text or "")
        if len(snippet_text) > 3600:
            snippet_text = snippet_text[:2200] + "\n...\n" + snippet_text[-1200:]
        chunk_blocks.append(
            "\n".join(
                [
                    f"[{snippet.snippet_id}] lines {snippet.start_line}-{snippet.end_line}",
                    snippet_text,
                ]
            ).strip()
        )

    extra_reject = ""
    if field_name == "capex":
        extra_reject = (
            "\n额外排除：资本承诺/Capital commitments、Contracted/Authorized、"
            "未在财务报表中确认(not recognised in the financial statements)、"
            "购建固定资产累计支出、借款费用资本化说明。"
        )
    elif field_name == "parent_netprofit":
        extra_reject = (
            "\n额外排除：股利、未分配利润、每股收益、利润分配方案、仅提到“净利润”但不是目标表格行的附注说明。"
        )
    elif field_name == "operating_cashflow":
        extra_reject = "\n额外排除：现金流入/流出小计、补充资料、调节表、原因说明段落。"

    user_prompt = f"""
任务：判断下列 Markdown 窗口是否真正包含 {year} 年目标字段。

目标字段：
{target}

可接受口径：
{chr(10).join(f"- {item}" for item in prompt_accept)}

不可接受口径：
{chr(10).join(f"- {item}" for item in prompt_reject)}{extra_reject}

判断标准：
- `found=true` 只在该窗口里已经出现目标字段本身，或其紧邻续行，且能够支持后续定位到该字段值时返回。
- 如果只是同主题讨论、原因解释、会计政策、资本承诺、股利附注、审计意见、摘要文字，必须返回 `found=false`。
- `matched_label` 填窗口里最接近目标字段的原文标签；没有就填 null。
- `evidence_excerpt` 尽量摘出窗口里最关键的 1 到 4 行，优先包含标签和对应数值；没有就填 null。
- `confidence` 取 0 到 100 的整数。明显命中 85+；模糊但较像 60 到 84；不应命中则 0 到 40。
- 每个 snippet_id 都必须返回一条记录。

窗口如下：

{chr(10).join(chunk_blocks)}
""".strip()
    return system_prompt, user_prompt, schema


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(str(value or "").strip()))
        except Exception:
            return int(default)


def select_markdown_judge_candidates(
    *,
    markdown_text: str,
    field_name: str,
    year: int,
    model: str,
    api_base_url: str,
    api_key: str,
    timeout: int,
) -> List[MarkdownJudgeCandidate]:
    windows = build_overlap_markdown_snippets(markdown_text)
    if not windows:
        return []
    kept: Dict[str, MarkdownJudgeCandidate] = {}
    for start in range(0, len(windows), MD_LLM_BATCH_SIZE):
        batch = windows[start : start + MD_LLM_BATCH_SIZE]
        if not batch:
            continue
        system_prompt, user_prompt, schema = build_markdown_judge_prompt(field_name=field_name, year=year, snippets=batch)
        try:
            payload = call_openai_json_schema(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_schema=schema,
                api_base_url=api_base_url,
                api_key=api_key,
                timeout=min(int(timeout), 90),
                schema_name="markdown_chunk_judge",
            )
        except Exception:
            continue
        id_map = {snippet.snippet_id: snippet for snippet in batch}
        rows = []
        if isinstance(payload, dict):
            rows = list(payload.get("candidates") or [])
        elif isinstance(payload, list):
            rows = list(payload)
        for row in rows:
            snippet_id = str((row or {}).get("snippet_id") or "").strip()
            snippet = id_map.get(snippet_id)
            if snippet is None:
                continue
            found = bool((row or {}).get("found"))
            confidence = max(0, min(100, _coerce_int((row or {}).get("confidence"), default=0)))
            if not found or confidence < 55:
                continue
            candidate = MarkdownJudgeCandidate(
                snippet=snippet,
                confidence=confidence,
                matched_label=str((row or {}).get("matched_label") or "").strip(),
                evidence_excerpt=str((row or {}).get("evidence_excerpt") or "").strip(),
                reason=str((row or {}).get("reason") or "").strip(),
            )
            current = kept.get(snippet_id)
            if current is None or int(candidate.confidence) > int(current.confidence):
                kept[snippet_id] = candidate
    ranked = sorted(
        kept.values(),
        key=lambda item: (
            -int(item.confidence),
            -len(_extract_numbers(item.evidence_excerpt)),
            item.snippet.start_line,
        ),
    )
    return ranked[:MD_LLM_TOP_K]


def locate_probe_in_snippet(
    snippet_text: str,
    *,
    matched_label: str,
    evidence_excerpt: str,
) -> Tuple[str, int, int]:
    lines = [str(line or "") for line in str(snippet_text or "").splitlines()]
    if not lines:
        return str(evidence_excerpt or "").strip(), 1, 1

    label_clean = _clean(matched_label)
    excerpt_clean = _clean(evidence_excerpt)
    excerpt_numbers = _extract_numbers(evidence_excerpt)
    best_idx = 0
    best_score = -1
    for idx, line in enumerate(lines):
        line_clean = _clean(line)
        if not line_clean:
            continue
        score = 0
        if label_clean and label_clean in line_clean:
            score += 8
        if excerpt_clean and excerpt_clean in line_clean:
            score += 8
        elif excerpt_clean and line_clean in excerpt_clean:
            score += 4
        for num in excerpt_numbers:
            if num and num in line:
                score += 3
        if score > best_score:
            best_idx = idx
            best_score = score

    if best_score <= 0:
        return str(evidence_excerpt or "").strip(), 1, min(len(lines), 4)

    start = max(0, best_idx - 1)
    end = min(len(lines), best_idx + 4)
    probe = "\n".join(lines[start:end]).strip()
    return probe, start + 1, end


def choose_anchor_window_with_markdown_llm(
    *,
    markdown_text: str,
    field_name: str,
    year: int,
    model: str,
    api_base_url: str,
    api_key: str,
    timeout: int,
) -> Tuple[Optional[AnchorWindow], List[Snippet], List[Dict[str, object]]]:
    candidates = select_markdown_judge_candidates(
        markdown_text=markdown_text,
        field_name=field_name,
        year=year,
        model=model,
        api_base_url=api_base_url,
        api_key=api_key,
        timeout=timeout,
    )
    if not candidates:
        return None, [], []

    windows: List[AnchorWindow] = []
    debug_rows: List[Dict[str, object]] = []
    snippets: List[Snippet] = []
    for candidate in candidates:
        probe_text, local_start, local_end = locate_probe_in_snippet(
            candidate.snippet.text,
            matched_label=candidate.matched_label,
            evidence_excerpt=candidate.evidence_excerpt,
        )
        if field_name == "capex" and any(_clean(token) in _clean(probe_text) for token in CAPEX_FALSE_POSITIVE_TOKENS):
            continue
        matched_terms = [candidate.matched_label] if candidate.matched_label else list(candidate.snippet.matched_terms)
        numbers = _extract_numbers(probe_text) or _extract_numbers(candidate.evidence_excerpt)
        anchor_window = AnchorWindow(
            start_line=candidate.snippet.start_line + local_start - 1,
            end_line=min(candidate.snippet.end_line, candidate.snippet.start_line + local_end - 1),
            text=probe_text or candidate.evidence_excerpt or candidate.snippet.text,
            matched_terms=matched_terms,
            numbers=numbers,
            score=40.0 + float(candidate.confidence) / 2.0,
        )
        windows.append(anchor_window)
        snippets.append(candidate.snippet)
        debug_rows.append(
            {
                "snippet_id": candidate.snippet.snippet_id,
                "start_line": candidate.snippet.start_line,
                "end_line": candidate.snippet.end_line,
                "confidence": candidate.confidence,
                "matched_label": candidate.matched_label,
                "evidence_excerpt": candidate.evidence_excerpt,
                "reason": candidate.reason,
            }
        )
    if not windows:
        return None, [], debug_rows
    windows.sort(key=lambda item: (-float(item.score), -len(item.numbers), item.start_line))
    return windows[0], snippets, debug_rows


def _count_term_hits(text: str, terms: Sequence[str]) -> List[str]:
    clean_text = _clean(text)
    return [term for term in terms if _clean(term) in clean_text]


def _window_quality_adjustment(
    text: str,
    *,
    field_name: str,
    full_hits: Sequence[str],
    fragment_hits: Sequence[str],
    numbers: Sequence[str],
) -> float:
    clean_text = _clean(text)
    score = _context_score_adjustment(text, field_name)
    if any(_clean(token) in clean_text for token in TABLE_LIKE_TOKENS):
        score += 10.0
    if any(re.search(rf"{re.escape(str(year))}", text) for year in ("2024", "2023", "2022", "2021")):
        score += 4.0
    if re.search(r"(元|万元|亿元|百万元|千元|RMB|CNY)", text, flags=re.I):
        score += 5.0
    if len(numbers) >= 2:
        score += min(len(numbers), 6) * 2.0
    if full_hits:
        score += 10.0
    if field_name == "capex":
        if fragment_hits and not full_hits and len(numbers) == 0:
            score -= 28.0
        if any(_clean(token) in clean_text for token in NARRATIVE_PENALTY_TOKENS):
            score -= 32.0
        if any(_clean(token) in clean_text for token in CAPEX_FALSE_POSITIVE_TOKENS):
            score -= 80.0
        if "投资活动现金流出小计" in text or "investment activities" in clean_text:
            score += 6.0
    if field_name == "parent_netprofit":
        if any(_clean(token) in clean_text for token in ["主要会计数据", "财务指标", "净利润", "利润表"]):
            score += 8.0
    if field_name == "operating_cashflow":
        if any(_clean(token) in clean_text for token in ["经营活动产生的现金流量净额", "现金流量表", "net cash"]):
            score += 8.0
    return score


def is_viable_anchor_window(window: Optional[AnchorWindow], field_name: str) -> bool:
    if window is None:
        return False
    cfg = FIELD_CONFIGS[field_name]
    clean_text = _clean(window.text)
    primary_terms = list(cfg.get("primary_terms") or [])
    page_hint_terms = list(cfg.get("page_hint_terms") or [])
    has_primary = any(term in window.matched_terms for term in primary_terms)
    has_page_hint = any(_clean(term) in clean_text for term in page_hint_terms)
    has_numbers = len(window.numbers) >= 1
    has_table_cue = any(_clean(token) in clean_text for token in TABLE_LIKE_TOKENS)
    if any(_clean(token) in clean_text for token in AUDIT_REPORT_PENALTY_TOKENS):
        return False
    if field_name == "capex":
        if any(_clean(token) in clean_text for token in CAPEX_FALSE_POSITIVE_TOKENS):
            return False
        if float(window.score) < 0.0:
            return False
        if any(_clean(token) in clean_text for token in NARRATIVE_PENALTY_TOKENS) and len(window.numbers) < 2 and not has_primary:
            return False
        if not has_primary and not has_page_hint and len(window.numbers) < 2:
            return False
        return True
    if field_name in {"parent_netprofit", "operating_cashflow"}:
        if float(window.score) < -10.0:
            return False
        if not has_primary and not has_page_hint and (not has_table_cue or not has_numbers):
            return False
    if field_name == "total_shares":
        if float(window.score) < -10.0:
            return False
        if not has_primary and not has_page_hint and not has_numbers:
            return False
    return True


def load_smoke_docs(csv_path: Path, *, fulltext_root: Path, year: int, codes: Sequence[str]) -> List[SmokeDoc]:
    wanted = {str(code).strip() for code in codes if str(code).strip()}
    docs: Dict[str, SmokeDoc] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("year") or "").strip() != str(int(year)):
                continue
            stock_code = str(row.get("stock_code") or "").strip()
            if wanted and stock_code not in wanted:
                continue
            if stock_code in docs:
                continue
            report_name = str(row.get("report_name") or "").strip()
            md_path = Path(str(row.get("markdown_path") or "").strip())
            raw_json_path = Path(str(row.get("raw_json_path") or "").strip())
            pdf_path = fulltext_root / str(int(year)) / f"{report_name}.pdf"
            if not md_path.exists():
                raise FileNotFoundError(f"Markdown not found: {md_path}")
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            docs[stock_code] = SmokeDoc(
                year=int(year),
                stock_code=stock_code,
                report_name=report_name,
                md_path=md_path,
                raw_json_path=raw_json_path,
                pdf_path=pdf_path,
            )
    order = {code: idx for idx, code in enumerate(codes)}
    return sorted(docs.values(), key=lambda x: order.get(x.stock_code, 10**9))


def read_current_raw_field(raw_json_path: Path, field_name: str) -> Dict[str, object]:
    if not raw_json_path.exists():
        return {}
    try:
        data = json.loads(raw_json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    raw = data.get("raw") or {}
    return _coerce_field_payload(raw.get(field_name))


def build_anchor_windows(markdown_text: str, field_name: str) -> List[AnchorWindow]:
    cfg = FIELD_CONFIGS[field_name]
    primary_terms = list(cfg.get("primary_terms") or [])
    fragment_terms = list(cfg.get("fragment_terms") or [])
    hint_terms = list(cfg.get("page_hint_terms") or [])
    lines = markdown_text.splitlines()
    seen_ranges = set()
    windows: List[AnchorWindow] = []
    for idx, line in enumerate(lines):
        clean_line = _clean(line)
        if not clean_line:
            continue
        full_hits: List[str] = []
        fragment_hits: List[str] = []
        score = 0.0
        for term in primary_terms:
            if _clean(term) in clean_line:
                full_hits.append(term)
                score += 20.0
        for term in fragment_terms:
            if _clean(term) in clean_line:
                fragment_hits.append(term)
                score += 7.0
        hits = full_hits + fragment_hits
        if not hits:
            continue
        start = max(0, idx - 1)
        end = min(len(lines) - 1, idx + 2)
        while end < len(lines) - 1 and end < idx + 4:
            probe = "\n".join(lines[start : end + 1])
            if len(_extract_numbers(probe)) >= 2:
                break
            end += 1
        key = (start, end)
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        text = "\n".join(lines[start : end + 1]).strip()
        clean_text = _clean(text)
        hint_bonus = 0.0
        for term in hint_terms:
            if _clean(term) in clean_text:
                hint_bonus += 2.0
        numbers = _extract_numbers(text)
        total_score = score + hint_bonus + min(len(numbers), 8)
        total_score += _window_quality_adjustment(
            text,
            field_name=field_name,
            full_hits=full_hits,
            fragment_hits=fragment_hits,
            numbers=numbers,
        )
        windows.append(
            AnchorWindow(
                start_line=start + 1,
                end_line=end + 1,
                text=text,
                matched_terms=sorted(set(hits)),
                numbers=numbers,
                score=total_score,
            )
        )
    windows.sort(key=lambda x: (-x.score, -len(x.numbers), x.start_line))
    return windows


def choose_anchor_window(markdown_text: str, snippets: Sequence[Snippet], field_name: str) -> Optional[AnchorWindow]:
    candidates: List[AnchorWindow] = []
    for snippet_rank, snippet in enumerate(snippets):
        snippet_windows = build_anchor_windows(snippet.text, field_name)
        for window in snippet_windows:
            text = window.text
            adjusted_score = window.score + max(0.0, 8.0 - snippet_rank * 2.0)
            candidates.append(
                AnchorWindow(
                    start_line=snippet.start_line + window.start_line - 1,
                    end_line=snippet.start_line + window.end_line - 1,
                    text=text,
                    matched_terms=window.matched_terms,
                    numbers=window.numbers,
                    score=adjusted_score,
                )
            )
    windows = build_anchor_windows(markdown_text, field_name)
    candidates.extend(windows)
    if candidates:
        candidates.sort(key=lambda x: (-x.score, -len(x.numbers), x.start_line))
        return candidates[0]
    return None


def choose_anchor_window_from_pdf_text(doc: fitz.Document, field_name: str) -> Tuple[Optional[AnchorWindow], List[int]]:
    candidates: List[Tuple[float, int, AnchorWindow]] = []
    for page_index in range(doc.page_count):
        page_text = doc[page_index].get_text("text") or ""
        page_windows = build_anchor_windows(page_text, field_name)
        if not page_windows:
            continue
        for window in page_windows[:5]:
            page_score, number_hits = score_pdf_page(
                page_text,
                field_name=field_name,
                anchor_numbers=window.numbers[:2],
                marker_pages=[],
                page_index=page_index,
            )
            combined = float(window.score) + float(page_score) + float(number_hits * 8) + 12.0
            candidates.append((combined, page_index, window))
    if not candidates:
        return None, []
    candidates.sort(key=lambda item: (-item[0], -len(item[2].numbers), item[1]))
    combined, page_index, window = candidates[0]
    return (
        AnchorWindow(
            start_line=window.start_line,
            end_line=window.end_line,
            text=window.text,
            matched_terms=list(window.matched_terms),
            numbers=list(window.numbers),
            score=float(combined),
        ),
        [page_index + 1],
    )


def _context_score_adjustment(text: str, field_name: str) -> float:
    clean_text = _clean(text)
    score = 0.0
    quarter_tokens = [
        "第一季度",
        "第二季度",
        "第三季度",
        "第四季度",
        "一季度",
        "二季度",
        "三季度",
        "四季度",
        "1-3月",
        "1-6月",
        "1-9月",
        "quarter1",
        "quarter2",
        "quarter3",
        "quarter4",
        "quarter 1",
        "quarter 2",
        "quarter 3",
        "quarter 4",
        "季度报告",
        "半年度",
    ]
    if any(_clean(token) in clean_text for token in quarter_tokens):
        score -= 60.0
    if any(_clean(token) in clean_text for token in AUDIT_REPORT_PENALTY_TOKENS):
        score -= 90.0
    if "季度报告相关财务指标" in text or "与公司已披露季度报告" in text:
        score -= 60.0
    if ("2024" in text and "2023" in text) or ("2024年" in text and "2023年" in text):
        score += 8.0
    if field_name == "operating_cashflow" and any(token in clean_text for token in [_clean("现金流量表"), _clean("cash flow statement")]):
        score += 14.0
    if field_name == "parent_netprofit" and any(token in clean_text for token in [_clean("主要会计数据"), _clean("财务指标"), _clean("利润表")]):
        score += 12.0
    if field_name == "total_shares" and any(
        token in clean_text
        for token in [
            _clean("\u80a1\u672c\u53d8\u52a8"),
            _clean("\u80a1\u4efd\u53d8\u52a8\u60c5\u51b5"),
            _clean("changes in share capital"),
            _clean("share capital"),
        ]
    ):
        score += 12.0
    if field_name == "capex" and any(
        token in clean_text
        for token in [
            _clean("投资活动产生的现金流量"),
            _clean("投资活动现金流出小计"),
            _clean("cashflow from investing activities"),
            _clean("property, plant and equipment"),
            _clean("construction in progress"),
        ]
    ):
        score += 16.0
    return score


def infer_target_number_hint(anchor_window: AnchorWindow, field_name: str) -> TargetNumberHint:
    if field_name == "capex":
        capex_hint = capex_smoke.infer_target_number_hint(anchor_window)
        return TargetNumberHint(
            current_value=capex_hint.current_value,
            prior_value=capex_hint.prior_value,
            source=capex_hint.source,
            evidence_line=capex_hint.evidence_line,
        )

    cfg = FIELD_CONFIGS[field_name]
    primary_terms = list(cfg.get("primary_terms") or [])
    fragment_terms = list(cfg.get("fragment_terms") or [])
    continuation_terms = list(cfg.get("continuation_terms") or [])
    prefer_next = bool(cfg.get("prefer_next_line_when_split"))
    lines = [str(line or "") for line in anchor_window.text.splitlines()]

    def _line_has_anchor(text: str) -> bool:
        clean_text = _clean(text)
        return any(_clean(term) in clean_text for term in (primary_terms + fragment_terms))

    def _continuation_hint(text: str) -> bool:
        clean_text = _clean(text)
        return any(_clean(term) in clean_text for term in continuation_terms)

    def _subtotal_hint(text: str) -> bool:
        clean_text = _clean(text)
        return any(token in clean_text for token in [_clean("小计"), _clean("subtotal")])

    if field_name == "total_shares":
        for idx, line in enumerate(lines):
            if not _line_has_anchor(line):
                continue
            numbers_in_line = _extract_numbers(line)
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            next_numbers = _extract_numbers(next_line) if next_line else []
            if numbers_in_line:
                return TargetNumberHint(
                    current_value=numbers_in_line[0],
                    prior_value=(numbers_in_line[1] if len(numbers_in_line) >= 2 else None),
                    source="same_line_total_shares",
                    evidence_line=line,
                )
            if next_line and next_numbers and (_continuation_hint(next_line) or "<br>" in line or _subtotal_hint(line)):
                return TargetNumberHint(
                    current_value=next_numbers[0],
                    prior_value=(next_numbers[1] if len(next_numbers) >= 2 else None),
                    source="next_line_total_shares",
                    evidence_line=next_line,
                )
        for line in lines:
            numbers = _extract_numbers(line)
            if numbers and _continuation_hint(line):
                return TargetNumberHint(
                    current_value=numbers[0],
                    prior_value=(numbers[1] if len(numbers) >= 2 else None),
                    source="continuation_line_total_shares",
                    evidence_line=line,
                )
        fallback_numbers = anchor_window.numbers[:2]
        return TargetNumberHint(
            current_value=fallback_numbers[0] if len(fallback_numbers) >= 1 else None,
            prior_value=fallback_numbers[1] if len(fallback_numbers) >= 2 else None,
            source="fallback_total_shares",
            evidence_line=anchor_window.text,
        )

    for idx, line in enumerate(lines):
        if not _line_has_anchor(line):
            continue
        numbers_in_line = _extract_numbers(line)
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        next_numbers = _extract_numbers(next_line) if next_line else []
        if prefer_next and next_line and len(next_numbers) >= 2 and _continuation_hint(next_line):
            if _subtotal_hint(line) or "<br>" in line or not numbers_in_line:
                return TargetNumberHint(next_numbers[0], next_numbers[1], "preferred_next_line", next_line)
        if len(numbers_in_line) >= 2:
            return TargetNumberHint(numbers_in_line[0], numbers_in_line[1], "same_line", line)
        if next_line and len(next_numbers) >= 2 and (_continuation_hint(next_line) or not numbers_in_line):
            return TargetNumberHint(next_numbers[0], next_numbers[1], "next_line", next_line)

    for line in lines:
        numbers = _extract_numbers(line)
        if len(numbers) >= 2 and _continuation_hint(line):
            return TargetNumberHint(numbers[0], numbers[1], "continuation_line_fallback", line)

    fallback_numbers = anchor_window.numbers[:2]
    return TargetNumberHint(
        current_value=fallback_numbers[0] if len(fallback_numbers) >= 1 else None,
        prior_value=fallback_numbers[1] if len(fallback_numbers) >= 2 else None,
        source="fallback_first_two",
        evidence_line=anchor_window.text,
    )


def score_pdf_page(page_text: str, *, field_name: str, anchor_numbers: Sequence[str], marker_pages: Sequence[int], page_index: int) -> Tuple[float, int]:
    cfg = FIELD_CONFIGS[field_name]
    hint_terms = list(cfg.get("page_hint_terms") or []) + list(cfg.get("primary_terms") or [])
    text = str(page_text or "")
    clean_text = _clean(text)
    number_hits = sum(1 for num in anchor_numbers if num and num in text)
    score = float(number_hits * 100)
    if page_index in marker_pages:
        score += 60.0
    hint_hits = 0
    for term in hint_terms:
        if _clean(term) in clean_text:
            hint_hits += 1
    score += min(hint_hits, 5) * 10.0
    if any(_clean(token) in clean_text for token in AUDIT_REPORT_PENALTY_TOKENS):
        score -= 120.0
    if field_name == "total_shares":
        if any(
            token in clean_text
            for token in [
                _clean("\u671f\u672b\u603b\u80a1\u672c"),
                _clean("\u80a1\u4efd\u603b\u6570"),
                _clean("total share capital"),
                _clean("total number of shares"),
                _clean("changes in share capital"),
            ]
        ):
            score += 40.0
    if field_name == "capex":
        if any(
            token in clean_text
            for token in [
                _clean("投资活动产生的现金流量"),
                _clean("投资活动现金流出小计"),
                _clean("cashflow from investing activities"),
                _clean("net cashflow from investing activities"),
                _clean("cash paid to acquire property, plant and equipment"),
                _clean("construction in progress"),
            ]
        ):
            score += 45.0
    if field_name == "operating_cashflow":
        if any(
            token in clean_text
            for token in [
                _clean("经营活动产生的现金流量净额"),
                _clean("cashflow from operating activities"),
                _clean("net cash inflow from operating activities"),
            ]
        ):
            score += 36.0
    return score, number_hits


def rank_candidate_pages(doc: fitz.Document, *, field_name: str, anchor_numbers: Sequence[str], marker_pages: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for page_index in range(doc.page_count):
        text = doc[page_index].get_text("text") or ""
        score, number_hits = score_pdf_page(
            text,
            field_name=field_name,
            anchor_numbers=anchor_numbers,
            marker_pages=marker_pages,
            page_index=page_index,
        )
        if score <= 0 and number_hits <= 0 and page_index not in marker_pages:
            continue
        rows.append(
            {
                "page_index": page_index,
                "page_number": page_index + 1,
                "score": score,
                "number_hits": number_hits,
            }
        )
    rows.sort(key=lambda x: (-float(x["score"]), -int(x["number_hits"]), int(x["page_index"])))
    return rows


def cluster_numeric_hits(hit_map: Dict[str, List[fitz.Rect]]) -> List[Dict[str, object]]:
    flat_hits: List[Tuple[str, fitz.Rect]] = []
    for num, rects in hit_map.items():
        for rect in rects:
            flat_hits.append((num, rect))
    flat_hits.sort(key=lambda item: (item[1].y0 + item[1].y1) / 2.0)
    clusters: List[Dict[str, object]] = []
    for num, rect in flat_hits:
        y_center = (rect.y0 + rect.y1) / 2.0
        placed = False
        for cluster in clusters:
            if abs(y_center - float(cluster["mean_y"])) <= 28.0:
                cluster["items"].append((num, rect))
                cluster["mean_y"] = sum((r.y0 + r.y1) / 2.0 for _, r in cluster["items"]) / len(cluster["items"])
                placed = True
                break
        if not placed:
            clusters.append({"items": [(num, rect)], "mean_y": y_center})

    def _cluster_key(cluster: Dict[str, object]) -> Tuple[int, int, float]:
        items = cluster["items"]
        distinct = len({num for num, _ in items})
        total = len(items)
        mean_y = float(cluster["mean_y"])
        return (distinct, total, -mean_y)

    return sorted(clusters, key=_cluster_key, reverse=True)


def compute_crop_rect(page: fitz.Page, anchor_numbers: Sequence[str]) -> Tuple[fitz.Rect, Dict[str, List[List[float]]]]:
    hit_map: Dict[str, List[fitz.Rect]] = {}
    for num in anchor_numbers:
        rects = page.search_for(num)
        if rects:
            hit_map[num] = rects
    if not hit_map:
        return fitz.Rect(page.rect), {}
    clusters = cluster_numeric_hits(hit_map)
    rects = [rect for _, rect in clusters[0]["items"]]
    union = fitz.Rect(rects[0])
    for rect in rects[1:]:
        union |= rect
    avg_height = sum(rect.height for rect in rects) / max(len(rects), 1)
    crop = fitz.Rect(
        page.rect.x0 + 8.0,
        max(page.rect.y0, union.y0 - max(95.0, avg_height * 3.2)),
        page.rect.x1 - 8.0,
        min(page.rect.y1, union.y1 + max(60.0, avg_height * 1.9)),
    )
    serializable = {
        num: [[float(r.x0), float(r.y0), float(r.x1), float(r.y1)] for r in rects]
        for num, rects in hit_map.items()
    }
    return crop, serializable


def compute_label_crop_rect(page: fitz.Page, field_name: str) -> Tuple[Optional[fitz.Rect], Dict[str, List[List[float]]]]:
    cfg = FIELD_CONFIGS[field_name]
    terms = sorted(
        {str(x) for x in list(cfg.get("primary_terms") or []) + list(cfg.get("fragment_terms") or [])},
        key=len,
        reverse=True,
    )
    for term in terms:
        rects = page.search_for(term)
        if not rects:
            continue
        union = fitz.Rect(rects[0])
        for rect in rects[1:]:
            union |= rect
        avg_height = sum(rect.height for rect in rects) / max(len(rects), 1)
        crop = fitz.Rect(
            page.rect.x0 + 8.0,
            max(page.rect.y0, union.y0 - max(70.0, avg_height * 2.6)),
            page.rect.x1 - 8.0,
            min(page.rect.y1, union.y1 + max(55.0, avg_height * 1.8)),
        )
        serializable = {
            term: [[float(r.x0), float(r.y0), float(r.x1), float(r.y1)] for r in rects]
        }
        return crop, serializable
    return None, {}


def salvage_value_from_page_text(page: fitz.Page, field_name: str) -> Optional[TargetNumberHint]:
    cfg = FIELD_CONFIGS[field_name]
    primary_terms = sorted((str(x) for x in (cfg.get("primary_terms") or [])), key=len, reverse=True)
    fragment_terms = sorted((str(x) for x in (cfg.get("fragment_terms") or [])), key=len, reverse=True)
    page_text = page.get_text("text") or ""
    lines = [str(line or "").strip() for line in (page.get_text("text") or "").splitlines()]
    lines = [line for line in lines if line]
    page_unit = (
        infer_share_unit_from_text(page_text)
        if field_name == "total_shares"
        else infer_money_unit_from_text(page_text)
    )
    for idx, line in enumerate(lines):
        clean_line = _clean(line)
        matched_primary = [term for term in primary_terms if _clean(term) in clean_line]
        matched_terms = list(matched_primary)
        matched_terms.extend([term for term in fragment_terms if _clean(term) in clean_line])
        if not matched_terms:
            continue
        numbers_in_line = _extract_numbers(line)
        if field_name in {"parent_netprofit", "operating_cashflow"} and (not matched_primary) and len(numbers_in_line) < 2:
            continue
        if field_name == "total_shares" and (not matched_primary) and len(numbers_in_line) < 1:
            continue
        probe = "\n".join(lines[idx : min(len(lines), idx + 6)])
        if field_name == "capex":
            clean_probe = _clean(probe)
            if any(_clean(token) in clean_probe for token in CAPEX_FALSE_POSITIVE_TOKENS):
                continue
            if any(_clean(token) in clean_probe for token in NARRATIVE_PENALTY_TOKENS) and len(_extract_numbers(probe)) < 2:
                continue
        probe_numbers = _extract_numbers(probe)
        temp_window = AnchorWindow(
            start_line=idx + 1,
            end_line=min(len(lines), idx + 6),
            text=probe,
            matched_terms=matched_terms,
            numbers=probe_numbers,
            score=0.0,
        )
        hint = infer_target_number_hint(temp_window, field_name)
        if hint.current_value is not None:
            return TargetNumberHint(
                current_value=hint.current_value,
                prior_value=hint.prior_value,
                source="page_text_salvage",
                evidence_line=hint.evidence_line or probe,
                unit=page_unit,
            )
    return None


def render_rect_to_png_bytes(page: fitz.Page, rect: fitz.Rect, *, dpi: int) -> bytes:
    matrix = fitz.Matrix(float(dpi) / 72.0, float(dpi) / 72.0)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    return pix.tobytes("png")


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def call_lm_studio_vision_json(
    *,
    model: str,
    api_base_url: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    images_b64: Sequence[str],
    response_schema: Dict[str, object],
    timeout: int,
) -> Dict[str, object]:
    url = _resolve_openai_chat_url(api_base_url)
    headers = {"Content-Type": "application/json"}
    if str(api_key or "").strip():
        headers["Authorization"] = f"Bearer {api_key}"
    content = [{"type": "text", "text": user_prompt}]
    for image_b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": 260,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "hybrid_field_extract", "schema": response_schema},
        },
    }
    response = requests.post(url, headers=headers, json=payload, timeout=(30, int(timeout)))
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"missing_choices: {json.dumps(data, ensure_ascii=False)[:300]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not str(content or "").strip():
        reasoning = message.get("reasoning_content")
        if str(reasoning or "").strip():
            content = reasoning
    if not str(content or "").strip():
        raise RuntimeError("empty_message_content")
    try:
        return _extract_json_from_content(content)
    except Exception as exc:
        preview = str(content or "").replace("\r", " ").replace("\n", "\\n")
        raise RuntimeError(f"invalid_json_response:{preview[:800]}") from exc


def build_vision_schema(field_name: str) -> Dict[str, object]:
    return {
        "type": "object",
        "properties": {field_name: build_field_schema()},
        "required": [field_name],
        "additionalProperties": False,
    }


def build_prompt(
    *,
    doc: SmokeDoc,
    field_name: str,
    snippets: Sequence[Snippet],
    anchor_window: AnchorWindow,
    target_hint: TargetNumberHint,
    chosen_page_number: int,
) -> Tuple[str, str]:
    cfg = FIELD_CONFIGS[field_name]
    target_numbers = [x for x in [target_hint.current_value, target_hint.prior_value] if x]
    snippet_ids = ", ".join(snippet.snippet_id for snippet in snippets[:2]) if snippets else "无"
    accept_lines = "\n".join(f"- {line}" for line in (cfg.get("prompt_accept") or []))
    reject_lines = "\n".join(f"- {line}" for line in (cfg.get("prompt_reject") or []))
    system_prompt = (
        "你是一名严格的财务报表视觉抽取助手。"
        "你将看到来自上市公司年报 PDF 的局部高清截图和整页截图，以及极少量 markdown 锚点线索。"
        "只能依据图片和线索判断。"
        "必须只输出 JSON。"
    )
    user_prompt = f"""
你现在只提取一个字段：{field_name}。
目标：{cfg.get("prompt_target")}

可接受口径：
{accept_lines}

排除项：
{reject_lines}

你会看到两张图片：
1. 以数字锚点为中心的局部高清裁剪图
2. 同一候选页的整页图

候选页：第 {chosen_page_number} 页。

Markdown 锚点如下：
[ANCHOR_LINES {anchor_window.start_line}-{anchor_window.end_line}]
{anchor_window.text.strip()}

目标行数字锚点（优先匹配）：
{" / ".join(target_numbers) if target_numbers else "无"}

可用 snippet id：
{snippet_ids}

要求：
- 只取 {doc.year} 年这一列，不要取上年对比列。
- 若目标行被拆成两行，请把两行合并判断。
- evidence 最多一句，尽量短。
- snippet_ids 填你参考到的 snippet id；若未实际参考 snippet，可填空数组。
- 只输出 JSON，不要输出任何解释。
""".strip()
    return system_prompt, user_prompt


def normalize_result(field_name: str, payload: Dict[str, object]) -> Dict[str, object]:
    field_payload = _coerce_field_payload((payload.get(field_name) if isinstance(payload, dict) else None))
    normalized_value = None
    field_kind = str(FIELD_CONFIGS[field_name].get("kind"))
    if field_kind == "money":
        normalized_value = normalize_money_to_yuan(field_payload.get("value"), str(field_payload.get("unit") or ""))
    elif field_kind == "shares":
        total_shares_wan = normalize_total_shares_to_wan(field_payload.get("value"), str(field_payload.get("unit") or ""))
        normalized_value = (float(total_shares_wan) * 10000.0) if total_shares_wan is not None else None
    return {
        "field": field_payload,
        "normalized_value": normalized_value,
        "ok": normalized_value is not None,
    }


def evaluate_against_baseline(field_name: str, current_raw_field: Dict[str, object], normalized_value: Optional[float]) -> bool:
    if field_name == "total_shares":
        baseline_wan = normalize_total_shares_to_wan(current_raw_field.get("value"), str(current_raw_field.get("unit") or ""))
        baseline = (float(baseline_wan) * 10000.0) if baseline_wan is not None else None
    else:
        baseline = normalize_money_to_yuan(current_raw_field.get("value"), str(current_raw_field.get("unit") or ""))
    if baseline is None:
        return normalized_value is not None
    if normalized_value is None:
        return False
    return abs(float(normalized_value) - float(baseline)) <= 0.5


def evaluate_result(
    *,
    field_name: str,
    current_raw_field: Dict[str, object],
    page_text_salvage: Dict[str, object],
    normalized_value: Optional[float],
) -> bool:
    if field_name in {"parent_netprofit", "operating_cashflow"}:
        salvage_value = normalize_money_to_yuan(
            page_text_salvage.get("current_value"),
            str((page_text_salvage.get("unit") or current_raw_field.get("unit") or "")),
        )
        if salvage_value is not None and normalized_value is not None:
            return abs(float(normalized_value) - float(salvage_value)) <= 0.5
    if field_name == "total_shares":
        salvage_wan = normalize_total_shares_to_wan(
            page_text_salvage.get("current_value"),
            str((page_text_salvage.get("unit") or current_raw_field.get("unit") or "")),
        )
        salvage_value = (float(salvage_wan) * 10000.0) if salvage_wan is not None else None
        if salvage_value is not None and normalized_value is not None:
            return abs(float(normalized_value) - float(salvage_value)) <= 0.5
    return evaluate_against_baseline(field_name, current_raw_field, normalized_value)


def build_direct_field_payload(
    field_name: str,
    *,
    value: object,
    unit: Optional[str],
    evidence: str,
    snippet_ids: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    return {
        field_name: {
            "value": value,
            "unit": unit,
            "evidence": str(evidence or "")[:240],
            "snippet_ids": [str(x) for x in (snippet_ids or []) if str(x).strip()],
        }
    }


def choose_direct_page_text_hit(
    doc: fitz.Document,
    *,
    ranked_pages: Sequence[Dict[str, object]],
    field_name: str,
    current_raw_field: Dict[str, object],
    max_pages: int = 5,
) -> Optional[Dict[str, object]]:
    for page_info in list(ranked_pages)[: max(1, int(max_pages))]:
        page = doc[int(page_info["page_index"])]
        hint = salvage_value_from_page_text(page, field_name)
        if hint is None or hint.current_value in (None, ""):
            continue
        page_text = page.get_text("text") or ""
        if field_name == "total_shares":
            unit = hint.unit or infer_share_unit_from_text(page_text, str(current_raw_field.get("unit") or ""))
        else:
            unit = hint.unit or infer_money_unit_from_text(page_text, str(current_raw_field.get("unit") or ""))
        payload = build_direct_field_payload(
            field_name,
            value=hint.current_value,
            unit=unit,
            evidence=hint.evidence_line or page_text[:160],
        )
        normalized = normalize_result(field_name, payload)
        if normalized.get("normalized_value") is None:
            continue
        return {
            "page_info": dict(page_info),
            "hint": hint,
            "payload": payload,
            "normalized": normalized,
        }
    return None


def run_one_field(
    doc_meta: SmokeDoc,
    *,
    field_name: str,
    out_dir: Path,
    model: str,
    api_base_url: str,
    api_key: str,
    timeout: int,
    dpi_crop: int,
    dpi_page: int,
    max_snippets: int,
    max_chars: int,
) -> Dict[str, object]:
    task_out_dir = out_dir / f"{doc_meta.stock_code}_{doc_meta.year}" / field_name
    task_out_dir.mkdir(parents=True, exist_ok=True)
    markdown_text = doc_meta.md_path.read_text(encoding="utf-8", errors="ignore")
    snippets = retrieve_snippets(
        markdown_text=markdown_text,
        field_name=field_name,
        year=doc_meta.year,
        max_snippets=max_snippets,
        max_chars=max_chars,
    )
    current_raw_field = read_current_raw_field(doc_meta.raw_json_path, field_name)
    marker_pages: List[int] = []
    for snippet in snippets:
        marker_pages.extend(_parse_page_markers(snippet.text))
    marker_pages = sorted({max(1, p) for p in marker_pages})
    doc = fitz.open(doc_meta.pdf_path)
    try:
        anchor_source: Optional[str] = "markdown"
        pdf_fallback_pages: List[int] = []
        md_llm_candidates: List[Dict[str, object]] = []
        llm_markdown_snippets: List[Snippet] = []
        anchor_window = choose_anchor_window(markdown_text, snippets, field_name)
        markdown_viable = is_viable_anchor_window(anchor_window, field_name)
        needs_markdown_llm = ENABLE_MARKDOWN_LLM_FALLBACK and (
            anchor_window is None
            or not markdown_viable
            or (field_name == "capex" and len(anchor_window.numbers) == 0)
        )
        llm_anchor_window: Optional[AnchorWindow] = None
        if needs_markdown_llm:
            llm_anchor_window, llm_markdown_snippets, md_llm_candidates = choose_anchor_window_with_markdown_llm(
                markdown_text=markdown_text,
                field_name=field_name,
                year=doc_meta.year,
                model=model,
                api_base_url=api_base_url,
                api_key=api_key,
                timeout=timeout,
            )
            for snippet in llm_markdown_snippets:
                marker_pages.extend(_parse_page_markers(snippet.text))
            marker_pages = sorted({max(1, p) for p in marker_pages})
        llm_markdown_viable = llm_anchor_window is not None and (
            len(llm_anchor_window.numbers) >= 1
            or (float(llm_anchor_window.score) >= 82.0 and bool(llm_anchor_window.matched_terms))
        )
        if llm_markdown_viable and (
            anchor_window is None
            or not markdown_viable
            or float(llm_anchor_window.score) >= float(anchor_window.score) + 4.0
        ):
            anchor_window = llm_anchor_window
            anchor_source = "markdown_llm_fallback"
            markdown_viable = True
        needs_pdf_competition = (
            anchor_window is None
            or not markdown_viable
            or float(anchor_window.score) < 12.0
            or (field_name == "capex" and len(anchor_window.numbers) == 0)
        )
        pdf_anchor_window: Optional[AnchorWindow] = None
        if needs_pdf_competition:
            pdf_anchor_window, pdf_fallback_pages = choose_anchor_window_from_pdf_text(doc, field_name)
            if pdf_fallback_pages:
                marker_pages = sorted({*marker_pages, *[max(1, int(page_no)) for page_no in pdf_fallback_pages]})
        pdf_viable = is_viable_anchor_window(pdf_anchor_window, field_name)
        if pdf_viable and (
            anchor_window is None
            or not markdown_viable
            or float(pdf_anchor_window.score) >= float(anchor_window.score) + 6.0
        ):
            anchor_window = pdf_anchor_window
            anchor_source = "pdf_text_fallback"
        elif pdf_anchor_window is not None and pdf_fallback_pages and (anchor_window is None or not markdown_viable):
            anchor_window = pdf_anchor_window
            anchor_source = "pdf_text_weak_fallback"
        elif anchor_window is not None and markdown_viable:
            anchor_source = "markdown"
        else:
            anchor_window = None
        if anchor_window is None:
            fallback_page_numbers = sorted(
                {
                    max(1, int(page_no))
                    for page_no in [*marker_pages, *pdf_fallback_pages]
                    if int(page_no) >= 1 and int(page_no) <= int(doc.page_count)
                }
            )
            fallback_ranked_pages = [
                {"page_index": int(page_no) - 1, "page_number": int(page_no), "score": 0.0, "number_hits": 0}
                for page_no in fallback_page_numbers
            ]
            direct_page_text_hit = choose_direct_page_text_hit(
                doc,
                ranked_pages=fallback_ranked_pages,
                field_name=field_name,
                current_raw_field=current_raw_field,
                max_pages=max(1, len(fallback_ranked_pages)),
            )
            if direct_page_text_hit is not None:
                best_page = dict(direct_page_text_hit["page_info"])
                page_text_salvage = direct_page_text_hit["hint"]
                raw_model_payload = direct_page_text_hit["payload"]
                normalized = direct_page_text_hit["normalized"]
                result = {
                    "task": {
                        "year": doc_meta.year,
                        "stock_code": doc_meta.stock_code,
                        "report_name": doc_meta.report_name,
                        "field_name": field_name,
                        "pdf_path": str(doc_meta.pdf_path),
                        "markdown_path": str(doc_meta.md_path),
                        "raw_json_path": str(doc_meta.raw_json_path),
                    },
                    "anchor_source": "page_text_direct_without_anchor",
                    "md_llm_candidates": md_llm_candidates,
                    "anchor_window": None,
                    "target_number_hint": None,
                    "marker_pages_from_snippets": marker_pages,
                    "pdf_fallback_pages": pdf_fallback_pages,
                    "candidate_pages_top5": fallback_ranked_pages[:5],
                    "chosen_page": {
                        "page_index": int(best_page["page_index"]),
                        "page_number": int(best_page["page_number"]),
                        "crop_rect": None,
                        "numeric_hit_map": {},
                        "label_crop_used": False,
                    },
                    "images": {},
                    "current_raw_field": current_raw_field,
                    "page_text_salvage": {
                        "current_value": page_text_salvage.current_value,
                        "prior_value": page_text_salvage.prior_value,
                        "source": page_text_salvage.source,
                        "evidence_line": page_text_salvage.evidence_line,
                        "unit": page_text_salvage.unit,
                    },
                    "model_response": raw_model_payload,
                    "normalized": normalized,
                    "elapsed_sec": 0.0,
                    "direct_extraction": True,
                }
                (task_out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                return result
            result = {
                "task": {"year": doc_meta.year, "stock_code": doc_meta.stock_code, "field_name": field_name},
                "error": "no_anchor_window",
                "anchor_source": None,
                "marker_pages_from_snippets": marker_pages,
                "md_llm_candidates": md_llm_candidates,
                "pdf_fallback_pages": pdf_fallback_pages,
                "current_raw_field": current_raw_field,
            }
            (task_out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result

        target_hint = infer_target_number_hint(anchor_window, field_name)
        anchor_numbers = [x for x in [target_hint.current_value, target_hint.prior_value] if x] or anchor_window.numbers

        ranked_pages = rank_candidate_pages(
            doc,
            field_name=field_name,
            anchor_numbers=anchor_numbers,
            marker_pages=[p - 1 for p in marker_pages],
        )
        if not ranked_pages:
            result = {
                "task": {"year": doc_meta.year, "stock_code": doc_meta.stock_code, "field_name": field_name},
                "error": "no_candidate_pdf_page",
                "anchor_source": anchor_source,
                "md_llm_candidates": md_llm_candidates,
                "anchor_window": {
                    "start_line": anchor_window.start_line,
                    "end_line": anchor_window.end_line,
                    "matched_terms": anchor_window.matched_terms,
                    "numbers": anchor_window.numbers,
                    "score": anchor_window.score,
                    "text": anchor_window.text,
                },
                "target_number_hint": {
                    "current_value": target_hint.current_value,
                    "prior_value": target_hint.prior_value,
                    "source": target_hint.source,
                    "evidence_line": target_hint.evidence_line,
                    "unit": target_hint.unit,
                },
                "marker_pages_from_snippets": marker_pages,
                "pdf_fallback_pages": pdf_fallback_pages,
                "current_raw_field": current_raw_field,
            }
            (task_out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result

        direct_started_at = time.time()
        direct_page_text_hit = choose_direct_page_text_hit(
            doc,
            ranked_pages=ranked_pages,
            field_name=field_name,
            current_raw_field=current_raw_field,
            max_pages=5,
        )
        if direct_page_text_hit is not None:
            best_page = dict(direct_page_text_hit["page_info"])
            page_text_salvage = direct_page_text_hit["hint"]
            raw_model_payload = direct_page_text_hit["payload"]
            normalized = direct_page_text_hit["normalized"]
            page_index = int(best_page["page_index"])
            elapsed_sec = round(time.time() - direct_started_at, 2)
            result = {
                "task": {
                    "year": doc_meta.year,
                    "stock_code": doc_meta.stock_code,
                    "report_name": doc_meta.report_name,
                    "field_name": field_name,
                    "pdf_path": str(doc_meta.pdf_path),
                    "markdown_path": str(doc_meta.md_path),
                    "raw_json_path": str(doc_meta.raw_json_path),
                },
                "anchor_source": f"{anchor_source}_page_text_direct",
                "md_llm_candidates": md_llm_candidates,
                "anchor_window": {
                    "start_line": anchor_window.start_line,
                    "end_line": anchor_window.end_line,
                    "matched_terms": anchor_window.matched_terms,
                    "numbers": anchor_window.numbers,
                    "score": anchor_window.score,
                    "text": anchor_window.text,
                },
                "target_number_hint": {
                    "current_value": target_hint.current_value,
                    "prior_value": target_hint.prior_value,
                    "source": target_hint.source,
                    "evidence_line": target_hint.evidence_line,
                    "unit": target_hint.unit,
                },
                "marker_pages_from_snippets": marker_pages,
                "pdf_fallback_pages": pdf_fallback_pages,
                "candidate_pages_top5": ranked_pages[:5],
                "chosen_page": {
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "crop_rect": None,
                    "numeric_hit_map": {},
                    "label_crop_used": False,
                },
                "images": {},
                "current_raw_field": current_raw_field,
                "page_text_salvage": {
                    "current_value": page_text_salvage.current_value,
                    "prior_value": page_text_salvage.prior_value,
                    "source": page_text_salvage.source,
                    "evidence_line": page_text_salvage.evidence_line,
                    "unit": page_text_salvage.unit,
                },
                "model_response": raw_model_payload,
                "normalized": normalized,
                "elapsed_sec": elapsed_sec,
                "direct_extraction": True,
            }
            (task_out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result

        best_page = ranked_pages[0]
        page_index = int(best_page["page_index"])
        page = doc[page_index]
        page_text = page.get_text("text") or ""
        label_crop_rect, label_hit_map = compute_label_crop_rect(page, field_name)
        crop_rect, numeric_hit_map = compute_crop_rect(page, anchor_numbers)
        label_crop_used = False
        if label_crop_rect is not None:
            if field_name in {"operating_cashflow", "parent_netprofit", "total_shares"}:
                crop_rect = label_crop_rect
                numeric_hit_map = label_hit_map
                label_crop_used = True
            elif field_name == "capex" and (not anchor_numbers or not numeric_hit_map):
                crop_rect = label_crop_rect
                numeric_hit_map = label_hit_map
                label_crop_used = True
        crop_png = render_rect_to_png_bytes(page, crop_rect, dpi=dpi_crop)
        page_png = render_rect_to_png_bytes(page, fitz.Rect(page.rect), dpi=dpi_page)
        crop_path = task_out_dir / "crop.png"
        page_path = task_out_dir / "page.png"
        write_bytes(crop_path, crop_png)
        write_bytes(page_path, page_png)

        prompt_snippets: List[Snippet] = list(snippets)
        seen_snippet_ids = {snippet.snippet_id for snippet in prompt_snippets}
        for snippet in llm_markdown_snippets:
            if snippet.snippet_id in seen_snippet_ids:
                continue
            prompt_snippets.append(snippet)
            seen_snippet_ids.add(snippet.snippet_id)

        system_prompt, user_prompt = build_prompt(
            doc=doc_meta,
            field_name=field_name,
            snippets=prompt_snippets,
            anchor_window=anchor_window,
            target_hint=target_hint,
            chosen_page_number=page_index + 1,
        )
        schema = build_vision_schema(field_name)
        t0 = time.time()
        raw_model_payload = call_lm_studio_vision_json(
            model=model,
            api_base_url=api_base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=[base64.b64encode(crop_png).decode("utf-8"), base64.b64encode(page_png).decode("utf-8")],
            response_schema=schema,
            timeout=timeout,
        )
        page_text_salvage = salvage_value_from_page_text(page, field_name)
        if page_text_salvage is not None and page_text_salvage.current_value is not None:
            payload = _coerce_field_payload(raw_model_payload.get(field_name))
            payload["value"] = page_text_salvage.current_value
            if payload.get("unit") in (None, ""):
                if field_name == "total_shares":
                    inferred_unit = infer_share_unit_from_text(
                        page_text,
                        page_text_salvage.unit or str(current_raw_field.get("unit") or ""),
                    )
                else:
                    inferred_unit = infer_money_unit_from_text(
                        page_text,
                        page_text_salvage.unit or str(current_raw_field.get("unit") or ""),
                    )
                if inferred_unit:
                    payload["unit"] = inferred_unit
            if payload.get("evidence") in (None, ""):
                payload["evidence"] = page_text_salvage.evidence_line[:160]
            raw_model_payload[field_name] = payload
        elapsed_sec = round(time.time() - t0, 2)
        normalized = normalize_result(field_name, raw_model_payload)

        result = {
            "task": {
                "year": doc_meta.year,
                "stock_code": doc_meta.stock_code,
                "report_name": doc_meta.report_name,
                "field_name": field_name,
                "pdf_path": str(doc_meta.pdf_path),
                "markdown_path": str(doc_meta.md_path),
                "raw_json_path": str(doc_meta.raw_json_path),
            },
            "anchor_source": anchor_source,
            "md_llm_candidates": md_llm_candidates,
            "anchor_window": {
                "start_line": anchor_window.start_line,
                "end_line": anchor_window.end_line,
                "matched_terms": anchor_window.matched_terms,
                "numbers": anchor_window.numbers,
                "score": anchor_window.score,
                "text": anchor_window.text,
            },
            "target_number_hint": {
                "current_value": target_hint.current_value,
                "prior_value": target_hint.prior_value,
                "source": target_hint.source,
                "evidence_line": target_hint.evidence_line,
                "unit": target_hint.unit,
            },
            "marker_pages_from_snippets": marker_pages,
            "pdf_fallback_pages": pdf_fallback_pages,
            "candidate_pages_top5": ranked_pages[:5],
            "chosen_page": {
                "page_index": page_index,
                "page_number": page_index + 1,
                "crop_rect": [float(crop_rect.x0), float(crop_rect.y0), float(crop_rect.x1), float(crop_rect.y1)],
                "numeric_hit_map": numeric_hit_map,
                "label_crop_used": label_crop_used,
            },
            "images": {"crop_path": str(crop_path), "page_path": str(page_path)},
            "current_raw_field": current_raw_field,
            "page_text_salvage": {
                "current_value": (page_text_salvage.current_value if page_text_salvage else None),
                "prior_value": (page_text_salvage.prior_value if page_text_salvage else None),
                "source": (page_text_salvage.source if page_text_salvage else None),
                "evidence_line": (page_text_salvage.evidence_line if page_text_salvage else None),
                "unit": (page_text_salvage.unit if page_text_salvage else None),
            },
            "model_response": raw_model_payload,
            "normalized": normalized,
            "elapsed_sec": elapsed_sec,
        }
        (task_out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        doc.close()


def write_progress(out_dir: Path, rows: Sequence[Dict[str, object]], total: int) -> None:
    done = len(rows)
    ok = sum(1 for row in rows if row.get("ok"))
    failed = done - ok
    lines = [
        f"updated_at={time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"total={total}",
        f"done={done}",
        f"ok={ok}",
        f"failed={failed}",
    ]
    for row in rows:
        lines.append(
            f"{row.get('stock_code')},{row.get('field_name')},{row.get('page_number')},{row.get('value')},{row.get('ok')},{row.get('elapsed_sec')}"
        )
    (out_dir / "progress_status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test hybrid PDF+Markdown extraction for multiple fields")
    parser.add_argument("--partial-csv", default=str(REPO_ROOT / ".tmp_gemma_markdown_repair_runner" / "partial_tasks_latest.csv"))
    parser.add_argument("--fulltext-root", default=str(REPO_ROOT / "年报" / "下载年报_fulltext"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / ".tmp_gemma_pdf_hybrid_fields_smoke"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--codes", nargs="+", default=["000166", "000651", "000776"])
    parser.add_argument("--fields", nargs="+", default=["parent_netprofit", "total_shares", "operating_cashflow", "capex"])
    parser.add_argument("--model", default="google/gemma-4-26b-a4b")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--dpi-crop", type=int, default=220)
    parser.add_argument("--dpi-page", type=int, default=150)
    parser.add_argument("--max-snippets", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=12000)
    args = parser.parse_args()

    fields = [str(x).strip() for x in args.fields if str(x).strip()]
    unsupported = [field for field in fields if field not in FIELD_CONFIGS]
    if unsupported:
        raise RuntimeError(f"Unsupported fields: {unsupported}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_smoke_docs(
        Path(args.partial_csv),
        fulltext_root=Path(args.fulltext_root),
        year=int(args.year),
        codes=list(args.codes),
    )
    if not docs:
        raise RuntimeError("No smoke docs matched the requested year/codes.")

    total = len(docs) * len(fields)
    rows: List[Dict[str, object]] = []
    write_progress(out_dir, rows, total)

    for doc_meta in docs:
        for field_name in fields:
            result = run_one_field(
                doc_meta,
                field_name=field_name,
                out_dir=out_dir,
                model=str(args.model),
                api_base_url=str(args.api_base_url),
                api_key=str(args.api_key),
                timeout=int(args.timeout),
                dpi_crop=int(args.dpi_crop),
                dpi_page=int(args.dpi_page),
                max_snippets=int(args.max_snippets),
                max_chars=int(args.max_chars),
            )
            normalized = result.get("normalized") or {}
            field_payload = normalized.get("field") or {}
            current_raw_field = result.get("current_raw_field") or {}
            page_text_salvage = result.get("page_text_salvage") or {}
            normalized_value = normalized.get("normalized_value")
            ok_match = evaluate_result(
                field_name=field_name,
                current_raw_field=current_raw_field,
                page_text_salvage=page_text_salvage,
                normalized_value=normalized_value,
            )
            rows.append(
                {
                    "year": doc_meta.year,
                    "stock_code": doc_meta.stock_code,
                    "report_name": doc_meta.report_name,
                    "field_name": field_name,
                    "page_number": ((result.get("chosen_page") or {}).get("page_number")),
                    "current_raw_value": current_raw_field.get("value"),
                    "value": field_payload.get("value") if isinstance(field_payload, dict) else None,
                    "unit": field_payload.get("unit") if isinstance(field_payload, dict) else None,
                    "normalized_value": normalized_value,
                    "ok": ok_match,
                    "elapsed_sec": result.get("elapsed_sec"),
                }
            )
            write_progress(out_dir, rows, total)

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "stock_code",
                "report_name",
                "field_name",
                "page_number",
                "current_raw_value",
                "value",
                "unit",
                "normalized_value",
                "ok",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Smoke finished: {summary_path}")
    print(f"ok={sum(1 for row in rows if row.get('ok'))}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
