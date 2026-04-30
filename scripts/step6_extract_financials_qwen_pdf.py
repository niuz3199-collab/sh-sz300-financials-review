#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step6 (PDF/Qwen): Extract annual-report key fields from downloaded PDF reports using local Ollama Qwen3.5 vision.

What it does
- Traverse `年报/下载年报_fulltext/<year>/*.pdf`
- For each (stock_code, report_year) PDF:
  - Render 3 key pages to high-res images (default DPI=200)
  - Call local Ollama model `qwen3.5:9b` (multimodal) to extract:
      1) 归属于母公司股东的净利润 (归母净利润)
      2) 期末总股本 / 期末普通股股份总数 (总股本)
      3) 经营活动产生的现金流量净额
      4) 购建固定资产、无形资产和其他长期资产支付的现金 (CapEx)
  - Normalize units:
      - net profit / cashflows / capex => 元
      - total shares => 万股 + also compute shares (股) for downstream market-cap calc
- Save per-year CSV to `<base_dir>/<year>/{year}_财报数据.csv` (or a custom filename)
- Save raw JSON evidence to `<out-dir>/raw_json/<year>/<code>.json`
- Maintain a run log `<out-dir>/extract_log.csv` for resume.

Why this script
- Your requirement is to follow Qwen recommended workflow: PDF -> high-res images -> model.
- Sample script `scripts/qwen_pdf_extract_sample.py` validated the approach.

Notes
- This is CPU/GPU heavy: expect long runtime for full dataset. Use `--resume`.
- For scanned PDFs with no text layer, page auto-location may fail; those tasks are logged as `no_key_pages`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import requests


OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OPENAI_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
DEFAULT_OLLAMA_NUM_CTX = 8192


@dataclass(frozen=True)
class Task:
    year: int
    stock_code: str
    pdf_path: Path


def resolve_ollama_num_ctx(default: int = DEFAULT_OLLAMA_NUM_CTX) -> int:
    raw = str(os.environ.get("OLLAMA_NUM_CTX", "") or "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return int(default)


def normalize_stock_code(raw: str) -> str:
    text = str(raw or "").strip()
    m = re.search(r"\d+", text)
    if not m:
        return ""
    return m.group(0).zfill(6)


def infer_code_name(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if not code:
        return ""
    if code.startswith(("60", "68", "90", "70")):
        return f"sh.{code}"
    if code.startswith(("00", "20", "30")):
        return f"sz.{code}"
    if code.startswith(("8", "4")):
        return f"bj.{code}"
    return f"sz.{code}"


def _clean_for_match(text: str) -> str:
    # Normalize for cheap substring matching (case-insensitive for English, no-op for Chinese).
    return re.sub(r"\s+", "", text or "").lower()


def _find_pages_containing_all(
    doc: fitz.Document,
    keywords: Iterable[str],
    *,
    max_pages: int = 10,
    page_range: Optional[range] = None,
) -> List[int]:
    keys = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    if not keys:
        return []
    hits: List[int] = []
    rng = page_range if page_range is not None else range(doc.page_count)
    for i in rng:
        t = _clean_for_match(doc[i].get_text("text"))
        if all(k in t for k in keys):
            hits.append(i)
            if len(hits) >= int(max_pages):
                break
    return hits


def _find_pages_containing_any(
    doc: fitz.Document,
    keywords: Iterable[str],
    *,
    max_pages: int = 10,
    page_range: Optional[range] = None,
) -> List[int]:
    keys = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    if not keys:
        return []
    hits: List[int] = []
    rng = page_range if page_range is not None else range(doc.page_count)
    for i in rng:
        t = _clean_for_match(doc[i].get_text("text"))
        if any(k in t for k in keys):
            hits.append(i)
            if len(hits) >= int(max_pages):
                break
    return hits


def _find_first_anchor_page(
    doc: fitz.Document,
    keywords: Iterable[str],
    *,
    page_range: Optional[range] = None,
) -> Optional[int]:
    hits = _find_pages_containing_any(doc, keywords, max_pages=1, page_range=page_range)
    return hits[0] if hits else None


def _section_ranges_for_legacy_report(doc: fitz.Document) -> Dict[str, range]:
    search_range = range(0, min(80, doc.page_count))
    summary_anchor = _find_first_anchor_page(
        doc,
        [
            "主要会计数据",
            "会计数据和业务数据摘要",
            "会计数据与业务数据摘要",
            "财务摘要",
            "主要财务数据",
            "financialhighlight",
            "financialhighlights",
            "financialhighlightsandbusinesshighlights",
            "majoraccountingdata",
            "extractsofaccountingandoperatingdata",
            "keyaccountingdataandfinancialindicator",
        ],
        page_range=search_range,
    )
    shares_anchor = _find_first_anchor_page(
        doc,
        [
            "股本变动及股东情况",
            "股份变动情况",
            "股本变动情况",
            "股东情况",
            "changesinsharecapitalandparticularsabouttheshareholders",
            "changesinsharecapital",
            "statementofchangesinsharecapital",
            "particularsabouttheshareholders",
        ],
        page_range=search_range,
    )
    financial_anchor = _find_first_anchor_page(
        doc,
        [
            "财务报告",
            "财务会计报告",
            "法定财务报告",
            "会计报告",
            "financialstatements",
            "financialstatement",
            "reportoftheinternationalauditors",
            "reportoftheindependentauditors",
            "auditorsreport",
        ],
        page_range=search_range,
    )

    def make_range(start: Optional[int], fallback_start: int, end_candidates: List[Optional[int]]) -> range:
        start_page = start if start is not None else fallback_start
        end_page = doc.page_count
        for candidate in end_candidates:
            if candidate is not None and candidate > start_page:
                end_page = min(end_page, candidate)
        end_page = max(start_page + 1, end_page)
        return range(max(0, start_page), min(doc.page_count, end_page))

    return {
        "summary": make_range(summary_anchor, 0, [shares_anchor, financial_anchor]),
        "shares": make_range(shares_anchor, 0, [financial_anchor]),
        "financials": make_range(financial_anchor, 0, []),
    }


def _pick_best_scored_page(
    doc: fitz.Document,
    page_indexes: Iterable[int],
    *,
    scorer,
) -> Optional[int]:
    best_page: Optional[int] = None
    best_score = -10**9
    seen: set[int] = set()
    for page_index in page_indexes:
        if page_index in seen:
            continue
        seen.add(page_index)
        text = _clean_for_match(doc[page_index].get_text("text"))
        score = scorer(text, page_index)
        if score > best_score:
            best_score = score
            best_page = page_index
    return best_page


def _score_shares_page_text(text: str, year: int, *, legacy: bool) -> int:
    year_str = str(int(year))
    text_num = text.replace(",", "").replace("，", "")
    score = 0

    if f"{year_str}年12月31日" in text or f"截至{year_str}年12月31日" in text:
        score += 4
    if f"31december{year_str}" in text or f"december31{year_str}" in text:
        score += 4

    if any(k in text for k in ["股份总数", "总股本", "期末总股本", "期末普通股股份总数", "期末股份总数"]):
        score += 30
    if any(k in text for k in ["totalnumberofshares", "totalshares", "totalsharecapital"]):
        score += 28
    if "sharecapital" in text or "issuedsharecapital" in text or "股本" in text:
        score += 6
    if "实收资本" in text:
        score += 4

    if re.search(r"\d{8,}", text_num):
        score += 3
    if any(k in text for k in ["单位：股", "单位:股", "数量单位股", "unit:share", "unit:shares"]):
        score += 6
    if "amountattheyear-end" in text:
        score += 12
    if "本次变动前" in text and "本次变动后" in text:
        score += 8

    if "股份变动情况表" in text or "股本变动情况表" in text:
        score += 10
    elif "股份变动情况" in text or "股本变动情况" in text:
        score += 6
    if "changeinsharecapital" in text or "changesinsharecapital" in text:
        score += 8

    if "股东权益变动情况" in text and "期末数" in text and "股本" in text:
        score += 12

    if any(k in text for k in ["利润分配", "分配预案", "每10股", "profitdistribution", "dividend"]):
        score -= 8
    if any(
        k in text
        for k in [
            "前三年历次股票发行情况",
            "股票发行与上市情况",
            "前次股票发行情况",
            "内部职工股情况",
            "前10名股东持股情况",
            "股东情况介绍",
            "top10shareholders",
        ]
    ):
        score -= 30
    if "期末股份总数" in text and ("每股收益" in text or "全面摊薄" in text or "加权平均" in text or "eps=" in text):
        score -= 40
    if "股东权益变动情况" in text and "股份变动情况" not in text and "股本变动情况" not in text:
        score -= 6

    if legacy and "股份总数" in text and ("本次变动后" in text or "期末数" in text or "amountattheyear-end" in text):
        score += 10
    return score


def _pick_legacy_income_page(doc: fitz.Document, year: int, ranges: Dict[str, range]) -> Optional[int]:
    year_str = str(int(year))
    summary_titles = [
        "主要会计数据",
        "会计数据和业务数据摘要",
        "会计数据与业务数据摘要",
        "财务摘要",
        "主要财务数据",
        "financialhighlight",
        "financialhighlights",
        "financialhighlightsandbusinesshighlights",
        "majoraccountingdata",
        "extractsofaccountingandoperatingdata",
        "keyaccountingdataandfinancialindicator",
    ]
    table_titles = [
        "利润及利润分配表",
        "利润分配表",
        "利润表",
        "statementofincome",
        "incomestatement",
        "consolidatedstatementofincome",
        "statementofprofitappropriation",
        "profitappropriationstatement",
    ]
    profit_keys = [
        "净利润",
        "netprofit",
        "profitfortheyear",
        "profitattributabletoshareholdersoftheparentcompany",
        "profitattributabletoshareholdersoftheparent",
        "netprofitattributabletoshareholdersoftheparentcompany",
        "netprofitattributabletoshareholdersoftheparent",
    ]

    def score_summary(text: str, _page_index: int) -> int:
        score = 0
        if any(_clean_for_match(k) in text for k in summary_titles):
            score += 8
        if any(_clean_for_match(k) in text for k in profit_keys):
            score += 8
        if year_str in text:
            score += 2
        if "rmb" in text or "货币单位" in text or "单位" in text:
            score += 1
        if re.search(r"\d{6,}", text.replace(",", "")):
            score += 1
        return score

    summary_candidates = _find_pages_containing_any(
        doc,
        summary_titles + profit_keys,
        max_pages=12,
        page_range=ranges["summary"],
    )
    summary_hits = [
        page_index
        for page_index in summary_candidates
        if any(_clean_for_match(k) in _clean_for_match(doc[page_index].get_text("text")) for k in profit_keys)
    ]
    if summary_hits:
        best_summary = _pick_best_scored_page(doc, summary_hits, scorer=score_summary)
        if best_summary is not None:
            return best_summary

    def score_financial(text: str, _page_index: int) -> int:
        score = 0
        if "利润及利润分配表" in text:
            score += 12
        if "利润分配表" in text:
            score += 10
        if "利润表" in text:
            score += 6
        if "statementofincome" in text or "consolidatedstatementofincome" in text:
            score += 12
        if "incomestatement" in text:
            score += 10
        if "statementofprofitappropriation" in text or "profitappropriationstatement" in text:
            score += 10
        if any(_clean_for_match(k) in text for k in profit_keys):
            score += 10
        if year_str in text:
            score += 2
        if "rmb" in text or "货币单位" in text or "单位" in text:
            score += 1
        return score

    financial_candidates = _find_pages_containing_any(
        doc,
        table_titles + profit_keys,
        max_pages=20,
        page_range=ranges["financials"],
    )
    financial_hits = [
        page_index
        for page_index in financial_candidates
        if any(_clean_for_match(k) in _clean_for_match(doc[page_index].get_text("text")) for k in table_titles)
        and any(_clean_for_match(k) in _clean_for_match(doc[page_index].get_text("text")) for k in profit_keys)
    ]
    if financial_hits:
        return _pick_best_scored_page(doc, financial_hits, scorer=score_financial)

    return None


def _pick_legacy_shares_page(doc: fitz.Document, year: int, ranges: Dict[str, range]) -> Optional[int]:
    candidate_pages = _find_pages_containing_any(
        doc,
        [
            "股本变动情况",
            "股份变动情况",
            "股份变动情况表",
            "总股本",
            "期末总股本",
            "期末普通股股份总数",
            "期末股份总数",
            "本次变动前",
            "本次变动后",
            "股份总数",
            "股东权益变动情况",
            "sharecapital",
            "totalnumberofshares",
            "totalshares",
            "changeinsharecapital",
            "changesinsharecapital",
            "amountattheyear-end",
        ],
        max_pages=30,
        page_range=ranges["shares"],
    )
    if not candidate_pages:
        return None

    def score(text: str, _page_index: int) -> int:
        return _score_shares_page_text(text, year, legacy=True)

    return _pick_best_scored_page(doc, candidate_pages, scorer=score)


def _collect_legacy_shares_context_pages(doc: fitz.Document, year: int, primary_page: int) -> List[int]:
    ranges = _section_ranges_for_legacy_report(doc)
    pages: List[int] = []
    primary_page = int(primary_page)
    primary_text = _clean_for_match(doc[primary_page].get_text("text"))

    prev_page = primary_page - 1
    if prev_page in ranges["shares"]:
        prev_text = _clean_for_match(doc[prev_page].get_text("text"))
        if (
            any(k in primary_text for k in ["股份总数", "总股本", "期末总股本", "期末普通股股份总数", "amountattheyear-end", "totalnumberofshares", "totalshares"])
            and any(k in prev_text for k in ["股份变动情况表", "股本变动情况表", "股份变动情况", "股本变动情况", "本次变动前", "本次变动后", "单位：股", "单位:股", "数量单位股", "unit:share", "unit:shares"])
        ):
            pages.append(prev_page)

    pages.append(primary_page)

    next_page = primary_page + 1
    if next_page in ranges["shares"]:
        next_text = _clean_for_match(doc[next_page].get_text("text"))
        if (
            any(k in primary_text for k in ["股份变动情况表", "股本变动情况表", "股份变动情况", "股本变动情况", "本次变动前", "本次变动后"])
            and not any(k in primary_text for k in ["股份总数", "总股本", "期末总股本", "期末普通股股份总数", "amountattheyear-end", "totalnumberofshares", "totalshares"])
        ) or any(k in next_text for k in ["股份总数", "总股本", "期末总股本", "期末普通股股份总数", "amountattheyear-end", "totalnumberofshares", "totalshares"]):
            pages.append(next_page)

    deduped: List[int] = []
    for page_index in pages:
        if 0 <= int(page_index) < doc.page_count and int(page_index) not in deduped:
            deduped.append(int(page_index))
    return deduped[:3]


def _resolve_field_page_from_evidence(field_payload: Dict[str, object], page_indexes: Iterable[int], fallback_page: Optional[int]) -> Optional[int]:
    indexes = [int(page_index) for page_index in page_indexes]
    evidence = str((field_payload or {}).get("evidence") or "")
    match = re.search(r"PAGE_(\d+)_TEXT", evidence, flags=re.IGNORECASE)
    if match:
        mapped_index = int(match.group(1)) - 1
        if 0 <= mapped_index < len(indexes):
            return indexes[mapped_index] + 1
    if fallback_page is None:
        return None
    return int(fallback_page) + 1


def _infer_unit_from_page_text(doc: fitz.Document, page_indexes: Iterable[int], *, field_kind: str) -> Optional[str]:
    texts = []
    seen: set[int] = set()
    for page_index in page_indexes:
        page_index = int(page_index)
        if page_index in seen or page_index < 0 or page_index >= doc.page_count:
            continue
        seen.add(page_index)
        texts.append(doc[page_index].get_text("text") or "")
    if not texts:
        return None

    raw_text = "\n".join(texts)
    lower = raw_text.lower().replace("’", "'")
    compact = re.sub(r"\s+", "", lower)

    if field_kind == "shares":
        if "数量单位股" in compact or "单位股" in compact or "unit:share" in compact or "unit:shares" in compact:
            return "股"
        if "单位万股" in compact:
            return "万股"
        if "单位亿股" in compact:
            return "亿股"
        return None

    if "rmb'000" in lower or "rmb'000" in compact or "rmb000" in compact or "thousandrmb" in compact:
        return "RMB'000"
    if ("rmb" in compact or "cny" in compact) and "million" in compact:
        return "RMB million"
    if "单位人民币元" in compact or "人民币元" in compact or "单位元" in compact:
        return "人民币元"
    if "单位万元" in compact or "人民币万元" in compact:
        return "万元"
    if "单位亿元" in compact or "人民币亿元" in compact:
        return "亿元"
    return None


def _coerce_field_payload(field_payload: object) -> Dict[str, object]:
    if isinstance(field_payload, dict):
        return dict(field_payload)
    if field_payload in (None, ""):
        return {}
    if isinstance(field_payload, (int, float, str)):
        return {"value": field_payload, "unit": None, "evidence": None}
    if isinstance(field_payload, (list, tuple)):
        if len(field_payload) == 1 and isinstance(field_payload[0], (int, float, str)):
            return {"value": field_payload[0], "unit": None, "evidence": None}
        try:
            evidence = json.dumps(field_payload, ensure_ascii=False)
        except Exception:
            evidence = str(field_payload)
        return {"value": None, "unit": None, "evidence": evidence}
    return {"value": str(field_payload), "unit": None, "evidence": None}


def _canonicalize_field_payload(
    field_payload: object,
    *,
    doc: fitz.Document,
    page_indexes: Iterable[int],
    field_kind: str,
) -> Dict[str, object]:
    payload = _coerce_field_payload(field_payload)
    if payload.get("value") in (None, ""):
        for alt_key in ["amount", "number", "amount_value", "numeric_value", "year_end_value", "amount_at_year_end"]:
            alt_value = payload.get(alt_key)
            if alt_value not in (None, ""):
                payload["value"] = alt_value
                break
    if payload.get("unit") in (None, ""):
        for alt_key in ["currency", "currency_unit", "measurement_unit", "share_unit", "unit_text"]:
            alt_value = payload.get(alt_key)
            if alt_value not in (None, ""):
                payload["unit"] = alt_value
                break
    if payload.get("unit") in (None, ""):
        inferred_unit = _infer_unit_from_page_text(doc, page_indexes, field_kind=field_kind)
        if inferred_unit:
            payload["unit"] = inferred_unit
    payload.setdefault("value", None)
    payload.setdefault("unit", None)
    payload.setdefault("evidence", None)
    return payload


def pick_key_pages(doc: fitz.Document, year: int) -> Dict[str, int]:
    """
    Return page indexes (0-based) for:
    - income: page that contains both "利润表" and "归属于…股东/所有者的净利润"（含银行“归属于本行股东的净利润”等变体）
    - cfo: page that contains "现金流量表" + "经营活动产生的现金流量净额"
    - capex: page that contains "现金流量表" + capex line
    - shares: page that contains "总股本" (prefer early pages)
    """
    out: Dict[str, int] = {}

    # Parent net profit can be shown as different labels in different industries/templates (e.g. banks).
    parent_profit_keys = [
        "归属于母公司股东的净利润",
        "归属于母公司所有者的净利润",
        "归属于本公司股东的净利润",
        "归属于本行股东的净利润",
        "归属于本集团股东的净利润",
        "归属于母公司普通股股东的净利润",
    ]
    # Some annual reports are English-only; match the row label (whitespace removed, lowercased).
    # Avoid overly broad keys like "shareholders" which may hit balance-sheet equity sections.
    parent_profit_keys_en = [
        "netprofitattributabletoshareholdersoftheparentcompany",
        "netprofitattributabletoshareholdersoftheparent",
        "profitattributabletoshareholdersoftheparentcompany",
        "profitattributabletoshareholdersoftheparent",
        "netprofitattributabletotheownersoftheparentcompany",
        "netprofitattributabletotheownersoftheparent",
        "profitattributabletotheownersoftheparentcompany",
        "profitattributabletotheownersoftheparent",
        "profitattributabletotheequityholdersoftheparentcompany",
        "profitattributabletotheequityholdersoftheparent",
    ]
    parent_profit_keys_all = parent_profit_keys + parent_profit_keys_en

    # Prefer statements pages (利润表) if they directly contain the parent-net-profit label.
    income_hits: List[int] = []
    for k in parent_profit_keys:
        income_hits = _find_pages_containing_all(doc, ["利润表", k], max_pages=3)
        if income_hits:
            break
        income_hits = _find_pages_containing_all(doc, ["合并利润表", k], max_pages=3)
        if income_hits:
            break

    # If not found, prefer early "会计数据和财务指标/关键指标" section to avoid picking footnotes.
    if not income_hits:
        early_range = range(0, min(60, doc.page_count))
        income_hits = _find_pages_containing_any(doc, parent_profit_keys_all, max_pages=5, page_range=early_range)

    # Last resort: anywhere in the PDF
    if not income_hits:
        income_hits = _find_pages_containing_any(doc, parent_profit_keys_all, max_pages=5)

    if not income_hits and int(year) <= 2006:
        legacy_ranges = _section_ranges_for_legacy_report(doc)
        legacy_income_page = _pick_legacy_income_page(doc, year, legacy_ranges)
        if legacy_income_page is not None:
            income_hits = [legacy_income_page]

    if income_hits:
        out["income"] = income_hits[0]

    # Cashflow statement may span pages; pick CFO and CapEx separately when needed
    cashflow_title_keys = [
        "现金流量表",
        "cashflowstatement",
        "cashflowstatements",
        "cashflowsstatement",
    ]
    cfo_keys = [
        "经营活动产生的现金流量净额",
        "netcashgeneratedfromoperatingactivities",
        "netcashflowsfromoperatingactivities",
        "netcashflowfromoperatingactivities",
        "netcashprovidedbyoperatingactivities",
        "netcashprovidedfromoperatingactivities",
        "netcashprovidebyoperatingactivities",
        "netcashfromoperatingactivities",
        "netcashinflowfromoperatingactivities",
        "netcashflowsarisingfromoperatingactivities",
    ]
    capex_variants = [
        "购建固定资产、无形资产及其他长期资产支付的现金",
        "购建固定资产、无形资产和其他长期资产支付的现金",
        "购建固定资产、无形资产及其他长期资产所支付的现金",
        "购建固定资产、无形资产和其他长期资产所支付的现金",
    ]
    capex_variants_en = [
        "paymentsfortheacquisitionandconstructionoffixedassets",
        "cashpaidfortheacquisitionandconstructionoffixedassets",
        "paymentsforthepurchaseoffixedassets",
        "purchaseoffixedassets,intangibleassetsandotherlong-termassets",
        "purchaseofproperty,plantandequipment",
        "purchasesofproperty,plantandequipment",
        "acquisitionofproperty,plantandequipment",
        "cashpaidtoacquireproperty,plantandequipment",
        "cashpaidtoacquireproperty,plantandequipmentandconstructioninprogress",
        "fixedassets,intangibleassetsandotherlong-termassets",
        "intangibleassetsandotherlong-termassets",
        "capitalexpenditure",
    ]
    capex_title_keys_all = capex_variants + capex_variants_en + ["购建固定资产", "fixedassets", "property,plantandequipment"]

    capex_keys_all = capex_variants + capex_variants_en + [
        "购建固定资产",
        "购建固定资产所支付的现金",
        "购建固定资产、无形资产等长期资产所支付的现金",
        "购建固定资产及其他长期资产所支付的现金",
    ]

    cfo_hits: List[int] = []
    for title in cashflow_title_keys:
        for k in cfo_keys:
            cfo_hits = _find_pages_containing_all(doc, [title, k], max_pages=3)
            if cfo_hits:
                break
        if cfo_hits:
            break
    if not cfo_hits:
        cfo_hits = _find_pages_containing_any(doc, cfo_keys, max_pages=3)

    capex_hits: List[int] = []
    for title in cashflow_title_keys:
        for k in capex_title_keys_all:
            capex_hits = _find_pages_containing_all(doc, [title, k], max_pages=3)
            if capex_hits:
                break
        if capex_hits:
            break
    if not capex_hits:
        capex_hits = _find_pages_containing_any(doc, capex_keys_all, max_pages=5)

    both = sorted(set(cfo_hits).intersection(capex_hits))
    if both:
        out["cfo"] = both[0]
        out["capex"] = both[0]
    else:
        if cfo_hits:
            out["cfo"] = cfo_hits[0]
        if capex_hits:
            out["capex"] = capex_hits[0]

    legacy_ranges = _section_ranges_for_legacy_report(doc) if int(year) <= 2006 else None
    if legacy_ranges is not None:
        legacy_shares_page = _pick_legacy_shares_page(doc, year, legacy_ranges)
        if legacy_shares_page is not None:
            out["shares"] = legacy_shares_page

    # Shares: prefer first 30 pages, but fall back to full doc
    year_str = str(int(year))
    shares_keys = [
        "总股本",
        "期末总股本",
        "期末普通股股份总数",
        "期末股份总数",
        "股份总数",
        "股本",
        "实收资本",
        "sharecapital",
        "totalsharecapital",
        "issuedsharecapital",
        "totalnumberofshares",
        "ordinaryshares",
        "totalshares",
        "sharebase",
    ]

    # Shares info can appear either early (profit distribution) or later (Changes in Shares / Share capital note).
    shares_candidates_early = _find_pages_containing_any(
        doc,
        shares_keys,
        max_pages=25,
        page_range=range(0, min(80, doc.page_count)),
    )
    shares_candidates_all = _find_pages_containing_any(doc, shares_keys, max_pages=50)
    shares_candidates = sorted(set(shares_candidates_early + shares_candidates_all))

    if shares_candidates and "shares" not in out:
        best_page = shares_candidates[0]
        best_score = -10**9
        for i in shares_candidates:
            t = _clean_for_match(doc[i].get_text("text"))
            score = _score_shares_page_text(t, year, legacy=False)
            if "sharebase" in t:
                score += 2
            if any(
                k in t
                for k in [
                    "股权激励",
                    "员工持股",
                    "持股计划",
                    "equityincentive",
                    "equityincentiveplan",
                    "equityincentiveplans",
                    "employeestockownershipplan",
                    "employeestockownershipplans",
                ]
            ):
                score -= 20

            if score > best_score:
                best_score = score
                best_page = i

        out["shares"] = best_page

    return out


def render_page_to_b64_png(doc: fitz.Document, page_index: int, *, dpi: int) -> str:
    page = doc[int(page_index)]
    pix = page.get_pixmap(dpi=int(dpi))
    png_bytes = pix.tobytes("png")
    import base64

    return base64.b64encode(png_bytes).decode("utf-8")


def extract_page_text(doc: fitz.Document, page_index: int) -> str:
    text = doc[int(page_index)].get_text("text") or ""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_model_inputs(
    doc: fitz.Document,
    page_indexes: Iterable[int],
    *,
    backend: str,
    dpi: int,
) -> Dict[str, List[str]]:
    indexes = [int(i) for i in page_indexes]
    if str(backend).strip().lower() == "ollama":
        return {"images_b64": [render_page_to_b64_png(doc, i, dpi=dpi) for i in indexes], "page_texts": []}
    return {"images_b64": [], "page_texts": [extract_page_text(doc, i) for i in indexes]}


def adapt_prompt_for_backend(user_prompt: str, *, backend: str) -> str:
    if str(backend).strip().lower() == "ollama":
        return user_prompt
    prefix = (
        "下面提供的不是图片，而是从年报关键页直接提取的原始文本。"
        "你只能依据给定文本抽取字段，不要猜测；如果文本没有明确给出，返回 null 并在 evidence 说明原因。"
    )
    return f"{prefix}\n\n{user_prompt.strip()}"


def _parse_number(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    neg = False
    if text.startswith("(") and text.endswith(")"):
        neg = True
        text = text[1:-1].strip()
    text = (
        text.replace(",", "")
        .replace("，", "")
        .replace("\u00a0", "")
        .replace("\u202f", "")
        .replace(" ", "")
    )
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3}){2,}", text):
        text = text.replace(".", "")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        num = float(m.group(0))
    except Exception:
        return None
    if neg and num > 0:
        num = -num
    return num


def normalize_total_shares_to_wan(value, unit: str) -> Optional[float]:
    num = _parse_number(value)
    if num is None:
        return None

    def _validate(wan: Optional[float]) -> Optional[float]:
        if wan is None:
            return None
        shares = float(wan) * 10000.0
        if shares < 1_000_000.0 or shares > 1_000_000_000_000.0:
            return None
        return float(wan)

    u = (unit or "").strip()
    u_lower = u.lower()
    value_text = str(value or "").strip()
    value_lower = value_text.lower()

    if "billion" in value_lower:
        return _validate(num * 100000.0)
    if "亿股" in value_text or "億股" in value_text:
        return _validate(num * 10000.0)
    if "万股" in value_text or "萬股" in value_text:
        return _validate(num)
    if (
        "百万股" in value_text
        or "百萬股" in value_text
        or "million shares" in value_lower
        or "million share" in value_lower
    ):
        return _validate((num * 1e6) / 10000.0)
    if "百万元" in value_text or "百萬元" in value_text or "million" in value_lower:
        return _validate((num * 1e6) / 10000.0)
    if "thousand shares" in value_lower or "thousand share" in value_lower:
        return _validate((num * 1000.0) / 10000.0)

    if u in {"万股"}:
        return _validate(num)
    if u in {"亿股"}:
        return _validate(num * 10000.0)
    if u in {"股"}:
        return _validate(num / 10000.0)
    if "share" in u_lower:
        # Support English unit labels: shares / thousand shares / million shares, etc.
        if "thousand" in u_lower or "000" in u_lower:
            return _validate((num * 1000.0) / 10000.0)
        if "million" in u_lower:
            return _validate((num * 1e6) / 10000.0)
        return _validate(num / 10000.0)
    if num > 1e7:
        return _validate(num / 10000.0)
    return _validate(num)


def normalize_money_to_yuan(value, unit: str) -> Optional[float]:
    num = _parse_number(value)
    if num is None:
        return None
    u = (unit or "").strip()
    u_lower = u.lower()
    # Common in bank reports
    if "百万元" in u:
        return num * 1e6
    if "千元" in u:
        return num * 1e3
    # Common in English financial statements: RMB'000 (thousand RMB) / RMB million, etc.
    if ("rmb" in u_lower or "cny" in u_lower) and ("000" in u_lower or "thousand" in u_lower):
        return num * 1e3
    if ("rmb" in u_lower or "cny" in u_lower) and "million" in u_lower:
        return num * 1e6
    if ("rmb" in u_lower or "cny" in u_lower) and "billion" in u_lower:
        return num * 1e9
    if ("rmb" in u_lower or "cny" in u_lower or "renminbi" in u_lower or "yuan" in u_lower) and not any(
        token in u_lower for token in ["000", "thousand", "million", "billion"]
    ):
        return num
    if u in {"元", "人民币元"}:
        return num
    if u in {"万元"}:
        return num * 1e4
    if u in {"亿元"}:
        return num * 1e8
    if "亿" in u:
        return num * 1e8
    if "万" in u:
        return num * 1e4
    # If unit inference failed, preserve the raw magnitude instead of blindly
    # guessing 万元/亿元 and exploding the value by 1e4/1e8.
    return num


def _resolve_openai_chat_url(api_base_url: str) -> str:
    base = str(api_base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("Missing --api-base-url for backend=openai_text")
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


def ollama_chat_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images_b64: List[str],
    response_format,
    timeout: int,
) -> Dict:
    ollama_num_ctx = resolve_ollama_num_ctx()
    payload = {
        "model": model,
        "stream": False,
        "format": response_format,
        "options": {"temperature": 0, "num_ctx": int(ollama_num_ctx)},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt, "images": images_b64},
        ],
    }

    last_exc: Optional[BaseException] = None
    for attempt in range(2):
        try:
            r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=int(timeout))
            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content") or ""
            return _extract_json_from_content(content)
        except Exception as exc:
            last_exc = exc
            if attempt < 1:
                time.sleep(1.0 + attempt)
                continue
            break

    raise RuntimeError(f"Model returned non-JSON content: {str(last_exc)[:300]!r}") from last_exc


def openai_text_chat_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    page_texts: List[str],
    timeout: int,
    api_base_url: str,
    api_key: str,
) -> Dict:
    if not str(api_key or "").strip():
        raise RuntimeError("Missing API key for backend=openai_text")

    url = _resolve_openai_chat_url(api_base_url)
    prompt_parts = [
        user_prompt.strip(),
        "",
        "以下是从年报关键页直接提取的原始文本。请只根据这些文本抽取，不要补充或猜测：",
    ]
    for idx, text in enumerate(page_texts, start=1):
        prompt_parts.append(f"[PAGE_{idx}_TEXT]")
        prompt_parts.append(str(text or "").strip() or "[EMPTY_PAGE_TEXT]")
        prompt_parts.append("")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(prompt_parts).strip()},
    ]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 600,
        "stream": False,
    }

    last_exc: Optional[BaseException] = None
    attempts = [True, False, False]
    for attempt_index, use_json_mode in enumerate(attempts):
        payload = dict(base_payload)
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=(30, int(timeout)))
            if use_json_mode and int(r.status_code) >= 400:
                response_text = str(r.text or "")
                if "response_format" in response_text.lower() or "json_object" in response_text.lower():
                    time.sleep(2.0 + attempt_index)
                    continue
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"missing_choices: {json.dumps(data, ensure_ascii=False)[:300]}")
            message = choices[0].get("message") or {}
            return _extract_json_from_content(message.get("content"))
        except Exception as exc:
            last_exc = exc
            if attempt_index < len(attempts) - 1:
                time.sleep(3.0 + attempt_index * 2.0)
                continue
            break

    raise RuntimeError(f"Model returned non-JSON content: {str(last_exc)[:300]!r}") from last_exc


def chat_json(
    *,
    backend: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images_b64: List[str],
    page_texts: List[str],
    response_format,
    timeout: int,
    api_base_url: str = "",
    api_key: str = "",
) -> Dict:
    backend_name = str(backend or "").strip().lower()
    if backend_name == "ollama":
        return ollama_chat_json(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64,
            response_format=response_format,
            timeout=timeout,
        )
    if backend_name == "openai_text":
        return openai_text_chat_json(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            page_texts=page_texts,
            timeout=timeout,
            api_base_url=api_base_url,
            api_key=api_key,
        )
    raise RuntimeError(f"Unsupported backend: {backend}")


def _null_field(reason: str) -> Dict[str, object]:
    return {"value": None, "unit": None, "evidence": reason}


def extract_from_pdf(
    task: Task,
    *,
    backend: str,
    model: str,
    dpi: int,
    timeout: int,
    api_base_url: str = "",
    api_key: str = "",
    debug: bool = False,
) -> Dict:
    with fitz.open(str(task.pdf_path)) as doc:
        pages = pick_key_pages(doc, task.year)
        income_page = pages.get("income")
        cfo_page = pages.get("cfo")
        capex_page = pages.get("capex")
        shares_page = pages.get("shares")
        income_page_indexes: List[int] = [income_page] if income_page is not None else []
        cfo_page_indexes: List[int] = [cfo_page] if cfo_page is not None else []
        capex_page_indexes: List[int] = [capex_page] if capex_page is not None else []
        shares_page_indexes: List[int] = [shares_page] if shares_page is not None else []

        if debug:
            def _p(v: Optional[int]) -> str:
                return "-" if v is None else str(int(v) + 1)

            print(
                f"[pages] {task.stock_code} {task.year}: "
                f"income={_p(income_page)} cfo={_p(cfo_page)} capex={_p(capex_page)} shares={_p(shares_page)}"
            )

        system_prompt = (
            "你是一名严格的财务报表信息抽取助手。"
            "你将看到上市公司年报PDF的关键页面图片。"
            "你必须只输出JSON，不要输出任何额外文字。"
            "若无法确定字段，输出 null 并在 evidence 说明原因。"
        )

        field_schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number", "null"]},
                "unit": {"type": ["string", "null"]},
                "evidence": {"type": ["string", "null"]},
            },
            "required": ["value", "unit", "evidence"],
            "additionalProperties": False,
        }

        # 1) income statement (parent net profit)
        if income_page is None:
            income_res = {"parent_netprofit": _null_field("no_key_page: income")}
        else:
            income_inputs = build_model_inputs(doc, income_page_indexes, backend=backend, dpi=dpi)
            income_prompt = f"""
你将看到一页年报图片：合并利润表（通常含两栏：{task.year}年 与 上年对比）。
请仅提取本报告年度（{task.year}年）的“归属于…股东/所有者的净利润”这一行的数值与单位。
可接受的行名（出现其一即可）：
- 归属于母公司股东的净利润
- 归属于母公司所有者的净利润
- 归属于本公司股东的净利润
- 归属于本行股东的净利润
- 归属于本集团股东的净利润
（若为英文报表，也可能写作：Net profit attributable to shareholders of the parent company / Profit attributable to shareholders of the parent company 等）
如果表中存在“归属于少数股东的净利润/少数股东损益”，不要选它。
不要提取上一年对比列；不要提取“扣非净利润”等其他口径。
输出JSON：{{"parent_netprofit": {{...}}}}
""".strip()
            if int(task.year) <= 2006:
                income_prompt = f"""
你将看到 {task.year} 年旧版年报的关键页图片，图片可能来自“主要会计数据”摘要页、利润表，或利润及利润分配表/利润分配表。
请提取能够代表 {task.year} 年度净利润口径的数值与单位，输出到字段 parent_netprofit。
优先级如下：
1) 若页面中存在“归属于母公司股东的净利润”或同义口径，优先使用它；
2) 若为 2007 年前旧准则报表且没有归母口径，允许使用“净利润”行；
3) 若利润表没有直接给出净利润，但利润分配表/利润及利润分配表有“净利润”行，也允许使用该值。
4) 若为英文年报，允许使用 “Net profit / Profit for the year / Statement of income / Income statement / Statement of profit appropriation / Major accounting data / Financial highlight” 中对应本报告年度的净利润值。
注意：
- 只取 {task.year} 年这一列，不要取上一年对比列；
- 不要提取“未分配利润”“利润总额”等其他口径；
- 只输出 JSON：{{"parent_netprofit": {{...}}}}
""".strip()
            income_format = {
                "type": "object",
                "properties": {"parent_netprofit": field_schema},
                "required": ["parent_netprofit"],
                "additionalProperties": False,
            }
            income_res = chat_json(
                backend=backend,
                model=model,
                system_prompt=system_prompt,
                user_prompt=adapt_prompt_for_backend(income_prompt, backend=backend),
                images_b64=income_inputs["images_b64"],
                page_texts=income_inputs["page_texts"],
                response_format=income_format,
                timeout=timeout,
                api_base_url=api_base_url,
                api_key=api_key,
            )

        # 2) cashflow statement (CFO + CapEx)
        operating_cashflow_res: Dict[str, object] = {}
        capex_res: Dict[str, object] = {}
        if cfo_page is None and capex_page is None:
            operating_cashflow_res = _null_field("no_key_page: cfo")
            capex_res = _null_field("no_key_page: capex")
        elif cfo_page is not None and capex_page is not None and int(capex_page) == int(cfo_page):
            cfo_page_indexes = [cfo_page]
            capex_page_indexes = [cfo_page]
            cash_inputs = build_model_inputs(doc, cfo_page_indexes, backend=backend, dpi=dpi)
            cash_prompt = f"""
你将看到一页年报图片：合并现金流量表（通常含两栏：{task.year}年 与 上年对比）。
请仅提取本报告年度（{task.year}年）的两项（都来自主表，不要来自补充资料）：
1) “经营活动产生的现金流量净额”（英文可能为 Net cash generated from operating activities / Net cash flows from operating activities 等）
2) “购建固定资产、无形资产及其他长期资产支付的现金”（英文可能为 Payments for the acquisition and construction of fixed assets, intangible assets and other long-term assets 等；有时标题中“及”会写作“和”，或写作“所支付的现金”）
输出JSON：{{"operating_cashflow": {{...}}, "capex": {{...}}}}
""".strip()
            if int(task.year) <= 2006:
                cash_prompt = f"""
你将看到 {task.year} 年旧版年报现金流量表相关原始文本，可能是英文报表。请仅提取本报告年度（{task.year}年）的两项，且都必须来自主表，不要来自补充资料：
1) “经营活动产生的现金流量净额” / Net cash generated from operating activities / Net cash flows from operating activities；
2) capex：优先取“购建固定资产、无形资产及其他长期资产支付的现金”或同义中文行。若旧版英文主表没有更完整长项名，但明确存在唯一资本开支现金流行，如 “Purchases of property, plant and equipment” / “Purchase of property, plant and equipment” / “Payments for purchase of fixed assets”，允许将其作为 capex fallback。
只取 {task.year} 年这一列。输出 JSON：{{"operating_cashflow": {{...}}, "capex": {{...}}}}
""".strip()
            cash_format = {
                "type": "object",
                "properties": {
                    "operating_cashflow": field_schema,
                    "capex": field_schema,
                },
                "required": ["operating_cashflow", "capex"],
                "additionalProperties": False,
            }
            cash_res = chat_json(
                backend=backend,
                model=model,
                system_prompt=system_prompt,
                user_prompt=adapt_prompt_for_backend(cash_prompt, backend=backend),
                images_b64=cash_inputs["images_b64"],
                page_texts=cash_inputs["page_texts"],
                response_format=cash_format,
                timeout=timeout,
                api_base_url=api_base_url,
                api_key=api_key,
            )
            operating_cashflow_res = cash_res.get("operating_cashflow") or {}
            capex_res = cash_res.get("capex") or {}
        else:
            if cfo_page is None:
                operating_cashflow_res = _null_field("no_key_page: cfo")
            else:
                cfo_page_indexes = [cfo_page]
                cfo_inputs = build_model_inputs(doc, cfo_page_indexes, backend=backend, dpi=dpi)
                cfo_prompt = f"""
你将看到一页年报图片：合并现金流量表（经营活动部分）。
请仅提取本报告年度（{task.year}年）的“经营活动产生的现金流量净额”（主表；英文可能为 Net cash generated from operating activities / Net cash flows from operating activities）。
输出JSON：{{"operating_cashflow": {{...}}}}
""".strip()
                cfo_format = {
                    "type": "object",
                    "properties": {"operating_cashflow": field_schema},
                    "required": ["operating_cashflow"],
                    "additionalProperties": False,
                }
                cfo_res = chat_json(
                    backend=backend,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=adapt_prompt_for_backend(cfo_prompt, backend=backend),
                    images_b64=cfo_inputs["images_b64"],
                    page_texts=cfo_inputs["page_texts"],
                    response_format=cfo_format,
                    timeout=timeout,
                    api_base_url=api_base_url,
                    api_key=api_key,
                )
                operating_cashflow_res = cfo_res.get("operating_cashflow") or {}

            if capex_page is None:
                capex_res = _null_field("no_key_page: capex")
            else:
                capex_page_indexes = [capex_page]
                cap_inputs = build_model_inputs(doc, capex_page_indexes, backend=backend, dpi=dpi)
                cap_prompt = f"""
你将看到一页年报图片：合并现金流量表（投资活动部分）。
请仅提取本报告年度（{task.year}年）的“购建固定资产、无形资产及其他长期资产支付的现金”（主表；英文可能为 Payments for the acquisition and construction of fixed assets, intangible assets and other long-term assets；有时写作“...和...”“...所支付的现金”）。
输出JSON：{{"capex": {{...}}}}
""".strip()
                if int(task.year) <= 2006:
                    cap_prompt = f"""
你将看到 {task.year} 年旧版年报现金流量表投资活动部分的原始文本。请提取本报告年度（{task.year}年）的 capex，优先取主表中的“购建固定资产、无形资产及其他长期资产支付的现金”或同义中文行。
若旧版英文主表没有更完整长项名，但明确存在唯一资本开支现金流行，如 “Purchases of property, plant and equipment” / “Purchase of property, plant and equipment” / “Payments for purchase of fixed assets”，允许将其作为 capex fallback。
不要使用补充资料。输出 JSON：{{"capex": {{...}}}}
""".strip()
                cap_format = {
                    "type": "object",
                    "properties": {"capex": field_schema},
                    "required": ["capex"],
                    "additionalProperties": False,
                }
                cap_res = chat_json(
                    backend=backend,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=adapt_prompt_for_backend(cap_prompt, backend=backend),
                    images_b64=cap_inputs["images_b64"],
                    page_texts=cap_inputs["page_texts"],
                    response_format=cap_format,
                    timeout=timeout,
                    api_base_url=api_base_url,
                    api_key=api_key,
                )
                capex_res = cap_res.get("capex") or {}

        # 3) shares
        if shares_page is None:
            shares_res = {"total_shares": _null_field("no_key_page: shares")}
        else:
            shares_prompt_pages = [shares_page]
            if str(backend).strip().lower() == "openai_text" and int(task.year) <= 2006:
                shares_prompt_pages = _collect_legacy_shares_context_pages(doc, task.year, shares_page)
            shares_page_indexes = list(shares_prompt_pages)
            shares_inputs = build_model_inputs(doc, shares_prompt_pages, backend=backend, dpi=dpi)
            shares_prompt = f"""
你将看到一页年报图片（通常在重要提示/利润分配预案/股本信息中）。
请提取截至{task.year}年12月31日的“总股本/期末总股本/期末普通股股份总数”的数值与单位（英文可能为 Total share capital / Total number of shares at end of period）。
注意必须是期末，不是期初，不是加权平均。
输出JSON：{{"total_shares": {{...}}}}
""".strip()
            if int(task.year) <= 2006:
                shares_prompt = f"""
你将看到 {task.year} 年旧版年报“股本变动及股东情况”章节的 1-3 页连续原始文本，可能是同一张股本/股份变动情况表的跨页内容。
请提取截至 {task.year} 年 12 月 31 日的期末总股本，输出到 total_shares。
优先锚点：
1) “股份总数 / 总股本 / 期末总股本 / 期末普通股股份总数”；
2) 英文表格中的 “Total shares / Total share capital / Amount at the year-end”；
3) 若是“股东权益变动情况”表，只有在明确给出“期末数”且项目为“股本 / Share capital”时才可使用。
排除项：
- 历次股票发行情况、配股基数、历史年度总股本；
- 股东持股数、内部职工股、高管持股；
- 每股收益计算公式中的“期末股份总数”。
如果表格跨页，请综合全部页面判断，并尽量给出单位。只输出 JSON：{{"total_shares": {{...}}}}
""".strip()
            shares_format = {
                "type": "object",
                "properties": {"total_shares": field_schema},
                "required": ["total_shares"],
                "additionalProperties": False,
            }
            shares_res = chat_json(
                backend=backend,
                model=model,
                system_prompt=system_prompt,
                user_prompt=adapt_prompt_for_backend(shares_prompt, backend=backend),
                images_b64=shares_inputs["images_b64"],
                page_texts=shares_inputs["page_texts"],
                response_format=shares_format,
                timeout=timeout,
                api_base_url=api_base_url,
                api_key=api_key,
            )

        if str(backend).strip().lower() == "openai_text" and int(task.year) <= 2006:
            current_total_shares = _canonicalize_field_payload(
                shares_res.get("total_shares"),
                doc=doc,
                page_indexes=shares_page_indexes,
                field_kind="shares",
            )
            if shares_page is not None and normalize_total_shares_to_wan(current_total_shares.get("value"), str(current_total_shares.get("unit") or "")) is None:
                retry_pages = _collect_legacy_shares_context_pages(doc, task.year, shares_page)
                retry_inputs = build_model_inputs(doc, retry_pages, backend=backend, dpi=dpi)
                shares_retry_prompt = f"""
你现在只做一件事：从下面这些来自 {task.year} 年年报“股本变动及股东情况”章节的文本中，定位报告期末总股本。
优先提取“股份总数 / 总股本 / 期末总股本 / 期末普通股股份总数 / Total shares / Total share capital / Amount at the year-end”对应的期末数值。
不要使用历史发行页里的“1997年末总股本”等旧数据；不要使用股东持股数；不要使用每股收益公式中的“期末股份总数”。
若表格跨页，请综合所有页面，并尽量补足单位。只输出 JSON：{{"total_shares": {{...}}}}
""".strip()
                shares_retry = chat_json(
                    backend=backend,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=adapt_prompt_for_backend(shares_retry_prompt, backend=backend),
                    images_b64=retry_inputs["images_b64"],
                    page_texts=retry_inputs["page_texts"],
                    response_format=shares_format,
                    timeout=timeout,
                    api_base_url=api_base_url,
                    api_key=api_key,
                )
                retry_total_shares = _canonicalize_field_payload(
                    shares_retry.get("total_shares"),
                    doc=doc,
                    page_indexes=retry_pages,
                    field_kind="shares",
                )
                if normalize_total_shares_to_wan(retry_total_shares.get("value"), str(retry_total_shares.get("unit") or "")) is not None:
                    shares_page_indexes = list(retry_pages)
                    shares_res = {"total_shares": retry_total_shares}

            current_capex = _canonicalize_field_payload(
                capex_res,
                doc=doc,
                page_indexes=capex_page_indexes,
                field_kind="money",
            )
            if capex_page is not None and normalize_money_to_yuan(current_capex.get("value"), str(current_capex.get("unit") or "")) is None:
                cap_retry_inputs = build_model_inputs(doc, [capex_page], backend=backend, dpi=dpi)
                cap_retry_prompt = f"""
你现在只做一件事：从 {task.year} 年现金流量表主表文本里提取 capex。
优先使用“购建固定资产、无形资产及其他长期资产支付的现金”或同义中文行。
若旧版英文主表没有完整长项名，但明确存在唯一资本开支现金流行，如 “Purchases of property, plant and equipment” / “Purchase of property, plant and equipment” / “Payments for purchase of fixed assets”，允许将其作为 capex fallback。
不要使用补充资料。只输出 JSON：{{"capex": {{...}}}}
""".strip()
                cap_retry = chat_json(
                    backend=backend,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=adapt_prompt_for_backend(cap_retry_prompt, backend=backend),
                    images_b64=cap_retry_inputs["images_b64"],
                    page_texts=cap_retry_inputs["page_texts"],
                    response_format={"type": "object", "properties": {"capex": field_schema}, "required": ["capex"], "additionalProperties": False},
                    timeout=timeout,
                    api_base_url=api_base_url,
                    api_key=api_key,
                )
                retry_capex = _canonicalize_field_payload(
                    cap_retry.get("capex"),
                    doc=doc,
                    page_indexes=capex_page_indexes,
                    field_kind="money",
                )
                if normalize_money_to_yuan(retry_capex.get("value"), str(retry_capex.get("unit") or "")) is not None:
                    capex_res = retry_capex

        parent_netprofit_field = _canonicalize_field_payload(
            income_res.get("parent_netprofit"),
            doc=doc,
            page_indexes=income_page_indexes,
            field_kind="money",
        )
        total_shares_field = _canonicalize_field_payload(
            shares_res.get("total_shares"),
            doc=doc,
            page_indexes=shares_page_indexes,
            field_kind="shares",
        )
        operating_cashflow_field = _canonicalize_field_payload(
            operating_cashflow_res,
            doc=doc,
            page_indexes=cfo_page_indexes,
            field_kind="money",
        )
        capex_field = _canonicalize_field_payload(
            capex_res,
            doc=doc,
            page_indexes=capex_page_indexes,
            field_kind="money",
        )

        raw = {
            "code": task.stock_code,
            "year": task.year,
            "parent_netprofit": dict(
                parent_netprofit_field,
                page=_resolve_field_page_from_evidence(parent_netprofit_field, income_page_indexes, income_page),
            ),
            "total_shares": dict(
                total_shares_field,
                page=_resolve_field_page_from_evidence(total_shares_field, shares_page_indexes, shares_page),
            ),
            "operating_cashflow": dict(
                operating_cashflow_field,
                page=_resolve_field_page_from_evidence(operating_cashflow_field, cfo_page_indexes, cfo_page),
            ),
            "capex": dict(
                capex_field,
                page=_resolve_field_page_from_evidence(capex_field, capex_page_indexes, capex_page),
            ),
        }

    pn = raw.get("parent_netprofit") or {}
    ts = raw.get("total_shares") or {}
    ocf = raw.get("operating_cashflow") or {}
    cap = raw.get("capex") or {}

    net_profit_yuan = normalize_money_to_yuan(pn.get("value"), str(pn.get("unit") or ""))
    operating_cashflow_yuan = normalize_money_to_yuan(ocf.get("value"), str(ocf.get("unit") or ""))
    capex_yuan = normalize_money_to_yuan(cap.get("value"), str(cap.get("unit") or ""))
    if capex_yuan is not None and capex_yuan < 0:
        capex_yuan = -capex_yuan

    total_shares_wan = normalize_total_shares_to_wan(ts.get("value"), str(ts.get("unit") or ""))
    total_shares_shares = (total_shares_wan * 10000.0) if total_shares_wan is not None else None

    return {
        "task": {
            "stock_code": task.stock_code,
            "year": task.year,
            "pdf_path": str(task.pdf_path),
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
            "income": (income_page + 1) if income_page is not None else None,
            "cfo": (cfo_page + 1) if cfo_page is not None else None,
            "capex": (capex_page + 1) if capex_page is not None else None,
            "shares": (shares_page + 1) if shares_page is not None else None,
        },
    }


def _extract_code_from_filename(pdf_path: Path) -> str:
    m = re.search(r"(\d{6})", pdf_path.name)
    return m.group(1) if m else ""


def collect_tasks(pdf_root: Path, *, start_year: int, end_year: int) -> List[Task]:
    tasks_map: Dict[Tuple[int, str], Task] = {}
    if not pdf_root.exists():
        raise FileNotFoundError(f"PDF root not found: {pdf_root}")

    for year_dir in sorted([p for p in pdf_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if year < int(start_year) or year > int(end_year):
            continue
        for pdf_path in year_dir.glob("*.pdf"):
            code = _extract_code_from_filename(pdf_path)
            code = normalize_stock_code(code)
            if not code:
                continue
            key = (year, code)
            cur = tasks_map.get(key)
            if cur is None:
                tasks_map[key] = Task(year=year, stock_code=code, pdf_path=pdf_path)
                continue
            # if duplicates exist, keep the larger one
            try:
                if pdf_path.stat().st_size > cur.pdf_path.stat().st_size:
                    tasks_map[key] = Task(year=year, stock_code=code, pdf_path=pdf_path)
            except Exception:
                pass

    out = list(tasks_map.values())
    out.sort(key=lambda t: (t.year, t.stock_code))
    return out


def append_csv_row(path: Path, row: Dict[str, object], *, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def write_run_config(
    out_dir: Path,
    *,
    backend: str,
    model: str,
    timeout: int,
    pdf_root: Path,
    csv_name: str,
    start_year: int,
    end_year: int,
    api_base_url: str = "",
    api_key_env: str = "",
    ollama_num_ctx: Optional[int] = None,
) -> None:
    payload = {
        "backend": str(backend or "").strip(),
        "model": str(model or "").strip(),
        "timeout_seconds": int(timeout),
        "pdf_root": str(pdf_root),
        "csv_name": str(csv_name),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "input_mode": "page_images" if str(backend).strip().lower() == "ollama" else "page_text",
        "api_base_url": str(api_base_url or "").strip(),
        "api_key_env": str(api_key_env or "").strip(),
        "ollama_num_ctx": int(ollama_num_ctx) if ollama_num_ctx else None,
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_done_set_from_log(log_path: Path) -> set[Tuple[int, str]]:
    if not log_path.exists():
        return set()
    done: set[Tuple[int, str]] = set()
    try:
        with log_path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    y = int(str(row.get("year") or "").strip())
                except Exception:
                    continue
                code = normalize_stock_code(row.get("stock_code") or "")
                status = str(row.get("status") or "").strip().lower()
                if status in {"ok", "partial"} and code:
                    done.add((y, code))
    except Exception:
        return set()
    return done


def load_done_codes_from_year_csv(year_csv: Path) -> set[str]:
    if not year_csv.exists():
        return set()
    done: set[str] = set()
    try:
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
    parser = argparse.ArgumentParser(description="Step6 (PDF/Qwen): 提取年报关键字段（归母净利润/总股本/经营现金流/CapEx）")
    parser.add_argument("--base-dir", default=".", help="项目根目录（包含 2006/2007/... 年度文件夹的目录）")
    parser.add_argument("--pdf-root", default="年报/下载年报_fulltext", help="PDF 年报根目录（按年份分子目录）")
    parser.add_argument("--out-dir", default=".cache/qwen_pdf_financials", help="输出目录（raw_json + log）")
    parser.add_argument("--backend", choices=["ollama", "openai_text"], default="ollama", help="模型调用后端：本地 Ollama 视觉模型，或外部 OpenAI 兼容文本 API")
    parser.add_argument("--model", default="qwen3.5:9b", help="模型名（Ollama 模型名，或外部 API 的 model）")
    parser.add_argument("--api-base-url", default="", help="backend=openai_text 时使用的 OpenAI 兼容 API base URL")
    parser.add_argument("--api-key-env", default="SCNET_API_KEY", help="backend=openai_text 时，读取 API key 的环境变量名")
    parser.add_argument("--dpi", type=int, default=200, help="backend=ollama 时渲染图片 DPI（默认 200）")
    parser.add_argument("--timeout", type=int, default=900, help="单次模型请求超时（秒）")
    parser.add_argument("--start-year", type=int, default=2001, help="起始年份（含）")
    parser.add_argument("--end-year", type=int, default=2024, help="结束年份（含）")
    parser.add_argument(
        "--csv-name",
        default="{year}_财报数据_qwen.csv",
        help="写入到年度目录的 CSV 文件名模板（默认不覆盖既有 Step6 输出）",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在 raw_json 和 CSV 记录")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过日志/CSV中已 ok 的任务")
    parser.add_argument("--start", type=int, default=0, help="从第 N 个任务开始（用于断点）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个任务（0 表示全部）")
    parser.add_argument("--sleep", type=float, default=0.0, help="每个任务完成后额外 sleep 秒数")
    parser.add_argument("--jitter", type=float, default=0.0, help="sleep 的随机抖动秒数")
    parser.add_argument("--debug", action="store_true", help="输出每个任务选择的页码")
    parser.add_argument("--ollama-num-ctx", type=int, default=DEFAULT_OLLAMA_NUM_CTX, help="backend=ollama num_ctx")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    pdf_root = (Path(args.pdf_root) if Path(args.pdf_root).is_absolute() else (base_dir / args.pdf_root)).resolve()
    out_dir = (Path(args.out_dir) if Path(args.out_dir).is_absolute() else (base_dir / args.out_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = str(args.backend).strip().lower()
    api_base_url = str(args.api_base_url or "").strip()
    api_key_env = str(args.api_key_env or "").strip()
    api_key = ""
    if backend == "openai_text":
        if not api_base_url:
            raise RuntimeError("backend=openai_text requires --api-base-url")
        api_key = str(os.environ.get(api_key_env, "") or "").strip()
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is empty; cannot call external API")
    elif backend == "ollama":
        os.environ["OLLAMA_NUM_CTX"] = str(int(args.ollama_num_ctx))

    log_path = out_dir / "extract_log.csv"
    raw_root = out_dir / "raw_json"
    write_run_config(
        out_dir,
        backend=backend,
        model=str(args.model),
        timeout=int(args.timeout),
        pdf_root=pdf_root,
        csv_name=str(args.csv_name),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        api_base_url=api_base_url,
        api_key_env=api_key_env,
        ollama_num_ctx=int(args.ollama_num_ctx) if backend == "ollama" else None,
    )

    tasks = collect_tasks(pdf_root, start_year=int(args.start_year), end_year=int(args.end_year))
    if not tasks:
        raise RuntimeError(f"No PDF tasks found under {pdf_root}")

    done_from_log: set[Tuple[int, str]] = set()
    if bool(args.resume) and not bool(args.overwrite):
        done_from_log = load_done_set_from_log(log_path)

    start = max(0, int(args.start))
    end = len(tasks) if int(args.limit) <= 0 else min(len(tasks), start + int(args.limit))
    tasks = tasks[start:end]

    print(f"[tasks] total={len(tasks)} pdf_root={pdf_root}", flush=True)
    print(f"[out] out_dir={out_dir}", flush=True)
    print(f"[log] {log_path}", flush=True)
    print(f"[backend] backend={backend} model={args.model} timeout={int(args.timeout)}", flush=True)
    if backend == "openai_text":
        print(f"[api] base_url={api_base_url} key_env={api_key_env}", flush=True)
    else:
        print(f"[ollama] num_ctx={int(args.ollama_num_ctx)}", flush=True)

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

    for idx, t in enumerate(tasks, start=1):
        if (t.year, t.stock_code) in done_from_log:
            skip += 1
            continue

        year_dir = base_dir / str(t.year)
        year_dir.mkdir(parents=True, exist_ok=True)
        csv_name = str(args.csv_name).format(year=t.year)
        year_csv = year_dir / csv_name

        if bool(args.resume) and not bool(args.overwrite):
            if t.year not in year_done_cache:
                year_done_cache[t.year] = load_done_codes_from_year_csv(year_csv)
            if t.stock_code in year_done_cache[t.year]:
                skip += 1
                continue

        raw_json_path = raw_root / str(t.year) / f"{t.stock_code}.json"
        if raw_json_path.exists() and not bool(args.overwrite):
            skip += 1
            continue

        try:
            extracted = extract_from_pdf(
                t,
                backend=backend,
                model=str(args.model),
                dpi=int(args.dpi),
                timeout=int(args.timeout),
                api_base_url=api_base_url,
                api_key=api_key,
                debug=bool(args.debug),
            )

            raw_json_path.parent.mkdir(parents=True, exist_ok=True)
            raw_json_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

            norm = extracted.get("normalized") or {}
            raw = extracted.get("raw") or {}

            total_shares_wan = norm.get("total_shares_wan")
            share_capital_shares = norm.get("total_shares_shares")
            row = {
                "year": t.year,
                "stock_code": t.stock_code,
                "code_name": infer_code_name(t.stock_code),
                "stock_name": "",
                "parent_netprofit": norm.get("parent_netprofit_yuan"),
                "share_capital": share_capital_shares,
                "share_capital_wan": total_shares_wan,
                "netcash_operate": norm.get("operating_cashflow_yuan"),
                "construct_long_asset": norm.get("capex_yuan"),
                "pdf_path": str(t.pdf_path),
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

            pages_info = extracted.get("pages") or {}
            missing_pages = [k for k in ("income", "shares", "cfo", "capex") if not pages_info.get(k)]

            status = "ok" if not missing_fields else "partial"
            message_parts: List[str] = []
            if missing_pages:
                message_parts.append(f"missing_pages={','.join(missing_pages)}")
            if missing_fields:
                message_parts.append(f"missing_fields={','.join(missing_fields)}")
            message = "; ".join(message_parts)

            append_csv_row(
                log_path,
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "year": t.year,
                    "stock_code": t.stock_code,
                    "code_name": infer_code_name(t.stock_code),
                    "pdf_path": str(t.pdf_path),
                    "status": status,
                    "message": message,
                    "raw_json_path": str(raw_json_path),
                },
                fieldnames=log_fields,
            )

            if bool(args.resume) and not bool(args.overwrite):
                year_done_cache.setdefault(t.year, set()).add(t.stock_code)

            if status == "ok":
                ok += 1
            else:
                partial += 1
        except Exception as exc:
            fail += 1
            append_csv_row(
                log_path,
                {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "year": t.year,
                    "stock_code": t.stock_code,
                    "code_name": infer_code_name(t.stock_code),
                    "pdf_path": str(t.pdf_path),
                    "status": "error",
                    "message": str(exc),
                    "raw_json_path": "",
                },
                fieldnames=log_fields,
            )

        if idx % 10 == 0:
            print(f"[progress] {idx}/{len(tasks)} ok={ok} partial={partial} skip={skip} fail={fail}", flush=True)

        if float(args.sleep) > 0:
            time.sleep(max(0.0, float(args.sleep) + random.random() * float(args.jitter)))

    print(f"[done] ok={ok} partial={partial} skip={skip} fail={fail} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
