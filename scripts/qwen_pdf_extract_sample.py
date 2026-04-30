#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample run: use local Ollama Qwen3.5 model to extract key fields from an annual-report PDF.

Why this script exists
- You asked to follow the "PDF -> high-res images -> model" workflow.
- Your model is running locally via Ollama (http://127.0.0.1:11434).
- This script renders only a few "key pages" to reduce tokens/time.

Outputs
- Prints raw + normalized values to stdout
- Optionally writes a JSON file
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import requests


OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"


@dataclass(frozen=True)
class PageHit:
    page_index: int  # 0-based
    reason: str


def _clean_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _find_pages_containing_all(doc: fitz.Document, keywords: Iterable[str], *, max_pages: int = 10) -> List[int]:
    keys = [k for k in (keywords or []) if str(k).strip()]
    if not keys:
        return []
    hits: List[int] = []
    for i in range(doc.page_count):
        t = _clean_for_match(doc[i].get_text("text"))
        if all(k in t for k in keys):
            hits.append(i)
            if len(hits) >= int(max_pages):
                break
    return hits


def _find_pages_containing_any(doc: fitz.Document, keywords: Iterable[str], *, max_pages: int = 10) -> List[int]:
    keys = [k for k in (keywords or []) if str(k).strip()]
    if not keys:
        return []
    hits: List[int] = []
    for i in range(doc.page_count):
        t = _clean_for_match(doc[i].get_text("text"))
        if any(k in t for k in keys):
            hits.append(i)
            if len(hits) >= int(max_pages):
                break
    return hits


def pick_key_pages(doc: fitz.Document) -> List[PageHit]:
    """
    Best-effort heuristics to locate the key pages that contain the 4 fields.
    We intentionally keep this conservative: a few pages only.
    """
    hits: List[PageHit] = []

    # 1) Net profit to parent: prefer consolidated income statement.
    income_candidates = _find_pages_containing_all(
        doc,
        ["归属于母公司股东的净利润", "利润表"],
        max_pages=5,
    )
    if not income_candidates:
        income_candidates = _find_pages_containing_any(doc, ["归属于母公司股东的净利润"], max_pages=5)
    if income_candidates:
        hits.append(PageHit(income_candidates[0], "归母净利润/利润表"))

    # 2) Cashflow: operating cashflow net + capex line.
    cash_both = _find_pages_containing_all(
        doc,
        ["经营活动产生的现金流量净额", "购建固定资产、无形资产和其他长期资产支付的现金"],
        max_pages=3,
    )
    for i in cash_both[:1]:
        hits.append(PageHit(i, "现金流量表(经营现金流+资本支出)"))

    # If they are split across pages, add one page for each.
    if not cash_both:
        cfo_pages = _find_pages_containing_any(doc, ["经营活动产生的现金流量净额"], max_pages=3)
        capex_pages = _find_pages_containing_any(doc, ["购建固定资产、无形资产和其他长期资产支付的现金"], max_pages=3)
        if cfo_pages:
            hits.append(PageHit(cfo_pages[0], "现金流量表(经营现金流)"))
        if capex_pages:
            hits.append(PageHit(capex_pages[0], "现金流量表(资本支出)"))

    # 3) Shares: period-end total shares / total share capital.
    share_candidates = _find_pages_containing_any(
        doc,
        ["期末普通股股份总数", "期末总股本", "总股本"],
        max_pages=8,
    )
    if share_candidates:
        hits.append(PageHit(share_candidates[0], "总股本/普通股股份总数"))

    # De-dup while keeping order
    seen: set[int] = set()
    out: List[PageHit] = []
    for h in hits:
        if h.page_index in seen:
            continue
        seen.add(h.page_index)
        out.append(h)
    return out


def render_pages_to_base64_png(doc: fitz.Document, page_indexes: List[int], *, dpi: int) -> List[str]:
    images_b64: List[str] = []
    for idx in page_indexes:
        page = doc[int(idx)]
        pix = page.get_pixmap(dpi=int(dpi))
        png_bytes = pix.tobytes("png")
        images_b64.append(base64.b64encode(png_bytes).decode("utf-8"))
    return images_b64


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
    text = text.replace(",", "").replace("，", "")
    # Keep digits, dot, minus
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
    """
    Normalize total shares to 万股.
    Supports unit in: 股 / 万股 / 亿股 (or empty).
    """
    num = _parse_number(value)
    if num is None:
        return None
    u = (unit or "").strip()
    if u in {"万股"}:
        return num
    if u in {"亿股"}:
        return num * 10000.0
    if u in {"股"}:
        return num / 10000.0
    # Heuristic: if very small and looks like 亿股
    if 0 < num < 1000:
        # 1.256 -> 12560 万股
        return num * 10000.0
    # If very large, assume 股
    if num > 1e7:
        return num / 10000.0
    return num


def normalize_money_to_yuan(value, unit: str) -> Optional[float]:
    """
    Normalize money fields to 元 (CNY).
    Supports unit in: 元 / 万元 / 亿元 (or empty).
    """
    num = _parse_number(value)
    if num is None:
        return None
    u = (unit or "").strip()
    if u in {"元", "人民币元"}:
        return num
    if u in {"万元"}:
        return num * 1e4
    if u in {"亿元"}:
        return num * 1e8
    # Fallback heuristics (from your prompt)
    if "亿" in u:
        return num * 1e8
    if "万" in u:
        return num * 1e4
    if abs(num) > 1e10:
        return num * 1e8
    if 1e6 < abs(num) < 1e10:
        return num * 1e4
    return num


def ollama_chat_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    images_b64: List[str],
    *,
    response_format,
    options: Optional[Dict] = None,
    timeout: int = 300,
) -> Dict:
    payload = {
        "model": model,
        "stream": False,
        "format": response_format,
        "options": options or {"temperature": 0},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt, "images": images_b64},
        ],
    }
    r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=int(timeout))
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content") or ""
    try:
        return json.loads(content)
    except Exception as exc:
        raise RuntimeError(f"Model returned non-JSON content: {content[:2000]!r}") from exc


def _guess_code_year_from_filename(pdf_path: Path) -> Tuple[str, Optional[int]]:
    name = pdf_path.name
    m = re.search(r"(?P<code>\d{6}).*?(?P<year>\d{4})", name)
    if not m:
        return "", None
    return m.group("code"), int(m.group("year"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample: use Ollama Qwen to extract key annual-report fields from PDF images")
    parser.add_argument("--pdf", required=True, help="PDF path")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama model name (default qwen3.5:9b)")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI for page images (default 200)")
    parser.add_argument("--code", default="", help="Stock code (6-digit); optional, inferred from filename if empty")
    parser.add_argument("--year", type=int, default=0, help="Report year; optional, inferred from filename if 0")
    parser.add_argument("--prompt-file", default="参考提示词.txt", help="Prompt template path (optional, used as reference)")
    parser.add_argument("--out-json", default="", help="Write final normalized JSON to this path (optional)")
    parser.add_argument("--debug", action="store_true", help="Print picked page numbers before calling model")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    code_guess, year_guess = _guess_code_year_from_filename(pdf_path)
    code = str(args.code).strip() or code_guess
    year = int(args.year) if int(args.year) > 0 else (int(year_guess) if year_guess else 0)

    doc = fitz.open(str(pdf_path))
    page_hits = pick_key_pages(doc)
    if not page_hits:
        raise RuntimeError("Could not locate any key pages from PDF text layer; try a different PDF for sample.")

    page_indexes = [h.page_index for h in page_hits]
    if bool(args.debug):
        print("[pages] picked:")
        for h in page_hits:
            print(f"  - p{h.page_index + 1}: {h.reason}")

    system_prompt = (
        "你是一名严格的财务报表信息抽取助手。"
        "你将看到上市公司年报PDF的关键页面图片（利润表/现金流量表/股本信息等）。"
        "你必须只输出JSON，不要输出任何额外文字。"
        "如果无法确定某个字段，输出 null 并在 evidence 里说明原因。"
    )

    # Split into smaller calls (1-2 images per call) to improve accuracy.
    page_by_reason = {h.reason: h.page_index for h in page_hits}

    def field_schema_no_page() -> Dict:
        return {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "number", "null"]},
                "unit": {"type": ["string", "null"]},
                "evidence": {"type": ["string", "null"]},
            },
            "required": ["value", "unit", "evidence"],
            "additionalProperties": False,
        }

    # Income statement (parent net profit)
    income_page = next((h.page_index for h in page_hits if "归母净利润" in h.reason), None)
    if income_page is None:
        raise RuntimeError("No income statement page found.")
    income_img = render_pages_to_base64_png(doc, [income_page], dpi=int(args.dpi))
    income_prompt = f"""
你将看到一页年报图片：合并利润表（含两栏：{year}年 与 上年对比）。

请仅提取本报告年度（{year}年）的“归属于母公司股东的净利润”这一行对应的数值与单位。
注意不要提取上一年对比列；不要提取“扣除非经常性损益后的净利润”等其他口径。

输出JSON，仅包含字段 parent_netprofit。
""".strip()
    income_format = {
        "type": "object",
        "properties": {"parent_netprofit": field_schema_no_page()},
        "required": ["parent_netprofit"],
        "additionalProperties": False,
    }
    income_res = ollama_chat_json(
        str(args.model),
        system_prompt,
        income_prompt,
        income_img,
        response_format=income_format,
        options={"temperature": 0},
        timeout=900,
    )

    # Cashflow statement (OCF + CapEx)
    cash_page = next((h.page_index for h in page_hits if "现金流量表" in h.reason), None)
    if cash_page is None:
        raise RuntimeError("No cashflow page found.")
    cash_img = render_pages_to_base64_png(doc, [cash_page], dpi=int(args.dpi))
    cash_prompt = f"""
你将看到一页年报图片：合并现金流量表（含两栏：{year}年 与 上年对比）。

请仅提取本报告年度（{year}年）的两项：
1) “经营活动产生的现金流量净额”
2) “购建固定资产、无形资产和其他长期资产支付的现金”

注意：两项都应来自现金流量表主表，不要来自补充资料/调节表。

输出JSON，仅包含字段 operating_cashflow 与 capex。
""".strip()
    cash_format = {
        "type": "object",
        "properties": {
            "operating_cashflow": field_schema_no_page(),
            "capex": field_schema_no_page(),
        },
        "required": ["operating_cashflow", "capex"],
        "additionalProperties": False,
    }
    cash_res = ollama_chat_json(
        str(args.model),
        system_prompt,
        cash_prompt,
        cash_img,
        response_format=cash_format,
        options={"temperature": 0},
        timeout=900,
    )

    # Total shares page
    shares_page = next((h.page_index for h in page_hits if "总股本" in h.reason), None)
    if shares_page is None:
        raise RuntimeError("No total shares page found.")
    shares_img = render_pages_to_base64_png(doc, [shares_page], dpi=int(args.dpi))
    shares_prompt = f"""
你将看到一页年报图片（非三张主表，通常在“重要提示/利润分配预案”等段落中）。

请提取“截至{year}年12月31日，公司总股本为XXX”的总股本数值与单位。
注意：这里的总股本通常以 万股/股 为单位，务必带上单位。

输出JSON，仅包含字段 total_shares。
""".strip()
    shares_format = {
        "type": "object",
        "properties": {"total_shares": field_schema_no_page()},
        "required": ["total_shares"],
        "additionalProperties": False,
    }
    shares_res = ollama_chat_json(
        str(args.model),
        system_prompt,
        shares_prompt,
        shares_img,
        response_format=shares_format,
        options={"temperature": 0},
        timeout=900,
    )

    extracted = {
        "code": code,
        "year": year if year else None,
        "parent_netprofit": dict(income_res.get("parent_netprofit") or {}, page=income_page + 1),
        "total_shares": dict(shares_res.get("total_shares") or {}, page=shares_page + 1),
        "operating_cashflow": dict(cash_res.get("operating_cashflow") or {}, page=cash_page + 1),
        "capex": dict(cash_res.get("capex") or {}, page=cash_page + 1),
    }

    # Normalize units
    pn = extracted.get("parent_netprofit") or {}
    ts = extracted.get("total_shares") or {}
    ocf = extracted.get("operating_cashflow") or {}
    cap = extracted.get("capex") or {}

    parent_netprofit_yuan = normalize_money_to_yuan(pn.get("value"), str(pn.get("unit") or ""))
    operating_cashflow_yuan = normalize_money_to_yuan(ocf.get("value"), str(ocf.get("unit") or ""))
    capex_yuan = normalize_money_to_yuan(cap.get("value"), str(cap.get("unit") or ""))
    if capex_yuan is not None and capex_yuan < 0:
        capex_yuan = -capex_yuan

    total_shares_wan = normalize_total_shares_to_wan(ts.get("value"), str(ts.get("unit") or ""))

    normalized = {
        "code": code,
        "year": year if year else None,
        "raw": extracted,
        "normalized": {
            "total_shares_wan": total_shares_wan,
            "net_profit_yuan": parent_netprofit_yuan,
            "operating_cashflow_yuan": operating_cashflow_yuan,
            "capex_yuan": capex_yuan,
        },
    }

    print(json.dumps(normalized, ensure_ascii=False, indent=2))

    if str(args.out_json).strip():
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
