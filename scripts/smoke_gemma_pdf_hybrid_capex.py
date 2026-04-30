#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smoke test for hybrid Markdown + PDF-vision CapEx extraction.

Why this exists
- Some markdown-only failures are not true missing values.
- The markdown often still contains enough clues to recover the row:
  - split row labels
  - exact current/prior year numbers
  - occasional page/image anchors
- We can use those clues to localize the right PDF page and crop a tighter
  visual region for a local vision model.

What this script does
1. Load a small set of capex-partial tasks from partial_tasks_latest.csv.
2. Find capex clue lines in the markdown and extract numeric anchors.
3. Scan the restored full PDF for the page that contains the same anchors.
4. Crop a full-width band around the matched numeric row.
5. Send the crop + full page to LM Studio vision with JSON schema output.
6. Save smoke outputs for manual inspection.

This is intentionally scoped to smoke testing a few samples, not the full
production rerun yet.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.step6_extract_financials_from_markdown import (  # noqa: E402
    FIELD_SPECS,
    Snippet,
    _coerce_field_payload,
    _extract_json_from_content,
    _null_field,
    build_field_schema,
    retrieve_snippets,
)
from scripts.step6_extract_financials_qwen_pdf import (  # noqa: E402
    OPENAI_CHAT_COMPLETIONS_SUFFIX,
    normalize_money_to_yuan,
)


CAPEX_FULL_TERMS = [
    "购建固定资产、无形资产及其他长期资产支付的现金",
    "购建固定资产、无形资产和其他长期资产支付的现金",
    "购建固定资产、无形资产及其他长期资产所支付的现金",
    "购建固定资产、无形资产和其他长期资产所支付的现金",
    "cash paid for the purchase and construction of fixed assets, intangible assets, and other long-term assets",
    "payments for the acquisition and construction of fixed assets, intangible assets and other long-term assets",
]

CAPEX_FRAGMENT_TERMS = [
    "购建固定资产",
    "长期资产所支付的现金",
    "长期资产支付的现金",
    "fixed assets, intangible assets",
    "other long-term assets",
    "purchase and construction of fixed assets",
]

PAGE_HINT_TERMS = [
    "合并现金流量表",
    "现金流量表",
    "投资活动产生的现金流量",
    "投资活动现金流出小计",
    "consolidated cash flow statement",
    "cash flow statement",
    "cash flows from investment activities",
    "investment activities",
]

NUMBER_PATTERN = re.compile(r"-?\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?")
PAGE_MARKER_PATTERNS = [
    re.compile(r"_page_(\d+)_", re.I),
    re.compile(r"page-(\d+)-", re.I),
]


@dataclass(frozen=True)
class SmokeTask:
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


def _parse_page_markers(text: str) -> List[int]:
    out: List[int] = []
    for pattern in PAGE_MARKER_PATTERNS:
        for raw in pattern.findall(str(text or "")):
            try:
                out.append(int(raw))
            except Exception:
                continue
    return sorted(set(out))


def load_smoke_tasks(
    csv_path: Path,
    *,
    fulltext_root: Path,
    year: int,
    codes: Sequence[str],
) -> List[SmokeTask]:
    wanted = {str(code).strip() for code in codes if str(code).strip()}
    tasks: List[SmokeTask] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("year") or "").strip() != str(int(year)):
                continue
            stock_code = str(row.get("stock_code") or "").strip()
            if wanted and stock_code not in wanted:
                continue
            missing_fields = str(row.get("missing_fields") or "")
            if "capex" not in missing_fields:
                continue
            report_name = str(row.get("report_name") or "").strip()
            md_path = Path(str(row.get("markdown_path") or "").strip())
            raw_json_path = Path(str(row.get("raw_json_path") or "").strip())
            pdf_path = fulltext_root / str(int(year)) / f"{report_name}.pdf"
            if not md_path.exists():
                raise FileNotFoundError(f"Markdown not found: {md_path}")
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            tasks.append(
                SmokeTask(
                    year=int(year),
                    stock_code=stock_code,
                    report_name=report_name,
                    md_path=md_path,
                    raw_json_path=raw_json_path,
                    pdf_path=pdf_path,
                )
            )
    tasks = sorted(tasks, key=lambda x: (x.year, x.stock_code, x.report_name))
    if wanted:
        order = {code: idx for idx, code in enumerate(codes)}
        tasks = sorted(tasks, key=lambda x: order.get(x.stock_code, 10**9))
    return tasks


def build_anchor_windows(markdown_text: str) -> List[AnchorWindow]:
    lines = markdown_text.splitlines()
    windows: List[AnchorWindow] = []
    seen_ranges = set()
    for idx, line in enumerate(lines):
        norm_line = _clean(line)
        if not norm_line:
            continue
        hits: List[str] = []
        score = 0.0
        for term in CAPEX_FULL_TERMS:
            if _clean(term) in norm_line:
                hits.append(term)
                score += 20.0
        for term in CAPEX_FRAGMENT_TERMS:
            if _clean(term) in norm_line:
                hits.append(term)
                score += 8.0
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
        numbers = _extract_numbers(text)
        hint_bonus = 0.0
        clean_window = _clean(text)
        for hint in PAGE_HINT_TERMS:
            if _clean(hint) in clean_window:
                hint_bonus += 2.0
        windows.append(
            AnchorWindow(
                start_line=start + 1,
                end_line=end + 1,
                text=text,
                matched_terms=sorted(set(hits)),
                numbers=numbers,
                score=score + hint_bonus + min(len(numbers), 8),
            )
        )
    windows.sort(key=lambda x: (-x.score, -len(x.numbers), x.start_line))
    return windows


def choose_anchor_window(markdown_text: str, snippets: Sequence[Snippet]) -> Optional[AnchorWindow]:
    windows = build_anchor_windows(markdown_text)
    if windows:
        return windows[0]
    # Fallback: try only within retrieved snippets if the full markdown scan missed.
    for snippet in snippets:
        snippet_windows = build_anchor_windows(snippet.text)
        if snippet_windows:
            window = snippet_windows[0]
            window.start_line = snippet.start_line + window.start_line - 1
            window.end_line = snippet.start_line + window.end_line - 1
            return window
    return None


def infer_target_number_hint(anchor_window: AnchorWindow) -> TargetNumberHint:
    lines = [str(line or "") for line in anchor_window.text.splitlines()]

    def _line_has_capex_term(text: str) -> bool:
        clean_text = _clean(text)
        return any(_clean(term) in clean_text for term in (CAPEX_FULL_TERMS + CAPEX_FRAGMENT_TERMS))

    def _continuation_hint(text: str) -> bool:
        clean_text = _clean(text)
        return any(
            token in clean_text
            for token in [
                _clean("长期资产"),
                _clean("other long-term assets"),
                _clean("and other long-term assets"),
            ]
        )

    def _subtotal_hint(text: str) -> bool:
        clean_text = _clean(text)
        return any(token in clean_text for token in [_clean("小计"), _clean("subtotal")])

    for idx, line in enumerate(lines):
        if not _line_has_capex_term(line):
            continue
        numbers_in_line = _extract_numbers(line)
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        next_numbers = _extract_numbers(next_line) if next_line else []
        if next_line and len(next_numbers) >= 2 and _continuation_hint(next_line):
            if _subtotal_hint(line) or "<br>" in line or not numbers_in_line:
                return TargetNumberHint(
                    current_value=next_numbers[0],
                    prior_value=next_numbers[1],
                    source="preferred_next_line",
                    evidence_line=next_line,
                )
        if len(numbers_in_line) >= 2:
            return TargetNumberHint(
                current_value=numbers_in_line[0],
                prior_value=numbers_in_line[1],
                source="same_line",
                evidence_line=line,
            )
        if idx + 1 < len(lines):
            if len(next_numbers) >= 2 and (_continuation_hint(next_line) or not _extract_numbers(line)):
                return TargetNumberHint(
                    current_value=next_numbers[0],
                    prior_value=next_numbers[1],
                    source="next_line",
                    evidence_line=next_line,
                )
            combo_numbers = _extract_numbers(f"{line}\n{next_line}")
            if len(combo_numbers) >= 2 and _continuation_hint(next_line):
                return TargetNumberHint(
                    current_value=combo_numbers[-2],
                    prior_value=combo_numbers[-1],
                    source="combined_lines",
                    evidence_line=f"{line} / {next_line}",
                )

    for line in lines:
        numbers_in_line = _extract_numbers(line)
        if len(numbers_in_line) >= 2 and _continuation_hint(line):
            return TargetNumberHint(
                current_value=numbers_in_line[0],
                prior_value=numbers_in_line[1],
                source="continuation_line_fallback",
                evidence_line=line,
            )

    fallback_numbers = anchor_window.numbers[:2]
    return TargetNumberHint(
        current_value=fallback_numbers[0] if len(fallback_numbers) >= 1 else None,
        prior_value=fallback_numbers[1] if len(fallback_numbers) >= 2 else None,
        source="fallback_first_two",
        evidence_line=anchor_window.text,
    )


def read_current_raw_capex(raw_json_path: Path) -> Dict[str, object]:
    if not raw_json_path.exists():
        return {}
    try:
        data = json.loads(raw_json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    raw = data.get("raw") or {}
    capex = raw.get("capex") or {}
    return _coerce_field_payload(capex)


def score_pdf_page(
    page_text: str,
    *,
    anchor_numbers: Sequence[str],
    marker_pages: Sequence[int],
    page_number: int,
) -> Tuple[float, int]:
    text = str(page_text or "")
    clean_text = _clean(text)
    number_hits = sum(1 for num in anchor_numbers if num and num in text)
    score = float(number_hits * 100)
    if page_number in marker_pages:
        score += 60.0
    for hint in PAGE_HINT_TERMS:
        if _clean(hint) in clean_text:
            score += 10.0
    if "subtotalofcashoutflowsfrominvestmentactivities" in clean_text or "投资活动现金流出小计" in clean_text:
        score += 4.0
    return score, number_hits


def rank_candidate_pages(
    doc: fitz.Document,
    *,
    anchor_numbers: Sequence[str],
    marker_pages: Sequence[int],
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for page_index in range(doc.page_count):
        text = doc[page_index].get_text("text") or ""
        score, number_hits = score_pdf_page(
            text,
            anchor_numbers=anchor_numbers,
            marker_pages=marker_pages,
            page_number=page_index,
        )
        if score <= 0 and number_hits <= 0 and page_index not in marker_pages:
            continue
        candidates.append(
            {
                "page_index": page_index,
                "page_number": page_index + 1,
                "score": score,
                "number_hits": number_hits,
            }
        )
    candidates.sort(key=lambda x: (-float(x["score"]), -int(x["number_hits"]), int(x["page_index"])))
    return candidates


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
    best_cluster = clusters[0]
    best_items = best_cluster["items"]
    rects = [rect for _, rect in best_items]
    union = fitz.Rect(rects[0])
    for rect in rects[1:]:
        union |= rect

    avg_height = sum(rect.height for rect in rects) / max(len(rects), 1)
    margin_top = max(95.0, avg_height * 3.2)
    margin_bottom = max(60.0, avg_height * 1.9)
    crop = fitz.Rect(
        page.rect.x0 + 8.0,
        max(page.rect.y0, union.y0 - margin_top),
        page.rect.x1 - 8.0,
        min(page.rect.y1, union.y1 + margin_bottom),
    )
    serializable_hits = {
        num: [[float(r.x0), float(r.y0), float(r.x1), float(r.y1)] for r in rects]
        for num, rects in hit_map.items()
    }
    return crop, serializable_hits


def render_rect_to_png_bytes(page: fitz.Page, rect: fitz.Rect, *, dpi: int) -> bytes:
    matrix = fitz.Matrix(float(dpi) / 72.0, float(dpi) / 72.0)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    return pix.tobytes("png")


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def build_vision_schema() -> Dict[str, object]:
    return {
        "type": "object",
        "properties": {"capex": build_field_schema()},
        "required": ["capex"],
        "additionalProperties": False,
    }


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
        "max_tokens": 350,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "hybrid_capex_extract", "schema": response_schema},
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
        raise RuntimeError(f"invalid_json_response:{preview[:1000]}") from exc


def build_prompt(
    *,
    task: SmokeTask,
    snippets: Sequence[Snippet],
    anchor_window: AnchorWindow,
    target_hint: TargetNumberHint,
    chosen_page_number: int,
) -> Tuple[str, str]:
    anchor_text = (
        f"[ANCHOR_LINES {anchor_window.start_line}-{anchor_window.end_line}]\n"
        f"{anchor_window.text.strip()}"
    )
    target_numbers = [x for x in [target_hint.current_value, target_hint.prior_value] if x]
    anchor_numbers = " / ".join(target_numbers) if target_numbers else "无"
    snippet_ids = ", ".join(snippet.snippet_id for snippet in snippets[:2]) if snippets else "无"
    system_prompt = (
        "你是一名严格的财务报表视觉抽取助手。"
        "你将看到来自上市公司年报 PDF 的局部高清截图和整页截图，以及极少量 markdown 锚点线索。"
        "只能依据图片和提供的线索判断，不要猜测。"
        "必须把最终答案放在 JSON 中。"
    )
    user_prompt = f"""
你现在只提取一个字段：capex。
目标是本报告年度（{task.year}年）的资本支出现金流，优先口径为：
- 购建固定资产、无形资产及其他长期资产支付的现金
- 购建固定资产、无形资产和其他长期资产支付的现金
- 购建固定资产、无形资产及其他长期资产所支付的现金
- 购建固定资产、无形资产和其他长期资产所支付的现金
- 英文等价行：Cash paid for the purchase and construction of fixed assets, intangible assets, and other long-term assets

不要把以下项目当成 capex：
- 投资活动现金流入/流出小计
- 投资支付的现金 / Cash paid for investments
- 其他与投资活动有关的现金

你会看到两张图片：
1. 以数字锚点为中心的局部高清裁剪图
2. 同一候选页的整页图

候选页：第 {chosen_page_number} 页。

Markdown 锚点如下：
{anchor_text}

目标行数字锚点（优先匹配）：
{anchor_numbers}

可用 snippet id：
{snippet_ids}

要求：
- 只取 {task.year} 年这一列，不要取上年对比列。
- 若目标行被拆成两行，请把两行合并判断。
- 不要把相邻的“投资所支付的现金 / Cash paid for investments”误判为 capex。
- 优先选择与目标行数字锚点一致的那一行。
- 若页面明确写明金额单位为人民币元/元/CNY/RMB，可将 unit 写为“元”或等价英文单位。
- evidence 最多一句，尽量短。
- snippet_ids 填你参考到的 snippet id；若未实际参考 snippet，可填空数组。
- 只输出 JSON，不要输出任何解释。
""".strip()
    return system_prompt, user_prompt


def normalize_result(payload: Dict[str, object]) -> Dict[str, object]:
    capex = _coerce_field_payload((payload.get("capex") if isinstance(payload, dict) else None))
    unit = str(capex.get("unit") or "")
    normalized_yuan = normalize_money_to_yuan(capex.get("value"), unit)
    return {
        "capex": capex,
        "normalized": {"capex_yuan": normalized_yuan},
        "ok": normalized_yuan is not None,
    }


def run_one_task(
    task: SmokeTask,
    *,
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
    task_out_dir = out_dir / f"{task.stock_code}_{task.year}"
    task_out_dir.mkdir(parents=True, exist_ok=True)

    markdown_text = task.md_path.read_text(encoding="utf-8", errors="ignore")
    snippets = retrieve_snippets(
        markdown_text=markdown_text,
        field_name="capex",
        year=task.year,
        max_snippets=max_snippets,
        max_chars=max_chars,
    )
    anchor_window = choose_anchor_window(markdown_text, snippets)
    if anchor_window is None:
        result = {
            "task": {
                "year": task.year,
                "stock_code": task.stock_code,
                "report_name": task.report_name,
            },
            "error": "no_anchor_window",
            "current_raw_capex": read_current_raw_capex(task.raw_json_path),
        }
        (task_out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    marker_pages: List[int] = []
    for snippet in snippets:
        marker_pages.extend(_parse_page_markers(snippet.text))
    marker_pages = sorted({max(1, p) for p in marker_pages})
    target_hint = infer_target_number_hint(anchor_window)
    target_numbers = [x for x in [target_hint.current_value, target_hint.prior_value] if x]

    doc = fitz.open(task.pdf_path)
    ranked_pages = rank_candidate_pages(
        doc,
        anchor_numbers=anchor_window.numbers,
        marker_pages=[p - 1 for p in marker_pages],
    )
    if not ranked_pages:
        result = {
            "task": {
                "year": task.year,
                "stock_code": task.stock_code,
                "report_name": task.report_name,
            },
            "error": "no_candidate_pdf_page",
            "anchor_window": {
                "start_line": anchor_window.start_line,
                "end_line": anchor_window.end_line,
                "matched_terms": anchor_window.matched_terms,
                "numbers": anchor_window.numbers,
            },
            "current_raw_capex": read_current_raw_capex(task.raw_json_path),
        }
        (task_out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    best_page = ranked_pages[0]
    page_index = int(best_page["page_index"])
    page = doc[page_index]
    crop_rect, numeric_hit_map = compute_crop_rect(page, target_numbers or anchor_window.numbers)

    crop_png = render_rect_to_png_bytes(page, crop_rect, dpi=dpi_crop)
    full_page_png = render_rect_to_png_bytes(page, fitz.Rect(page.rect), dpi=dpi_page)

    crop_path = task_out_dir / "crop.png"
    page_path = task_out_dir / "page.png"
    write_bytes(crop_path, crop_png)
    write_bytes(page_path, full_page_png)

    system_prompt, user_prompt = build_prompt(
        task=task,
        snippets=snippets,
        anchor_window=anchor_window,
        target_hint=target_hint,
        chosen_page_number=page_index + 1,
    )
    schema = build_vision_schema()

    t0 = time.time()
    raw_model_payload = call_lm_studio_vision_json(
        model=model,
        api_base_url=api_base_url,
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images_b64=[
            base64.b64encode(crop_png).decode("utf-8"),
            base64.b64encode(full_page_png).decode("utf-8"),
        ],
        response_schema=schema,
        timeout=timeout,
    )
    elapsed_sec = round(time.time() - t0, 2)

    normalized = normalize_result(raw_model_payload)
    current_raw_capex = read_current_raw_capex(task.raw_json_path)

    result = {
        "task": {
            "year": task.year,
            "stock_code": task.stock_code,
            "report_name": task.report_name,
            "pdf_path": str(task.pdf_path),
            "markdown_path": str(task.md_path),
            "raw_json_path": str(task.raw_json_path),
        },
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
        },
        "marker_pages_from_snippets": marker_pages,
        "candidate_pages_top5": ranked_pages[:5],
        "chosen_page": {
            "page_index": page_index,
            "page_number": page_index + 1,
            "crop_rect": [float(crop_rect.x0), float(crop_rect.y0), float(crop_rect.x1), float(crop_rect.y1)],
            "numeric_hit_map": numeric_hit_map,
        },
        "images": {
            "crop_path": str(crop_path),
            "page_path": str(page_path),
        },
        "current_raw_capex": current_raw_capex,
        "model_response": raw_model_payload,
        "normalized": normalized,
        "elapsed_sec": elapsed_sec,
    }
    (task_out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


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
            f"{row.get('stock_code')},{row.get('page_number')},{row.get('value')},{row.get('ok')},{row.get('elapsed_sec')}"
        )
    (out_dir / "progress_status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test hybrid PDF+Markdown capex extraction with LM Studio vision")
    parser.add_argument(
        "--partial-csv",
        default=str(REPO_ROOT / ".tmp_gemma_markdown_repair_runner" / "partial_tasks_latest.csv"),
        help="partial tasks csv",
    )
    parser.add_argument(
        "--fulltext-root",
        default=str(REPO_ROOT / "年报" / "下载年报_fulltext"),
        help="restored fulltext PDF root",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / ".tmp_gemma_pdf_hybrid_smoke"),
        help="output directory",
    )
    parser.add_argument("--year", type=int, default=2024, help="target year for smoke tasks")
    parser.add_argument(
        "--codes",
        nargs="+",
        default=["000166", "000651", "000776"],
        help="stock codes to smoke test",
    )
    parser.add_argument("--model", default="google/gemma-4-26b-a4b", help="LM Studio model id")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:1234/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="lm-studio", help="API key placeholder for local server")
    parser.add_argument("--timeout", type=int, default=240, help="request timeout seconds")
    parser.add_argument("--dpi-crop", type=int, default=220, help="crop render dpi")
    parser.add_argument("--dpi-page", type=int, default=150, help="full page render dpi")
    parser.add_argument("--max-snippets", type=int, default=3, help="retrieved snippets for prompt context")
    parser.add_argument("--max-chars", type=int, default=16000, help="max markdown chars for snippets")
    args = parser.parse_args()

    partial_csv = Path(args.partial_csv)
    fulltext_root = Path(args.fulltext_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_smoke_tasks(
        partial_csv,
        fulltext_root=fulltext_root,
        year=int(args.year),
        codes=list(args.codes),
    )
    if not tasks:
        raise RuntimeError("No smoke tasks matched the requested year/codes.")

    summary_rows: List[Dict[str, object]] = []
    write_progress(out_dir, summary_rows, len(tasks))

    for task in tasks:
        result = run_one_task(
            task,
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
        capex = normalized.get("capex") or {}
        summary_rows.append(
            {
                "year": task.year,
                "stock_code": task.stock_code,
                "report_name": task.report_name,
                "page_number": ((result.get("chosen_page") or {}).get("page_number")),
                "value": (capex.get("value") if isinstance(capex, dict) else None),
                "unit": (capex.get("unit") if isinstance(capex, dict) else None),
                "capex_yuan": ((normalized.get("normalized") or {}).get("capex_yuan") if isinstance(normalized, dict) else None),
                "ok": bool(normalized.get("ok")) if isinstance(normalized, dict) else False,
                "elapsed_sec": result.get("elapsed_sec"),
                "crop_path": ((result.get("images") or {}).get("crop_path")),
                "page_path": ((result.get("images") or {}).get("page_path")),
            }
        )
        write_progress(out_dir, summary_rows, len(tasks))

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "stock_code",
                "report_name",
                "page_number",
                "value",
                "unit",
                "capex_yuan",
                "ok",
                "elapsed_sec",
                "crop_path",
                "page_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Smoke finished: {summary_path}")
    ok = sum(1 for row in summary_rows if row.get("ok"))
    print(f"ok={ok}/{len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
