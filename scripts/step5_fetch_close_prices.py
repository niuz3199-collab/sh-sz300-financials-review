import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent


@dataclass(frozen=True)
class ClosePriceRow:
    year: int
    target_date: str
    stock_code: str
    code_name: str
    stock_name: str
    price_date: str
    close: float
    note: str


def _parse_code_name(code_name: str) -> Tuple[str, str]:
    """
    code_name examples:
    - sh.600000
    - sz.000001
    """
    text = (code_name or "").strip()
    if "." not in text:
        raise ValueError(f"Invalid code_name: {code_name!r}")
    market_prefix, code = text.split(".", 1)
    market_prefix = market_prefix.lower().strip()
    code = code.strip()
    if not code.isdigit():
        raise ValueError(f"Invalid code in code_name: {code_name!r}")
    if market_prefix not in {"sh", "sz"}:
        raise ValueError(f"Unsupported market prefix in code_name: {code_name!r}")
    return market_prefix, code.zfill(6)


def _to_secid(code_name: str) -> str:
    market_prefix, code = _parse_code_name(code_name)
    market = "1" if market_prefix == "sh" else "0"
    return f"{market}.{code}"


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, float):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _fetch_close_on_or_before(
    session: requests.Session,
    *,
    code_name: str,
    target: date,
    lookback_days: int,
    timeout: int = 20,
) -> Optional[Tuple[str, float]]:
    secid = _to_secid(code_name)
    beg = (target - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end = target.strftime("%Y%m%d")
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": "101",
        # 用不复权的收盘价（历史极早期数据在前复权下可能出现异常负数）
        "fqt": "0",
        "beg": beg,
        "end": end,
    }
    resp = session.get(EASTMONEY_KLINE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = (resp.json() or {}).get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return None

    picked: Optional[Tuple[str, float]] = None
    for line in klines:
        # 2024-04-30,6.85,6.97,7.11,6.83,1054728,813284484.00
        parts = (line or "").split(",")
        if len(parts) < 3:
            continue
        d_str = parts[0].strip()
        close = _safe_float(parts[2])
        if not d_str or close is None:
            continue
        try:
            d = date.fromisoformat(d_str)
        except Exception:
            continue
        if d <= target:
            picked = (d_str, float(close))

    return picked


def get_close_price(
    session: requests.Session,
    *,
    code_name: str,
    target_date: str,
    max_lookback_days: int = 120,
    step_days: int = 20,
    timeout: int = 20,
    max_retries: int = 3,
    retry_backoff: float = 0.6,
) -> Optional[Tuple[str, float, str]]:
    """
    Return (price_date, close, note).
    If 4/30 is non-trading day or suspended, fallback to last trading day <= target_date.
    """
    target = date.fromisoformat(target_date)
    for lookback in range(step_days, max_lookback_days + 1, step_days):
        picked = None
        for attempt in range(1, max(1, int(max_retries)) + 1):
            try:
                picked = _fetch_close_on_or_before(
                    session,
                    code_name=code_name,
                    target=target,
                    lookback_days=lookback,
                    timeout=timeout,
                )
                break
            except Exception:
                if attempt >= int(max_retries):
                    picked = None
                    break
                time.sleep(float(retry_backoff) * attempt)
        if picked:
            price_date, close = picked
            note = "" if price_date == target_date else f"fallback_to_{price_date}"
            return price_date, close, note
    return None


def load_existing_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if df.empty:
        return {}
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in df.to_dict(orient="records"):
        stock_code = str(row.get("stock_code", "")).zfill(6)
        code_name = str(row.get("code_name", "")).strip()
        close = _safe_float(row.get("close"))
        if not stock_code or not code_name or close is None:
            continue
        out[(stock_code, code_name)] = dict(row)
    return out


def load_name_map(history_json_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Return: {year_str: {code_name: stock_name}}
    """
    if not history_json_path.exists():
        return {}
    obj = json.loads(history_json_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}
    for year_str, block in (obj or {}).items():
        stocks = (block or {}).get("stocks") or []
        year_map: Dict[str, str] = {}
        for item in stocks:
            code = (item or {}).get("code")
            name = (item or {}).get("code_name")
            if code and name:
                year_map[str(code)] = str(name)
        out[str(year_str)] = year_map
    return out


def load_kept_constituents(status_csv_path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(status_csv_path, dtype=str, encoding="utf-8")
    df_year = df[df["year"].astype(str) == str(year)].copy()
    status = df_year.get("剔除状态")
    if status is None:
        kept = df_year
    else:
        kept = df_year[status.isna() | (status.astype(str).str.strip() == "")]
    kept = kept[["stock_code", "code_name"]].copy()
    kept["stock_code"] = kept["stock_code"].astype(str).str.zfill(6)
    kept["code_name"] = kept["code_name"].astype(str).str.strip()
    kept = kept.drop_duplicates(subset=["code_name"]).sort_values("code_name")
    return kept.reset_index(drop=True)


def resolve_path(base_dir: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def write_markdown(path: Path, *, title: str, df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"共 {len(df)} 只（已剔除不满10年上市样本）")
    lines.append("")

    headers = ["序号", "股票代码", "股票名称", "收盘价", "交易日期", "备注"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for idx, row in enumerate(df.to_dict(orient="records"), start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("stock_code", "")),
                    str(row.get("stock_name", "")),
                    str(row.get("close", "")),
                    str(row.get("price_date", "")),
                    str(row.get("note", "")),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step5: 抓取剔除后的沪深300成分股在当年4月30日的收盘价")
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="项目根目录（包含 2006/2007/... 年度文件夹的目录）",
    )
    parser.add_argument("--start-year", type=int, default=2006, help="起始年份（含）")
    parser.add_argument("--end-year", type=int, default=2024, help="结束年份（含）")
    parser.add_argument("--status-csv", default="hs300_with_ipo_status.csv", help="剔除状态文件")
    parser.add_argument("--history-json", default="hs300_history.json", help="用于补全股票名称")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在输出文件")
    parser.add_argument("--sleep", type=float, default=0.15, help="每次请求基础休眠秒数")
    parser.add_argument("--jitter", type=float, default=0.25, help="随机抖动秒数")
    parser.add_argument("--lookback-days", type=int, default=120, help="最多回看多少天以应对非交易日/停牌")
    parser.add_argument("--max-retries", type=int, default=3, help="single request retries before giving up")
    parser.add_argument("--retry-backoff", type=float, default=0.6, help="seconds multiplier between retries")
    parser.add_argument("--reuse-existing", action="store_true", help="keep existing close values when a refetch fails")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    status_csv_path = resolve_path(base_dir, args.status_csv).resolve()
    history_json_path = resolve_path(base_dir, args.history_json).resolve()
    name_map = load_name_map(history_json_path)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://quote.eastmoney.com/",
            "Accept": "application/json,text/plain,*/*",
        }
    )

    for year in range(int(args.start_year), int(args.end_year) + 1):
        year_dir = base_dir / str(year)
        if not year_dir.exists():
            year_dir.mkdir(parents=True, exist_ok=True)
            print(f"[创建] 年度目录: {year_dir}")

        out_csv = year_dir / f"{year}_0430收盘价.csv"
        out_md = year_dir / f"{year}_0430收盘价.md"
        if not args.overwrite and out_csv.exists() and out_md.exists():
            print(f"[跳过] 已存在: {out_csv} / {out_md}")
            continue

        target_date = f"{year}-04-30"
        print(f"\n--- {year}: 抓取 {target_date} 收盘价 ---")

        kept = load_kept_constituents(status_csv_path, year)
        rows: List[ClosePriceRow] = []
        existing_rows = load_existing_rows(out_csv) if args.reuse_existing and out_csv.exists() else {}

        year_name_map = name_map.get(str(year), {})

        for i, rec in enumerate(kept.to_dict(orient="records"), start=1):
            stock_code = str(rec.get("stock_code", "")).zfill(6)
            code_name = str(rec.get("code_name", "")).strip()
            stock_name = year_name_map.get(code_name, "")

            result = get_close_price(
                session,
                code_name=code_name,
                target_date=target_date,
                max_lookback_days=int(args.lookback_days),
                max_retries=int(args.max_retries),
                retry_backoff=float(args.retry_backoff),
            )
            if result is None:
                existing = existing_rows.get((stock_code, code_name))
                if existing:
                    rows.append(
                        ClosePriceRow(
                            year=year,
                            target_date=target_date,
                            stock_code=stock_code,
                            code_name=code_name,
                            stock_name=stock_name or str(existing.get("stock_name", "") or ""),
                            price_date=str(existing.get("price_date", "") or ""),
                            close=float(existing.get("close")),
                            note=str(existing.get("note", "") or "reused_existing_price"),
                        )
                    )
                else:
                    rows.append(
                        ClosePriceRow(
                            year=year,
                            target_date=target_date,
                            stock_code=stock_code,
                            code_name=code_name,
                            stock_name=stock_name,
                            price_date="",
                            close=float("nan"),
                            note="no_price",
                        )
                    )
            else:
                price_date, close, note = result
                rows.append(
                    ClosePriceRow(
                        year=year,
                        target_date=target_date,
                        stock_code=stock_code,
                        code_name=code_name,
                        stock_name=stock_name,
                        price_date=price_date,
                        close=close,
                        note=note,
                    )
                )

            if i % 20 == 0:
                print(f"[进度] {i}/{len(kept)}")

            time.sleep(max(0.0, float(args.sleep) + random.random() * float(args.jitter)))

        out_df = pd.DataFrame([r.__dict__ for r in rows])
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        write_markdown(out_md, title=f"剔除后成分股收盘价 - {year}年4月30日", df=out_df)
        print(f"[完成] 输出: {out_csv} / {out_md}")


if __name__ == "__main__":
    main()
