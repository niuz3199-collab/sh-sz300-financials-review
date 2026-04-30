# 中国 A 股年报因子提取与策略回测

这个目录是从原始工作区中抽出来的 GitHub 上传版，只保留了适合公开托管的源码、说明文档、配置模板和少量轻量参考数据。

项目目标是把中国 A 股上市公司年报中的关键财务字段批量结构化，进一步计算沪深300口径的 `CAPE` 与 `FCF Yield`，并基于这些因子做年度资产配置与策略回测。

## 这份上传版包含什么

- `scripts/`
  - 年报抓取、PDF/Markdown 提取、Gemma/Qwen 修复、监控、回填、因子计算、回测与实验脚本。
- `docs/`
  - 项目说明、原始任务描述、提示词参考、数据目录约定。
- `configs/`
  - 本地模型运行配置模板。
- `reference_data/`
  - 轻量参考文件，便于理解字段与流程。
- `sample_data/`
  - 仅保留说明，不包含真实年报、缓存或批量结果。

## 这份上传版刻意排除了什么

- 原始年报 PDF、Markdown、图片切片。
- `.cache/`、`.tmp*/`、监控日志、临时运行目录。
- 按年份生成的批量 CSV/JSON 结果。
- 大体量实验输出、人工处理文件夹、压缩包。

这样做的原因很直接：

- GitHub 不适合承载几十 GB 的工作缓存和原始文档。
- 年报 PDF 存在体积和再分发风险。
- 运行期产物应留在本地数据工作区，而不是源码仓库。

## 仓库边界

这份目录更适合作为“公开代码镜像”或“研究复现说明仓库”，不是完整数据仓库。

- 许可边界见 [LICENSE.md](LICENSE.md)
- 上传前自查见 [docs/上传前检查清单.md](docs/上传前检查清单.md)
- 脚本导航见 [scripts/README.md](scripts/README.md)

## 环境要求

- Windows + PowerShell
- Python 3.10+
- 本地或远端 OpenAI 兼容接口
- 如果走 PDF 图片或局部截图路线，需要 `PyMuPDF`

安装依赖：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 环境变量

见 `.env.example`。当前脚本主要会读取这些变量：

- `LM_STUDIO_API_KEY`
- `OPENAI_API_KEY`
- `SCNET_API_KEY`
- `NAAPI_API_KEY`

本地 LM Studio 通常可配合：

- `api_base_url=http://127.0.0.1:1234/v1`
- `api_key_env=LM_STUDIO_API_KEY`

## 推荐的数据分离方式

建议把这个 GitHub 仓库作为“代码仓库”，把真正的大文件和运行结果放到单独的“数据工作区”。

例如：

```text
C:\repo\fire-github\
  scripts\
  docs\
  configs\

D:\fire-data\
  年报\下载年报_fulltext\
  .cache\qwen_pdf_markdown_remaining\output_markdown\
  2001\
  2002\
  ...
  2025\
  CPI指数汇总.csv
  hs300_with_ipo_status.csv
  hs300_history.json
```

多数脚本都支持通过 `--base-dir`、`--pdf-root`、`--markdown-root`、`--year-csv-root` 指向本地数据工作区，所以代码和数据不需要继续耦合在同一个目录里。

详细约定见 [docs/数据目录约定.md](docs/数据目录约定.md)。

## 常见运行方式

### 1. Markdown 路线提取

```powershell
python scripts\step6_extract_financials_from_markdown.py `
  --base-dir D:\fire-data `
  --markdown-root D:\fire-data\.cache\qwen_pdf_markdown_remaining\output_markdown `
  --out-dir D:\fire-data\.tmp_gemma_markdown_financials `
  --year-csv-root D:\fire-data `
  --csv-name "{year}_gemma_markdown.csv" `
  --model "google/gemma-4-26b-a4b" `
  --api-base-url "http://127.0.0.1:1234/v1" `
  --api-key-env "LM_STUDIO_API_KEY"
```

### 2. Markdown 修复与监控

```powershell
python scripts\repair_gemma_markdown_financials.py `
  --base-dir D:\fire-data `
  --markdown-root D:\fire-data\.cache\qwen_pdf_markdown_remaining\output_markdown `
  --out-dir D:\fire-data\.tmp_gemma_markdown_financials_full `
  --year-csv-root D:\fire-data `
  --model "google/gemma-4-26b-a4b" `
  --api-base-url "http://127.0.0.1:1234/v1"
```

```powershell
python scripts\monitor_gemma_markdown_progress.py `
  --out-dir D:\fire-data\.tmp_gemma_markdown_financials_full
```

### 3. PDF + Markdown 混合修复

```powershell
python scripts\repair_gemma_pdf_hybrid_fields.py `
  --base-dir D:\fire-data `
  --fulltext-root D:\fire-data\年报\下载年报_fulltext `
  --markdown-root D:\fire-data\.cache\qwen_pdf_markdown_remaining\output_markdown `
  --out-dir D:\fire-data\.tmp_gemma_pdf_hybrid_repair `
  --api-base-url "http://127.0.0.1:1234/v1" `
  --api-key-env "LM_STUDIO_API_KEY"
```

```powershell
python scripts\monitor_gemma_pdf_hybrid_progress.py `
  --out-dir D:\fire-data\.tmp_gemma_pdf_hybrid_repair
```

### 4. 计算年度配置比例

```powershell
python scripts\step7_compute_allocation.py `
  --base-dir D:\fire-data `
  --cpi-csv D:\fire-data\CPI指数汇总.csv `
  --start-year 2006 `
  --end-year 2024 `
  --overwrite
```

### 5. 回测

```powershell
python scripts\step8_backtest.py `
  --base-dir D:\fire-data `
  --start-year 2006 `
  --end-year 2024 `
  --out-dir backtest_output `
  --overwrite
```

## 说明

- `scripts/*.ps1` 保留了原始 Windows 批处理风格，适合在本机快速恢复任务。
- 更推荐直接调用 `python scripts\...`，因为这样更容易把“代码仓库”和“数据工作区”拆开。
- `reference_data/` 里的文件是轻量参考输入，不是完整实验产物。

## 文档入口

- [docs/项目说明.md](docs/项目说明.md)
- [docs/任务描述.txt](docs/任务描述.txt)
- [docs/参考提示词.txt](docs/参考提示词.txt)
- [docs/数据目录约定.md](docs/数据目录约定.md)
- [docs/上传前检查清单.md](docs/上传前检查清单.md)
- [docs/GitHub发布步骤.md](docs/GitHub发布步骤.md)
