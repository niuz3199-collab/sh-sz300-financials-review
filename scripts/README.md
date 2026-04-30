# Scripts Overview

这个目录保留了原项目中的主要自动化脚本，按职责大致可以分成几组。

## 1. 原始提取

- `step6_extract_financials_qwen_pdf.py`
  - 直接从 PDF 年报提取关键字段。
- `step6_extract_financials_from_markdown.py`
  - 基于 Markdown 年报文本提取关键字段。
- `step6_extract_financials.py`
  - 较早版本的提取流程。

## 2. 修复与回填

- `repair_gemma_markdown_financials.py`
  - 用本地或兼容 OpenAI 的模型对 Markdown 提取失败样本做修复。
- `repair_gemma_pdf_hybrid_fields.py`
  - 结合 Markdown 定位与 PDF 局部高精度图片继续修复失败字段。
- `merge_gemma_hybrid_backfill.py`
  - 合并修复结果。
- `apply_company_avg_backfill.py`
  - 按公司历史均值规则做可接受缺失的正式回填。
- `import_financial_data_json.py`
  - 把人工或外部补录的 `financial_data.json` 合并回主结果。

## 3. 监控

- `monitor_gemma_markdown_progress.py`
- `monitor_gemma_pdf_hybrid_progress.py`
- `monitor_marker_progress.py`
- `monitor_qwen_progress.py`
- `monitor_qwen_sharded_progress.py`

这些脚本负责根据输出目录中的日志、状态文件和结果文件生成实时进度视图。

## 4. 数据准备

- `materialize_pdf_manifest.py`
- `prepare_manual_hardest_pdf_folder.py`
- `step5_fetch_close_prices.py`
- `refill_close_prices_tx.py`

## 5. 因子与回测

- `step7_compute_allocation.py`
  - 计算沪深300口径 CAPE、FCF Yield 与年度配置比例。
- `step8_backtest.py`
  - 跑策略回测。
- `run_allocation_transform_experiments.py`
  - 测不同映射函数与参数。
- `analyze_factor_return_models.py`
  - 分析因子与未来收益率关系。
- `analyze_factor_drawdown_models.py`
  - 分析因子与未来回撤关系。
- `run_drawdown_driven_allocation_experiments.py`
  - 基于预测回撤驱动配置实验。
- `evaluate_live_period_strategies.py`
  - 单独评估 live 区间。

## 6. Windows 启动脚本

`run_*.ps1` 和 `start_*.ps1` 保留了原本机工作流的快速入口，适合在 Windows + PowerShell 环境下恢复任务。

如果你要长期维护这个 GitHub 版，建议逐步把这些入口统一收敛到更少的 Python CLI 上，再把 PowerShell 只保留为薄封装。
