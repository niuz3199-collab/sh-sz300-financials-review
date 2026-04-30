# GitHub 发布步骤

以下步骤默认你要上传的目录就是：

`C:\Users\1120230509\Desktop\fire\github_upload_fire_20260430_122545`

## 1. 进入目录

```powershell
cd C:\Users\1120230509\Desktop\fire\github_upload_fire_20260430_122545
```

## 2. 检查当前状态

```powershell
git status
```

你应该看到的是一个干净的小型代码仓库，而不是几十 GB 的数据工作区。

## 3. 首次提交

```powershell
git add .
git commit -m "Initialize public code-only version of fire annual-report pipeline"
```

## 4. 关联远端仓库

如果你已经在 GitHub 上创建了空仓库：

```powershell
git remote add origin <your-github-repo-url>
git push -u origin main
```

例如：

```powershell
git remote add origin https://github.com/<your-name>/<repo-name>.git
git push -u origin main
```

## 5. 建议仓库描述

GitHub 仓库简介可以直接写：

`Code-only public mirror for a China A-share annual-report extraction and CAPE/FCF backtesting pipeline. Raw PDFs, caches, and large outputs are excluded.`

## 6. 如果 push 前还想再确认

可以先看会被提交的文件清单：

```powershell
git diff --cached --name-only
```

如果这里出现这些内容，说明你推错目录了：

- `.cache/`
- `.tmp*/`
- `年报/`
- `2001/` 到 `2025/`
- `*.pdf`
- 批量 `raw_json/`
- 大型实验输出目录
