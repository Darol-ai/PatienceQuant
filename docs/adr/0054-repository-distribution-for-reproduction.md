# ADR-0054 仓库分发：代码进 git，数据走 Release 附件，一键复现并对照快照

状态：已采纳（2026-09-24）

## 背景

项目要上传到 GitHub（Darol-ai/PatienceQuant，公开），供老师复现和检查。数据文件（本地行情库、旧模型）此前走 Git LFS，这次待推送约 681MB，加上每日指标约 1.2GB。GitHub 免费 LFS 每月只有 1GB 下载流量，一次完整克隆就用掉大半，超额后别人克隆到的只是占位符，系统无法运行。数据库里存着大模型 key，不能上传；设计文档（ADR、术语表）原来在仓库外。

## 决定

1. **数据包**：`backend/data` 下的数据文件和每日指标打成一个 tar（约 1.2GB），作为 GitHub Release 附件发布（没有流量限制）。`scripts/fetch_data.py`（`make data`）只用标准库下载、续传、校验 sha256 并解压。仓库不再用 LFS 跟踪数据；推送时跳过 LFS 上传，历史提交里的数据文件在 GitHub 上是缺失的占位符，最新代码不依赖它们。老师不需要 tushare token 也能得到含每日指标的完整数据。
2. **一键复现**：`backend/scripts/reproduce.py`（`make reproduce`）初始化数据库、训练内置模型、计算全部成绩卡，再与仓库里的 `docs/成绩卡快照.json` 按 kind 逐项对比。依赖版本锁定在 `backend/requirements-lock.txt`。
3. **设计文档进仓库**：ADR 与术语表复制到仓库 `docs/adr/`、`docs/CONTEXT.md`，上传前扫描过敏感信息。
4. **训练依赖默认安装**：`make setup` 装 `training` 组（Linux 上是 GPU 版 torch）；Python 要求提高到 3.10 以上（xgboost 3.4）。补上漏声明的 pyarrow、scipy、scikit-learn。
5. **对外说明以展示系统为主**：README、使用说明、演示轨迹、PPT 照实列出成绩卡数字，但措辞中性，不写自我否定式的评价；分析过程在 ADR 里如实保留。
6. **仓库保持公开**。不做推送前的全新环境验证（用户决定）。

## 后果

- 老师的步骤：克隆 → `make setup` → `make data` → `make reproduce`（可选）→ `make backend` / `make frontend`。
- 数据更新时需要重新打包、发新 Release，并更新 `fetch_data.py` 里的标签和 sha256。
- XGBoost 在 GPU 与 CPU 上的结果不逐位相同，LSTM 在不同显卡上可能有细微差别；LightGBM 与因子策略应当完全一致。
