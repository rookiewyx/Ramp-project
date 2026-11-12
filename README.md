# RAMP 报销审核 Agent

[![CI](https://github.com/rookiewyx/Ramp-project/actions/workflows/ci.yml/badge.svg)](https://github.com/rookiewyx/Ramp-project/actions/workflows/ci.yml)

对标 Ramp Policy Agent 的企业报销审核系统：员工提交报销单和发票后，Agent 完成票据识别、查重、对照公司制度逐条审核，低风险的建议通过，有疑问的转人工，并给出引用制度条款的理由。

## 快速开始（Windows）

在 `D:\RAMP` 打开 PowerShell：

```powershell
# 1. 一键配置环境（uv + Python 3.11 虚拟环境 + 依赖 + .env + 自检）
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1

# 2. 下载公开票据数据集（默认只下评测子集，约 1.5 GB 以内）
.venv\Scripts\python scripts\download_datasets.py

# 3. 生成带标注的合成报销数据（默认 300 条）
.venv\Scripts\python scripts\gen_synthetic.py

# 4. （可选）启动 MySQL / Redis / Jaeger，需要先开 Docker Desktop
docker compose up -d
```

配置脚本把 uv 缓存、uv 下载的 Python、HF 缓存都放在项目目录里，不占 C 盘。

## 数据

| 数据 | 来源 | 用途 |
|---|---|---|
| CORD v2 | `naver-clova-ix/cord-v2`，CC BY 4.0 | 真实拍摄小票 + 字段标注，评测扫描件识别 |
| Invoice Checkmark | `AlroWilde/invoice-checkmark-annotations` | 真实发票（含中文）+ 字段标注，中文票据识别抽样评测 |
| 合成报销数据 | `scripts/gen_synthetic.py` | 按 `policies/expense_policy.md` 生成，带期望结论和触发条款，评测制度审核 |

合成数据说明：
- 每条报销带 `expected.decision`（approve / review / reject）和 `expected.violations`（触发的条款及原因）。
- 发票有文字版 PDF 和拍照 JPG 两种形态，分别对应“直接解析”和“视觉模型识别”两条路径。
- 包含难例：上海住宿 580 元（合规但超过其他城市标准）、超额打车但写明原因、大额会议费附预审批。
- 所有发票带“样例数据 非真实发票”水印，公司、商户、税号均为虚构。

## 模块一：分级票据识别（`src/ramp_agent/extraction/`）

```
文件 ─► 文字版 PDF？─是─► 规则解析 ─► 校验通过？─是─► 采用（零模型调用，毫秒级）
          │否                              │否
          ▼                                ▼
     图片 / 渲染成图片 ─────────────► 视觉模型 ─► 校验 ─► 采用；没通过的字段标为 uncertain，转人工复核
```

| 文件 | 作用 |
|---|---|
| `schema.py` | 识别结果结构：字段、走了哪一级、可疑字段、模型调用次数和 tokens |
| `normalize.py` | 金额、日期、税率、号码的写法统一 |
| `validate.py` | 交叉校验：必填字段、号码/税号格式、明细合计 = 价税合计 |
| `pdf_text.py` | 第一级：PyMuPDF 读文字层 + 规则解析 |
| `vlm.py` | 第二级：OpenAI 兼容接口调用视觉模型（AutoDL vLLM 或云端），JSON 解析失败自动重试一次 |
| `router.py` | 分级路由与降级 |

评测：

```powershell
.venv\Scripts\python scripts\eval_extraction.py --no-vlm     # 只评文字版 PDF，不需要密钥
.venv\Scripts\python scripts\eval_extraction.py              # 配好 VLM_* 后，扫描件一起评
.venv\Scripts\python scripts\eval_extraction.py --force-vlm  # 全部走视觉模型，作为成本对照组
.venv\Scripts\python scripts\eval_extraction.py --dataset cn # 中文票据数据集
.venv\Scripts\python scripts\eval_extraction.py --rescore reports\xxx.jsonl  # 改了校验逻辑后重评旧结果，不调模型
.venv\Scripts\python scripts\check_vlm.py                    # 检查视觉模型配置
.venv\Scripts\python -m pytest                                # 单元测试

# 独立测试集：换个随机种子生成没见过的数据，避免“看着错例调规则”导致结果偏乐观
.venv\Scripts\python scripts\gen_synthetic.py --seed 7 --out data/synthetic_test
.venv\Scripts\python scripts\eval_extraction.py --synthetic-dir data/synthetic_test
```

每轮评测的结果和改动记录在 `docs/eval_log.md`。

核心指标是**静默错误率**：字段读错了、却没被标记为需要复核的比例。读错但被标记的只多一次人工复核，静默错误会直接导致审核误判。

## 模块二：制度审核（`src/ramp_agent/policy/`）

```
报销单 + 识别结果 ─► 规则引擎（R1–R9、金额一致）──► 汇总 ─► 建议通过 / 需人工审核 / 建议驳回（附条款原文）
                         │  需要理解语义的问题                ▲
                         └──► 大模型（每笔最多一次）──────────┘
```

- **代码能算清楚的交给代码**：金额标准、日期、抬头税号、查重、预审批。禁报关键词由代码筛出，再交给大模型确认商品本身是否就是这类物品（“红酒杯”不是酒）。
- **需要理解语义的才交给大模型**，且每笔报销最多调用一次：变相的烟酒娱乐罚款（“53%vol 酱香型”）、打车超标理由是否成立、说明里写的招待人数。提示词直接引用 `policy.yaml` 的条款原文，只附这次要问的条款。
- **员工填写的文字只是数据**：说明和明细里的“已获特批”、要求给出某种结论的话，不影响判断。
- **不确定性传递**：模块一标为可疑的字段，依赖它的条款只能给出“无法确认 → 转人工”，不会据此驳回。
- **保守**：大模型不可用或答不上来，对应条款记为无法确认，转人工，绝不自动通过。

| 文件 | 作用 |
|---|---|
| `policies/policy.yaml` | 结构化制度：阈值、禁报/正常名单、条款原文、违规结论 |
| `config.py` | 加载制度 |
| `models.py` | 报销单 `Claim`、条款结果 `RuleResult`、审核结论 `Review` |
| `reviewer.py` | 规则引擎、语义问题收集、汇总与解释 |
| `semantic.py` | 大模型语义判断（只回答具体问题，结论由代码组装） |
| `registry.py` | 发票登记查重（现在用内存实现，接服务时换成 Redis） |

评测：

```powershell
.venv\Scripts\python scripts\check_llm.py                                         # 检查大模型配置
.venv\Scripts\python scripts\eval_policy.py --no-llm                              # 只用规则
.venv\Scripts\python scripts\eval_policy.py                                       # 规则 + 大模型
.venv\Scripts\python scripts\eval_policy.py --data evalsets/policy_challenge.jsonl # 语义难例集
.venv\Scripts\python scripts\eval_policy.py --data data/synthetic_test/expenses.jsonl --extraction reports\<识别结果>.jsonl  # 端到端
```

`evalsets/policy_challenge.jsonl`（由 `scripts/gen_policy_challenge.py` 生成）是 27 条只有理解语义才能判对的报销，用来衡量大模型的作用。

## 模块三：审核服务（`src/ramp_agent/service/`）

```
员工 ─► POST /v1/expenses ─► 校验 + 存发票 + 落库 ─► 202 受理 ─► 队列（Redis）
                                                                 │
          财务 ◄─ GET /v1/expenses?decision=review&pending=true   ▼
          财务 ─► POST /v1/expenses/{id}/override        worker：领取 → 识别 → 审核 → 落库 + 审计
```

| 能力 | 做法 |
|---|---|
| 异步审核 | 提交即返回受理编号，审核在 Celery worker 里进行；识别要调视觉模型，几秒的事不能卡住接口 |
| 幂等提交 | `Idempotency-Key`：客户端超时重试只受理一次；同一个键换了内容返回 409 |
| 不重复审核 | 消息可能被投递两次（worker 崩溃重投），领取用条件更新，同一时间只有一个 worker 能拿到；已完成的跳过 |
| 崩溃恢复 | `acks_late`：做完才确认；处理中超过 10 分钟没完成，可被别的 worker 接手；`requeue` 补投投递失败的单子 |
| 重试与降级 | 模型临时故障先按 5s、10s、20s 重试，最后一次才降级成“转人工”；其他故障重试用尽后标记 failed |
| 并发查重 | 发票号做数据库主键，靠唯一约束判重，多个 worker 同时审核同一张发票也只有一笔能登记 |
| 文件安全 | 按文件头判断类型（不信扩展名），限制大小，按内容哈希存储（同一文件只存一份，不暴露原始文件名） |
| 个人信息脱敏 | 发给大模型前，把说明和明细里的手机号、身份证号、银行卡号、邮箱、员工姓名换成占位符 |
| 审计与追溯 | 每笔报销的提交、审核、复核都记审计日志；审核记录保存当时的代码、提示词、制度版本 |
| 人机协同 | 财务复核给出最终结论，推翻系统结论必须写理由；`/v1/stats` 统计自动通过率和人机一致率 |

**快速体验**（不需要 Docker，审核在接口进程里同步执行）：

```powershell
.venv\Scripts\python -m ramp_agent.service api --eager
# 另开一个窗口
.venv\Scripts\python scripts\demo_service.py -n 20
```

打开 http://127.0.0.1:8080 就是**财务审核工作台**：
- 左边是待办，按“待复核 / 系统驳回 / 系统通过 / 已复核”分栏；
- 中间是报销单和发票的逐项对照，不一致的标红、识别存疑的标黄，下面是发票原件；
- 右边是系统结论、违反和无法确认的条款（附制度原文）、审核过程和操作记录；
- 底部给出最终结论，推翻系统结论时必须写理由。

键盘就能完成复核：`J` / `K` 切换上下一笔，`A` 通过，`R` 驳回，`Ctrl+Enter` 提交。

接口文档在 http://127.0.0.1:8080/docs。

**完整模式**（Redis 队列 + worker，可以开多个 worker）：

```powershell
docker compose up -d redis            # 或者 docker compose up -d，同时启动 MySQL、Jaeger
.venv\Scripts\python -m ramp_agent.service api
.venv\Scripts\python -m ramp_agent.service worker     # 另开窗口；可以开多个
.venv\Scripts\python -m ramp_agent.service requeue    # 补投卡住的单子，可以放进定时任务
```

换成 MySQL：在 `.env` 里设置 `DATABASE_URL=<MYSQL_URL 的值>`。发票文件可以放在任何 S3 兼容存储里：设置 `STORAGE_BACKEND=s3` 和 `S3_*`。

**整套跑在容器里**（MySQL + Redis + Jaeger + 迁移 + API + 两组 worker）：

```powershell
docker compose --profile app up -d --build
```

- 国内网络下载 Python 包慢的话，先设置包源再构建：`$env:PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"`。
- 启动顺序由健康检查保证：MySQL 就绪后先跑一次 `migrate` 容器升级表结构，成功退出后再启动 API 和 worker。
- API 和 worker 设置了 `AUTO_MIGRATE=false`，只检查表结构版本，不一致就拒绝启动，避免多个进程同时改表。
- 模型密钥从 `.env` 读取。如果 `.env` 里的模型地址是 `127.0.0.1`（比如 AutoDL 的端口转发），容器里访问不到，要改成 `host.docker.internal`。
- 打开 http://127.0.0.1:8080 是工作台，http://127.0.0.1:16686 是 Jaeger 链路追踪。
- 发票文件默认存在 API 和 worker 共享的卷里。想用 S3 兼容存储：`$env:COMPOSE_STORAGE="s3"; docker compose --profile app --profile s3 up -d --build`，会多启动一个 RustFS。MinIO 官方镜像已于 2026 年 9 月从 Docker Hub 删除，所以不再使用。

**数据库迁移**（Alembic，`service/migrations/`）：表结构的每次变化都对应一个迁移脚本。

```powershell
.venv\Scripts\python -m ramp_agent.service initdb              # 升级到最新（新库就是建表）
.venv\Scripts\python -m ramp_agent.service migrate             # 查看当前版本和最新版本
.venv\Scripts\python -m ramp_agent.service migrate -m "说明"   # 改了 db.py 的模型后，自动生成迁移脚本
.venv\Scripts\python -m ramp_agent.service migrate --sql       # 只导出升级 SQL，给 DBA 审核
```

- 测试会检查“迁移脚本建出来的表”和“模型”完全一致：改了模型却忘了写迁移，测试会失败。
- 引入迁移之前用 `create_all` 建的库，会被识别并直接标记为第一版，数据不动。

**持续集成**（`.github/workflows/ci.yml`）：每次推送都会跑三项检查。

- ruff 代码检查和全部测试；
- 服务与迁移测试在 MySQL 8.4 和 PostgreSQL 16 上各跑一遍；
- 构建 Docker 镜像，启动后冒烟测试。

在 MySQL 上跑迁移测试时发现过一个问题：回滚脚本先删索引再删表，MySQL 不允许删除外键正在用的索引（错误 1553）。只在 SQLite 上测是发现不了的。

生产上分开起两组 worker：拍照件走 `vlm` 队列（`--queues vlm`，并发按视觉模型配额设置），文字版 PDF 和无附件的走 `default` 队列（`--queues default`）。视觉模型排队时，秒级就能审完的单子不受影响。

**压测与链路追踪**（详见 [`docs/perf_log.md`](docs/perf_log.md)）

在 2 个 vCPU 的容器里，用模拟模型（延迟取实测值）压到 **20 笔/秒（7.2 万笔/小时）**：

- 端到端 P50 0.19 秒，P95 5.8 秒；
- 文字版 PDF P50 80 毫秒，拍照件 P50 4.3 秒；
- 零失败。

链路追踪显示，worker 91% 的时间花在视觉模型上：瓶颈在模型配额，不在服务本身。

压测中发现并修复了 6 个问题：

- 任务限速把吞吐卡在 1 笔/秒；
- 长连接导致链路串号；
- SQLAlchemy 2.1 下 SQL 埋点静默失效；
- 模型配额不足时的重试风暴；
- 拍照件拖慢文字版 PDF；
- `async def` 里的同步调用堵塞事件循环。

| 文件 | 作用 |
|---|---|
| `src/ramp_agent/telemetry.py` | 链路追踪：API → 队列 → worker → 模型 → SQL 一条链路；模型调用按 GenAI 语义约定记录，不记录提示词内容 |
| `src/ramp_agent/limits.py` | 模型调用并发上限：在调用方排队，不把请求打过去等着被 429 |
| `scripts/mock_models.py` | 模拟模型服务：可设延迟、长尾、故障率、并发配额 |
| `tests/load/locustfile.py` | 压测脚本：员工提交并等结论（上报端到端耗时），财务刷待办和复核 |
| `scripts/trace_report.py` | 链路汇总：排队、处理、各环节、各类发票的 P50 / P95 |

| 文件 | 作用 |
|---|---|
| `api.py` | FastAPI 接口：提交、查询、发票原件、财务复核、重新审核、审计、指标、健康检查 |
| `static/workbench.html` | 财务审核工作台（单文件，无前端构建） |
| `pipeline.py` | 审核流水线：领取 → 取文件 → 识别 → 审核 → 落库，可重入 |
| `tasks.py` | Celery 任务：两个队列（拍照件单独排队）、重试、补投 |
| `db.py` | 业务库：报销单、审核记录、财务复核、审计日志、发票登记 |
| `migrate.py`、`migrations/` | 数据库迁移（Alembic）：升级、版本检查、生成迁移脚本、导出 SQL |
| `storage.py` | 发票文件存储：本地目录，或任何 S3 兼容存储（RustFS、云厂商对象存储） |
| `registry.py` | 发票查重登记：数据库唯一约束（默认）或 Redis SET NX |
| `../pii.py` | 个人信息脱敏 |

## 评测平台（`src/ramp_agent/evalhub/`）

把散落的评测脚本升级成可追溯的评测体系：

| 能力 | 做法 |
|---|---|
| 版本化评测集 | 每个评测集有 v1、v2……；冻结后不能改，新增用例用 `fork` 派生新版本，保证同一版本上的分数可比 |
| 盲标标注 | 网页只显示报销单和发票，不显示系统结论或其他人的标注；支持多人标注、`agreement` 算 Cohen's kappa |
| 实验追踪 | 每次运行入库，记录代码、提示词、制度文件的哈希，以及模型名和参数 |
| 对比 | 逐条列出“修好的 / 改坏的”，McNemar 检验判断差异是否显著，准确率附 95% 置信区间 |
| 回归门禁 | 误通过、误驳回、静默错误比基线多，或有用例从对变错，就不通过（退出码 1） |

存储默认是 `data/evalhub.db`（SQLite），服务化时可以把 `EVALHUB_URL` 改成 MySQL。

```powershell
.venv\Scripts\python -m ramp_agent.evalhub init                     # 导入项目自带的评测集
.venv\Scripts\python -m ramp_agent.evalhub serve                    # 标注页面 http://127.0.0.1:8000
.venv\Scripts\python -m ramp_agent.evalhub agreement policy-hard    # 你和出题人的一致性
.venv\Scripts\python -m ramp_agent.evalhub freeze policy-hard
.venv\Scripts\python -m ramp_agent.evalhub run policy --dataset policy-hard --note "基线"
.venv\Scripts\python -m ramp_agent.evalhub compare 1 2
.venv\Scripts\python -m ramp_agent.evalhub gate 1 2
.venv\Scripts\python -m ramp_agent.evalhub gate 1 2 --waive H004 --reason "标注人也拿不准的灰色用例"   # 豁免必须写理由
.venv\Scripts\python -m ramp_agent.evalhub export policy-hard -o evalsets/policy_hard_v1.labeled.jsonl
.venv\Scripts\python -m ramp_agent.evalhub list
```

数据库不入 git，标注成果用 `export` 导出到 `evalsets/` 再提交。

| 评测集 | 用途 |
|---|---|
| `policy-dev` / `policy-test` | 合成数据，规则类条款的回归测试 |
| `policy-challenge` | 27 条语义难例。出题人和提示词同源，偏易，只作冒烟测试 |
| `policy-hard` | 102 条难题：理由灰区、关键词误伤、变相禁报、人数提取、提示词注入、边界值、抬头变体。人工盲标，与出题人 kappa 0.80。基线 78.4% → 修复后 91.2%，误通过 0。修复是看着这批题做的，分数偏乐观（见 `docs/eval_log.md`） |
| `policy-hard2` | 103 条第二批难题（留出集），句子、注入写法、边界维度都和第一批不同。盲标（kappa 0.77）、冻结后只跑一次：**规则 + 大模型 78.6%（95% 区间 69.8–85.5%），误通过 1，误驳回 6；只用规则 43.7%**。这是对外报告的数字 |
| `extract-dev` / `extract-test` | 识别评测 |

## 数据闭环：线上数据回流评测集（`evalhub/harvest.py`）

评测集是人出的题，总有没想到的情况；线上每天都在产生真实的考题。`evalhub harvest` 从业务库里收三类单子：

| 类别 | 是什么 | 作用 |
|---|---|---|
| 系统定论被推翻 | 系统建议通过或驳回，财务复核时改了 | 系统确定答错的地方 |
| 转人工后财务定论 | 系统拿不准，财务给出了结论 | 系统在哪些问题上没把握，人最后怎么判 |
| 自动结论抽检 | 系统自动通过或驳回、没人复核过的，随机抽 | 只看财务推翻的单子有盲区：自动通过的单子错了没人发现，抽检才能估计这部分的错误率 |

```powershell
.venv\Scripts\python -m ramp_agent.evalhub harvest                # 收集，追加到评测集 policy-prod
.venv\Scripts\python -m ramp_agent.evalhub harvest --dry-run      # 只统计不写入
.venv\Scripts\python -m ramp_agent.evalhub serve                  # 第二个人盲标（抽检的单子必须标）
.venv\Scripts\python -m ramp_agent.evalhub agreement policy-prod  # 财务和第二个人不一致的，讨论后定答案
.venv\Scripts\python -m ramp_agent.evalhub freeze policy-prod
.venv\Scripts\python -m ramp_agent.evalhub run policy --dataset policy-prod
```

- **财务的复核结论**作为标注人 `finance` 导入。财务看过系统结论，可能被带偏，所以再由第二个人盲标；两人意见不一致的记为有争议，不参与打分。
- **写入前统一脱敏**：去掉员工姓名，工号换成哈希，说明和明细里的手机号等替换掉。用例编号由线上编号哈希得到，同一笔报销只收一次。
- **版本**：评测集已冻结时，新数据进派生出的新版本，旧版本上的分数不受影响。
- **不收**：发票识别失败的（字段本身不可靠，属于识别问题），以及触发了重复报销的（取决于当时的发票登记状态，离线评测复现不了）。报告里会列出跳过的条数。
- **报告**：除了新增条数，还会列出“系统 → 财务”的结论变化，以及被推翻的单子涉及哪些条款，直接指出该先改哪一块。
- **看分数时注意**：“转人工后财务定论”这一组，当前系统的准确率按定义是 0（系统当时答的就是“转人工”）。它衡量的是系统还有多少提升空间：以后的版本能把多少原本要人看的单子判对。

## 目录

```
policies/expense_policy.md   报销制度（条款 R1–R9，是审核和打标的依据）
scripts/setup.ps1            Windows 一键环境配置
scripts/check_env.py         环境自检
scripts/download_datasets.py 公开数据集下载与格式转换
scripts/gen_synthetic.py     合成报销数据生成
src/ramp_agent/config.py     配置（读取 .env）
src/ramp_agent/extraction/   模块一：分级票据识别
scripts/eval_extraction.py   识别评测
tests/                       单元测试
docs/eval_log.md             评测记录（每轮改动与指标变化）
src/ramp_agent/policy/       模块二：制度审核
policies/policy.yaml         结构化制度
evalsets/                    语义难例集（入库）
scripts/eval_policy.py       审核评测（快速版，不入库）
src/ramp_agent/evalhub/      评测平台（harvest.py：线上数据回流）
scripts/gen_policy_hard.py   难题集生成（第一批）
scripts/gen_policy_hard_v2.py 难题集生成（第二批，留出集）
src/ramp_agent/service/      模块三：审核服务
scripts/demo_service.py      审核服务演示
scripts/mock_models.py       压测用模拟模型
scripts/trace_report.py      链路追踪汇总
tests/load/locustfile.py     压测脚本
docs/perf_log.md             压测与链路追踪记录
docker-compose.yml           依赖（MySQL / Redis / Jaeger）+ app 部分（迁移、API、两组 worker）+ 可选的 RustFS
Dockerfile                   服务镜像（API 和 worker 共用）
.github/workflows/ci.yml     持续集成
src/ramp_agent/service/migrations/  数据库迁移脚本
data/                        数据（不入库）
```

## 技术栈

FastAPI · Celery · MySQL · Redis · S3 兼容对象存储 · SQLAlchemy · Alembic · locust · OpenAI 兼容接口（DeepSeek / Qwen / AutoDL vLLM）· PyMuPDF · Qwen3-VL · OpenTelemetry · Jaeger · Docker · Docker Compose · GitHub Actions · pytest
