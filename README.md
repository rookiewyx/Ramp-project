# RAMP 报销审核 Agent

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
src/ramp_agent/evalhub/      评测平台
scripts/gen_policy_hard.py   难题集生成（第一批）
scripts/gen_policy_hard_v2.py 难题集生成（第二批，留出集）
docker-compose.yml           MySQL / Redis / Jaeger
data/                        数据（不入库）
```

## 技术栈

FastAPI · MySQL · Redis · SQLAlchemy · OpenAI 兼容接口（DeepSeek / Qwen / AutoDL vLLM）· PyMuPDF · Qwen3-VL · OpenTelemetry · Jaeger · Docker Compose · pytest · locust
