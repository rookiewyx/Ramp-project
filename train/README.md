# 微调小模型识别交通票据

用 SCID 训练集微调 Qwen3.5-4B（可选 2B 对照），在 SCID testB 上和现在的大模型接口（qwen3-vl-plus + v3）比较准确率、静默错误、每张票的成本和耗时。

微调后的模型用 vLLM 起成 OpenAI 兼容接口，项目只改 `VLM_BASE_URL` 就能切换，评测、校验、转人工的代码一行不改，所以对比是同一套流程只换模型。

```
本机（Windows）                                   租用的 GPU 机器（Linux）
  build_scid_sft.py → SCID/sft.zip  ── 上传 ──►  setup.sh（装环境、下模型）
                                                   train.sh（LoRA 微调）
                                                   serve.sh（合并权重、vLLM 起服务，端口 8000）
  evalhub run extraction  ◄── SSH 隧道 8001 → 8000 ──┘
```

## 一、本机准备（不用 GPU）

### 1. 生成训练数据

```powershell
cd D:\RAMP
.venv\Scripts\python scripts\build_scid_sft.py
```

输出 `SCID/sft/`（train.jsonl、val.jsonl、images/、stats.json）和打包好的 `SCID/sft.zip`。先看一眼 stats.json：各票种数量、标为 null 的字段数、丢掉了多少张、为什么丢。

### 2. 准备 testB（最终对比用，之前谁都没碰过）

testB 的图片是 rar。用 7-Zip 右键“解压到当前文件夹”，解压到 `SCID\testB\` 下面就行（脚本会自动找解压出来的图片文件夹，不用再打包）。注意：只改扩展名不行，rar 改名成 .zip 还是 rar。然后：

```powershell
.venv\Scripts\python scripts\build_scid_evalset.py --split testB
.venv\Scripts\python -m ramp_agent.evalhub import SCID/eval/testB_600/samples.jsonl --dataset scid-testB --task samples --freeze
```

### 3. 先跑接口的成绩（对照组，可以在租机器之前跑）

```powershell
$env:TICKET_PROMPT="v3"
.venv\Scripts\python -m ramp_agent.evalhub run extraction --dataset scid-testB --note "testB 接口 qwen3-vl-plus v3"
```

## 二、租机器

- 显卡：4B 用 LoRA，24 GB 显存够（4090，`BS=2 GRAD_ACC=8`）；RTX PRO 6000（96 GB）可以开 `BS=16 GRAD_ACC=1`，更快。
- 镜像：PyTorch 2.x、CUDA 12.x、Python 3.12。
- 硬盘：数据盘留 60 GB 以上（模型 + 合并后的权重 + 检查点）。
- 预计：装环境 30 分钟（有两个包要现场编译），训练 1–3 小时，评测 30 分钟。**用完记得关机**。

## 三、在 GPU 机器上

把 `train/` 目录和 `SCID/sft.zip` 上传到机器的家目录（AutoDL 可以用网页上传或 scp）。

```bash
cd ~ && unzip -q sft.zip          # 解压出 ~/sft
bash train/setup.sh               # 装环境、下载 Qwen3.5-4B
bash train/train.sh               # 训练；日志在 ~/train_Qwen3.5-4B.log
```

训练时看日志里的 `eval_loss`，应该一路下降；如果后半段回升，说明过拟合了，用回升前的检查点。

先起**基础模型**测一次（不微调的小模型是什么水平），再起微调后的：

```bash
bash train/serve.sh base                                            # 基础模型
bash train/serve.sh ~/output/Qwen3.5-4B/v0-xxxx/checkpoint-xxxx     # 微调后（换成实际目录）
```

## 四、在本机评测

在本机另开一个 PowerShell，建 SSH 隧道（端口和地址换成租用平台给的）：

```powershell
ssh -CNg -L 8001:127.0.0.1:8000 -p <端口> root@<地址>
```

再开一个 PowerShell 跑评测。这几个环境变量只对当前窗口生效，不会改 `.env`：

```powershell
cd D:\RAMP
$env:VLM_BASE_URL="http://127.0.0.1:8001/v1"
$env:VLM_API_KEY="EMPTY"
$env:VLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'

# 基础模型（serve.sh base 起着的时候）：用 v2 的长提示词
$env:VLM_MODEL="base"; $env:TICKET_PROMPT="v2"
.venv\Scripts\python -m ramp_agent.evalhub run extraction --dataset scid-testB --workers 16 --note "testB Qwen3.5-4B 基础模型"

# 微调后（serve.sh <检查点> 起着的时候）：用训练时的短提示词
$env:VLM_MODEL="ramp-ticket"; $env:TICKET_PROMPT="ft"
.venv\Scripts\python -m ramp_agent.evalhub run extraction --dataset scid-testB --workers 16 --note "testB Qwen3.5-4B 微调"
```

`--workers 16` 是为了测吞吐：自部署的每张票成本 = GPU 每小时价格 × 这次运行的耗时 ÷ 600。记下租的显卡每小时多少钱。

## 五、规矩

- testB 每种做法只跑一次，跑完不改；
- 调参只看训练时的 `eval_loss`（val.jsonl 来自 train），不拿 testB 调；
- SCID 是非商业许可，`SCID/` 不进仓库，训练出的权重也不要公开发布。
