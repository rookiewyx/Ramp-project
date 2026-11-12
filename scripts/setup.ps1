<#
  RAMP 项目一键环境配置（Windows PowerShell 5.1 / PowerShell 7 均可）

  用法（在 D:\RAMP 下打开 PowerShell）：
    powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
  可选参数：
    -StartServices   顺带用 Docker 启动 MySQL / Redis / Jaeger
    -SkipDocker      不检查 Docker
    -Mirror <url>    PyPI 镜像，默认清华源

  为了不占 C 盘：uv 缓存和 uv 下载的 Python 都放在项目目录内。
#>
param(
    [switch]$StartServices,
    [switch]$SkipDocker,
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
)

# 原生命令（uv/docker）写 stderr 不代表失败，统一靠 $LASTEXITCODE 判断
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    [!]  $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "    [X]  $msg" -ForegroundColor Red; exit 1 }

# ---------- 1. 磁盘空间 ----------
Step "检查磁盘空间"
$drive = (Get-Item $Root).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
Write-Host "    $($drive.Name): 盘剩余 $freeGB GB"
if ($freeGB -lt 3) { Fail "剩余空间不足 3 GB，环境+评测数据集至少需要约 3 GB，请先清理磁盘。" }
elseif ($freeGB -lt 8) { Warn "剩余空间偏少：建议数据集只下评测子集（下载脚本默认就是）。" }
else { Ok "空间充足" }

# 让 uv 的缓存和 Python 都落在项目所在盘，不占 C 盘
$env:UV_CACHE_DIR          = Join-Path $Root ".uv_cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root ".uv_python"
$env:UV_INDEX_URL          = $Mirror
$env:UV_LINK_MODE          = "copy"

# ---------- 2. 安装 / 定位 uv ----------
Step "准备 uv（Python 包与环境管理器）"
$script:UseUvModule = $false
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Ok "已找到 uv：$((uv --version) 2>&1)"
} else {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) {
        Write-Host "    用现有 Python 通过镜像安装 uv ..."
        & python -m pip install -U uv -i $Mirror --quiet
        if ($LASTEXITCODE -eq 0) { $script:UseUvModule = $true; Ok "uv 已通过 pip 安装" }
    }
    if (-not $script:UseUvModule -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "    改用官方安装脚本安装 uv ..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "uv 安装失败，请手动安装：pip install uv" }
        Ok "uv 已安装"
    }
}
function Invoke-Uv {
    if ($script:UseUvModule) { & python -m uv @args } else { & uv @args }
    if ($LASTEXITCODE -ne 0) { Fail "uv $($args -join ' ') 执行失败" }
}

# ---------- 3. 虚拟环境（Python 3.11） ----------
Step "创建虚拟环境 .venv（Python 3.11）"
if (Test-Path ".venv\Scripts\python.exe") {
    Ok ".venv 已存在，跳过创建"
} else {
    # 优先复用本机已装的 3.11；没有的话 uv 会从 GitHub 下载到 .uv_python
    if ($script:UseUvModule) { & python -m uv venv .venv --python 3.11 } else { & uv venv .venv --python 3.11 }
    if ($LASTEXITCODE -ne 0) {
        Fail ("创建 3.11 环境失败（多半是从 GitHub 下载 Python 超时）。可以先用 conda 装好 3.11：`n" +
              "      conda create -n py311 python=3.11 -y`n" +
              "    然后把那个 python.exe 的路径传给 uv，例如：`n" +
              "      uv venv .venv --python D:\anaconda3\envs\py311\python.exe`n" +
              "    再重新运行本脚本。")
    }
    Ok ".venv 创建完成"
}
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"

# ---------- 4. 安装依赖 ----------
Step "安装项目依赖（核心 + 数据 + 开发工具），镜像：$Mirror"
Invoke-Uv pip install --python $VenvPy -e ".[data,dev]"
Ok "依赖安装完成"

# ---------- 5. .env ----------
Step "生成 .env"
if (Test-Path ".env") { Ok ".env 已存在，未覆盖" }
else { Copy-Item ".env.example" ".env"; Ok "已从 .env.example 复制，记得填写 LLM_API_KEY" }

# ---------- 6. Docker（MySQL / Redis / Jaeger） ----------
if (-not $SkipDocker) {
    Step "检查 Docker"
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $docker) {
        Warn "未找到 docker。存储服务需要 Docker Desktop；现阶段可以先跳过，后面开发服务层时再装。"
    } else {
        $null = docker info 2>&1
        if ($LASTEXITCODE -ne 0) {
            Warn "Docker 已安装但未运行，请先打开 Docker Desktop。"
        } elseif ($StartServices) {
            Write-Host "    启动 MySQL / Redis / Jaeger（首次需要拉镜像）..."
            docker compose up -d
            if ($LASTEXITCODE -eq 0) { Ok "服务已启动；Jaeger 界面：http://localhost:16686" }
            else { Warn "启动失败。若是拉镜像超时，请在 Docker Desktop 里配置镜像加速后重试。" }
        } else {
            Ok "Docker 可用。需要时运行：docker compose up -d"
        }
    }
}

# ---------- 7. 自检 ----------
Step "环境自检"
& $VenvPy scripts\check_env.py
Write-Host "`n全部完成。下一步：" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\python scripts\download_datasets.py   # 下载公开票据数据集"
Write-Host "  .venv\Scripts\python scripts\gen_synthetic.py       # 生成带标注的合成报销数据"
