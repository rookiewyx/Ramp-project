# 审核服务镜像：API 和 worker 用同一个镜像，启动命令不同（见 docker-compose.yml 的 app 部分）
#
# 只依赖一个基础镜像（Docker Hub 的 python），其余都从 Python 包源下载：
# 不写 “# syntax=” 指令、不从 ghcr.io 拷 uv，构建时就不用再去连 auth.docker.io 和 ghcr.io，国内网络更稳。
# 包源可以换成国内镜像：构建参数 PIP_INDEX（docker compose 会读环境变量 PIP_INDEX）。
FROM python:3.11-slim-bookworm

ARG PIP_INDEX=https://pypi.org/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DEFAULT_INDEX=${PIP_INDEX}
WORKDIR /app

RUN pip install --index-url "${PIP_INDEX}" "uv==0.13.6"

# 依赖单独一层：只改代码时这一层走缓存，不用重新下载安装
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r pyproject.toml

# 代码和制度文件。以可编辑方式安装，项目根目录就是 /app，制度文件按 /app/policies 读取
COPY src ./src
COPY policies ./policies
RUN uv pip install --no-deps -e .

# 不用 root 运行。data/objects 先建好并交给 app 用户：
# docker 第一次把命名卷挂到这里时会沿用这个目录的属主，否则 app 用户写不进发票文件
RUN useradd --create-home --uid 10001 app \
    && mkdir -p data/objects reports \
    && chown -R app:app data reports
USER app

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
CMD ["python", "-m", "ramp_agent.service", "api", "--host", "0.0.0.0", "--port", "8080"]
