"""下载公开票据数据集，并转换成“图片 + jsonl 标注”的统一格式。

数据集（均通过 hf-mirror.com 下载，国内可直连）：
  1. CORD v2（naver-clova-ix/cord-v2，CC BY 4.0）
     真实拍摄的小票，带字段级标注。用于评测票据识别（扫描件 → 结构化字段）。
     默认只下 validation + test（各 100 张，体积小）；--cord full 额外下载 800 张训练集。
  2. Invoice Checkmark Annotations（AlroWilde/invoice-checkmark-annotations）
     约 600 张真实发票图片（含约 200 张中文），带字段标注。用于中文票据识别的抽样评测。

用法：
  .venv\\Scripts\\python scripts\\download_datasets.py               # 默认
  .venv\\Scripts\\python scripts\\download_datasets.py --cord full   # CORD 全量
  .venv\\Scripts\\python scripts\\download_datasets.py --skip-invoices
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from pathlib import Path

# 必须在导入 huggingface_hub 之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))  # 缓存放项目盘，不占 C 盘

CORD_REPO = "naver-clova-ix/cord-v2"
INVOICE_REPO = "AlroWilde/invoice-checkmark-annotations"


def check_space(min_gb: float) -> None:
    free = shutil.disk_usage(ROOT).free / 1024**3
    print(f"[disk] 剩余 {free:.1f} GB，本次预计需要约 {min_gb:.1f} GB")
    if free < min_gb:
        sys.exit("[disk] 空间不足，已停止。可以先清理磁盘，或去掉部分数据集再试。")


def download(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    print(f"[download] {repo_id} -> {local_dir}（镜像 {os.environ['HF_ENDPOINT']}）")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        max_workers=4,
    )
    return Path(path)


def split_name(parquet: Path) -> str:
    # 形如 train-00000-of-00004-xxxx.parquet / validation-00000-of-00001.parquet
    return parquet.name.split("-")[0]


def image_bytes(cell) -> bytes | None:
    """parquet 里的 Image 特征一般是 {'bytes': ..., 'path': ...} 结构。"""
    if cell is None:
        return None
    if isinstance(cell, dict):
        return cell.get("bytes")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def convert_cord(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    """CORD parquet -> out_dir/{split}/images/*.png + out_dir/{split}.jsonl"""
    import pyarrow.parquet as pq
    from PIL import Image

    counts: dict[str, int] = {}
    parquets = sorted(raw_dir.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"{raw_dir} 下没有 parquet 文件，下载可能不完整")

    writers: dict[str, io.TextIOWrapper] = {}
    try:
        for pqf in parquets:
            split = split_name(pqf)
            img_dir = out_dir / split / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            if split not in writers:
                writers[split] = open(out_dir / f"{split}.jsonl", "w", encoding="utf-8")  # noqa: SIM115  在 finally 里统一关闭
                counts[split] = 0
            table = pq.read_table(pqf)
            for row in table.to_pylist():
                idx = counts[split]
                data = image_bytes(row.get("image"))
                if data is None:
                    continue
                img_path = img_dir / f"{split}_{idx:04d}.png"
                Image.open(io.BytesIO(data)).convert("RGB").save(img_path)
                gt = json.loads(row.get("ground_truth") or "{}")
                record = {
                    "id": f"cord_{split}_{idx:04d}",
                    "image": os.path.relpath(img_path, ROOT).replace("\\", "/"),
                    "fields": gt.get("gt_parse", {}),  # menu / sub_total / total 等结构化字段
                    "meta": gt.get("meta", {}),
                }
                writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
            print(f"[convert] {pqf.name}: {split} 累计 {counts[split]} 条")
    finally:
        for w in writers.values():
            w.close()
    return counts


def summarize_dir(path: Path) -> dict[str, int]:
    """按扩展名统计文件数，便于确认下载结果。"""
    stats: dict[str, int] = {}
    for f in path.rglob("*"):
        if f.is_file() and ".cache" not in f.parts:
            stats[f.suffix.lower() or "<none>"] = stats.get(f.suffix.lower() or "<none>", 0) + 1
    return dict(sorted(stats.items(), key=lambda kv: -kv[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description="下载公开票据数据集")
    ap.add_argument("--cord", choices=["eval", "full", "skip"], default="eval",
                    help="eval=只下验证+测试集（默认）；full=含训练集；skip=不下载")
    ap.add_argument("--skip-invoices", action="store_true", help="不下载多语言发票数据集")
    args = ap.parse_args()

    need = {"eval": 0.6, "full": 2.5, "skip": 0.0}[args.cord] + (0 if args.skip_invoices else 1.0)
    check_space(need + 0.5)

    manifest: dict[str, object] = {"hf_endpoint": os.environ["HF_ENDPOINT"]}

    if args.cord != "skip":
        patterns = ["README.md", "data/validation-*", "data/test-*"]
        if args.cord == "full":
            patterns.append("data/train-*")
        cord_raw = download(CORD_REPO, RAW / "cord-v2", patterns)
        counts = convert_cord(cord_raw, PROCESSED / "cord-v2")
        manifest["cord-v2"] = {"license": "CC BY 4.0", "splits": counts,
                               "path": "data/processed/cord-v2"}

    if not args.skip_invoices:
        inv_raw = download(INVOICE_REPO, RAW / "invoice-checkmark")
        manifest["invoice-checkmark"] = {"files": summarize_dir(inv_raw),
                                         "path": "data/raw/invoice-checkmark"}

    (ROOT / "data").mkdir(exist_ok=True)
    out = ROOT / "data" / "MANIFEST.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 数据清单写入 {out}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
