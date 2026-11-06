# data/

- `raw/`        下载的原始数据（download_datasets.py）
- `processed/`  转换后的“图片 + jsonl”（download_datasets.py）
- `synthetic/`  合成报销数据（gen_synthetic.py），已预先生成一份 300 条，可直接使用或重新生成
- `MANIFEST.json` 本地数据清单

以上目录都在 .gitignore 里，不会提交到 GitHub。
