"""分级识别路由：能用便宜的路径就不用贵的，便宜路径的结果靠校验兜底。

    文件 ──► 是 PDF 且有文字层？──是──► 规则解析 ──► 校验通过？──是──► 采用（零模型调用）
                    │否                                    │否
                    ▼                                       ▼
              图片 / 渲染成图片 ──────────────────────► 视觉模型 ──► 校验 ──► 采用；没通过的字段标为 uncertain
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from ..config import get_settings
from .pdf_text import has_text_layer, parse_einvoice_text, read_pdf_text, render_first_page
from .schema import ExtractionResult, InvoiceFields, Tier
from .validate import validate
from .vlm import VLMBackend

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InvoiceExtractor:
    def __init__(self, vlm: VLMBackend | None = None, *, strict_cn: bool = True,
                 force_vlm: bool = False, known_tax_ids: tuple[str, ...] | None = None):
        """
        vlm:           视觉模型后端；为 None 时扫描件直接标记失败（转人工）
        strict_cn:     是否按中国发票格式校验号码和税号（评测外文票据时关掉）
        force_vlm:     所有文件都走视觉模型，用来和分级方案做成本/准确率对比
        known_tax_ids: 公司自己的税号，用于发现易混字符误读；默认取配置里的公司税号
        """
        self.vlm = vlm
        self.strict_cn = strict_cn
        self.force_vlm = force_vlm
        if known_tax_ids is None:
            known_tax_ids = (get_settings().company_tax_id,) if strict_cn else ()
        self.known_tax_ids = known_tax_ids

    def revalidate(self, fields: InvoiceFields) -> tuple[list[str], list[str]]:
        """对识别结果做校验，返回 (未通过的校验项, 可疑字段)。"""
        return validate(fields, strict_cn=self.strict_cn, known_tax_ids=self.known_tax_ids)

    def extract(self, path: str | Path | None) -> ExtractionResult:
        t0 = time.perf_counter()
        result = self._extract(Path(path) if path else None)
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    def _extract(self, path: Path | None) -> ExtractionResult:
        if path is None or not path.exists():
            return ExtractionResult(fields=InvoiceFields(), tier=Tier.FAILED, error="no_file")

        suffix = path.suffix.lower()
        partial: InvoiceFields | None = None
        fallback_from: Tier | None = None

        if suffix == ".pdf":
            text = read_pdf_text(path)
            if has_text_layer(text) and not self.force_vlm:
                fields = parse_einvoice_text(text)
                failed, suspicious = self.revalidate(fields)
                if not failed:
                    return ExtractionResult(fields=fields, tier=Tier.PDF_TEXT)
                logger.debug("文字解析未通过校验 {}: {}", path.name, failed)
                partial, fallback_from = fields, Tier.PDF_TEXT
                if self.vlm is None:      # 没有视觉模型可升级：保留解析结果，标出可疑字段
                    return ExtractionResult(fields=fields, tier=Tier.PDF_TEXT,
                                            checks_failed=failed, uncertain_fields=suspicious)
            image = render_first_page(path)
        elif suffix in IMAGE_SUFFIXES:
            image = path.read_bytes()
        else:
            return ExtractionResult(fields=InvoiceFields(), tier=Tier.FAILED,
                                    error=f"unsupported_type:{suffix}")

        if self.vlm is None:
            return ExtractionResult(fields=partial or InvoiceFields(), tier=Tier.FAILED,
                                    fallback_from=fallback_from, error="vlm_unavailable")
        try:
            out = self.vlm.extract(image)
        except Exception as exc:  # noqa: BLE001  模型超时/格式错误都不能让整单崩掉
            logger.warning("视觉模型识别失败 {}: {}", path.name, exc)
            return ExtractionResult(fields=partial or InvoiceFields(), tier=Tier.FAILED,
                                    fallback_from=fallback_from, model_calls=1,
                                    error=f"vlm_error:{type(exc).__name__}")

        failed, suspicious = self.revalidate(out.fields)
        return ExtractionResult(fields=out.fields, tier=Tier.VLM, fallback_from=fallback_from,
                                checks_failed=failed, uncertain_fields=suspicious, model_calls=1,
                                prompt_tokens=out.prompt_tokens,
                                completion_tokens=out.completion_tokens)
