"""发票文件存储：本地目录或 S3 兼容存储（RustFS、云厂商对象存储……；客户端用 minio-py，它是通用的 S3 客户端）。

按内容哈希命名（invoices/ab/abcdef….pdf）：同一个文件传两次只存一份，
文件名也不会泄露员工上传时用的原始文件名。
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Protocol

from ..config import ROOT, Settings

# 只认文件头，不信客户端声明的 Content-Type 和扩展名
MAGIC = {b"%PDF-": ".pdf", b"\xff\xd8\xff": ".jpg", b"\x89PNG\r\n\x1a\n": ".png"}
CONTENT_TYPES = {".pdf": "application/pdf", ".jpg": "image/jpeg", ".png": "image/png"}


def sniff(data: bytes) -> str | None:
    """按文件头判断类型，返回扩展名；不支持的类型返回 None。"""
    for magic, suffix in MAGIC.items():
        if data.startswith(magic):
            return suffix
    return None


def object_key(data: bytes, suffix: str) -> tuple[str, str]:
    digest = hashlib.sha256(data).hexdigest()
    return f"invoices/{digest[:2]}/{digest}{suffix}", digest


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> bytes: ...
    def ping(self) -> bool: ...


class LocalStore:
    def __init__(self, base: str | Path):
        base = Path(base)
        self.base = base if base.is_absolute() else ROOT / base
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.base / key).resolve()
        if self.base.resolve() not in p.parents:          # 防止 key 里带 ../ 跳出存储目录
            raise ValueError(f"非法的对象键：{key}")
        return p

    def put(self, key: str, data: bytes, content_type: str) -> None:
        p = self._path(key)
        if p.exists():                                     # 内容寻址：已存在就是同一个文件
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)                                     # 先写临时文件再改名，不会留下写了一半的文件

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def ping(self) -> bool:
        return self.base.is_dir()


class S3Store:
    def __init__(self, s: Settings):
        from minio import Minio

        self.client = Minio(s.s3_endpoint, access_key=s.s3_access_key, secret_key=s.s3_secret_key,
                            secure=s.s3_secure)
        self.bucket = s.s3_bucket
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(self.bucket, key, io.BytesIO(data), len(data), content_type=content_type)

    def get(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def ping(self) -> bool:
        return self.client.bucket_exists(self.bucket)


def make_store(s: Settings) -> ObjectStore:
    if s.storage_backend == "s3":
        return S3Store(s)
    if s.storage_backend == "local":
        return LocalStore(s.storage_dir)
    raise ValueError(f"未知的 STORAGE_BACKEND：{s.storage_backend}")
