"""从上传的研报文件里提取纯文本。支持 .pdf/.txt/.md；截断到 max_chars，
避免把整份报告塞进 LLM 上下文。"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path


def extract_text(filename: str, content: bytes, max_chars: int = 12000) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        text = content.decode("utf-8", errors="ignore")
    elif suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"不支持的文件类型：{suffix}，目前只支持 .pdf/.txt/.md")

    text = text.strip()
    if not text:
        raise ValueError("没有从文件里提取到任何文本")
    return text[:max_chars]
