import pytest

from app.ai.report_extraction import extract_text


def test_extract_text_from_plain_text_file():
    content = "这是一份研报的正文内容。".encode("utf-8")

    text = extract_text("report.txt", content)

    assert text == "这是一份研报的正文内容。"


def test_extract_text_from_markdown_file():
    content = "# 标题\n正文".encode("utf-8")

    text = extract_text("report.md", content)

    assert text == "# 标题\n正文"


def test_extract_text_truncates_to_max_chars():
    content = ("A" * 100).encode("utf-8")

    text = extract_text("report.txt", content, max_chars=10)

    assert text == "A" * 10


def test_extract_text_rejects_unsupported_file_type():
    with pytest.raises(ValueError):
        extract_text("report.docx", b"whatever")


def test_extract_text_rejects_empty_content():
    with pytest.raises(ValueError):
        extract_text("report.txt", b"   ")
