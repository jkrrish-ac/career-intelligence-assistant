from pathlib import Path

import pytest

from app.core.exceptions import DocumentParseError, UnsupportedFileTypeError
from app.rag.parsers import parse_document, parse_txt


def test_parse_txt_success(tmp_path: Path):
    path = tmp_path / "resume.txt"
    path.write_text("Backend engineer with 5 years of experience.")
    assert parse_txt(path) == "Backend engineer with 5 years of experience."


def test_parse_txt_raises_on_empty_file(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    with pytest.raises(DocumentParseError):
        parse_txt(path)


def test_parse_txt_raises_on_whitespace_only_file(tmp_path: Path):
    path = tmp_path / "whitespace.txt"
    path.write_text("   \n\n   ")
    with pytest.raises(DocumentParseError):
        parse_txt(path)


def test_parse_document_rejects_unsupported_extension(tmp_path: Path):
    path = tmp_path / "resume.rtf"
    path.write_text("some content")
    with pytest.raises(UnsupportedFileTypeError):
        parse_document(path)


def test_parse_document_dispatches_by_extension(tmp_path: Path):
    path = tmp_path / "jd.txt"
    path.write_text("We are hiring a backend engineer.")
    assert parse_document(path) == "We are hiring a backend engineer."
