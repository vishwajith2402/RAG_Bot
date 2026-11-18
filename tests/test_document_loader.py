import pytest

from src.document_loader import load_document


def test_load_text_file():
    pages = load_document("notes.txt", b"Alpha policy\nBeta clause")

    assert len(pages) == 1
    assert pages[0].document_name == "notes.txt"
    assert "Alpha policy" in pages[0].text


def test_load_html_file_strips_tags():
    pages = load_document("page.html", b"<html><body><h1>Contract</h1><p>Renewal terms</p></body></html>")

    assert "Contract" in pages[0].text
    assert "<h1>" not in pages[0].text


def test_extensionless_text_file_is_accepted():
    pages = load_document("README", b"This is a readable text document.")

    assert pages[0].text == "This is a readable text document."


def test_binary_file_without_text_is_rejected():
    with pytest.raises(ValueError, match="extractable text"):
        load_document("image.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

