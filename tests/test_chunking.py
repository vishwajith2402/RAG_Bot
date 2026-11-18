from src.chunking import chunk_pages
from src.schemas import Page


def test_chunk_pages_preserves_page_metadata():
    page = Page("doc1", "contract.pdf", 7, "First sentence. Second sentence. Third sentence.")
    chunks = chunk_pages([page], chunk_size=25, overlap=0)

    assert chunks
    assert all(chunk.document_name == "contract.pdf" for chunk in chunks)
    assert all(chunk.page_number == 7 for chunk in chunks)

