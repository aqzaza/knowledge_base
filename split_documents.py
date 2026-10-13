"""文本切分"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from read_document import load_directory


def split_documents(documents: list[Document],chunk_size: int = 500,chunk_overlap: int = 80) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    # TODO: return the result of splitting the input Documents.
    return splitter.split_documents(documents)


if __name__ == "__main__":
    corpus_dir = Path(__file__).resolve().parent / "corpus"
    documents = load_directory(corpus_dir)
    chunk_size = 500
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=80)
    if not chunks:
        raise ValueError("No chunks produced.")
    assert all(0 < len(chunk.page_content) <= chunk_size for chunk in chunks)

    print("Documents:", len(documents))
    print("Chunks:", len(chunks))
    print("Max characters:", max(len(chunk.page_content) for chunk in chunks))
    for index, chunk in enumerate(chunks[:3]):
        print("Chunk:", index)
        print("Source:", chunk.metadata["source"])
        print("Start index:", chunk.metadata["start_index"])
        print("Characters:", len(chunk.page_content))
        print("Preview:", chunk.page_content[:160])
