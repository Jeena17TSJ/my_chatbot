from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data"


def load_documents(docs_path: str | Path = DEFAULT_DATA_PATH) -> list[Document]:
    """Load UTF-8 text references from the project's data directory."""
    path = Path(docs_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Medical reference directory not found: {path}")

    loader = DirectoryLoader(
        path=str(path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=False,
    )
    documents = loader.load()
    if not documents:
        raise FileNotFoundError(
            f"No .txt medical reference files found in {path}. Add UTF-8 text files and try again."
        )
    return documents
