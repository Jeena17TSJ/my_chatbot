from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(docs: list[Document], query: str) -> list[Document]:
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = _get_model().predict(pairs)
    ranked = sorted(
        zip(scores, docs),
        key=lambda pair: float(pair[0]),
        reverse=True,
    )
    return [document for _, document in ranked]
