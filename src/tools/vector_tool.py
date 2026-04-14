from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings
from src.state import DocsetChunk


@lru_cache(maxsize=1)
def _embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    model_path = Path(settings.embedding_local_dir)
    if not model_path.exists():
        raise RuntimeError(
            f"Local embedding model not found: {model_path}. "
            "Please download BAAI/bge-m3 to this directory first."
        )
    return HuggingFaceEmbeddings(model_name=str(model_path))


def _vectorstore(collection_id: str) -> Chroma:
    settings = get_settings()
    return Chroma(
        persist_directory=str(settings.chroma_dir),
        embedding_function=_embeddings(),
        collection_name=collection_id,
    )


def _serialize_asset_refs(asset_refs: List[str]) -> str:
    refs = [str(x).strip() for x in (asset_refs or []) if str(x).strip()]
    return "||".join(refs)


def _deserialize_asset_refs(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        return [x for x in s.split("||") if x]
    return []


def index_docset_chunks(collection_id: str, chunks: List[DocsetChunk]) -> None:
    store = _vectorstore(collection_id)
    existing = store.get()
    if existing and existing.get("ids"):
        store.delete(ids=existing["ids"])

    docs = [
        Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "page": chunk.page,
                "asset_refs": _serialize_asset_refs(chunk.asset_refs),
            },
        )
        for chunk in chunks
    ]
    ids = [chunk.chunk_id for chunk in chunks]
    if docs:
        store.add_documents(docs, ids=ids)


def retrieve_docset_chunks(collection_id: str, query: str, top_k: int) -> List[Dict]:
    store = _vectorstore(collection_id)
    retriever = store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    result: List[Dict] = []
    for doc in docs:
        md = doc.metadata or {}
        result.append(
            DocsetChunk(
                chunk_id=str(md.get("chunk_id", "")),
                text=doc.page_content,
                section=str(md.get("section", "Unknown")),
                page=int(md.get("page", 0) or 0),
                asset_refs=_deserialize_asset_refs(md.get("asset_refs")),
            ).model_dump()
        )
    return result

