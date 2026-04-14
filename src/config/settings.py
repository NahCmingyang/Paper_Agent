from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    embedding_model: str
    embedding_local_dir: Path
    retrieval_top_k: int
    max_retry: int
    arxiv_top_n: int
    arxiv_delay_seconds: float
    arxiv_num_retries: int
    arxiv_page_size: int
    workspace_dir: Path
    session_root: Path
    chroma_dir: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    workspace_dir = Path(__file__).resolve().parents[2]
    session_root = workspace_dir / "runtime_sessions"
    chroma_dir = workspace_dir / "chroma_db"
    embedding_local_dir = Path(
        os.getenv("EMBEDDING_LOCAL_DIR", str(workspace_dir / "models" / "bge-m3"))
    ).resolve()

    session_root.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Force offline local embedding usage by default
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip(),
        embedding_local_dir=embedding_local_dir,
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "4")),
        max_retry=int(os.getenv("RAG_MAX_RETRY", "1")),
        arxiv_top_n=int(os.getenv("ARXIV_TOP_N", "5")),
        arxiv_delay_seconds=float(os.getenv("ARXIV_DELAY_SECONDS", "1.0")),
        arxiv_num_retries=int(os.getenv("ARXIV_NUM_RETRIES", "2")),
        arxiv_page_size=int(os.getenv("ARXIV_PAGE_SIZE", "20")),
        workspace_dir=workspace_dir,
        session_root=session_root,
        chroma_dir=chroma_dir,
    )

