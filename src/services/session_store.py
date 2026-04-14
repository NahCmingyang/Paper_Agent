from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict

from src.config import get_settings


def ensure_session_dirs(session_id: str) -> Dict[str, Path]:
    settings = get_settings()
    session_root = settings.session_root / session_id
    uploads = session_root / "uploads"
    docset = session_root / "docset"
    assets = session_root / "assets"
    downloads = session_root / "downloads"

    for p in (uploads, docset, assets, downloads):
        p.mkdir(parents=True, exist_ok=True)

    return {
        "session_root": session_root,
        "uploads": uploads,
        "docset": docset,
        "assets": assets,
        "downloads": downloads,
    }


def persist_uploaded_pdf(session_id: str, source_path: str) -> str:
    dirs = ensure_session_dirs(session_id)
    src = Path(source_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Uploaded file not found: {src}")

    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", src.name)
    dst = dirs["uploads"] / safe_name
    shutil.copy2(src, dst)
    return str(dst.resolve())


def collection_id_for_session(session_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", session_id)
    return f"paper_agent_{safe}"

