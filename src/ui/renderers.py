from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import chainlit as cl


def build_sidebar_elements(assets: List[Dict]) -> List[cl.Image]:
    elements: List[cl.Image] = []
    for asset in assets:
        path = Path(str(asset.get("path", "")))
        if not path.exists():
            continue

        caption = (asset.get("caption") or "").strip()
        if not caption:
            caption = f"{asset.get('type', 'asset')} {asset.get('id', '')}"
        page = asset.get("page", "N/A")
        title = f"{asset.get('id', path.stem)} | p{page} | {caption}"

        # Put title in the image element name so title/image stay 1:1.
        elements.append(
            cl.Image(
                path=str(path.resolve()),
                name=title,
                display="side",
                size="large",
            )
        )
    return elements


def render_final_answer(answer: str) -> str:
    cleaned = (answer or "").strip()
    if not cleaned:
        return "暂无输出。"
    return cleaned

