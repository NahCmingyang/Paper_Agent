from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.state import DocsetChunk, SidebarAsset


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _to_top_left_rect(bbox: Dict, page: fitz.Page) -> fitz.Rect:
    page_rect = page.rect
    height = page_rect.height

    left = float(bbox.get("l", 0.0))
    right = float(bbox.get("r", page_rect.width))
    top_bl = float(bbox.get("t", height))
    bottom_bl = float(bbox.get("b", 0.0))

    y0 = height - top_bl
    y1 = height - bottom_bl
    rect = fitz.Rect(min(left, right), min(y0, y1), max(left, right), max(y0, y1))
    return rect & page_rect


def _build_text_map(doc_dict: Dict) -> Dict[str, str]:
    text_map: Dict[str, str] = {}
    for item in doc_dict.get("texts", []):
        ref = item.get("self_ref")
        txt = str(item.get("text", "")).strip()
        if ref and txt:
            text_map[ref] = txt
    return text_map


def _resolve_caption(item: Dict, text_map: Dict[str, str]) -> str:
    captions = item.get("captions", [])
    resolved: List[str] = []
    for c in captions:
        ref = c.get("$ref")
        if ref and ref in text_map:
            resolved.append(text_map[ref])
    return " ".join(resolved).strip()


def _extract_assets(pdf_path: Path, doc_dict: Dict, assets_dir: Path) -> List[SidebarAsset]:
    assets: List[SidebarAsset] = []
    assets_dir.mkdir(parents=True, exist_ok=True)
    text_map = _build_text_map(doc_dict)
    pdf = fitz.open(pdf_path)

    try:
        for asset_type, key in (("figure", "pictures"), ("table", "tables")):
            for idx, item in enumerate(doc_dict.get(key, []), start=1):
                prov = (item.get("prov") or [{}])[0]
                page_no = int(prov.get("page_no", 0) or 0)
                if page_no <= 0 or page_no > len(pdf):
                    continue
                bbox = prov.get("bbox") or {}
                caption = _resolve_caption(item, text_map)
                asset_id = f"{asset_type}-{idx}"

                page = pdf[page_no - 1]
                rect = _to_top_left_rect(bbox, page)
                if rect.width < 2 or rect.height < 2:
                    continue

                output_path = assets_dir / f"{asset_id}.png"
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
                pix.save(str(output_path))

                assets.append(
                    SidebarAsset(
                        id=asset_id,
                        type=asset_type,
                        path=str(output_path.resolve()),
                        caption=caption,
                        page=page_no,
                    )
                )
    finally:
        pdf.close()

    return assets


def _build_docset_chunks(doc_dict: Dict, assets: List[SidebarAsset]) -> List[DocsetChunk]:
    page_entries: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    current_section = "Unknown"

    for item in doc_dict.get("texts", []):
        raw_text = str(item.get("text", "")).strip()
        if not raw_text:
            continue
        label = str(item.get("label", "text"))
        prov = (item.get("prov") or [{}])[0]
        page = int(prov.get("page_no", 0) or 0)
        if page <= 0:
            continue
        if label == "section_header":
            current_section = raw_text
        page_entries[page].append((current_section, raw_text))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    assets_by_page: Dict[int, List[str]] = defaultdict(list)
    for asset in assets:
        if asset.page:
            assets_by_page[asset.page].append(asset.id)

    chunks: List[DocsetChunk] = []
    for page in sorted(page_entries):
        section = next((s for s, _ in page_entries[page] if s and s != "Unknown"), "Unknown")
        page_text = "\n".join(text for _, text in page_entries[page])
        sub_chunks = splitter.split_text(page_text)

        for i, text in enumerate(sub_chunks, start=1):
            chunk_id = f"p{page}-c{i}"
            chunks.append(
                DocsetChunk(
                    chunk_id=chunk_id,
                    text=text,
                    section=section,
                    page=page,
                    asset_refs=assets_by_page.get(page, []),
                )
            )
    return chunks


def parse_pdf_to_docset(pdf_path: str, docset_dir: Path, assets_dir: Path):
    pdf = Path(pdf_path).resolve()
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    docset_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    converter = DocumentConverter()
    result = converter.convert(str(pdf))
    doc = result.document
    markdown = doc.export_to_markdown()
    doc_dict = doc.export_to_dict()

    docset_md = docset_dir / "docset.md"
    docset_json = docset_dir / "docset.json"

    docset_md.write_text(markdown, encoding="utf-8")
    docset_json.write_text(json.dumps(doc_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    assets = _extract_assets(pdf, doc_dict, assets_dir)
    chunks = _build_docset_chunks(doc_dict, assets)

    return {
        "doc_hash": _sha256(pdf),
        "docset_path": str(docset_md.resolve()),
        "docset_json_path": str(docset_json.resolve()),
        "chunks": [chunk.model_dump() for chunk in chunks],
        "assets": [asset.model_dump() for asset in assets],
    }

