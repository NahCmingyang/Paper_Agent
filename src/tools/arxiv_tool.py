from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import List

import arxiv

from src.config import get_settings
from src.state import ArxivPaperMeta


def _safe_file_id(raw_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_id)


def search_arxiv_papers(query: str, max_results: int | None = None) -> List[ArxivPaperMeta]:
    settings = get_settings()
    limit = max_results or settings.arxiv_top_n

    client = arxiv.Client(
        page_size=max(settings.arxiv_page_size, limit),
        delay_seconds=max(0.0, settings.arxiv_delay_seconds),
        num_retries=max(0, settings.arxiv_num_retries),
    )
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: List[ArxivPaperMeta] = []
    for result in client.results(search):
        papers.append(
            ArxivPaperMeta(
                paper_id=result.get_short_id(),
                title=result.title.strip(),
                summary=result.summary.strip(),
                pdf_url=result.pdf_url,
                authors=[a.name for a in result.authors],
                published=result.published.strftime("%Y-%m-%d") if result.published else "",
            )
        )
    return papers


def download_arxiv_papers(papers: List[ArxivPaperMeta], download_dir: Path) -> List[ArxivPaperMeta]:
    download_dir.mkdir(parents=True, exist_ok=True)
    updated: List[ArxivPaperMeta] = []

    for paper in papers:
        filename = f"{_safe_file_id(paper.paper_id)}.pdf"
        path = download_dir / filename
        if not path.exists():
            urllib.request.urlretrieve(paper.pdf_url, str(path))

        updated.append(paper.model_copy(update={"local_pdf_path": str(path.resolve())}))

    return updated
