from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class SidebarAsset(BaseModel):
    id: str = Field(description="Asset id such as figure-1 or table-2")
    type: Literal["figure", "table"]
    path: str = Field(description="Absolute local file path")
    caption: str = ""
    page: Optional[int] = None
    section: Optional[str] = None


class ArxivPaperMeta(BaseModel):
    paper_id: str
    title: str
    summary: str
    pdf_url: str
    authors: List[str] = Field(default_factory=list)
    published: str = ""
    local_pdf_path: str = ""


class PaperAnalysis(BaseModel):
    paper_id: str
    title: str
    research_problem: str
    method: str
    dataset_or_task: str
    key_findings: str
    limitations: str
    conclusion: str
    source_url: str


class BatchComparisonReport(BaseModel):
    papers: List[PaperAnalysis]
    trend_summary: str
    selection_suggestion: str
    markdown: str


class DocsetChunk(BaseModel):
    chunk_id: str
    text: str
    section: str
    page: int
    asset_refs: List[str] = Field(default_factory=list)


class RetrievalJudgeResult(BaseModel):
    is_relevant: Literal["yes", "no"]
    reason: str


class GraphState(TypedDict, total=False):
    mode: str
    resolved_mode: str
    query: str
    original_query: str
    search_query: str
    retrieved_chunks: List[Dict]
    retrieval_attempt: int
    judge_result: Dict
    final_answer: str
    sidebar_assets: List[Dict]
    paper_batch_report: Dict

    session_id: str
    collection_id: str
    uploaded_pdf_path: str
    active_pdf_path: str
    docset_path: str
    docset_json_path: str
    doc_hash: str
    needs_ingest: bool

    arxiv_papers: List[Dict]
    analysis_results: List[Dict]
    error: str
