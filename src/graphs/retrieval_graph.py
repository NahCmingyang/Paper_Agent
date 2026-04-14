from __future__ import annotations

from typing import Dict, List

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.state import GraphState
from src.tools import invoke_structured, search_arxiv_papers


class PaperDigest(BaseModel):
    paper_id: str = Field(description="论文ID")
    concise_summary: str = Field(description="凝练摘要，1-2句")


class PaperDigestList(BaseModel):
    items: List[PaperDigest]


class LanguageGuard(BaseModel):
    is_consistent: bool = Field(description="输出语言是否与用户输入一致")
    corrected_output: str = Field(description="若不一致，给出修正后的完整输出")


def intent_or_mode_node(state: GraphState):
    return {"resolved_mode": "retrieval"}


def arxiv_search_node(state: GraphState):
    search_query = (state.get("search_query") or state.get("query") or "").strip()
    if not search_query:
        return {"arxiv_papers": [], "error": "检索模式下关键词不能为空。"}
    try:
        papers = search_arxiv_papers(query=search_query)
        return {"arxiv_papers": [p.model_dump() for p in papers], "search_query": search_query}
    except Exception as e:
        return {"arxiv_papers": [], "error": f"ArXiv 检索失败: {e}"}


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def _fallback_compact_abstract(text: str, max_len: int = 120) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1] + "…"


def _llm_summarize_abstracts(user_query: str, papers: List[Dict]) -> Dict[str, str]:
    language = "中文" if _is_chinese(user_query) else "English"
    items_text = []
    for p in papers:
        items_text.append(
            f"- paper_id: {p.get('paper_id','')}\n"
            f"  title: {p.get('title','')}\n"
            f"  abstract: {p.get('summary','')}"
        )

    prompt = f"""
你是论文摘要重写助手。请把论文摘要“凝练改写”为用户语言，不要照抄原句。
用户语言: {language}
用户意图: {user_query}

要求:
1. 每篇输出1-2句，突出研究问题、方法亮点、适用场景。
2. 避免直接复制原摘要中的连续长句。
3. 使用 paper_id 对齐输出。

论文列表:
{chr(10).join(items_text)}
"""
    result = invoke_structured(prompt, PaperDigestList, temperature=0.2)
    return {item.paper_id: item.concise_summary for item in result.items}


def summarize_candidates_node(state: GraphState):
    papers: List[Dict] = state.get("arxiv_papers", [])
    user_query = state.get("query", "")
    search_query = state.get("search_query", "")
    if not papers:
        return {"final_answer": "没有检索到相关论文，请换一个更具体的关键词。"}

    try:
        digest_map = _llm_summarize_abstracts(user_query=user_query, papers=papers[:5])
    except Exception:
        digest_map = {p.get("paper_id", ""): _fallback_compact_abstract(p.get("summary", "")) for p in papers[:5]}

    lines = [f"## 为你检索到最相关的 5 篇论文（检索词：{search_query or user_query}）", ""]
    for idx, p in enumerate(papers[:5], start=1):
        authors = ", ".join((p.get("authors") or [])[:3])
        if len(p.get("authors") or []) > 3:
            authors += " et al."
        concise = digest_map.get(p.get("paper_id", ""), _fallback_compact_abstract(p.get("summary", "")))
        lines.extend(
            [
                f"### {idx}. {p.get('title', 'Untitled')}",
                f"- 作者: {authors or 'N/A'}",
                f"- 发布日期: {p.get('published', 'N/A')}",
                f"- 凝练摘要: {concise}",
                f"- PDF: {p.get('pdf_url', '')}",
                "",
            ]
        )
    lines.append("请选择一篇你最感兴趣的论文，我会自动下载并进入精读。")
    return {"final_answer": "\n".join(lines)}


def language_guard_node(state: GraphState):
    user_query = state.get("query", "")
    draft = state.get("final_answer", "")
    if not draft:
        return {}

    prompt = f"""
检查以下回复是否与用户输入语言一致（中文对中文，英文对英文）。
若一致，is_consistent=true，corrected_output 原样返回。
若不一致，is_consistent=false，并将回复完整改写为与用户输入一致的语言。

用户输入:
{user_query}

回复草稿:
{draft}
"""
    try:
        result = invoke_structured(prompt, LanguageGuard, temperature=0.0)
        return {"final_answer": result.corrected_output if not result.is_consistent else draft}
    except Exception:
        return {}


def respond_node(state: GraphState):
    if state.get("error"):
        return {"final_answer": f"执行失败: {state['error']}"}
    return {"final_answer": state.get("final_answer", "")}


def create_retrieval_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("intent_or_mode_node", intent_or_mode_node)
    workflow.add_node("arxiv_search_node", arxiv_search_node)
    workflow.add_node("summarize_candidates_node", summarize_candidates_node)
    workflow.add_node("language_guard_node", language_guard_node)
    workflow.add_node("respond_node", respond_node)

    workflow.set_entry_point("intent_or_mode_node")
    workflow.add_edge("intent_or_mode_node", "arxiv_search_node")
    workflow.add_edge("arxiv_search_node", "summarize_candidates_node")
    workflow.add_edge("summarize_candidates_node", "language_guard_node")
    workflow.add_edge("language_guard_node", "respond_node")
    workflow.add_edge("respond_node", END)
    return workflow.compile()

