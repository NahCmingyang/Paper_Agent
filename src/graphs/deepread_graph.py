from __future__ import annotations

from typing import Dict, List

from langgraph.graph import END, StateGraph

from src.config import get_settings
from src.services import ensure_session_dirs
from src.state import DocsetChunk, GraphState, RetrievalJudgeResult
from src.tools import (
    index_docset_chunks,
    invoke_structured,
    invoke_text,
    normalize_yes_no,
    parse_pdf_to_docset,
    retrieve_docset_chunks,
    rewrite_for_rag,
)


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def _target_language(query: str) -> str:
    return "中文" if _is_chinese(query) else "English"


def _retrieved_context(chunks: List[Dict], max_chars: int = 4500) -> str:
    lines: List[str] = []
    for c in chunks:
        lines.append(
            f"[{c.get('chunk_id','')}] page={c.get('page',0)} section={c.get('section','Unknown')}\n{c.get('text','')}"
        )
    return "\n\n".join(lines)[:max_chars]


def upload_ingest_node(state: GraphState):
    uploaded = state.get("uploaded_pdf_path", "").strip()
    active = state.get("active_pdf_path", "").strip()
    if uploaded:
        return {"needs_ingest": True, "active_pdf_path": uploaded}
    if active:
        return {"needs_ingest": False}
    return {"error": "精读模式需要先上传 PDF。"}


def error_node(state: GraphState):
    return {"final_answer": f"执行失败: {state.get('error', '未知错误')}"}


def decide_after_upload(state: GraphState):
    if state.get("error"):
        return "error"
    return "continue"


def docling_parse_node(state: GraphState):
    if not state.get("needs_ingest", False):
        return {}

    session_id = state.get("session_id", "default")
    dirs = ensure_session_dirs(session_id)
    parsed = parse_pdf_to_docset(
        pdf_path=state["active_pdf_path"],
        docset_dir=dirs["docset"],
        assets_dir=dirs["assets"],
    )
    return {
        "doc_hash": parsed["doc_hash"],
        "docset_path": parsed["docset_path"],
        "docset_json_path": parsed["docset_json_path"],
        "retrieved_chunks": parsed["chunks"],
        "sidebar_assets": parsed["assets"],
    }


def index_node(state: GraphState):
    if not state.get("needs_ingest", False):
        return {}
    collection_id = state.get("collection_id", "")
    chunks = [DocsetChunk(**c) for c in state.get("retrieved_chunks", [])]
    index_docset_chunks(collection_id=collection_id, chunks=chunks)
    return {}


def retrieve_node(state: GraphState):
    settings = get_settings()
    try:
        chunks = retrieve_docset_chunks(
            collection_id=state.get("collection_id", ""),
            query=state.get("query", ""),
            top_k=settings.retrieval_top_k,
        )
        return {"retrieved_chunks": chunks}
    except Exception as e:
        return {"retrieved_chunks": [], "error": f"向量检索失败: {e}"}


def judge_relevance_node(state: GraphState):
    if state.get("error"):
        return {"judge_result": {"is_relevant": "no", "reason": state["error"]}}

    query = state.get("original_query") or state.get("query", "")
    context = _retrieved_context(state.get("retrieved_chunks", []))

    prompt = f"""
你是RAG检索评估器。判断检索片段是否足够回答用户问题，只能输出 yes 或 no。

用户问题:
{query}

检索片段:
{context}

若片段缺乏核心方法、结果、定义或结论信息，则判定 no。
"""
    try:
        result = invoke_structured(prompt, RetrievalJudgeResult, temperature=0.0)
        decision = normalize_yes_no(result.is_relevant)
        return {"judge_result": {"is_relevant": decision, "reason": result.reason}}
    except Exception as e:
        return {"judge_result": {"is_relevant": "no", "reason": f"评估失败: {e}"}}


def rewrite_query_node(state: GraphState):
    old_query = state.get("query", "")
    new_query = rewrite_for_rag(
        original_query=state.get("original_query", old_query),
        failed_query=old_query,
        retrieved_context=_retrieved_context(state.get("retrieved_chunks", [])),
    )
    return {
        "query": new_query,
        "retrieval_attempt": state.get("retrieval_attempt", 0) + 1,
    }


def _format_answer(query: str, chunks: List[Dict], strict_mode: str) -> str:
    language = _target_language(query)
    context = _retrieved_context(chunks)
    prompt = f"""
你是论文精读助教，请使用{language}回答用户。
不要输出英文原文大段摘抄，不要输出“低置信度/建议重试”这类系统提示语。

任务模式: {strict_mode}
用户问题:
{query}

检索片段:
{context}

输出要求:
1. 直接给出结构化结论（2-4条）
2. 每条可用括号标注证据页码，如 (p5)
3. 用自然表达总结，不要贴原始片段
"""
    return invoke_text(prompt, temperature=0.2)


def answer_node(state: GraphState):
    query = state.get("original_query") or state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    try:
        answer = _format_answer(query=query, chunks=chunks, strict_mode="normal")
    except Exception as e:
        answer = f"回答生成失败：{e}"

    related_ids = {ref for c in chunks for ref in c.get("asset_refs", [])}
    related_assets = [a for a in state.get("sidebar_assets", []) if a.get("id") in related_ids]
    return {
        "final_answer": answer,
        "sidebar_assets": related_assets or state.get("sidebar_assets", [])[:6],
    }


def fallback_answer_node(state: GraphState):
    if state.get("error"):
        return {"final_answer": f"执行失败: {state['error']}"}

    query = state.get("original_query") or state.get("query", "")
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        empty_msg = "目前没有命中可用证据。请给我一个更具体的问题。"
        if not _is_chinese(query):
            empty_msg = "No usable evidence was retrieved yet. Please ask a more specific question."
        return {"final_answer": empty_msg}

    try:
        concise = _format_answer(query=query, chunks=chunks, strict_mode="evidence-limited")
    except Exception:
        # Still avoid dumping raw chunks
        concise = "已基于当前检索内容提炼结论，但证据覆盖有限。可继续问具体方法或实验细节。"
        if not _is_chinese(query):
            concise = "A concise answer was produced from the retrieved evidence, but coverage is limited."

    return {"final_answer": concise}


def decide_after_judge(state: GraphState):
    settings = get_settings()
    if state.get("error"):
        return "fallback"
    decision = ((state.get("judge_result") or {}).get("is_relevant") or "no").lower()
    if decision == "yes":
        return "answer"
    if state.get("retrieval_attempt", 0) >= settings.max_retry:
        return "fallback"
    return "rewrite"


def create_deepread_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("upload_ingest_node", upload_ingest_node)
    workflow.add_node("error_node", error_node)
    workflow.add_node("docling_parse_node", docling_parse_node)
    workflow.add_node("index_node", index_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("judge_relevance_node", judge_relevance_node)
    workflow.add_node("rewrite_query_node", rewrite_query_node)
    workflow.add_node("answer_node", answer_node)
    workflow.add_node("fallback_answer_node", fallback_answer_node)

    workflow.set_entry_point("upload_ingest_node")
    workflow.add_conditional_edges(
        "upload_ingest_node",
        decide_after_upload,
        {"error": "error_node", "continue": "docling_parse_node"},
    )
    workflow.add_edge("docling_parse_node", "index_node")
    workflow.add_edge("index_node", "retrieve_node")
    workflow.add_edge("retrieve_node", "judge_relevance_node")
    workflow.add_conditional_edges(
        "judge_relevance_node",
        decide_after_judge,
        {
            "answer": "answer_node",
            "rewrite": "rewrite_query_node",
            "fallback": "fallback_answer_node",
        },
    )
    workflow.add_edge("rewrite_query_node", "retrieve_node")
    workflow.add_edge("error_node", END)
    workflow.add_edge("answer_node", END)
    workflow.add_edge("fallback_answer_node", END)
    return workflow.compile()

