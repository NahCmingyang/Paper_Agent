from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.graphs.chat_graph import create_chat_graph
from src.graphs.deepread_graph import create_deepread_graph
from src.graphs.retrieval_graph import create_retrieval_graph
from src.state import GraphState
from src.tools import invoke_structured


class RouteDecision(BaseModel):
    mode: Literal["chat", "retrieval", "deepread"] = Field(description="路由模式")
    reason: str = Field(description="简短判别理由")
    retrieval_query: str = Field(
        default="",
        description="若 mode=retrieval，提取出的检索关键词（例如 3DGS）",
    )


def _heuristic_extract_search_query(query: str) -> str:
    q = (query or "").strip()
    if re.search(r"\b3dgs\b", q, flags=re.IGNORECASE):
        return "3DGS"
    # remove common request words and keep key noun phrase-ish tokens
    cleaned = re.sub(r"[，。！？,.!?]", " ", q)
    cleaned = re.sub(
        r"(给我|帮我|找|推荐|一些|几篇|论文|文献|please|recommend|find|papers?)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else q


def _heuristic_fallback(query: str, has_pdf: bool) -> tuple[str, str]:
    q = query.lower()
    if has_pdf:
        return "deepread", ""
    has_paper = any(x in q for x in ["paper", "papers", "论文", "文献"])
    has_retrieve_action = any(x in q for x in ["推荐", "检索", "找", "搜", "recommend", "find"])
    if any(x in q for x in ["3dgs", "gaussian splatting", "论文推荐", "推荐论文", "arxiv", "找论文", "检索"]) or (
        has_paper and has_retrieve_action
    ):
        return "retrieval", _heuristic_extract_search_query(query)
    return "chat", ""


def _llm_route(query: str, has_pdf: bool) -> RouteDecision:
    prompt = f"""
你是模式判别与检索词提取器。请在以下模式中二选一或三选一：
- chat：普通聊天、解释、寒暄
- retrieval：用户要找论文/推荐论文/按关键词检索论文
- deepread：用户要对某篇论文进行精读问答（通常已上传PDF）

如果是 retrieval，请提取最核心检索词，去掉“给我找几篇”“推荐一下”等请求词。
示例：
- 输入：给我找几篇3DGS -> retrieval_query: 3DGS
- 输入：推荐一些多模态RAG论文 -> retrieval_query: 多模态 RAG

已上传PDF: {has_pdf}
用户输入: {query}
"""
    return invoke_structured(prompt, RouteDecision, temperature=0.0)


def intent_or_mode_node(state: GraphState):
    mode = (state.get("mode") or "auto").strip().lower()
    query = (state.get("query") or "").strip()
    has_pdf = bool(state.get("uploaded_pdf_path") or state.get("active_pdf_path"))

    if mode in {"chat", "retrieval", "deepread"}:
        resolved = mode
        search_query = _heuristic_extract_search_query(query) if mode == "retrieval" else ""
        reason = "explicit mode"
    else:
        try:
            decision = _llm_route(query=query, has_pdf=has_pdf)
            resolved = decision.mode
            search_query = (decision.retrieval_query or "").strip() if resolved == "retrieval" else ""
            if resolved == "retrieval" and not search_query:
                search_query = _heuristic_extract_search_query(query)
            reason = decision.reason
        except Exception:
            resolved, search_query = _heuristic_fallback(query=query, has_pdf=has_pdf)
            reason = "heuristic fallback"

    return {
        "resolved_mode": resolved,
        "search_query": search_query,
        "judge_result": {"route_reason": reason},
    }


def _route_mode(state: GraphState):
    return state.get("resolved_mode", "chat")


@lru_cache(maxsize=1)
def _chat_app():
    return create_chat_graph()


@lru_cache(maxsize=1)
def _retrieval_app():
    return create_retrieval_graph()


@lru_cache(maxsize=1)
def _deepread_app():
    return create_deepread_graph()


def run_chat_graph(state: GraphState):
    return _chat_app().invoke(state)


def run_retrieval_graph(state: GraphState):
    return _retrieval_app().invoke(state)


def run_deepread_graph(state: GraphState):
    return _deepread_app().invoke(state)


def create_router_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("intent_or_mode_node", intent_or_mode_node)
    workflow.add_node("chat_graph_runner", run_chat_graph)
    workflow.add_node("retrieval_graph_runner", run_retrieval_graph)
    workflow.add_node("deepread_graph_runner", run_deepread_graph)

    workflow.set_entry_point("intent_or_mode_node")
    workflow.add_conditional_edges(
        "intent_or_mode_node",
        _route_mode,
        {
            "chat": "chat_graph_runner",
            "retrieval": "retrieval_graph_runner",
            "deepread": "deepread_graph_runner",
        },
    )
    workflow.add_edge("chat_graph_runner", END)
    workflow.add_edge("retrieval_graph_runner", END)
    workflow.add_edge("deepread_graph_runner", END)
    return workflow.compile()
