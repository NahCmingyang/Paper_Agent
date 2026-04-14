from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Iterable, List

import chainlit as cl
from chainlit.input_widget import Select

from src.graphs import create_router_graph
from src.graphs.deepread_graph import create_deepread_graph
from src.services import collection_id_for_session, ensure_session_dirs, persist_uploaded_pdf
from src.state import ArxivPaperMeta
from src.tools import download_arxiv_papers
from src.ui.renderers import build_sidebar_elements, render_final_answer

MODE_LABEL_TO_KEY = {
    "自动识别": "auto",
    "普通对话": "chat",
    "论文检索": "retrieval",
    "论文精读": "deepread",
}
MODE_KEY_TO_LABEL = {v: k for k, v in MODE_LABEL_TO_KEY.items()}

ROUTER_APP = create_router_graph()


@lru_cache(maxsize=1)
def _deepread_app():
    return create_deepread_graph()


def _extract_pdf_paths(message: cl.Message) -> List[str]:
    paths: List[str] = []
    for element in (message.elements or []):
        path = getattr(element, "path", None)
        name = str(getattr(element, "name", "")).lower()
        mime = str(getattr(element, "mime", "")).lower()
        if not path:
            continue
        if str(path).lower().endswith(".pdf") or name.endswith(".pdf") or mime == "application/pdf":
            paths.append(str(path))
    return paths


def _chunk_text(text: str, size: int = 80) -> Iterable[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]


async def _send_streamed_message(text: str, actions: List[cl.Action] | None = None):
    msg = cl.Message(content="", actions=actions or [])
    await msg.send()
    for part in _chunk_text(text):
        await msg.stream_token(part)
    await msg.update()


async def _replace_sidebar(assets: List[dict]):
    old_sidebar_msg = cl.user_session.get("sidebar_message")
    if old_sidebar_msg is not None:
        try:
            await old_sidebar_msg.remove()
        except Exception:
            pass

    side_elements = build_sidebar_elements(assets)
    if not side_elements:
        cl.user_session.set("sidebar_message", None)
        return

    sidebar_msg = cl.Message(content="论文图表（最新）", elements=side_elements)
    await sidebar_msg.send()
    cl.user_session.set("sidebar_message", sidebar_msg)


async def _ensure_mode_settings(default_mode: str):
    await cl.ChatSettings(
        [
            Select(
                id="mode",
                label="运行模式",
                values=list(MODE_LABEL_TO_KEY.keys()),
                initial_value=MODE_KEY_TO_LABEL[default_mode],
            )
        ]
    ).send()


def _paper_actions(papers: List[dict]) -> List[cl.Action]:
    actions: List[cl.Action] = []
    for i, _ in enumerate(papers[:5], start=1):
        actions.append(
            cl.Action(
                name="deepread_selected_paper",
                payload={"index": i - 1},
                label=f"精读第{i}篇",
            )
        )
    return actions


@cl.on_chat_start
async def on_chat_start():
    session_id = uuid.uuid4().hex[:12]
    ensure_session_dirs(session_id)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("mode", "auto")
    cl.user_session.set("collection_id", collection_id_for_session(session_id))
    cl.user_session.set("active_pdf_path", "")
    cl.user_session.set("sidebar_assets", [])
    cl.user_session.set("last_retrieval_papers", [])
    cl.user_session.set("sidebar_message", None)

    await _ensure_mode_settings(default_mode="auto")
    await cl.Message(
        content=(
            "论文 Agent 已启动。\n\n"
            "- 普通对话：直接聊天，不触发论文流程。\n"
            "- 论文检索：返回最相关 5 篇论文（凝练摘要 + PDF 链接），可点选精读。\n"
            "- 论文精读：上传 PDF 后提问，执行 Docling + RAG + 质量判定。"
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    selected = settings.get("mode", "自动识别")
    mode = MODE_LABEL_TO_KEY.get(selected, "auto")
    cl.user_session.set("mode", mode)
    await cl.Message(content=f"已切换模式：{selected}").send()


async def _run_deepread(session_id: str, collection_id: str, active_pdf_path: str, query: str):
    state = {
        "mode": "deepread",
        "query": query,
        "original_query": query,
        "retrieval_attempt": 0,
        "session_id": session_id,
        "collection_id": collection_id,
        "uploaded_pdf_path": active_pdf_path,
        "active_pdf_path": active_pdf_path,
    }
    result = await cl.make_async(_deepread_app().invoke)(state)

    sidebar_assets = result.get("sidebar_assets", [])
    if sidebar_assets:
        cl.user_session.set("sidebar_assets", sidebar_assets)
        await _replace_sidebar(sidebar_assets)

    await _send_streamed_message(render_final_answer(result.get("final_answer", "")))


@cl.action_callback("deepread_selected_paper")
async def deepread_selected_paper(action: cl.Action):
    session_id = cl.user_session.get("session_id")
    collection_id = cl.user_session.get("collection_id")
    papers = cl.user_session.get("last_retrieval_papers", [])
    idx = int((action.payload or {}).get("index", -1))

    if idx < 0 or idx >= len(papers):
        await cl.Message(content="选择无效，请重新检索后再选择。").send()
        return

    paper = ArxivPaperMeta(**papers[idx])
    dirs = ensure_session_dirs(session_id)
    downloaded = download_arxiv_papers([paper], dirs["downloads"])[0]
    active_pdf_path = downloaded.local_pdf_path

    cl.user_session.set("active_pdf_path", active_pdf_path)
    cl.user_session.set("mode", "deepread")

    await cl.Message(content=f"已下载并开始精读：{paper.title}\n\nPDF: {paper.pdf_url}").send()
    await _run_deepread(
        session_id=session_id,
        collection_id=collection_id,
        active_pdf_path=active_pdf_path,
        query="请精读这篇论文，给出核心贡献、方法、关键实验与局限性。",
    )


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    mode = cl.user_session.get("mode", "auto")
    collection_id = cl.user_session.get("collection_id")

    uploaded_pdf_path = ""
    pdf_paths = _extract_pdf_paths(message)
    if pdf_paths:
        uploaded_pdf_path = persist_uploaded_pdf(session_id=session_id, source_path=pdf_paths[0])
        cl.user_session.set("active_pdf_path", uploaded_pdf_path)

    active_pdf_path = cl.user_session.get("active_pdf_path", "")
    query = (message.content or "").strip()

    if mode == "deepread" and not active_pdf_path and not uploaded_pdf_path:
        await cl.Message(content="精读模式请先上传 PDF，然后再提问。").send()
        return
    if mode == "retrieval" and not query:
        await cl.Message(content="检索模式请输入关键词。").send()
        return
    if not query and mode != "retrieval":
        query = "你好"

    initial_state = {
        "mode": mode,
        "query": query,
        "original_query": query,
        "retrieval_attempt": 0,
        "session_id": session_id,
        "collection_id": collection_id,
        "uploaded_pdf_path": uploaded_pdf_path,
        "active_pdf_path": active_pdf_path,
        "sidebar_assets": cl.user_session.get("sidebar_assets", []),
    }

    async with cl.Step(name="Agent Workflow") as step:
        result = await cl.make_async(ROUTER_APP.invoke)(initial_state)
        step.output = (
            f"模式: {result.get('resolved_mode', mode)} | "
            f"检索词: {result.get('search_query', '')} | "
            f"重试: {result.get('retrieval_attempt', 0)}"
        )

    sidebar_assets = result.get("sidebar_assets", [])
    if sidebar_assets:
        cl.user_session.set("sidebar_assets", sidebar_assets)
        await _replace_sidebar(sidebar_assets)

    if result.get("error"):
        await _send_streamed_message(f"执行失败: {result['error']}")
        return

    final_text = render_final_answer(result.get("final_answer", ""))
    if result.get("resolved_mode") == "retrieval":
        papers = result.get("arxiv_papers", [])
        cl.user_session.set("last_retrieval_papers", papers)
        await _send_streamed_message(final_text, actions=_paper_actions(papers))
        return

    await _send_streamed_message(final_text)

