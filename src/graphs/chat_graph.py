from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.state import GraphState
from src.tools import invoke_text


def chat_node(state: GraphState):
    query = (state.get("query") or "").strip()
    if not query:
        return {"final_answer": "你好，我在。你可以直接问我论文问题，或先聊聊你的研究方向。"}

    prompt = f"""
你是一个友好、专业的论文助教。
当前用户在普通对话模式下和你交流，不需要触发论文检索或PDF精读流程。
请直接自然地回答用户问题，中文输出，简洁清晰。

用户消息:
{query}
"""
    try:
        answer = invoke_text(prompt, temperature=0.4)
    except Exception as e:
        answer = f"当前无法调用对话模型（{e}）。请先设置 DEEPSEEK_API_KEY，或切换到论文检索/精读流程。"
    return {"final_answer": answer, "resolved_mode": "chat"}


def create_chat_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("chat_node", chat_node)
    workflow.set_entry_point("chat_node")
    workflow.add_edge("chat_node", END)
    return workflow.compile()
