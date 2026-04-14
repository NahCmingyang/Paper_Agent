from __future__ import annotations

from src.tools.llm_tool import invoke_text


def rewrite_for_rag(original_query: str, failed_query: str, retrieved_context: str) -> str:
    prompt = f"""
你是学术检索优化助手。请把失败检索词重写成更可检索的英文关键词短语。

原始问题:
{original_query}

上一次失败检索词:
{failed_query}

失败检索命中的片段:
{retrieved_context[:2000]}

要求:
1. 输出仅包含一个新检索词短语，不要解释。
2. 不要复用失败检索词的完整表达。
3. 聚焦论文中的方法、实验、结果术语。
"""
    return invoke_text(prompt, temperature=0.0).strip()

