from __future__ import annotations

from functools import lru_cache
from typing import Type, TypeVar

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.config import get_settings

T = TypeVar("T", bound=BaseModel)


def _ensure_api_key() -> str:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    return settings.deepseek_api_key


@lru_cache(maxsize=6)
def _cached_llm(temperature: float) -> ChatOpenAI:
    settings = get_settings()
    api_key = _ensure_api_key()
    return ChatOpenAI(
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=temperature,
    )


def invoke_text(prompt: str, temperature: float = 0.1) -> str:
    llm = _cached_llm(round(float(temperature), 2))
    result = llm.invoke([HumanMessage(content=prompt)])
    return str(result.content).strip()


def invoke_structured(prompt: str, schema: Type[T], temperature: float = 0.1) -> T:
    parser = PydanticOutputParser(pydantic_object=schema)
    chain = _cached_llm(round(float(temperature), 2)) | parser
    full_prompt = f"{prompt}\n\n输出格式要求:\n{parser.get_format_instructions()}"
    return chain.invoke([HumanMessage(content=full_prompt)])


def normalize_yes_no(value: str) -> str:
    cleaned = (value or "").strip().lower()
    if cleaned.startswith("y"):
        return "yes"
    return "no"

