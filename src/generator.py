"""
LLM 答案生成器。
  - 默认走 MiniMax-M3（OpenAI 兼容）
  - 无 key 时降级为 extractive 模式（直接拼接 top chunks）
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from src.config import MINIMAX_API_KEY, MINIMAX_BASE_URL, MINIMAX_MODEL, USE_LLM
from src.prompt import SYSTEM_PROMPT
from src.retriever import RetrievalHit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generator")


def _format_context(hits: list[RetrievalHit]) -> str:
    lines: list[str] = []
    for idx, h in enumerate(hits, start=1):
        meta = h.metadata
        header = f"[{idx}] 《{meta.get('sop_title','')}》 - {meta.get('section','')} (平台: {meta.get('platform','')}, 类别: {meta.get('category','')})"
        lines.append(header)
        lines.append(h.text)
        lines.append("")  # 空行分隔
    return "\n".join(lines)


def _format_references(hits: list[RetrievalHit]) -> list[dict]:
    return [
        {
            "n": idx,
            "sop_id": h.metadata.get("sop_id", ""),
            "sop_title": h.metadata.get("sop_title", ""),
            "section": h.metadata.get("section", ""),
            "platform": h.metadata.get("platform", ""),
            "category": h.metadata.get("category", ""),
            "score": round(h.score, 3),
            "recall_score": round(h.recall_score, 3),
        }
        for idx, h in enumerate(hits, start=1)
    ]


def _extractive_answer(query: str, hits: list[RetrievalHit]) -> str:
    """无 LLM 时的降级：直接拼接 top chunks + 引导。"""
    parts: list[str] = ["（未配置 LLM API key，当前为检索模式。以下为知识库中匹配度最高的片段，请人工综合。）", ""]
    for idx, h in enumerate(hits, start=1):
        meta = h.metadata
        parts.append(f"### 引用 [{idx}] {meta.get('sop_title','')}")
        parts.append(f"_平台_: {meta.get('platform','')} · _类别_: {meta.get('category','')} · _章节_: {meta.get('section','')}")
        parts.append("")
        parts.append(h.text)
        parts.append("")
    return "\n".join(parts)


def generate(query: str, hits: list[RetrievalHit]) -> dict:
    if not hits:
        return {
            "answer": "知识库无相关信息，建议联系业务团队核实。",
            "references": [],
            "refused": True,
            "mode": "refuse",
        }

    if not USE_LLM:
        return {
            "answer": _extractive_answer(query, hits),
            "references": _format_references(hits),
            "refused": False,
            "mode": "extractive",
        }

    # LLM 模式
    from openai import OpenAI

    client = OpenAI(api_key=MINIMAX_API_KEY, base_url=MINIMAX_BASE_URL)
    context = _format_context(hits)
    sys_prompt = SYSTEM_PROMPT.format(context=context)
    # 大多数 category 字段取第一个
    category = hits[0].metadata.get("category", "业务")
    sys_prompt = sys_prompt.replace("{category}", category)

    log.info("Calling LLM (%s) ...", MINIMAX_MODEL)
    try:
        resp = client.chat.completions.create(
            model=MINIMAX_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        log.exception("LLM call failed, falling back to extractive: %s", e)
        answer = _extractive_answer(query, hits) + f"\n\n> ⚠️ LLM 调用失败：{e}"

    return {
        "answer": answer,
        "references": _format_references(hits),
        "refused": False,
        "mode": "llm",
    }
