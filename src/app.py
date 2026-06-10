"""
Gradio Web UI 入口。
启动：python -m src.app
部署：直接 push 到 HF Space（gradio SDK）。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import gradio as gr

from src.config import MINIMAX_API_KEY, USE_LLM
from src.retriever import Retriever
from src.generator import generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")


EXAMPLE_QUERIES = [
    "亚马逊 Listing 被投诉版权侵权，第一步该做什么？",
    "TikTok Shop 差评率上升了，怎么分析根因？",
    "Shopee 退换货中 '与描述不符' 应该怎么处理？",
    "FBA 库存可售天数只剩 10 天，怎么决策？",
    "欧洲站 VAT 什么时候必须注册？",
    "新品期亚马逊广告应该怎么出价？",
    "选品时如何判断一个类目值不值得进？",
    "Temu 主图的尺寸和构图要求是什么？",
]


def build_index_if_needed() -> Retriever:
    if not (Path(__file__).parent.parent / "data" / "chroma" / "_bm25.pkl").exists():
        log.info("BM25 index not found, running ingest() ...")
        from src.ingest import main as ingest_main
        ingest_main()
    return Retriever()


def answer_query(retriever: Retriever, query: str) -> tuple[str, str, str]:
    if not query.strip():
        return "请输入问题", "", "empty"
    hits = retriever.search(query)
    if retriever.should_refuse(hits):
        return "知识库无相关信息，建议联系业务团队核实。", "[]", "refused"
    result = generate(query, hits)
    refs_str = json.dumps(result["references"], ensure_ascii=False, indent=2) if result["references"] else "[]"
    return result["answer"], refs_str, result["mode"]


def launch() -> None:
    retriever = build_index_if_needed()

    with gr.Blocks(title="跨境电商 SOP 助手") as demo:
        gr.Markdown(
            f"# 🛒 跨境电商 SOP 助手\n\n基于 BM25 的跨境电商业务知识库，覆盖 200+ 条真实业务 SOP。\n\n"
            f"运行模式：{'LLM 答案生成（MiniMax-M3）' if USE_LLM else '**降级模式**（未配置 LLM key，仅返回检索片段）'}"
        )
        gr.Markdown("### 示例问题（点击直接试）")
        gr.Examples(examples=[[q] for q in EXAMPLE_QUERIES], inputs=[], label="")

        with gr.Row():
            query_box = gr.Textbox(label="你的问题", placeholder="例如：亚马逊 Listing 被投诉版权侵权，第一步该做什么？", lines=2)
            submit_btn = gr.Button("🔍 提问", variant="primary")

        answer_box = gr.Markdown(label="答案")
        refs_box = gr.Textbox(label="引用来源（JSON）", lines=20, interactive=False)
        mode_box = gr.Textbox(label="模式", interactive=False)

        def on_submit(query: str):
            ans, refs, mode = answer_query(retriever, query)
            return ans, refs, mode

        submit_btn.click(on_submit, inputs=[query_box], outputs=[answer_box, refs_box, mode_box])
        query_box.submit(on_submit, inputs=[query_box], outputs=[answer_box, refs_box, mode_box])

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        prevent_thread_lock=True,
        show_error=True,
    )
    # Keep main thread alive
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    launch()
