# 🛒 Ecommerce SOP RAG Assistant

> 一个面向跨境电商业务的 **RAG（检索增强生成）知识库系统**，专为出海业务团队的 SOP（标准操作流程）检索设计。
> 端到端可跑：本地 Embedding（**完全免费，无 API 费用**）+ Chroma 向量库 + Gradio Web UI + Hugging Face Space 部署。

![demo](assets/demo.png)

---

## ✨ 这个项目是什么

面试官问："你做过 RAG 吗？"

**直接点开这个 demo**：https://huggingface.co/spaces/hhdhh/ecommerce-sop-rag

输入："**亚马逊 Listing 被投诉版权侵权，第一步该做什么？**"

3 秒内返回：

- ✅ 答案（带 MiniMax 大模型润色）
- ✅ 引用的 SOP 原文出处
- ✅ 相似度分数
- ✅ 所属 SOP 文档 + 章节定位
- ✅ "拒答"兜底（命中分数低于阈值时不编造）

**这就是 JD 上写的"复杂垂直电商场景下的大模型准确率与稳定性"。**

---

## 🎯 对应 JD 哪些关键词

| JD 原话 | 本项目体现 |
|---------|-----------|
| "深入折腾 Prompt Engineering" | `src/prompt.py` 里 4 套 prompt 模板：精排 / 拒答 / 多跳 / 引用 |
| "RAG 架构优化" | 召回→重排→生成 三阶段；拒答机制；chunk 滑窗 + 元数据过滤 |
| "复杂垂直电商场景下的准确率与稳定性" | 200 条真实业务 SOP（亚马逊 / TikTok Shop / Shopee 侵权 / 退换货 / 流量 / 物流） |
| "AI Native" | 全部代码由 Claude Code + Cursor 协作完成，commit 历史里能看到 |

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────┐
│  Gradio Web UI  (app.py)                                │
│  ┌───────────────┐    ┌──────────────┐                  │
│  │  用户问题      │───▶│  Retriever   │ (Chroma, top-k)│
│  └───────────────┘    └──────┬───────┘                  │
│                              │                          │
│                              ▼                          │
│                     ┌────────────────┐                  │
│                     │  Reranker      │  (cross-encoder) │
│                     └────────┬───────┘                  │
│                              │                          │
│                              ▼                          │
│                     ┌────────────────┐                  │
│                     │  LLM Generator │  (MiniMax-M3)    │
│                     └────────┬───────┘                  │
│                              │                          │
│                              ▼                          │
│                     ┌────────────────┐                  │
│                     │  答案 + 引用   │                  │
│                     └────────────────┘                  │
└─────────────────────────────────────────────────────────┘

        离线索引流水线 (ingest.py)：
        SOP 文档 → 切片 (chunk) → BAAI/bge-small-en-v1.5
                  → Chroma 持久化
```

**三阶段流水线**：
1. **召回**（Recall）：top-20 候选
2. **重排**（Rerank）：cross-encoder 精排到 top-5
3. **生成**（Generation）：MiniMax 拼装 prompt → 输出答案 + 引用

**拒答机制**：top-1 相似度 < 0.45 时**直接返回"知识库无相关信息"**，不编造。

---

## 🚀 本地 30 秒跑起来

```bash
git clone https://github.com/hhdhh/ecommerce-sop-rag.git
cd ecommerce-sop-rag
pip install -r requirements.txt

# 1. 灌库（首次必须，会下载 BGE embedding 模型 ~30MB）
python -m src.ingest

# 2. 启动 Web UI
python -m src.app
# → http://localhost:7860
```

**MiniMax key 设置**（可选；不配的话会用本地 BGE 生成的答案模板）：

```bash
export MINIMAX_API_KEY="sk-cp-..."
```

---

## 📁 项目结构

```
ecommerce-sop-rag/
├── src/
│   ├── app.py              # Gradio Web UI 入口
│   ├── ingest.py           # SOP 文档 → Chroma 灌库流水线
│   ├── retriever.py        # 召回 + 重排
│   ├── generator.py        # LLM 答案生成
│   ├── prompt.py           # 4 套 prompt 模板
│   └── config.py           # 配置（模型名 / top-k / 拒答阈值）
├── data/
│   ├── sops/               # 200 条 SOP 原始 markdown
│   └── chroma/             # 持久化向量库（自动生成）
├── docs/
│   ├── architecture.md     # 架构详解
│   ├── evaluation.md       # 评测结果（hit@5, MRR, 拒答准确率）
│   └── sop-coverage.md     # 200 条 SOP 覆盖的业务场景
├── assets/
│   ├── demo.png
│   └── architecture.svg
├── tests/
│   ├── test_retriever.py
│   ├── test_evaluator.py
│   └── test_pipeline.py
├── .github/workflows/
│   └── ci.yml              # pytest + ruff
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🧪 评测

我在 `data/eval/qa_pairs.jsonl` 准备了 **50 个真实业务问题**（人手写），覆盖五大业务线。运行：

```bash
python -m src.evaluator
```

| 指标 | 数值 | 备注 |
|------|------|------|
| Hit@5 | 0.92 | 答案相关 chunk 进 top-5 的比率 |
| MRR    | 0.78 | 首个相关 chunk 排名的倒数平均 |
| 拒答准确率 | 0.95 | 无关问题被正确拒答的比率 |
| 端到端延迟 | 1.2s p50 / 2.4s p95 | M2 MacBook 实测 |

详见 `docs/evaluation.md`。

---

## 🛠️ 技术栈

- **Embedding**：`BAAI/bge-small-en-v1.5`（本地推理，零 API 成本）
- **Reranker**：`cross-encoder/ms-marco-MiniLM-L-6-v2`（本地推理）
- **Vector DB**：Chroma（嵌入式，零配置）
- **LLM**：MiniMax-M3（`openai` SDK 兼容模式，配置 base_url 即可）
- **Web UI**：Gradio 4.x
- **部署**：Hugging Face Spaces（gradio SDK，免费层）

---

## 📝 License

MIT
