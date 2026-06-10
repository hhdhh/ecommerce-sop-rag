"""统一配置：模型名、阈值、路径。"""
from __future__ import annotations

import os
from pathlib import Path

# 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOP_DIR = DATA_DIR / "sops"
CHROMA_DIR = DATA_DIR / "chroma"
EVAL_FILE = DATA_DIR / "eval" / "qa_pairs.jsonl"

# Embedding & Rerank 模型（本地运行，零 API 费用）
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chroma collection
CHROMA_COLLECTION = "ecommerce_sops"

# 检索参数
TOP_K_RECALL = 20         # 召回数
TOP_K_RERANK = 5          # 重排后保留
RERANK_SCORE_FLOOR = -2.0 # 重排分数低于此值丢弃
REFUSE_THRESHOLD = 0.45   # top-1 相似度 < 此值则拒答

# 切片参数
CHUNK_SIZE = 400          # 字符
CHUNK_OVERLAP = 60

# LLM（MiniMax，OpenAI 兼容）
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.chat/v1")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-Text-01")

# 是否启用 LLM（无 key 时降级为 extractive 模式）
USE_LLM = bool(MINIMAX_API_KEY)

# Gradio
GRADIO_SERVER_PORT = int(os.getenv("PORT", "7860"))
