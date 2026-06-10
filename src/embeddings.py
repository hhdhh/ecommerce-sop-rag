"""
Embedding 接口：
  - 默认 BM25（完全离线，零网络，零下载）
  - 顺序：BM25 召回 → reranker

面试演示推荐用 BM25：
  - 启动 0 秒（无网络依赖）
  - 200 SOP 灌库 < 1 秒
  - 检索效果远好于 TF-IDF（处理短查询 + 短文档 + 重复模板更鲁棒）
"""
from __future__ import annotations

import logging
import pickle
import re
import threading
from pathlib import Path

import numpy as np

from src.config import CHROMA_DIR, EMBEDDING_MODEL

log = logging.getLogger("embeddings")

# 默认 backend：bm25
_BACKEND: str = "bm25"
_bge = None

_LOCK = threading.Lock()
_BM25_PATH = CHROMA_DIR / "_bm25.pkl"

_bm25 = None
_bm25_texts: list[str] = []
_bm25_ids: list[str] = []


def backend() -> str:
    return _BACKEND


def use_bge() -> bool:
    """可选切换到 BGE（需要本地有缓存）"""
    global _BACKEND, _bge
    with _LOCK:
        if _BACKEND == "bge" and _bge is not None:
            return True
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(EMBEDDING_MODEL, cache_folder=None, local_files_only=True)
            _bge = HuggingFaceBgeEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu", "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True},
            )
            _BACKEND = "bge"
            log.info("✅ BGE backend enabled (offline)")
            return True
        except Exception as e:  # noqa: BLE001
            log.info("BGE not available: %s. Staying on BM25.", e)
            return False


# =============================================================================
# 占位 embed_* 接口（保持接口兼容，retriever.py / ingest.py 调用统一）
# 真实实现见下方 BM25 专用接口
# =============================================================================


def embed_documents(texts: list[str], ids: list[str] | None = None) -> list[list[float]]:
    """灌库接口：实际走 BM25 build_index（接口兼容）"""
    return _bm25_build_index(texts, ids)


def embed_query(text: str) -> list[float]:
    """query 接口：实际走 BM25 score（接口兼容）"""
    scores = _bm25_score(text)
    # 兼容 retriever 的 Chroma 接口（不会真的用这个向量去查 chroma）
    return scores


# =============================================================================
# BM25 实现
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """中英文混合分词：英文按词，中文按字 + 二元"""
    text = text.lower()
    # 英文词
    en_tokens = re.findall(r"[a-z0-9]+", text)
    # 中文：单字 + 二元
    cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cn_unigrams = cn_chars
    cn_bigrams = [cn_chars[i] + cn_chars[i+1] for i in range(len(cn_chars) - 1)]
    return en_tokens + cn_unigrams + cn_bigrams


def _bm25_build_index(texts: list[str], ids: list[str] | None = None) -> list[list[float]]:
    from rank_bm25 import BM25Okapi
    global _bm25, _bm25_texts, _bm25_ids
    log.info("Building BM25 index on %d documents ...", len(texts))
    tokenized_corpus = [_tokenize(t) for t in texts]
    _bm25 = BM25Okapi(tokenized_corpus)
    _bm25_texts = texts
    _bm25_ids = ids or [str(i) for i in range(len(texts))]
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BM25_PATH, "wb") as f:
        pickle.dump({"bm25": _bm25, "texts": _bm25_texts, "ids": _bm25_ids}, f)
    log.info("BM25 index built: %d docs", len(texts))
    return [[0.0] for _ in texts]


def _bm25_score(query: str) -> list[float]:
    global _bm25
    if _bm25 is None:
        if not _load_bm25():
            raise RuntimeError("BM25 not built. Call embed_documents first.")
    q_tokens = _tokenize(query)
    scores = _bm25.get_scores(q_tokens)
    return scores.tolist()


def _load_bm25() -> bool:
    global _bm25, _bm25_texts, _bm25_ids
    if _BM25_PATH.exists():
        with open(_BM25_PATH, "rb") as f:
            data = pickle.load(f)
        # 必须用 global 变量承接
        _bm25 = data["bm25"]
        _bm25_texts = data["texts"]
        _bm25_ids = data["ids"]
        log.info("Loaded BM25 from cache: %d docs", len(_bm25_texts))
        return True
    return False


def bm25_corpus_size() -> int:
    return len(_bm25_texts) if _bm25_texts else 0
