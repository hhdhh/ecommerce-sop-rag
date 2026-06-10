"""
检索器：
  - BM25 召回 + 排序（默认，单阶段）
  - 可选：cross-encoder rerank（需要本地缓存）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import (
    CHROMA_DIR,
    TOP_K_RERANK,
    REFUSE_THRESHOLD,
)
from src import embeddings  # 用 module 引用，避免 from import 的快照问题
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("retriever")


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    score: float
    recall_score: float
    metadata: dict


class Retriever:
    _reranker = None
    _use_ce: bool = False

    def __init__(self) -> None:
        # 触发 BM25 索引加载（如果 pickle 在）
        if embeddings._bm25 is None:
            embeddings._load_bm25()
        n = len(embeddings._bm25_texts) if embeddings._bm25_texts else 0
        if n == 0:
            raise RuntimeError(
                "BM25 索引未构建。请先运行：python -m src.ingest"
            )
        log.info("Retriever ready with %d docs", n)

    def enable_reranker(self) -> bool:
        if Retriever._use_ce and Retriever._reranker is not None:
            return True
        try:
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder
            Retriever._reranker = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                model_kwargs={"local_files_only": True},
            )
            Retriever._use_ce = True
            log.info("Cross-encoder reranker enabled")
            return True
        except Exception as e:  # noqa: BLE001
            log.info("Cross-encoder unavailable: %s", e)
            return False

    def search(self, query: str, top_k: int = TOP_K_RERANK) -> list[RetrievalHit]:
        scores = embeddings._bm25_score(query)
        scores_arr = np.array(scores)
        if len(scores_arr) == 0:
            return []

        # top_k
        if len(scores_arr) <= top_k:
            top_idx = np.argsort(-scores_arr)
        else:
            top_idx_unsorted = np.argpartition(-scores_arr, top_k)[:top_k]
            top_idx = top_idx_unsorted[np.argsort(-scores_arr[top_idx_unsorted])]

        max_s = float(scores_arr[top_idx[0]]) if len(top_idx) > 0 and scores_arr[top_idx[0]] > 0 else 1.0
        min_s = float(min(scores_arr[top_idx])) if len(top_idx) > 0 else 0.0
        denom = (max_s - min_s) if max_s > min_s else 1.0

        # 元数据（懒加载一次）
        if not hasattr(Retriever, "_chroma_vs"):
            Retriever._chroma_vs = self._open_chroma()
        metadatas = self._fetch_metadatas(top_idx)

        hits: list[RetrievalHit] = []
        for idx in top_idx:
            raw = float(scores_arr[idx])
            norm = (raw - min_s) / denom if denom > 0 else 0.0
            hits.append(
                RetrievalHit(
                    chunk_id=embeddings._bm25_ids[idx] if idx < len(embeddings._bm25_ids) else str(idx),
                    text=embeddings._bm25_texts[idx],
                    score=raw,
                    recall_score=norm,
                    metadata=metadatas.get(int(idx), {}),
                )
            )
        return hits

    def _open_chroma(self):
        try:
            from langchain_community.vectorstores import Chroma
            from src.config import CHROMA_COLLECTION
            return Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=None,
                persist_directory=str(CHROMA_DIR),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Open chroma failed: %s", e)
            return None

    def _fetch_metadatas(self, indices) -> dict[int, dict]:
        vs = getattr(Retriever, "_chroma_vs", None)
        if vs is None:
            return {int(idx): {} for idx in indices}
        try:
            chroma_ids = [embeddings._bm25_ids[int(idx)] for idx in indices if int(idx) < len(embeddings._bm25_ids)]
            res = vs._collection.get(ids=chroma_ids, include=["metadatas"])
            result = {}
            for cid, meta in zip(res.get("ids", []), res.get("metadatas", [])):
                try:
                    idx = embeddings._bm25_ids.index(cid)
                    result[idx] = meta or {}
                except ValueError:
                    pass
            return result
        except Exception as e:  # noqa: BLE001
            log.debug("Fetch metadatas failed: %s", e)
            return {int(idx): {} for idx in indices}

    def should_refuse(self, hits: list[RetrievalHit]) -> bool:
        if not hits:
            return True
        return hits[0].recall_score < REFUSE_THRESHOLD
