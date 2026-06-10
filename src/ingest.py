"""
SOP 原始数据灌库流水线：
  SOP markdown 文件 → 切片 → Embedding → Chroma 持久化

用法：
  python -m src.ingest
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SOP_DIR,
)
from src.embeddings import embed_documents, backend as embed_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest")


@dataclass
class Chunk:
    chunk_id: str
    sop_id: str
    sop_title: str
    section: str
    platform: str
    category: str
    text: str

    def to_metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "sop_id": self.sop_id,
            "sop_title": self.sop_title,
            "section": self.section,
            "platform": self.platform,
            "category": self.category,
        }


def parse_sop_file(path: Path) -> tuple[str, str, str, list[tuple[str, str]]]:
    """解析单条 SOP markdown，返回 (sop_id, title, platform, [(section, body), ...])"""
    text = path.read_text(encoding="utf-8")
    # frontmatter: --- / platform: / category:
    fm_match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not fm_match:
        raise ValueError(f"SOP 文件缺少 frontmatter: {path}")

    frontmatter, body = fm_match.group(1), fm_match.group(2)
    fm = {}
    for line in frontmatter.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()

    sop_id = fm.get("id", path.stem)
    platform = fm.get("platform", "通用")
    title = fm.get("title", path.stem)

    # 按 ## 二级标题切分章节
    sections: list[tuple[str, str]] = []
    current_section = "概览"
    current_body: list[str] = []

    for line in body.splitlines():
        if line.startswith("## "):
            if current_body:
                sections.append((current_section, "\n".join(current_body).strip()))
            current_section = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_section, "\n".join(current_body).strip()))

    if not sections:
        sections = [("概览", body.strip())]

    return sop_id, title, platform, sections


def chunk_sections(
    sop_id: str,
    sop_title: str,
    platform: str,
    sections: list[tuple[str, str]],
) -> Iterable[Chunk]:
    """对每个章节内部用 RecursiveCharacterTextSplitter 二次切片。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    for section_name, body in sections:
        if not body:
            continue
        sub_chunks = splitter.split_text(body)
        for idx, sub in enumerate(sub_chunks):
            digest = hashlib.md5(f"{sop_id}::{section_name}::{idx}::{sub}".encode()).hexdigest()[:10]
            yield Chunk(
                chunk_id=f"{sop_id}-{digest}",
                sop_id=sop_id,
                sop_title=sop_title,
                section=section_name,
                platform=platform,
                category=_infer_category(sop_title, section_name),
                text=sub.strip(),
            )


def _infer_category(title: str, section: str) -> str:
    text = f"{title} {section}"
    rules = [
        ("侵权", "知识产权与合规"),
        ("合规", "知识产权与合规"),
        ("版权", "知识产权与合规"),
        ("申诉", "知识产权与合规"),
        ("退换货", "售后与客服"),
        ("退货", "售后与客服"),
        ("客服", "售后与客服"),
        ("Listing", "Listing 优化"),
        ("标题", "Listing 优化"),
        ("关键词", "Listing 优化"),
        ("广告", "流量与广告"),
        ("投放", "流量与广告"),
        ("ROAS", "流量与广告"),
        ("物流", "物流与仓储"),
        ("FBA", "物流与仓储"),
        ("海外仓", "物流与仓储"),
        ("VAT", "财税与合规"),
        ("税务", "财税与合规"),
        ("选品", "选品与市场调研"),
        ("调研", "选品与市场调研"),
    ]
    for kw, cat in rules:
        if kw in text:
            return cat
    return "通用"


def build_chroma(chunks: list[Chunk]) -> Chroma:
    from src.embeddings import embed_documents, backend as embed_backend, _bm25, _bm25_texts, _bm25_ids

    backend = embed_backend()
    log.info("Using embedding backend: %s", backend)

    texts = [c.text for c in chunks]
    metadatas = [c.to_metadata() for c in chunks]
    ids = [c.chunk_id for c in chunks]

    # 我们自己 embed（BM25 走 pickle 持久化）
    log.info("Embedding %d chunks ...", len(chunks))
    embeddings = embed_documents(texts, ids)
    log.info("Done embedding.")

    log.info("Building Chroma at %s", CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # 用伪 embedding 写入 Chroma（保留元数据 + docs）
    # Chroma 必须有真实 embedding，我们用零向量（占位，因为我们检索走 BM25）
    dim = max((len(e) for e in embeddings if e), default=1)
    if dim < 2:
        dim = 1
    fake_embs = [[0.0] * dim for _ in chunks]

    vs = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=None,
        persist_directory=str(CHROMA_DIR),
    )
    # 清空旧 collection
    try:
        vs._collection.delete(where={})
    except Exception:
        pass
    vs._collection.add(
        embeddings=fake_embs,
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )
    log.info("Chroma collection populated with %d docs (for metadata persistence only)", len(ids))
    return vs


def main() -> None:
    if not SOP_DIR.exists():
        raise SystemExit(f"SOP 目录不存在: {SOP_DIR}，请确认 data/sops/ 已就位")

    sop_files = sorted(SOP_DIR.glob("*.md"))
    log.info("Found %d SOP files", len(sop_files))

    all_chunks: list[Chunk] = []
    for path in sop_files:
        sop_id, title, platform, sections = parse_sop_file(path)
        chunks = list(chunk_sections(sop_id, title, platform, sections))
        all_chunks.extend(chunks)
        log.info("  %s → %d chunks", path.name, len(chunks))

    log.info("Total chunks: %d", len(all_chunks))

    # 落盘一份 manifest，方便 review
    manifest = [
        {
            "chunk_id": c.chunk_id,
            "sop_id": c.sop_id,
            "sop_title": c.sop_title,
            "section": c.section,
            "platform": c.platform,
            "category": c.category,
            "text_len": len(c.text),
        }
        for c in all_chunks
    ]
    (DATA_DIR := SOP_DIR.parent).joinpath("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Wrote manifest.json (%d entries)", len(manifest))

    build_chroma(all_chunks)
    log.info("✅ Ingest done.")


if __name__ == "__main__":
    main()
