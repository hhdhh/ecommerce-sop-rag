"""测试切片、解析等不依赖外部模型的逻辑。"""
from pathlib import Path

import pytest

from src.ingest import parse_sop_file, chunk_sections, _infer_category


DATA = Path(__file__).resolve().parent.parent / "data" / "sops"


def test_parse_sop_file():
    files = list(DATA.glob("*.md"))
    assert len(files) >= 100, f"expected >=100 SOPs, got {len(files)}"

    sample = files[0]
    sop_id, title, platform, sections = parse_sop_file(sample)
    assert sop_id.startswith("SOP-")
    assert title
    assert platform in {"Amazon", "TikTok Shop", "Shopee", "Temu", "通用"}
    assert len(sections) >= 2


def test_chunk_sections_basic():
    files = list(DATA.glob("*.md"))
    sample = files[0]
    sop_id, title, platform, sections = parse_sop_file(sample)
    chunks = list(chunk_sections(sop_id, title, platform, sections))
    assert len(chunks) >= 1
    for c in chunks:
        assert c.chunk_id
        assert c.text
        assert c.sop_id == sop_id
        assert c.platform == platform


def test_infer_category():
    assert _infer_category("亚马逊 Listing 优化", "标题结构") == "Listing 优化"
    assert _infer_category("亚马逊广告", "出价策略") == "流量与广告"
    assert _infer_category("Shopee 退货", "退款模板") == "售后与客服"
    assert _infer_category("亚马逊侵权", "申诉") == "知识产权与合规"
