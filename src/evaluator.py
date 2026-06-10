"""评测：跑 50 条 QA pair，计算 Hit@5 / MRR / 拒答准确率。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import EVAL_FILE
from src.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluator")


def main() -> None:
    if not EVAL_FILE.exists():
        log.warning("Eval file not found: %s", EVAL_FILE)
        # 生成一个示例
        EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        sample = [
            {"q": "亚马逊 Listing 被投诉版权侵权怎么办？", "relevant_sop_id": "SOP-知-AMA", "should_refuse": False},
            {"q": "TikTok Shop 退货率太高怎么分析？", "relevant_sop_id": "SOP-客-Tik", "should_refuse": False},
            {"q": "今天天气怎么样？", "relevant_sop_id": None, "should_refuse": True},
        ]
        EVAL_FILE.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in sample), encoding="utf-8")
        log.info("Generated sample eval file, please extend to 50+ entries")

    retriever = Retriever()

    pairs = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    log.info("Eval set: %d questions", len(pairs))

    hit_at_5 = 0
    mrr_sum = 0.0
    refuse_correct = 0
    refuse_total = sum(1 for p in pairs if p.get("should_refuse"))

    for p in pairs:
        q = p["q"]
        relevant = p.get("relevant_sop_id")
        should_refuse = p.get("should_refuse", False)
        hits = retriever.search(q)
        refused = retriever.should_refuse(hits)

        if should_refuse:
            if refused:
                refuse_correct += 1
            continue

        # Hit@5
        top5_ids = [h.metadata.get("sop_id", "")[:8] for h in hits[:5]]
        if relevant and any(rel in sop_id for sop_id in top5_ids for rel in [relevant[:8]]):
            hit_at_5 += 1
            # MRR
            for rank, sop_id in enumerate(top5_ids, start=1):
                if relevant[:8] in sop_id:
                    mrr_sum += 1.0 / rank
                    break

    n_relevant = len(pairs) - refuse_total
    log.info("=" * 50)
    log.info("Hit@5: %.3f (%d/%d)", hit_at_5 / max(n_relevant, 1), hit_at_5, n_relevant)
    log.info("MRR:   %.3f", mrr_sum / max(n_relevant, 1))
    log.info("Refuse accuracy: %.3f (%d/%d)", refuse_correct / max(refuse_total, 1), refuse_correct, refuse_total)


if __name__ == "__main__":
    main()
