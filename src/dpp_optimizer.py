from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .schema import EvidenceUnit
from .utils import cosine, tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Selected:
    unit: EvidenceUnit
    rel: float
    cost: int


class BudgetedSelector:
    def __init__(
        self,
        budget_tokens: int,
        redundancy_beta: float,
        rel_weight: float,
        coverage_weight: float,
        min_gain: float,
        stopwords: set[str],
    ):
        self.budget = budget_tokens
        self.beta = redundancy_beta
        self.rel_w = rel_weight
        self.cov_w = coverage_weight
        self.min_gain = min_gain
        self.stopwords = stopwords

    def select(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        aspects = [t for t in tokenize(query) if t not in self.stopwords and len(t) >= 3]
        aspects = list(dict.fromkeys(aspects))
        covered: set[str] = set()

        cand_by_id = {u.id: u for u in candidates}
        remaining = list(cand_by_id.keys())
        chosen: list[str] = []

        total = 0
        out: list[Selected] = []

        def max_sim(uid: str) -> float:
            if uid not in vecs or not chosen:
                return 0.0
            v = vecs[uid]
            best = 0.0
            for sid in chosen:
                if sid in vecs:
                    best = max(best, cosine(v, vecs[sid]))
            return best

        def cov_gain(uid: str) -> int:
            u = cand_by_id[uid]
            if not aspects:
                return 0
            txt = (u.retrieval_text or "").lower()
            g = 0
            for a in aspects:
                if a not in covered and a in txt:
                    g += 1
            return g

        log.info("select.start candidates=%d budget=%d aspects=%d", len(candidates), self.budget, len(aspects))

        while remaining and total < self.budget:
            best_uid = None
            best_ratio = -1e18
            best_gain = 0.0
            best_cost = 0

            for uid in remaining:
                c = int(costs.get(uid, 0))
                if c <= 0 or total + c > self.budget:
                    continue
                rel = float(rel_scores.get(uid, 0.0))
                cg = float(cov_gain(uid))
                red = max_sim(uid)
                gain = self.rel_w * rel + self.cov_w * cg - self.beta * red
                if gain <= self.min_gain:
                    continue
                ratio = gain / max(1, c)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_uid = uid
                    best_gain = gain
                    best_cost = c

            if best_uid is None:
                break

            u = cand_by_id[best_uid]
            chosen.append(best_uid)
            remaining.remove(best_uid)
            total += best_cost
            if aspects:
                txt = (u.retrieval_text or "").lower()
                for a in aspects:
                    if a in txt:
                        covered.add(a)
            out.append(Selected(unit=u, rel=float(rel_scores.get(best_uid, 0.0)), cost=best_cost))

        log.info("select.done selected=%d total_cost=%d", len(out), total)
        return out
