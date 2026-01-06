from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .schema import EvidenceUnit
from .utils import cosine

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
        min_gain: float,
        coverage_weight: float = 0.0,
        stopwords: set[str] | None = None,
    ):
        self.budget = budget_tokens
        self.beta = redundancy_beta
        self.rel_w = rel_weight
        self.min_gain = min_gain
        self.cov_w = coverage_weight
        self.stopwords = stopwords or set()

    def select(
        self,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        """
        Greedy selection with redundancy penalty.

        OLD IMPLEMENTATION (with coverage component - removed in refactoring):

        This version had query aspect coverage tracking to ensure diverse selection:

        def select(self, query: str, candidates, rel_scores, costs, vecs):
            # Extract query aspects for coverage tracking
            from .utils import tokenize
            aspects = [t for t in tokenize(query) if t not in self.stopwords and len(t) >= 3]
            aspects = list(dict.fromkeys(aspects))
            covered = set()

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

            # In selection loop, gain calculation was:
            cg = float(cov_gain(uid))
            gain = self.rel_w * rel + self.cov_w * cg - self.beta * red

            # After selecting a unit:
            if aspects:
                txt = (u.retrieval_text or "").lower()
                for a in aspects:
                    if a in txt:
                        covered.add(a)

        The coverage component (self.cov_w *ve cg) added significant positive value,
        helping orcome redundancy penalties even with low relevance scores.
        """
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

        log.info("select.start candidates=%d budget=%d", len(candidates), self.budget)

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
                red = max_sim(uid)
                gain = self.rel_w * rel - self.beta * red
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
            out.append(Selected(unit=u, rel=float(rel_scores.get(best_uid, 0.0)), cost=best_cost))

        log.info("select.done selected=%d total_cost=%d", len(out), total)
        return out

    def select_with_coverage(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        """
        Greedy selection with redundancy penalty AND query aspect coverage tracking.

        This method adds coverage gain to encourage selecting units that cover
        different aspects of the query, helping overcome redundancy penalties.
        """
        from .utils import tokenize

        # Extract query aspects for coverage tracking
        aspects = [t for t in tokenize(query) if t not in self.stopwords and len(t) >= 3]
        aspects = list(dict.fromkeys(aspects))  # Remove duplicates, preserve order
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

        log.info("select_with_coverage.start candidates=%d budget=%d aspects=%d",
                 len(candidates), self.budget, len(aspects))

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

            # Update covered aspects
            if aspects:
                txt = (u.retrieval_text or "").lower()
                for a in aspects:
                    if a in txt:
                        covered.add(a)

            out.append(Selected(unit=u, rel=float(rel_scores.get(best_uid, 0.0)), cost=best_cost))

        log.info("select_with_coverage.done selected=%d total_cost=%d covered_aspects=%d/%d",
                 len(out), total, len(covered), len(aspects))
        return out

    def select_top_until_budget(
        self,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        # Sort candidates by relevance score
        sorted_candidates = sorted(candidates, key=lambda u: rel_scores.get(u.id, 0.0), reverse=True)
        
        total = 0
        out: list[Selected] = []
        
        for u in sorted_candidates:
            uid = u.id
            c = int(costs.get(uid, 0))
            if c <= 0:
                continue
            if total + c > self.budget:
                break
            
            total += c
            out.append(Selected(unit=u, rel=float(rel_scores.get(uid, 0.0)), cost=c))
        
        log.info("select_top_until_budget.done selected=%d total_cost=%d", len(out), total)
        return out
