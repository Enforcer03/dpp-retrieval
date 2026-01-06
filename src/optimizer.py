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
    """
    Budgeted selection with multiple strategies for ablation studies.

    MAIN METHOD: select() uses absolute gain (no cost normalization)
    This avoids bias toward many cheap fragments over few complete answers.
    """

    # Central list of methods/baselines
    METHODS = ("greedy", "topk", "greedy_cov", "cost_norm", "dpp")

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

    def select_method(
        self,
        method: str,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
        *,
        coverage_query: str | None = None,
    ) -> list[Selected]:
        """
        Dispatcher so main/batch can loop cleanly.
        - query: default query used for greedy/cost_norm/dpp
        - coverage_query: optional query used for greedy_cov
        """
        if method == "greedy":
            return self.select(query=query, candidates=candidates, rel_scores=rel_scores, costs=costs, vecs=vecs)
        if method == "topk":
            return self.select_top_until_budget(candidates=candidates, rel_scores=rel_scores, costs=costs, vecs=vecs)
        if method == "greedy_cov":
            q = coverage_query if coverage_query is not None else query
            return self.select(query=q, candidates=candidates, rel_scores=rel_scores, costs=costs, vecs=vecs)
        if method == "cost_norm":
            return self.select_cost_normalized(
                query=query, candidates=candidates, rel_scores=rel_scores, costs=costs, vecs=vecs
            )
        if method == "dpp":
            return self.select_dpp(query=query, candidates=candidates, rel_scores=rel_scores, costs=costs, vecs=vecs)
        raise ValueError(f"Unknown selection method: {method}. Expected one of: {self.METHODS}")

    def select(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        """
        MAIN METHOD: Greedy selection with ABSOLUTE GAIN (no cost normalization).

        Combines:
        - Relevance (rel_weight * rel)
        - Coverage (coverage_weight * covered_aspects)
        - Diversity (-redundancy_beta * max_similarity)

        Selects by ABSOLUTE GAIN, not cost-normalized ratio.
        This prevents bias toward many cheap fragments.
        """
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

        log.info(
            "select.start candidates=%d budget=%d aspects=%d mode=absolute_gain",
            len(candidates),
            self.budget,
            len(aspects),
        )

        while remaining and total < self.budget:
            best_uid = None
            best_gain = -1e18
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

                if gain > best_gain:
                    best_gain = gain
                    best_uid = uid
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

        log.info(
            "select.done selected=%d total_cost=%d covered_aspects=%d/%d",
            len(out),
            total,
            len(covered),
            len(aspects),
        )
        return out

    def select_with_coverage(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        """Alias for select() - kept for backwards compatibility."""
        return self.select(query, candidates, rel_scores, costs, vecs)

    def select_cost_normalized(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        """
        BASELINE: Cost-normalized selection (for ablation comparison).
        OLD method that biases toward cheap fragments.
        """
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

        log.info(
            "select_cost_normalized.start candidates=%d budget=%d aspects=%d mode=cost_ratio",
            len(candidates),
            self.budget,
            len(aspects),
        )

        while remaining and total < self.budget:
            best_uid = None
            best_ratio = -1e18
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

        log.info(
            "select_cost_normalized.done selected=%d total_cost=%d covered_aspects=%d/%d",
            len(out),
            total,
            len(covered),
            len(aspects),
        )
        return out

    def select_top_until_budget(
        self,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        """
        BASELINE: Simple top-k by relevance.
        No diversity, no coverage - pure greedy relevance.
        """
        sorted_candidates = sorted(candidates, key=lambda u: rel_scores.get(u.id, 0.0), reverse=True)

        total = 0
        out: list[Selected] = []

        log.info("select_top_until_budget.start candidates=%d budget=%d", len(candidates), self.budget)

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

    def select_dpp(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        """
        BASELINE: DPP-style greedy MAP selection under budget.

        Uses SAME quality objective (relevance + coverage) and gets diversity from DPP geometry:
          q_i = rel_w*rel + cov_w*cov_gain
          f_i = sqrt(max(q_i, eps)) * v_i   (v_i is unit embedding)
        Greedy step: pick item with largest squared residual ||f_i - Proj_span(selected) f_i||^2.
        """
        aspects = [t for t in tokenize(query) if t not in self.stopwords and len(t) >= 3]
        aspects = list(dict.fromkeys(aspects))
        covered: set[str] = set()

        cand_by_id = {u.id: u for u in candidates}
        remaining = list(cand_by_id.keys())

        total = 0
        out: list[Selected] = []

        orth: list[np.ndarray] = []
        eps = 1e-12

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

        log.info(
            "select_dpp.start candidates=%d budget=%d aspects=%d mode=dpp_greedy",
            len(candidates),
            self.budget,
            len(aspects),
        )

        while remaining and total < self.budget:
            best_uid = None
            best_score = -1e18
            best_cost = 0
            best_residual: np.ndarray | None = None

            for uid in remaining:
                c = int(costs.get(uid, 0))
                if c <= 0 or total + c > self.budget:
                    continue
                v = vecs.get(uid)
                if v is None:
                    continue

                rel = float(rel_scores.get(uid, 0.0))
                cg = float(cov_gain(uid))
                q = self.rel_w * rel + self.cov_w * cg
                if q <= self.min_gain:
                    continue

                f = np.sqrt(max(q, eps)) * v
                r = f.copy()
                for b in orth:
                    r -= float(np.dot(b, r)) * b

                score = float(np.dot(r, r))
                if score > best_score:
                    best_score = score
                    best_uid = uid
                    best_cost = c
                    best_residual = r

            if best_uid is None:
                break

            if best_residual is not None:
                n = float(np.linalg.norm(best_residual))
                if n > 1e-10:
                    orth.append(best_residual / n)

            u = cand_by_id[best_uid]
            remaining.remove(best_uid)
            total += best_cost

            if aspects:
                txt = (u.retrieval_text or "").lower()
                for a in aspects:
                    if a in txt:
                        covered.add(a)

            out.append(Selected(unit=u, rel=float(rel_scores.get(best_uid, 0.0)), cost=best_cost))

        log.info(
            "select_dpp.done selected=%d total_cost=%d covered_aspects=%d/%d",
            len(out),
            total,
            len(covered),
            len(aspects),
        )
        return out
