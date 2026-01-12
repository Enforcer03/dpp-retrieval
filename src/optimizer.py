from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

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
    Budgeted selection with modular objectives + multiple optimizers.
    - objective controls marginal gain computation for greedy/cost_norm (and shared quality for DPP).
    """

    # optimizers (selection algorithms)
    METHODS = ("greedy", "topk", "greedy_cov", "cost_norm", "dpp")

    # objectives (marginal gain models)
    OBJECTIVES = ("mmr", "graphcut", "facility")  # extend safely

    def __init__(
        self,
        budget_tokens: int,
        redundancy_beta: float,
        rel_weight: float,
        coverage_weight: float,
        min_gain: float,
        stopwords: set[str],
        *,
        objective: str = "mmr",
        rep_weight: float = 0.15,  # only used by facility-location objective
    ):
        self.budget = budget_tokens
        self.beta = float(redundancy_beta)
        self.rel_w = float(rel_weight)
        self.cov_w = float(coverage_weight)
        self.min_gain = float(min_gain)
        self.stopwords = stopwords

        if objective not in self.OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective}. Expected one of: {self.OBJECTIVES}")
        self.objective = objective
        self.rep_w = float(rep_weight)

    # -------------------------
    # Public dispatch
    # -------------------------
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
        if method == "greedy":
            return self.select(query, candidates, rel_scores, costs, vecs)
        if method == "topk":
            return self.select_top_until_budget(candidates, rel_scores, costs)
        if method == "greedy_cov":
            return self.select(coverage_query or query, candidates, rel_scores, costs, vecs)
        if method == "cost_norm":
            return self.select_cost_normalized(query, candidates, rel_scores, costs, vecs)
        if method == "dpp":
            return self.select_dpp(query, candidates, rel_scores, costs, vecs)
        raise ValueError(f"Unknown selection method: {method}. Expected one of: {self.METHODS}")

    # -------------------------
    # Helpers: aspects + coverage
    # -------------------------
    def _aspects(self, query: str) -> list[str]:
        a = [t for t in tokenize(query) if t not in self.stopwords and len(t) >= 3]
        return list(dict.fromkeys(a))

    @staticmethod
    def _cov_gain(u: EvidenceUnit, aspects: list[str], covered: set[str]) -> int:
        if not aspects:
            return 0
        txt = (u.retrieval_text or "").lower()
        return sum(1 for a in aspects if a not in covered and a in txt)

    @staticmethod
    def _update_covered(u: EvidenceUnit, aspects: list[str], covered: set[str]) -> None:
        if not aspects:
            return
        txt = (u.retrieval_text or "").lower()
        for a in aspects:
            if a in txt:
                covered.add(a)

    # -------------------------
    # Helpers: similarities
    # -------------------------
    @staticmethod
    def _max_sim(uid: str, chosen: list[str], vecs: dict[str, np.ndarray]) -> float:
        v = vecs.get(uid)
        if v is None or not chosen:
            return 0.0
        best = 0.0
        for sid in chosen:
            sv = vecs.get(sid)
            if sv is not None:
                best = max(best, cosine(v, sv))
        return best

    @staticmethod
    def _sum_sim(uid: str, chosen: list[str], vecs: dict[str, np.ndarray]) -> float:
        v = vecs.get(uid)
        if v is None or not chosen:
            return 0.0
        s = 0.0
        for sid in chosen:
            sv = vecs.get(sid)
            if sv is not None:
                s += cosine(v, sv)
        return s

    @staticmethod
    def _facility_rep_gain(
        uid: str,
        all_ids: list[str],
        cur_rep: dict[str, float],
        vecs: dict[str, np.ndarray],
    ) -> float:
        """Marginal representativeness: sum_j max(0, sim(uid,j) - cur_rep[j])."""
        v = vecs.get(uid)
        if v is None:
            return 0.0
        g = 0.0
        for j in all_ids:
            w = vecs.get(j)
            if w is None:
                continue
            s = cosine(v, w)
            prev = cur_rep.get(j, 0.0)
            if s > prev:
                g += (s - prev)
        return g

    @staticmethod
    def _facility_update_rep(
        uid: str,
        all_ids: list[str],
        cur_rep: dict[str, float],
        vecs: dict[str, np.ndarray],
    ) -> None:
        v = vecs.get(uid)
        if v is None:
            return
        for j in all_ids:
            w = vecs.get(j)
            if w is None:
                continue
            s = cosine(v, w)
            if s > cur_rep.get(j, 0.0):
                cur_rep[j] = s

    # -------------------------
    # Modular objective: marginal gain
    # -------------------------
    def _quality(
        self,
        u: EvidenceUnit,
        *,
        rel_scores: dict[str, float],
        aspects: list[str],
        covered: set[str],
    ) -> float:
        rel = float(rel_scores.get(u.id, 0.0))
        cg = float(self._cov_gain(u, aspects, covered))
        return self.rel_w * rel + self.cov_w * cg

    def _marginal_gain(
        self,
        uid: str,
        *,
        cand_by_id: dict[str, EvidenceUnit],
        rel_scores: dict[str, float],
        aspects: list[str],
        covered: set[str],
        chosen: list[str],
        vecs: dict[str, np.ndarray],
        all_ids: list[str],
        rep_cache: dict[str, float] | None,
    ) -> float:
        u = cand_by_id[uid]
        q = self._quality(u, rel_scores=rel_scores, aspects=aspects, covered=covered)

        if self.objective == "mmr":
            return q - self.beta * self._max_sim(uid, chosen, vecs)

        if self.objective == "graphcut":
            return q - self.beta * self._sum_sim(uid, chosen, vecs)

        # facility-location style representativeness + optional mild redundancy control
        if self.objective == "facility":
            rep = self._facility_rep_gain(uid, all_ids, rep_cache or {}, vecs)
            red = self._max_sim(uid, chosen, vecs)  # keeps near-duplicates in check
            return q + self.rep_w * rep - (0.5 * self.beta) * red

        raise RuntimeError("unreachable")

    # -------------------------
    # Optimizers
    # -------------------------
    def _greedy_core(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
        *,
        cost_normalized: bool,
    ) -> list[Selected]:
        aspects = self._aspects(query)
        covered: set[str] = set()

        cand_by_id = {u.id: u for u in candidates}
        all_ids = list(cand_by_id.keys())
        remaining = all_ids.copy()
        chosen: list[str] = []

        total = 0
        out: list[Selected] = []

        rep_cache: dict[str, float] | None = {} if self.objective == "facility" else None

        log.info(
            "select.start method=%s objective=%s candidates=%d budget=%d aspects=%d",
            "cost_norm" if cost_normalized else "greedy",
            self.objective,
            len(candidates),
            self.budget,
            len(aspects),
        )

        while remaining and total < self.budget:
            best_uid, best_score, best_cost = None, -1e18, 0

            for uid in remaining:
                c = int(costs.get(uid, 0))
                if c <= 0 or total + c > self.budget:
                    continue

                gain = self._marginal_gain(
                    uid,
                    cand_by_id=cand_by_id,
                    rel_scores=rel_scores,
                    aspects=aspects,
                    covered=covered,
                    chosen=chosen,
                    vecs=vecs,
                    all_ids=all_ids,
                    rep_cache=rep_cache,
                )
                if gain <= self.min_gain:
                    continue

                score = (gain / max(1, c)) if cost_normalized else gain
                if score > best_score:
                    best_uid, best_score, best_cost = uid, score, c

            if best_uid is None:
                break

            u = cand_by_id[best_uid]
            chosen.append(best_uid)
            remaining.remove(best_uid)
            total += best_cost

            self._update_covered(u, aspects, covered)
            if rep_cache is not None:
                self._facility_update_rep(best_uid, all_ids, rep_cache, vecs)

            out.append(Selected(unit=u, rel=float(rel_scores.get(best_uid, 0.0)), cost=best_cost))

        log.info("select.done selected=%d total_cost=%d covered=%d/%d", len(out), total, len(covered), len(aspects))
        return out

    def select(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        """MAIN: greedy by absolute marginal gain (objective-controlled)."""
        return self._greedy_core(query, candidates, rel_scores, costs, vecs, cost_normalized=False)

    def select_cost_normalized(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
        vecs: dict[str, np.ndarray],
    ) -> list[Selected]:
        """BASELINE: greedy by (marginal gain / cost)."""
        return self._greedy_core(query, candidates, rel_scores, costs, vecs, cost_normalized=True)

    def select_top_until_budget(
        self,
        candidates: list[EvidenceUnit],
        rel_scores: dict[str, float],
        costs: dict[str, int],
    ) -> list[Selected]:
        """BASELINE: pure relevance until budget."""
        sorted_candidates = sorted(candidates, key=lambda u: rel_scores.get(u.id, 0.0), reverse=True)
        total, out = 0, []
        for u in sorted_candidates:
            c = int(costs.get(u.id, 0))
            if c <= 0:
                continue  # skip non-positive costs
            if total + c > self.budget:
                break  # stop on budget overflow
            total += c
            out.append(Selected(unit=u, rel=float(rel_scores.get(u.id, 0.0)), cost=c))
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
        BASELINE: DPP-greedy MAP under budget using shared quality q(u)=rel_w*rel + cov_w*cov_gain.
        (Objective switch does not apply; DPP defines its own diversity geometry.)
        """
        aspects = self._aspects(query)
        covered: set[str] = set()

        cand_by_id = {u.id: u for u in candidates}
        remaining = list(cand_by_id.keys())

        total, out = 0, []
        orth: list[np.ndarray] = []
        eps = 1e-12

        while remaining and total < self.budget:
            best_uid, best_score, best_cost, best_res = None, -1e18, 0, None

            for uid in remaining:
                c = int(costs.get(uid, 0))
                if c <= 0 or total + c > self.budget:
                    continue
                v = vecs.get(uid)
                if v is None:
                    continue

                q = self._quality(cand_by_id[uid], rel_scores=rel_scores, aspects=aspects, covered=covered)
                if q <= self.min_gain:
                    continue

                f = np.sqrt(max(q, eps)) * v
                r = f.copy()
                for b in orth:
                    r -= float(np.dot(b, r)) * b
                score = float(np.dot(r, r))

                if score > best_score:
                    best_uid, best_score, best_cost, best_res = uid, score, c, r

            if best_uid is None:
                break

            if best_res is not None:
                n = float(np.linalg.norm(best_res))
                if n > 1e-10:
                    orth.append(best_res / n)

            u = cand_by_id[best_uid]
            remaining.remove(best_uid)
            total += best_cost
            self._update_covered(u, aspects, covered)
            out.append(Selected(unit=u, rel=float(rel_scores.get(best_uid, 0.0)), cost=best_cost))

        return out
