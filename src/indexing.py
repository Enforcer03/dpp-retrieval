from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .schema import EvidenceUnit
from .utils import normalize_rows, rrf_fusion

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalHit:
    unit_id: str
    score: float


class CombinedRetriever:
    def __init__(
        self,
        units: list[EvidenceUnit],
        unit_vecs: dict[str, np.ndarray],
        page_vecs: dict[int, np.ndarray],
        rrf_k: int,
        table_boost: float = 0.0,
        figure_boost: float = 0.0,
        min_table_candidates: int = 0,
        min_figure_candidates: int = 0,
    ):
        self.units = units
        self.by_id = {u.id: u for u in units}
        self.unit_vecs = unit_vecs
        self.page_vecs = page_vecs
        self.rrf_k = rrf_k
        self.table_boost = float(table_boost)
        self.figure_boost = float(figure_boost)
        self.min_table_candidates = int(min_table_candidates)
        self.min_figure_candidates = int(min_figure_candidates)

        self._dense_ids = [u.id for u in units if u.id in unit_vecs]
        if self._dense_ids:
            mat = np.vstack([unit_vecs[i] for i in self._dense_ids]).astype(np.float32)
            self._dense_mat = normalize_rows(mat)
        else:
            self._dense_mat = np.zeros((0, 1), dtype=np.float32)

    def retrieve(self, query_vec: np.ndarray, top_pages: int, top_dense: int) -> tuple[list[RetrievalHit], list[int]]:
        """Retrieval using dense embeddings + page ranking."""

        # Dense retrieval via cosine similarity
        dense_rank: list[str] = []
        dense_sims = np.array([])
        if self._dense_mat.shape[0] > 0:
            q = query_vec.astype(np.float32)
            q = q / (np.linalg.norm(q) + 1e-12)
            dense_sims = (self._dense_mat @ q).astype(np.float32)
            dense_rank = [self._dense_ids[i] for i in np.argsort(-dense_sims)[:top_dense]]

        # Page-level ranking
        page_rank = self._rank_pages(query_vec, top_pages)
        page_set = set(page_rank)
        page_units = [u.id for u in self.units if u.page in page_set]

        # RRF fusion of dense results + page-filtered units (2 lists instead of 3)
        fused = rrf_fusion([dense_rank, page_units], k=self.rrf_k)

        # Apply type-specific boosts
        for uid in list(fused.keys()):
            u = self.by_id.get(uid)
            if not u:
                continue
            if u.type == "table_text":
                fused[uid] *= (1.0 + self.table_boost)
            elif u.type == "figure":
                fused[uid] *= (1.0 + self.figure_boost)

        # Ensure minimum type candidates using dense scores
        self._ensure_types(fused, dense_rank, dense_sims)

        hits = [RetrievalHit(unit_id=k, score=v) for k, v in sorted(fused.items(), key=lambda x: -x[1])]
        return hits, page_rank

    def _ensure_types(self, fused: dict[str, float], dense_rank: list[str], dense_sims: np.ndarray) -> None:
        """Ensure minimum number of table/figure candidates using dense scores."""
        if self.min_table_candidates <= 0 and self.min_figure_candidates <= 0:
            return

        table_have = sum(
            1 for uid in fused
            if self.by_id.get(uid) and self.by_id[uid].type == "table_text"
        )
        fig_have = sum(
            1 for uid in fused
            if self.by_id.get(uid) and self.by_id[uid].type == "figure"
        )

        # Build score lookup from dense_rank
        score_lookup = {}
        if len(dense_sims) > 0:
            for i, uid in enumerate(self._dense_ids):
                score_lookup[uid] = float(dense_sims[i])

        if table_have < self.min_table_candidates:
            need = self.min_table_candidates - table_have
            cand = []
            for uid in self._dense_ids:
                if uid in fused:
                    continue
                u = self.by_id.get(uid)
                if u and u.type == "table_text":
                    cand.append((uid, score_lookup.get(uid, 0.0)))
            cand.sort(key=lambda x: -x[1])
            base = min(fused.values()) if fused else 0.0
            for uid, sc in cand[:need]:
                fused[uid] = max(base * 0.9, sc + 1e-6) * (1.0 + self.table_boost)

        if fig_have < self.min_figure_candidates:
            need = self.min_figure_candidates - fig_have
            cand = []
            for uid in self.by_id.keys():
                if uid in fused:
                    continue
                u = self.by_id.get(uid)
                if u and u.type == "figure":
                    score = score_lookup.get(uid, 0.0)
                    cand.append((uid, score))
            cand.sort(key=lambda x: -x[1])
            base = min(fused.values()) if fused else 0.0
            for uid, _ in cand[:need]:
                fused[uid] = base * 0.9 + 1e-6

    def _rank_pages(self, query_vec: np.ndarray, top_pages: int) -> list[int]:
        if not self.page_vecs:
            return []
        q = query_vec.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        items = []
        for p, v in self.page_vecs.items():
            vv = v.astype(np.float32)
            vv = vv / (np.linalg.norm(vv) + 1e-12)
            items.append((p, float(vv @ q)))
        items.sort(key=lambda x: -x[1])
        return [p for p, _ in items[:top_pages]]
