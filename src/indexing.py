from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from .schema import EvidenceUnit
from .utils import normalize_rows, rrf_fusion, tokenize

log = logging.getLogger(__name__)


class BM25Index:
    def __init__(self, ids: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.ids = ids
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        self.dl: np.ndarray = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            m: dict[str, int] = {}
            for t in d:
                m[t] = m.get(t, 0) + 1
            self.tf.append(m)
            self.dl[i] = float(len(d))
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = float(self.dl.mean()) if len(docs) else 0.0
        self.N = len(docs)

    def score(self, query_tokens: list[str]) -> np.ndarray:
        if self.N == 0:
            return np.zeros(0, dtype=np.float32)
        scores = np.zeros(self.N, dtype=np.float32)
        for t in query_tokens:
            df = self.df.get(t, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            for i, tfm in enumerate(self.tf):
                f = tfm.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * (self.dl[i] / (self.avgdl + 1e-9)))
                scores[i] += float(idf * (f * (self.k1 + 1.0)) / (denom + 1e-9))
        return scores


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

        ids = [u.id for u in units]
        docs = [tokenize(u.retrieval_text) for u in units]
        self.bm25 = BM25Index(ids, docs)

        self._dense_ids = [u.id for u in units if u.id in unit_vecs]
        if self._dense_ids:
            mat = np.vstack([unit_vecs[i] for i in self._dense_ids]).astype(np.float32)
            self._dense_mat = normalize_rows(mat)
        else:
            self._dense_mat = np.zeros((0, 1), dtype=np.float32)

    def retrieve(self, query: str, query_vec: np.ndarray, top_pages: int, top_bm25: int, top_dense: int) -> tuple[list[RetrievalHit], list[int]]:
        qtoks = tokenize(query)
        bm = self.bm25.score(qtoks)
        bm_ids = self.bm25.ids
        bm_rank = [bm_ids[i] for i in np.argsort(-bm)[:top_bm25]]

        dense_rank: list[str] = []
        if self._dense_mat.shape[0] > 0:
            q = query_vec.astype(np.float32)
            q = q / (np.linalg.norm(q) + 1e-12)
            sims = (self._dense_mat @ q).astype(np.float32)
            dense_rank = [self._dense_ids[i] for i in np.argsort(-sims)[:top_dense]]

        page_rank = self._rank_pages(query_vec, top_pages)
        page_set = set(page_rank)
        page_units = [u.id for u in self.units if u.page in page_set]

        fused = rrf_fusion([bm_rank, dense_rank, page_units], k=self.rrf_k)

        for uid in list(fused.keys()):
            u = self.by_id.get(uid)
            if not u:
                continue
            if u.type == "table_text":
                fused[uid] *= (1.0 + self.table_boost)
            elif u.type == "figure":
                fused[uid] *= (1.0 + self.figure_boost)

        self._ensure_types(fused, bm, bm_ids)

        hits = [RetrievalHit(unit_id=k, score=v) for k, v in sorted(fused.items(), key=lambda x: -x[1])]
        return hits, page_rank

    def _ensure_types(self, fused: dict[str, float], bm: np.ndarray, bm_ids: list[str]) -> None:
        if self.min_table_candidates <= 0 and self.min_figure_candidates <= 0:
            return
        table_have = sum(1 for uid in fused if self.by_id.get(uid) and self.by_id[uid].type == "table_text")
        fig_have = sum(1 for uid in fused if self.by_id.get(uid) and self.by_id[uid].type == "figure")

        if table_have < self.min_table_candidates:
            need = self.min_table_candidates - table_have
            cand = []
            for i, uid in enumerate(bm_ids):
                if uid in fused:
                    continue
                u = self.by_id.get(uid)
                if u and u.type == "table_text":
                    cand.append((uid, float(bm[i])))
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
                    cand.append(uid)
            base = min(fused.values()) if fused else 0.0
            for uid in cand[:need]:
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
