from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

from .openai_client import OpenAIChatClient
from .utils import cosine

if TYPE_CHECKING:
    from .embedder import MultiModalEmbedder

log = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"\b(TODO|TBD|INSERT)\b|(\[[^\]]{0,60}\])|(<[^>]{1,60}>)", re.I)


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = float(np.linalg.norm(vec) + 1e-12)
    return vec / norm


def _truncate(s: str, max_chars: int) -> str:
    """Truncate string to max_chars with ellipsis."""
    s = (s or "").strip()
    if max_chars and len(s) > max_chars:
        return s[: max(0, max_chars - 30)] + "...[TRUNCATED]"
    return s


def _heuristic_multi_anchors(query: str, anchor_count: int, required_sections: list[str] | None = None) -> list[str]:
    """
    Heuristic fallback for anchor generation when LLM fails.
    Generates K diverse anchors targeting different decision facets.
    """
    q = (query or "").strip()
    secs = [s.strip() for s in (required_sections or []) if isinstance(s, str) and s.strip()]
    sec_hint = ", ".join(secs[:8]) if secs else ""

    # Define decision-oriented anchor templates
    templates = [
        f"{q} - Find specific constraints, requirements, and hard limits.",
        f"{q} - Retrieve quantitative metrics, numbers, comparisons, benchmarks from tables and figures.",
        f"{q} - Identify risks, limitations, failure cases, and negative results.",
        f"{q} - Extract trade-offs, alternatives, and competing approaches.",
        f"{q} - Locate implementation details, methodologies, experimental setups, and datasets.",
        f"{q} - Search for evidence in tables, figures, charts, and visualizations.",
        f"{q} - Find related work, baselines, and prior art comparisons.",
    ]

    if sec_hint:
        templates.append(f"{q} - Search in sections: {sec_hint}")

    # Return first K templates
    anchors = templates[:anchor_count] if anchor_count <= len(templates) else templates

    # If K > len(templates), pad with generic anchors
    while len(anchors) < anchor_count:
        anchors.append(f"{q} - Retrieve supporting evidence and details (aspect {len(anchors) + 1}).")

    return anchors


def _check_diversity(anchors: list[str], embedder: MultiModalEmbedder, threshold: float = 0.85) -> bool:
    """
    Check if anchors are sufficiently diverse by comparing pairwise cosine similarity.
    Returns True if all pairwise similarities are below threshold.
    """
    if len(anchors) <= 1:
        return True

    # Embed all anchors
    vecs = []
    for anchor in anchors:
        vec = embedder.embed_query(anchor) if hasattr(embedder, 'embed_query') else embedder.embed_text(["a"], [anchor]).vecs[0]
        vecs.append(normalize(vec))

    # Check pairwise similarities
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = cosine(vecs[i], vecs[j])
            if sim > threshold:
                log.warning("Anchors %d and %d too similar (cosine=%.3f > %.3f)", i, j, sim, threshold)
                return False

    return True


def generate_multi_anchors(
    client: OpenAIChatClient,
    model: str,
    temperature: float,
    query: str,
    user_instruction: str | None,
    anchor_count: int,
    required_sections: list[str] | None = None,
    max_chars_per_anchor: int = 300,
) -> list[str]:
    """
    Generate K focused anchor queries via LLM decomposition.

    Each anchor targets a distinct decision-oriented facet:
    - Constraints and requirements
    - Quantitative metrics and comparisons
    - Risk factors and limitations
    - Trade-offs and alternatives
    - Implementation details
    - Evidence from tables/figures

    Args:
        client: OpenAI client for LLM calls
        model: Model name (e.g., "gpt-4o")
        temperature: Sampling temperature
        query: User's original query
        user_instruction: Optional user instruction
        anchor_count: Number of anchors to generate (K)
        required_sections: Optional list of required sections
        max_chars_per_anchor: Max characters per anchor

    Returns:
        List of K anchor query strings
    """
    q = (query or "").strip()
    ui = (user_instruction or "").strip()
    sections = [s for s in (required_sections or []) if isinstance(s, str) and s.strip()]

    system = (
        "You are a decision-oriented query decomposer for technical document retrieval. "
        "Given a user query/instruction, decompose it into K focused anchor queries that help decision-makers. "
        "Each anchor should target a DISTINCT facet critical for decision-making:\n"
        "- Constraints and requirements (hard limits, must-haves)\n"
        "- Quantitative metrics and comparisons (numbers, benchmarks, tables)\n"
        "- Risk factors and limitations (failure cases, downsides)\n"
        "- Trade-offs and alternatives (competing approaches, pros/cons)\n"
        "- Implementation details (methodologies, setups, how-to)\n"
        "- Evidence from tables/figures/charts\n\n"
        "Each anchor should be:\n"
        "- 1-2 sentences, concrete and specific\n"
        "- Distinct from other anchors (no redundancy)\n"
        "- Optimized for retrieval (includes synonyms, technical terms)\n"
        "- Focused on ONE specific decision facet\n\n"
        "Do NOT answer the question. Do NOT summarize. Do NOT use placeholders (TODO/TBD/[])."
    )

    user = (
        f"USER_QUERY:\n{q}\n\n"
        + (f"USER_INSTRUCTION:\n{ui}\n\n" if ui and ui != q else "")
        + ("REQUIRED_SECTIONS:\n" + "\n".join([f"- {s}" for s in sections]) + "\n\n" if sections else "")
        + f"Generate exactly {anchor_count} focused anchor queries for retrieving decision-relevant evidence.\n"
        f"Format as a numbered list:\n"
        f"1. [First anchor - focus on constraints/requirements]\n"
        f"2. [Second anchor - focus on metrics/numbers]\n"
        f"...\n"
        f"{anchor_count}. [Final anchor]\n\n"
        f"Keep each anchor concise (<= {max_chars_per_anchor} characters)."
    )

    try:
        res = client.chat(model=model, system=system, user=user, temperature=temperature)
        text = (res.text or "").strip()

        # Parse numbered list
        lines = text.split("\n")
        anchors = []
        for line in lines:
            line = line.strip()
            # Match lines starting with "1.", "2.", etc.
            match = re.match(r"^(\d+)\.\s*(.+)$", line)
            if match:
                anchor_text = match.group(2).strip()
                # Remove any leading/trailing quotes
                anchor_text = anchor_text.strip('"\'')
                # Truncate to max chars
                anchor_text = _truncate(anchor_text, max_chars_per_anchor)
                if anchor_text and not _PLACEHOLDER.search(anchor_text):
                    anchors.append(anchor_text)

        # Validate we got K anchors
        if len(anchors) != anchor_count:
            log.warning("LLM returned %d anchors (expected %d), using heuristic fallback", len(anchors), anchor_count)
            anchors = _heuristic_multi_anchors(q, anchor_count, required_sections=sections)
        elif any(_PLACEHOLDER.search(a) for a in anchors):
            log.warning("LLM anchors contain placeholders, using heuristic fallback")
            anchors = _heuristic_multi_anchors(q, anchor_count, required_sections=sections)
        else:
            log.info("Generated %d anchors via LLM", len(anchors))

    except Exception as e:
        log.warning("LLM anchor generation failed (%s), using heuristic fallback", e)
        anchors = _heuristic_multi_anchors(q, anchor_count, required_sections=sections)

    return anchors


class MultiAnchorScorer:
    """
    Computes multi-anchor retrieval scores with per-anchor softmax normalization.

    Algorithm:
    1. For each anchor j, embed it: e_j = Emb(a_j)
    2. Compute similarities to all chunks: s_j = e_j^T F
    3. Apply softmax per anchor: p_j = softmax(s_j / τ)
    4. Aggregate by mean: S = (1/K) · Σ_j p_j

    The final score S_i for chunk i is the mean of its softmax-normalized scores
    across all anchors.
    """

    def __init__(self, embedder: MultiModalEmbedder, unit_vecs: dict[str, np.ndarray], temperature: float):
        """
        Initialize multi-anchor scorer.

        Args:
            embedder: Embedding model (must have embed_query method)
            unit_vecs: Dictionary mapping unit IDs to embedding vectors
            temperature: Softmax temperature τ (default: 1.0)
        """
        self.embedder = embedder
        self.unit_vecs = unit_vecs
        self.temperature = temperature

    def compute_scores(self, anchor_queries: list[str]) -> dict[str, float]:
        """
        Compute multi-anchor scores with per-anchor softmax normalization.

        Args:
            anchor_queries: List of K anchor query strings

        Returns:
            Dictionary mapping unit IDs to final aggregated scores
        """
        K = len(anchor_queries)
        if K == 0:
            log.warning("No anchor queries provided, returning empty scores")
            return {}

        unit_ids = list(self.unit_vecs.keys())
        N = len(unit_ids)

        if N == 0:
            log.warning("No unit vectors provided, returning empty scores")
            return {}

        log.info("Computing multi-anchor scores: K=%d anchors, N=%d chunks", K, N)

        # Store per-anchor probabilities
        all_probs = []

        for j, anchor_text in enumerate(anchor_queries):
            # Step 2: Embed anchor
            e_j = self.embedder.embed_query(anchor_text) if hasattr(self.embedder, 'embed_query') \
                  else self.embedder.embed_text(["a"], [anchor_text]).vecs[0]
            e_j = normalize(e_j)

            # Step 3: Compute similarities to all chunks
            s_j = np.array([float(e_j @ self.unit_vecs[uid]) for uid in unit_ids], dtype=np.float32)

            # Step 4: Per-anchor softmax normalization (CRITICAL)
            # Apply temperature scaling
            s_j_scaled = s_j / self.temperature

            # Numerical stability: subtract max before exp
            s_j_max = np.max(s_j_scaled)
            exp_scores = np.exp(s_j_scaled - s_j_max)

            # Normalize to get probabilities
            p_j = exp_scores / (np.sum(exp_scores) + 1e-12)

            # Verify normalization (should sum to ~1.0)
            total = np.sum(p_j)
            if not (0.99 <= total <= 1.01):
                log.warning("Anchor %d softmax sum %.4f (expected ~1.0)", j, total)

            all_probs.append(p_j)

            log.debug("Anchor %d: score range [%.4f, %.4f], softmax range [%.6f, %.6f]",
                     j, np.min(s_j), np.max(s_j), np.min(p_j), np.max(p_j))

        # Step 5: Anchor aggregation by mean
        # S = (1/K) · Σ_j p_j
        all_probs_arr = np.array(all_probs)  # Shape: (K, N)
        S = np.mean(all_probs_arr, axis=0)  # Shape: (N,)

        # SCALE SCORES: Multiply by N to match cosine similarity scale
        # Softmax probabilities average ~1/N, but optimizer expects cosine similarities (0-1 range)
        # Scaling by N brings scores from ~0.007 to ~1.0
        N = len(unit_ids)
        S_scaled = S * N

        log.info("Multi-anchor aggregation: mean score=%.6f, range [%.6f, %.6f] (before scaling)",
                 np.mean(S), np.min(S), np.max(S))
        log.info("Scaled multi-anchor scores by N=%d: range [%.4f, %.4f]",
                 N, np.min(S_scaled), np.max(S_scaled))

        # Convert to dictionary
        final_scores = {uid: float(S_scaled[i]) for i, uid in enumerate(unit_ids)}

        return final_scores
