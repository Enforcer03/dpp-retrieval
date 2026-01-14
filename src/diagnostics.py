"""Budgeted Retrieval Diagnostics - Comprehensive visualization module."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from src.schema import EvidenceUnit, SelectionResult

log = logging.getLogger(__name__)


class RetrievalDiagnostics:
    """Generate comprehensive diagnostic visualizations for budgeted retrieval."""

    def __init__(
        self,
        all_units: list[EvidenceUnit],
        candidates: list[EvidenceUnit],
        selected: list[SelectionResult],
        costs: dict[str, float],
        rel_scores: dict[str, float],
        budget_tokens: int,
    ):
        """Initialize diagnostics.

        Args:
            all_units: All evidence units from the document
            candidates: Retrieved candidate units
            selected: Final selected units with metadata
            costs: Cognitive cost for each unit
            rel_scores: Relevance scores for each unit
            budget_tokens: Budget limit in tokens
        """
        self.all_units = all_units
        self.candidates = candidates
        self.selected = selected
        self.costs = costs
        self.rel_scores = rel_scores
        self.budget_tokens = budget_tokens

        # Extract selected unit IDs
        self.selected_ids = {s.unit.id for s in selected}

    def generate_dashboard(self, output_path: Path) -> None:
        """Generate comprehensive diagnostic dashboard.

        Args:
            output_path: Path to save the visualization
        """
        log.info("Generating retrieval diagnostics dashboard...")

        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 11))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Chunk Selection Space
        ax2 = fig.add_subplot(gs[0, 1])  # Cost Distribution
        ax3 = fig.add_subplot(gs[0, 2])  # Selection Pattern Across Document
        ax4 = fig.add_subplot(gs[1, 0])  # Budget Allocation Over Selection
        ax5 = fig.add_subplot(gs[1, 1])  # Unit Type Distribution
        ax6 = fig.add_subplot(gs[1, 2])  # Selection Efficiency Comparison

        # Generate each plot
        self._plot_selection_space(ax1)
        self._plot_cost_distribution(ax2)
        self._plot_selection_pattern(ax3)
        self._plot_budget_allocation(ax4)
        self._plot_unit_type_distribution(ax5)
        self._plot_efficiency_comparison(ax6)

        # Add main title
        fig.suptitle('ICB-Sum: Budgeted Retrieval Diagnostics',
                    fontsize=16, fontweight='bold', y=0.98)

        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        log.info(f"Diagnostics saved to {output_path}")

    def _plot_selection_space(self, ax) -> None:
        """Plot 1: Chunk Selection Space (Importance vs Cognitive Cost)."""
        # Prepare data for candidates
        candidate_ids = {u.id for u in self.candidates}

        # All candidates data
        all_costs = []
        all_scores = []
        all_selected = []

        for u in self.candidates:
            cost = self.costs.get(u.id, 0)
            score = self.rel_scores.get(u.id, 0)
            is_selected = u.id in self.selected_ids

            all_costs.append(cost)
            all_scores.append(score)
            all_selected.append(is_selected)

        # Convert to numpy arrays
        all_costs = np.array(all_costs)
        all_scores = np.array(all_scores)
        all_selected = np.array(all_selected)

        # Plot not selected (gray)
        not_selected_mask = ~all_selected
        ax.scatter(all_scores[not_selected_mask], all_costs[not_selected_mask],
                  c='lightgray', s=80, alpha=0.4, edgecolors='gray', linewidths=0.5,
                  label='Not Selected')

        # Plot selected (green)
        selected_mask = all_selected
        ax.scatter(all_scores[selected_mask], all_costs[selected_mask],
                  c='#2ecc71', s=120, alpha=0.8, edgecolors='darkgreen', linewidths=1.5,
                  label='Selected')

        ax.set_xlabel('Importance (Relevance Score)', fontsize=11)
        ax.set_ylabel('Cognitive Cost', fontsize=11)
        ax.set_title('Chunk Selection Space\n(Green = Selected)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

    def _plot_cost_distribution(self, ax) -> None:
        """Plot 2: Cost Distribution (Histogram)."""
        # All candidates costs
        all_costs = [self.costs.get(u.id, 0) for u in self.candidates]

        # Selected costs
        selected_costs = [self.costs.get(s.unit.id, 0) for s in self.selected]

        # Create histogram
        bins = 30
        ax.hist(all_costs, bins=bins, color='skyblue', alpha=0.7,
               label='All Chunks', edgecolor='steelblue', linewidth=0.5)
        ax.hist(selected_costs, bins=bins, color='salmon', alpha=0.7,
               label='Selected', edgecolor='darkred', linewidth=0.5)

        ax.set_xlabel('Cognitive Cost', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Cost Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    def _plot_selection_pattern(self, ax) -> None:
        """Plot 3: Selection Pattern Across Document."""
        # Create chunk ordering based on document position
        # Sort all units by page and position
        sorted_units = sorted(self.all_units, key=lambda u: (u.page, u.bbox[1], u.bbox[0]))

        # Create chunk ID mapping (document order)
        chunk_order = {u.id: idx for idx, u in enumerate(sorted_units)}

        # Prepare data
        all_chunk_ids = []
        all_scores = []
        all_selected = []

        for u in sorted_units:
            if u.id in {c.id for c in self.candidates}:  # Only plot candidates
                chunk_id = chunk_order[u.id]
                score = self.rel_scores.get(u.id, 0)
                is_selected = u.id in self.selected_ids

                all_chunk_ids.append(chunk_id)
                all_scores.append(score)
                all_selected.append(is_selected)

        # Convert to numpy
        all_chunk_ids = np.array(all_chunk_ids)
        all_scores = np.array(all_scores)
        all_selected = np.array(all_selected)

        # Plot all chunks in gray
        ax.scatter(all_chunk_ids[~all_selected], all_scores[~all_selected],
                  c='lightgray', s=40, alpha=0.4, marker='o', linewidths=0)

        # Plot selected chunks in red
        ax.scatter(all_chunk_ids[all_selected], all_scores[all_selected],
                  c='#e74c3c', s=100, alpha=0.9, marker='o',
                  edgecolors='darkred', linewidths=1.5, label='Selected')

        # Add connecting lines for all chunks
        ax.plot(all_chunk_ids, all_scores, color='lightgray', alpha=0.3,
               linewidth=1, linestyle='-', zorder=0)

        ax.set_xlabel('Chunk ID (Document Order)', fontsize=11)
        ax.set_ylabel('Importance Score', fontsize=11)
        ax.set_title('Selection Pattern Across Document', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

    def _plot_budget_allocation(self, ax) -> None:
        """Plot 4: Budget Allocation Over Selection."""
        # Calculate cumulative cost
        cumulative_costs = []
        current_cost = 0

        for s in self.selected:
            current_cost += self.costs.get(s.unit.id, 0)
            cumulative_costs.append(current_cost)

        selection_order = list(range(len(self.selected)))

        # Plot cumulative cost
        ax.plot(selection_order, cumulative_costs,
               marker='o', color='steelblue', linewidth=2,
               markersize=6, markerfacecolor='white', markeredgewidth=2)
        ax.fill_between(selection_order, cumulative_costs,
                        alpha=0.3, color='skyblue')

        # Add budget limit line
        ax.axhline(y=self.budget_tokens, color='red', linestyle='--',
                  linewidth=2, label='Budget Limit', alpha=0.8)

        # Calculate budget usage percentage
        final_cost = cumulative_costs[-1] if cumulative_costs else 0
        usage_pct = 100 * final_cost / self.budget_tokens if self.budget_tokens > 0 else 0

        # Add usage text
        ax.text(0.98, 0.95, f'{usage_pct:.1f}% Used',
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))

        ax.set_xlabel('Selection Order', fontsize=11)
        ax.set_ylabel('Cumulative Cost', fontsize=11)
        ax.set_title('Budget Allocation Over Selection', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

    def _plot_unit_type_distribution(self, ax) -> None:
        """Plot 5: Unit Type Distribution in Selection."""
        # Count unit types in selection
        type_counts = {}
        for s in self.selected:
            unit_type = s.unit.type
            type_counts[unit_type] = type_counts.get(unit_type, 0) + 1

        # Sort by count
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        types = [t[0] for t in sorted_types]
        counts = [t[1] for t in sorted_types]

        # Define colors for each type
        type_colors = {
            'table': '#b19cd9',      # Purple
            'reference': '#fff494',   # Yellow
            'paragraph': '#77dd77',   # Green
            'figure': '#ff6b6b',      # Red
            'list': '#84b6f4',        # Blue
        }

        colors = [type_colors.get(t, '#cccccc') for t in types]

        # Create horizontal bar chart
        y_pos = np.arange(len(types))
        ax.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(types)
        ax.set_xlabel('Count', fontsize=11)
        ax.set_title('Unit Type Distribution in Selection', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        # Invert y-axis to match the style (table at top)
        ax.invert_yaxis()

    def _plot_efficiency_comparison(self, ax) -> None:
        """Plot 6: Selection Efficiency Comparison."""
        # Calculate efficiency metric: importance / cost
        all_efficiencies = []
        selected_efficiencies = []

        for u in self.candidates:
            cost = self.costs.get(u.id, 1)  # Avoid division by zero
            score = self.rel_scores.get(u.id, 0)
            efficiency = score / cost if cost > 0 else 0

            all_efficiencies.append(efficiency)

            if u.id in self.selected_ids:
                selected_efficiencies.append(efficiency)

        # Prepare data for box plot
        data_to_plot = [all_efficiencies, selected_efficiencies]
        labels = ['All Chunks', 'Selected']

        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                       widths=0.6, showmeans=True,
                       meanprops=dict(marker='*', markerfacecolor='red',
                                    markersize=15, markeredgecolor='darkred', linewidth=2),
                       medianprops=dict(color='black', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))

        # Color the boxes
        colors = ['lightgray', '#77dd77']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

        ax.set_ylabel('Efficiency (Importance / Cost)', fontsize=11)
        ax.set_title('Selection Efficiency Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add legend for mean
        ax.legend([bp['means'][0]], ['Mean'], loc='upper right',
                 framealpha=0.9, fontsize=10)


def _redundancy_metrics(sim: np.ndarray, dup_thresh: float = 0.90) -> dict:
    """Compute redundancy and diversity metrics from similarity matrix."""
    n = sim.shape[0]
    if n <= 1:
        return dict(
            n=n,
            mean_offdiag=np.nan,
            max_offdiag=np.nan,
            dup_rate=np.nan,
            dispersion=np.nan,
            novelty_curve=np.array([]),
            mean_novelty=np.nan,
        )

    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    iu = np.triu_indices(n, k=1)

    # Novelty curve: how novel is each chunk compared to previous ones
    novelty = np.zeros(n, dtype=np.float32)
    novelty[0] = 1.0
    for t in range(1, n):
        novelty[t] = 1.0 - float(np.max(sim[t, :t]))

    return dict(
        n=n,
        mean_offdiag=float(np.nanmean(off)),
        max_offdiag=float(np.nanmax(off)),
        dup_rate=float(np.mean(sim[iu] >= dup_thresh)),
        dispersion=float(np.nanmean(1.0 - off)),
        novelty_curve=novelty,
        mean_novelty=float(np.mean(novelty[1:])) if n > 1 else np.nan,
    )


def generate_method_comparison_diagnostics(
    all_units: list[EvidenceUnit],
    candidates: list[EvidenceUnit],
    selections: dict[str, list[SelectionResult]],
    costs: dict[str, float],
    rel_scores: dict[str, float],
    budget_tokens: int,
    embedder,
    output_path: Path,
) -> None:
    """Generate comprehensive comparison diagnostic for all selection methods.

    Args:
        all_units: All evidence units from the document
        candidates: Retrieved candidate units
        selections: Dict mapping method name to selected units
        costs: Cognitive cost for each unit
        rel_scores: Relevance scores for each unit (after multi-anchor fusion if enabled)
        budget_tokens: Budget limit in tokens
        embedder: Embedder instance to compute similarity between selected units
        output_path: Path to save the visualization
    """
    from sklearn.metrics.pairwise import cosine_similarity

    log.info("Generating method comparison diagnostics...")

    methods = ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"]
    nrows, ncols = len(methods), 6

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(36, 24),
        dpi=400,
        constrained_layout=True
    )
    fig.suptitle(
        "Selection Method Comparison: Redundancy, Diversity & Relevance",
        fontsize=18,
        fontweight="bold"
    )

    HM_VMIN, HM_VMAX = 0.0, 1.0

    for i, method in enumerate(methods):
        sel = selections[method]

        # ----- Existing diagnostics in columns 0-2 -----
        diag = RetrievalDiagnostics(
            all_units=all_units,
            candidates=candidates,
            selected=sel,
            costs=costs,
            rel_scores=rel_scores,
            budget_tokens=budget_tokens,
        )
        diag._plot_selection_space(axes[i, 0])
        diag._plot_cost_distribution(axes[i, 1])
        diag._plot_budget_allocation(axes[i, 2])

        # Add method label on left side
        axes[i, 0].text(
            -0.18, 0.5, method.upper(),
            transform=axes[i, 0].transAxes,
            fontsize=14, fontweight="bold", rotation=90, va="center"
        )

        # ----- Prepare selected data -----
        selected_ids = [s.unit.id for s in sel]
        selected_texts = [s.unit.context_text for s in sel]
        n = len(selected_ids)

        # If empty selection, clear remaining panels
        if n == 0:
            for j in [3, 4, 5]:
                axes[i, j].axis("off")
                axes[i, j].text(
                    0.5, 0.5, "No units selected",
                    ha="center", va="center", fontsize=12, color="gray"
                )
            continue

        # ----- Embed selected texts and compute similarity -----
        try:
            selected_res = embedder.embed_text(selected_ids, selected_texts)
            selected_vecs = {sid: v.astype(np.float32) for sid, v in zip(selected_res.ids, selected_res.vecs)}
            X = np.stack([selected_vecs[sid] for sid in selected_ids], axis=0)
            sim = cosine_similarity(X)

            metrics = _redundancy_metrics(sim, dup_thresh=0.90)

            # Avg similarity per chunk (excluding self)
            if n > 1:
                avg_sim = (sim.sum(axis=1) - 1.0) / (n - 1)
            else:
                avg_sim = np.array([np.nan], dtype=np.float32)
        except Exception as e:
            log.warning(f"Failed to compute similarity for {method}: {e}")
            # Clear panels and continue
            for j in [3, 4, 5]:
                axes[i, j].axis("off")
                axes[i, j].text(
                    0.5, 0.5, f"Error: {str(e)[:50]}",
                    ha="center", va="center", fontsize=10, color="red"
                )
            continue

        # ----- Relevance and cost per chunk -----
        rel_per_chunk = np.array([rel_scores.get(sid, 0.0) for sid in selected_ids], dtype=np.float32)
        cost_per_chunk = np.array([max(1, costs.get(sid, 1)) for sid in selected_ids], dtype=np.float32)
        rel_per_token = rel_per_chunk / cost_per_chunk

        # ----- Column 3: Relevance bars with rel/token overlay -----
        ax_rel = axes[i, 3]
        x = np.arange(n)

        ax_rel.bar(x, rel_per_chunk, color='steelblue', alpha=0.7, label='Relevance')
        ax_rel.set_xlabel("Selection index", fontsize=10)
        ax_rel.set_ylabel("Relevance", fontsize=10, color='steelblue')
        ax_rel.tick_params(axis='y', labelcolor='steelblue')
        ax_rel.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Secondary axis for rel/token
        ax_rel2 = ax_rel.twinx()
        ax_rel2.plot(x, rel_per_token, marker="o", linewidth=1.5, color='darkorange', label='Rel/Token')
        ax_rel2.set_ylabel("Relevance / Token", fontsize=10, color='darkorange')
        ax_rel2.tick_params(axis='y', labelcolor='darkorange')

        # ----- Column 4: Similarity heatmap -----
        ax_hm = axes[i, 4]
        im = ax_hm.imshow(sim, vmin=HM_VMIN, vmax=HM_VMAX, aspect="equal", cmap='viridis')
        ax_hm.set_xlabel("Chunk index", fontsize=10)
        ax_hm.set_ylabel("Chunk index", fontsize=10)
        cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.02)
        cbar.set_label("Cosine similarity", fontsize=9)

        # ----- Column 5: Summary metrics -----
        ax_txt = axes[i, 5]
        ax_txt.axis("off")

        used_tokens = float(np.sum(cost_per_chunk))
        budget = float(budget_tokens)
        total_rel = float(np.sum(rel_per_chunk))

        txt = (
            f"n selected: {n}\n"
            f"mean relevance/chunk: {float(np.mean(rel_per_chunk)):.3f}\n"
            f"median relevance/chunk: {float(np.median(rel_per_chunk)):.3f}\n"
            f"mean (rel/token): {float(np.mean(rel_per_token)):.5f}\n"
            f"Σ relevance: {total_rel:.3f}\n"
            f"budget used: {used_tokens:.0f}/{budget:.0f} ({(used_tokens/max(budget,1))*100:.1f}%)\n"
            f"\n--- Redundancy/Diversity ---\n"
            f"mean off-diag sim: {metrics['mean_offdiag']:.3f}\n"
            f"max off-diag sim:  {metrics['max_offdiag']:.3f}\n"
            f"dup rate @0.90:    {metrics['dup_rate']:.3f}\n"
            f"dispersion:        {metrics['dispersion']:.3f}\n"
            f"mean novelty:      {metrics['mean_novelty']:.3f}\n"
            f"mean avg-sim excl self: {float(np.nanmean(avg_sim)):.3f}\n"
        )
        ax_txt.text(
            0.02, 0.98, txt,
            ha="left", va="top",
            fontsize=10, family="monospace",
            transform=ax_txt.transAxes
        )

    # Column headers
    col_titles = [
        "Chunk Selection Space",
        "Cost Distribution",
        "Budget Allocation",
        "Relevance (+ Rel/Token)",
        "Similarity Heatmap",
        "Summary Metrics",
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Save figure
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()

    log.info(f"Method comparison diagnostics saved to {output_path}")

    # Return all metrics for logging/wandb
    all_metrics = {}
    for method in methods:
        sel = selections[method]
        if len(sel) == 0:
            continue

        selected_ids = [s.unit.id for s in sel]
        rel_per_chunk = np.array([rel_scores.get(sid, 0.0) for sid in selected_ids], dtype=np.float32)
        cost_per_chunk = np.array([max(1, costs.get(sid, 1)) for sid in selected_ids], dtype=np.float32)
        rel_per_token = rel_per_chunk / cost_per_chunk

        # Recompute similarity metrics if possible
        try:
            selected_texts = [s.unit.context_text for s in sel]
            selected_res = embedder.embed_text(selected_ids, selected_texts)
            selected_vecs = {sid: v.astype(np.float32) for sid, v in zip(selected_res.ids, selected_res.vecs)}
            X = np.stack([selected_vecs[sid] for sid in selected_ids], axis=0)
            sim = cosine_similarity(X)
            metrics = _redundancy_metrics(sim, dup_thresh=0.90)
        except Exception:
            metrics = {
                'mean_offdiag': np.nan,
                'max_offdiag': np.nan,
                'dup_rate': np.nan,
                'dispersion': np.nan,
                'mean_novelty': np.nan,
            }

        all_metrics[method] = {
            'n_selected': len(sel),
            'mean_relevance': float(np.mean(rel_per_chunk)),
            'median_relevance': float(np.median(rel_per_chunk)),
            'total_relevance': float(np.sum(rel_per_chunk)),
            'mean_rel_per_token': float(np.mean(rel_per_token)),
            'budget_used': float(np.sum(cost_per_chunk)),
            'budget_pct': float(np.sum(cost_per_chunk) / max(budget_tokens, 1) * 100),
            'mean_offdiag_sim': metrics['mean_offdiag'],
            'max_offdiag_sim': metrics['max_offdiag'],
            'dup_rate': metrics['dup_rate'],
            'dispersion': metrics['dispersion'],
            'mean_novelty': metrics['mean_novelty'],
        }

    return all_metrics


def generate_diagnostics(
    all_units: list[EvidenceUnit],
    candidates: list[EvidenceUnit],
    selected: list[SelectionResult],
    costs: dict[str, float],
    rel_scores: dict[str, float],
    budget_tokens: int,
    output_path: Path,
) -> None:
    """Generate and save diagnostic visualizations.

    Args:
        all_units: All evidence units from the document
        candidates: Retrieved candidate units
        selected: Final selected units with metadata
        costs: Cognitive cost for each unit
        rel_scores: Relevance scores for each unit
        budget_tokens: Budget limit in tokens
        output_path: Path to save the visualization
    """
    diagnostics = RetrievalDiagnostics(
        all_units=all_units,
        candidates=candidates,
        selected=selected,
        costs=costs,
        rel_scores=rel_scores,
        budget_tokens=budget_tokens,
    )

    diagnostics.generate_dashboard(output_path)


def generate_spider_chart(
    diagnostic_metrics: dict[str, dict[str, float]],
    output_path: Path = None,
) -> plt.Figure:
    """Generate spider/radar chart comparing selection methods.

    Args:
        diagnostic_metrics: Dict[method_name -> metrics_dict]
        output_path: Optional path to save the chart

    Returns:
        matplotlib Figure object
    """
    import numpy as np

    # Define metrics to visualize (must be normalized to 0-1 scale)
    metric_names = [
        'Relevance',
        'Efficiency',
        'Budget Use',
        'Novelty',
        'Diversity',
    ]

    methods = ['greedy', 'topk', 'greedy_cov', 'cost_norm', 'dpp']
    method_labels = {
        'greedy': 'Greedy',
        'topk': 'Top-K',
        'greedy_cov': 'Greedy+Cov',
        'cost_norm': 'Cost-Norm',
        'dpp': 'DPP',
    }

    # Color scheme for methods
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Prepare data matrix: methods × metrics
    raw_data = []
    valid_methods = []
    valid_colors = []

    for i, method in enumerate(methods):
        if method not in diagnostic_metrics:
            continue

        metrics = diagnostic_metrics[method]

        # Extract raw metric values
        mean_rel = metrics.get('mean_relevance', 0.0)
        mean_rel_per_tok = metrics.get('mean_rel_per_token', 0.0)
        budget_pct = metrics.get('budget_pct', 0.0)
        novelty = metrics.get('mean_novelty', 0.0)
        dispersion = metrics.get('dispersion', 0.0)

        raw_data.append([mean_rel, mean_rel_per_tok, budget_pct, novelty, dispersion])
        valid_methods.append(method_labels[method])
        valid_colors.append(colors[i])

    if not raw_data:
        log.warning("No valid methods in diagnostic_metrics, skipping spider chart")
        return None

    raw_data = np.array(raw_data)

    # Apply relative scaling: normalize each metric by its min/max across all methods
    data = np.zeros_like(raw_data)
    for col in range(raw_data.shape[1]):
        col_values = raw_data[:, col]
        col_min = col_values.min()
        col_max = col_values.max()

        # Avoid division by zero
        if col_max - col_min > 1e-9:
            data[:, col] = (col_values - col_min) / (col_max - col_min)
        else:
            # If all values are the same, set to 0.5
            data[:, col] = 0.5

    # Number of metrics
    num_vars = len(metric_names)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Complete the circle
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Plot each method
    for i, (method_name, method_data) in enumerate(zip(valid_methods, data)):
        # Complete the circle
        values = method_data.tolist()
        values += values[:1]

        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=method_name, color=valid_colors[i])
        ax.fill(angles, values, alpha=0.15, color=valid_colors[i])

    # Fix axis to go from 0 to 1
    ax.set_ylim(0, 1)

    # Set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=11, weight='bold')

    # Set radial labels
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9, color='gray')

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9)

    # Add title
    plt.title('Selection Method Comparison', size=14, weight='bold', pad=20)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        log.info(f"Spider chart saved to {output_path}")

    return fig


def generate_aggregated_visualizations(
    aggregated_metrics: dict[str, float],
    output_dir: Path,
    n_samples: int,
) -> None:
    """
    Generate research-grade visualizations from aggregated metrics.

    Args:
        aggregated_metrics: Dict with keys like "method_metric" and "method_metric_std"
        output_dir: Directory to save plots
        n_samples: Number of samples in the aggregation
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ['greedy', 'topk', 'greedy_cov', 'cost_norm', 'dpp']
    method_labels = {
        'greedy': 'Greedy',
        'topk': 'Top-K',
        'greedy_cov': 'Greedy+Cov',
        'cost_norm': 'Cost-Norm',
        'dpp': 'DPP',
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Extract metrics for each method
    method_data = {}
    for method in methods:
        if f"{method}_mean_relevance" not in aggregated_metrics:
            continue

        method_data[method] = {
            'relevance': (
                aggregated_metrics.get(f"{method}_mean_relevance", 0.0),
                aggregated_metrics.get(f"{method}_mean_relevance_std", 0.0)
            ),
            'efficiency': (
                aggregated_metrics.get(f"{method}_mean_rel_per_token", 0.0),
                aggregated_metrics.get(f"{method}_mean_rel_per_token_std", 0.0)
            ),
            'budget_pct': (
                aggregated_metrics.get(f"{method}_budget_pct", 0.0),
                aggregated_metrics.get(f"{method}_budget_pct_std", 0.0)
            ),
            'novelty': (
                aggregated_metrics.get(f"{method}_mean_novelty", 0.0),
                aggregated_metrics.get(f"{method}_mean_novelty_std", 0.0)
            ),
            'diversity': (
                aggregated_metrics.get(f"{method}_dispersion", 0.0),
                aggregated_metrics.get(f"{method}_dispersion_std", 0.0)
            ),
            'n_selected': (
                aggregated_metrics.get(f"{method}_n_selected", 0.0),
                aggregated_metrics.get(f"{method}_n_selected_std", 0.0)
            ),
        }

    if not method_data:
        log.warning("No valid method data for aggregated visualizations")
        return

    # 1. Aggregated Spider Chart
    _generate_aggregated_spider_chart(method_data, method_labels, colors, output_dir, n_samples)

    # 2. Bar charts with error bars
    _generate_metric_bar_charts(method_data, method_labels, colors, output_dir, n_samples)

    # 3. Budget utilization comparison
    _generate_budget_comparison(method_data, method_labels, colors, output_dir, n_samples)

    # 4. Efficiency vs Relevance scatter
    _generate_efficiency_relevance_scatter(method_data, method_labels, colors, output_dir, n_samples)

    # 5. Diversity vs Novelty comparison
    _generate_diversity_novelty_comparison(method_data, method_labels, colors, output_dir, n_samples)

    log.info(f"Aggregated visualizations saved to {output_dir}")


def _generate_aggregated_spider_chart(
    method_data: dict,
    method_labels: dict,
    colors: list,
    output_dir: Path,
    n_samples: int,
) -> None:
    """Generate aggregated spider chart with relative scaling."""
    import numpy as np

    metric_names = ['Relevance', 'Efficiency', 'Budget Use', 'Novelty', 'Diversity']
    metric_keys = ['relevance', 'efficiency', 'budget_pct', 'novelty', 'diversity']

    # Extract raw values (means only)
    methods = list(method_data.keys())
    raw_data = []
    valid_methods = []
    valid_colors = []

    method_to_color = dict(zip(['greedy', 'topk', 'greedy_cov', 'cost_norm', 'dpp'], colors))

    for method in methods:
        values = [method_data[method][key][0] for key in metric_keys]
        raw_data.append(values)
        valid_methods.append(method_labels[method])
        valid_colors.append(method_to_color[method])

    raw_data = np.array(raw_data)

    # Apply relative scaling
    data = np.zeros_like(raw_data)
    for col in range(raw_data.shape[1]):
        col_values = raw_data[:, col]
        col_min = col_values.min()
        col_max = col_values.max()

        if col_max - col_min > 1e-9:
            data[:, col] = (col_values - col_min) / (col_max - col_min)
        else:
            data[:, col] = 0.5

    # Create spider chart
    num_vars = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    for i, (method_name, method_vals) in enumerate(zip(valid_methods, data)):
        values = method_vals.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method_name, color=valid_colors[i])
        ax.fill(angles, values, alpha=0.15, color=valid_colors[i])

    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=12, weight='bold')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10, color='gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)
    plt.title(f'Selection Method Comparison (n={n_samples})', size=15, weight='bold', pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(output_dir / 'aggregated_spider.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_metric_bar_charts(
    method_data: dict,
    method_labels: dict,
    colors: list,
    output_dir: Path,
    n_samples: int,
) -> None:
    """Generate bar charts with error bars for key metrics."""
    import numpy as np

    metrics = {
        'relevance': ('Mean Relevance', 'Relevance Score'),
        'efficiency': ('Relevance per Token', 'Efficiency (×100)'),
        'novelty': ('Mean Novelty', 'Novelty Score'),
        'diversity': ('Diversity', 'Diversity Score'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    method_to_color = dict(zip(['greedy', 'topk', 'greedy_cov', 'cost_norm', 'dpp'], colors))

    for idx, (metric_key, (title, ylabel)) in enumerate(metrics.items()):
        ax = axes[idx]

        methods = list(method_data.keys())
        means = [method_data[m][metric_key][0] for m in methods]
        stds = [method_data[m][metric_key][1] for m in methods]
        labels = [method_labels[m] for m in methods]
        bar_colors = [method_to_color[m] for m in methods]

        # Scale efficiency for better visualization
        if metric_key == 'efficiency':
            means = [m * 100 for m in means]
            stds = [s * 100 for s in stds]

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)

        ax.set_xlabel('Method', fontsize=11, weight='bold')
        ax.set_ylabel(ylabel, fontsize=11, weight='bold')
        ax.set_title(title, fontsize=12, weight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    plt.suptitle(f'Performance Metrics Comparison (n={n_samples})', fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_metrics_bars.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_budget_comparison(
    method_data: dict,
    method_labels: dict,
    colors: list,
    output_dir: Path,
    n_samples: int,
) -> None:
    """Generate budget utilization comparison with number of units selected."""
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    method_to_color = dict(zip(['greedy', 'topk', 'greedy_cov', 'cost_norm', 'dpp'], colors))

    methods = list(method_data.keys())
    budget_means = [method_data[m]['budget_pct'][0] for m in methods]
    budget_stds = [method_data[m]['budget_pct'][1] for m in methods]
    n_sel_means = [method_data[m]['n_selected'][0] for m in methods]
    n_sel_stds = [method_data[m]['n_selected'][1] for m in methods]
    labels = [method_labels[m] for m in methods]
    bar_colors = [method_to_color[m] for m in methods]

    x = np.arange(len(methods))

    # Budget percentage
    ax1.bar(x, budget_means, yerr=budget_stds, capsize=5, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Budget Limit')
    ax1.set_xlabel('Method', fontsize=11, weight='bold')
    ax1.set_ylabel('Budget Used (%)', fontsize=11, weight='bold')
    ax1.set_title('Budget Utilization', fontsize=12, weight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10)

    # Number of units selected
    ax2.bar(x, n_sel_means, yerr=n_sel_stds, capsize=5, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Method', fontsize=11, weight='bold')
    ax2.set_ylabel('Number of Units', fontsize=11, weight='bold')
    ax2.set_title('Units Selected', fontsize=12, weight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    plt.suptitle(f'Budget and Selection Comparison (n={n_samples})', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_budget.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_efficiency_relevance_scatter(
    method_data: dict,
    method_labels: dict,
    colors: list,
    output_dir: Path,
    n_samples: int,
) -> None:
    """Generate efficiency vs relevance scatter plot."""
    import numpy as np

    method_to_color = dict(zip(['greedy', 'topk', 'greedy_cov', 'cost_norm', 'dpp'], colors))

    fig, ax = plt.subplots(figsize=(10, 8))

    for method in method_data.keys():
        rel_mean, rel_std = method_data[method]['relevance']
        eff_mean, eff_std = method_data[method]['efficiency']

        # Scale efficiency for better visualization
        eff_mean *= 100
        eff_std *= 100

        label = method_labels[method]
        color = method_to_color[method]

        # Plot point with error bars
        ax.errorbar(rel_mean, eff_mean, xerr=rel_std, yerr=eff_std,
                   fmt='o', markersize=12, capsize=5, capthick=2,
                   color=color, label=label, alpha=0.8, linewidth=2)

    ax.set_xlabel('Mean Relevance', fontsize=12, weight='bold')
    ax.set_ylabel('Efficiency (Relevance per Token ×100)', fontsize=12, weight='bold')
    ax.set_title(f'Efficiency vs Relevance Trade-off (n={n_samples})', fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_efficiency_relevance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def _generate_diversity_novelty_comparison(
    method_data: dict,
    method_labels: dict,
    colors: list,
    output_dir: Path,
    n_samples: int,
) -> None:
    """Generate diversity vs novelty scatter plot."""
    import numpy as np

    method_to_color = dict(zip(['greedy', 'topk', 'greedy_cov', 'cost_norm', 'dpp'], colors))

    fig, ax = plt.subplots(figsize=(10, 8))

    for method in method_data.keys():
        div_mean, div_std = method_data[method]['diversity']
        nov_mean, nov_std = method_data[method]['novelty']

        label = method_labels[method]
        color = method_to_color[method]

        # Plot point with error bars
        ax.errorbar(div_mean, nov_mean, xerr=div_std, yerr=nov_std,
                   fmt='o', markersize=12, capsize=5, capthick=2,
                   color=color, label=label, alpha=0.8, linewidth=2)

    ax.set_xlabel('Diversity Score', fontsize=12, weight='bold')
    ax.set_ylabel('Novelty Score', fontsize=12, weight='bold')
    ax.set_title(f'Diversity vs Novelty Analysis (n={n_samples})', fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_diversity_novelty.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
