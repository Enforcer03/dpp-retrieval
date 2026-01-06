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
