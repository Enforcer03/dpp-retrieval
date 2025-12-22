"""
Utility functions for Budgeted Dual-DPP.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def compute_diversity_score(embeddings: torch.Tensor) -> float:
    """
    Compute diversity score based on pairwise similarity.
    
    Args:
        embeddings: (N, D) tensor of normalized embeddings
    
    Returns:
        Diversity score in [0, 1] where higher = more diverse
    
    Formula:
        diversity = 1 - avg_pairwise_similarity
    """
    N = embeddings.shape[0]
    
    if N <= 1:
        return 1.0
    
    # Compute similarity matrix
    similarity_matrix = embeddings @ embeddings.T
    
    # Get off-diagonal elements
    mask = ~torch.eye(N, dtype=bool, device=embeddings.device)
    avg_similarity = similarity_matrix[mask].mean().item()
    
    return 1.0 - avg_similarity


def compute_coverage_score(
    selected_indices: List[int],
    all_relevances: torch.Tensor,
    top_k: int = 10
) -> float:
    """
    Compute coverage of top-K most relevant patches.
    
    Args:
        selected_indices: List of selected patch indices
        all_relevances: (N,) tensor of relevance scores for all patches
        top_k: Number of top patches to consider (default: 2 * |selected|)
    
    Returns:
        Coverage score in [0, 1]
    
    Formula:
        coverage = |selected ∩ top_K| / K
    """
    if top_k is None:
        top_k = min(len(selected_indices) * 2, len(all_relevances))
    
    # Get indices of top-K most relevant patches
    top_k_indices = torch.argsort(all_relevances, descending=True)[:top_k].tolist()
    
    # Compute intersection
    intersection = set(selected_indices) & set(top_k_indices)
    
    return len(intersection) / top_k


def analyze_selection(
    selected_indices: List[int],
    patch_embeddings: torch.Tensor,
    patch_costs: torch.Tensor,
    relevance_scores: torch.Tensor,
    budget: float
) -> Dict[str, Any]:
    """
    Comprehensive analysis of selected patches.
    
    Args:
        selected_indices: List of selected patch indices
        patch_embeddings: (N, D) all patch embeddings
        patch_costs: (N,) all patch costs
        relevance_scores: (N,) all relevance scores
        budget: Total budget
    
    Returns:
        Dictionary with analysis metrics
    """
    N = len(patch_embeddings)
    n_selected = len(selected_indices)
    
    # Cost analysis
    selected_costs = patch_costs[selected_indices]
    total_cost = selected_costs.sum().item()
    
    # Relevance analysis
    selected_relevance = relevance_scores[selected_indices]
    total_relevance = selected_relevance.sum().item()
    avg_relevance = selected_relevance.mean().item()
    
    # Diversity analysis
    selected_embeddings = patch_embeddings[selected_indices]
    diversity = compute_diversity_score(selected_embeddings)
    
    # Coverage analysis
    coverage = compute_coverage_score(selected_indices, relevance_scores)
    
    # Efficiency metrics
    selection_rate = n_selected / N
    budget_utilization = total_cost / budget
    avg_gain_per_cost = total_relevance / total_cost if total_cost > 0 else 0
    
    return {
        'n_patches': N,
        'n_selected': n_selected,
        'selection_rate': selection_rate,
        'total_cost': total_cost,
        'budget': budget,
        'budget_utilization': budget_utilization,
        'avg_cost_per_patch': selected_costs.mean().item(),
        'total_relevance': total_relevance,
        'avg_relevance': avg_relevance,
        'diversity_score': diversity,
        'coverage_score': coverage,
        'efficiency': avg_gain_per_cost
    }


def visualize_selection(
    selected_indices: List[int],
    patch_embeddings: torch.Tensor,
    patch_costs: torch.Tensor,
    relevance_scores: torch.Tensor,
    budget: float,
    save_path: str = None
) -> None:
    """
    Create visualization of selection results.
    
    Args:
        selected_indices: List of selected patch indices
        patch_embeddings: (N, D) all patch embeddings
        patch_costs: (N,) all patch costs
        relevance_scores: (N,) all relevance scores
        budget: Total budget
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Budgeted DPP Selection Analysis', fontsize=16, fontweight='bold')
    
    N = len(patch_embeddings)
    selected_mask = torch.zeros(N, dtype=bool)
    selected_mask[selected_indices] = True
    
    # Plot 1: Relevance vs Cost scatter
    ax = axes[0, 0]
    colors = ['#2ecc71' if m else '#bdc3c7' for m in selected_mask]
    sizes = [100 if m else 30 for m in selected_mask]
    alphas = [0.8 if m else 0.3 for m in selected_mask]
    
    relevances = relevance_scores.cpu().numpy()
    costs = patch_costs.cpu().numpy()
    
    for i in range(N):
        ax.scatter(relevances[i], costs[i], c=colors[i], s=sizes[i], 
                  alpha=alphas[i], edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Relevance Score', fontweight='bold')
    ax.set_ylabel('Cognitive Cost', fontweight='bold')
    ax.set_title('Relevance vs Cost')
    ax.grid(alpha=0.3)
    
    # Plot 2: Cost distribution
    ax = axes[0, 1]
    selected_costs = patch_costs[selected_indices].cpu().numpy()
    ax.hist(costs, bins=20, alpha=0.5, label='All Patches', color='#3498db')
    ax.hist(selected_costs, bins=20, alpha=0.7, label='Selected', color='#e74c3c')
    ax.set_xlabel('Cognitive Cost', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Cost Distribution')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 3: Cumulative budget usage
    ax = axes[1, 0]
    cumulative_cost = np.cumsum(selected_costs)
    x = range(len(selected_costs))
    ax.plot(x, cumulative_cost, 'o-', color='#3498db', linewidth=2, markersize=6)
    ax.axhline(budget, color='red', linestyle='--', linewidth=2, label='Budget Limit')
    ax.fill_between(x, 0, cumulative_cost, alpha=0.3, color='#3498db')
    ax.set_xlabel('Selection Order', fontweight='bold')
    ax.set_ylabel('Cumulative Cost', fontweight='bold')
    ax.set_title('Budget Utilization')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Selection metrics
    ax = axes[1, 1]
    metrics = analyze_selection(
        selected_indices, patch_embeddings, patch_costs, relevance_scores, budget
    )
    
    metric_names = ['Selection\nRate', 'Budget\nUtil.', 'Diversity', 'Coverage']
    metric_values = [
        metrics['selection_rate'],
        metrics['budget_utilization'],
        metrics['diversity_score'],
        metrics['coverage_score']
    ]
    
    bars = ax.bar(metric_names, metric_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Selection Metrics')
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_selection_summary(
    selected_indices: List[int],
    patch_embeddings: torch.Tensor,
    patch_costs: torch.Tensor,
    relevance_scores: torch.Tensor,
    budget: float
) -> None:
    """
    Print formatted summary of selection results.
    
    Args:
        selected_indices: List of selected patch indices
        patch_embeddings: (N, D) all patch embeddings
        patch_costs: (N,) all patch costs
        relevance_scores: (N,) all relevance scores
        budget: Total budget
    """
    metrics = analyze_selection(
        selected_indices, patch_embeddings, patch_costs, relevance_scores, budget
    )
    
    print("\n" + "="*80)
    print("SELECTION SUMMARY")
    print("="*80)
    
    print(f"\n📊 Basic Statistics:")
    print(f"   • Total patches:      {metrics['n_patches']}")
    print(f"   • Patches selected:   {metrics['n_selected']}")
    print(f"   • Selection rate:     {metrics['selection_rate']:.2%}")
    
    print(f"\n💰 Budget Analysis:")
    print(f"   • Budget:             {metrics['budget']:.2f}")
    print(f"   • Cost used:          {metrics['total_cost']:.2f}")
    print(f"   • Utilization:        {metrics['budget_utilization']:.2%}")
    print(f"   • Avg cost/patch:     {metrics['avg_cost_per_patch']:.3f}")
    
    print(f"\n🎯 Quality Metrics:")
    print(f"   • Total relevance:    {metrics['total_relevance']:.3f}")
    print(f"   • Avg relevance:      {metrics['avg_relevance']:.3f}")
    print(f"   • Diversity score:    {metrics['diversity_score']:.3f}")
    print(f"   • Coverage score:     {metrics['coverage_score']:.3f}")
    print(f"   • Efficiency:         {metrics['efficiency']:.3f}")
    
    print(f"\n📋 Selected Patch IDs (first 20):")
    print(f"   {selected_indices[:20]}")
    
    print("\n" + "="*80)