"""
Full pipeline demonstration for Budgeted Dual-DPP.

This script simulates a complete document retrieval workflow with synthetic data.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import DPPConfig, CognitiveProfiler, ColPaliEmbedder, BudgetedDualDPP
from src.config import ExperimentConfig
from src.utils import print_selection_summary, visualize_selection


def main():
    """
    Demonstrate the full Budgeted Dual-DPP pipeline.
    
    Pipeline:
        1. Generate synthetic document patches
        2. Compute cognitive costs
        3. Generate embeddings
        4. Optimize patch selection
        5. Analyze and visualize results
    """
    print("="*80)
    print("Layout-Aware Budgeted Dual-DPP: Full Pipeline Demo")
    print("="*80)
    
    # =========================================================================
    # Configuration
    # =========================================================================
    exp_config = ExperimentConfig(
        num_pages=100,
        patches_per_page=10,
        patch_size=224,
        embedding_dim=128,
        budget=50.0,
        batch_size=100,
        seed=42
    )
    
    dpp_config = DPPConfig(
        temperature=0.1,
        epsilon=1e-6,
        alpha=0.6,
        beta=0.4
    )
    
    torch.manual_seed(exp_config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_patches = exp_config.num_pages * exp_config.patches_per_page
    
    print(f"\n📄 Document Configuration:")
    print(f"   • Pages:              {exp_config.num_pages}")
    print(f"   • Patches per page:   {exp_config.patches_per_page}")
    print(f"   • Total patches:      {num_patches}")
    print(f"   • Patch size:         {exp_config.patch_size}×{exp_config.patch_size}")
    print(f"   • Device:             {device}")
    
    print(f"\n⚙️  Optimization Configuration:")
    print(f"   • Budget:             {exp_config.budget}")
    print(f"   • Temperature:        {dpp_config.temperature}")
    print(f"   • Alpha (entropy):    {dpp_config.alpha}")
    print(f"   • Beta (clutter):     {dpp_config.beta}")
    
    # =========================================================================
    # Stage 1: Generate Synthetic Data
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 1: SYNTHETIC DATA GENERATION")
    print(f"="*80)
    
    print(f"\n🖼️  Generating {num_patches} synthetic patch images...")
    patches = torch.rand(
        num_patches, 3, exp_config.patch_size, exp_config.patch_size, 
        device=device
    )
    print(f"✓ Patches generated: {patches.shape}")
    
    # =========================================================================
    # Stage 2: Cognitive Profiling
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 2: COGNITIVE PROFILING")
    print(f"="*80)
    
    profiler = CognitiveProfiler(
        alpha=dpp_config.alpha, 
        beta=dpp_config.beta
    ).to(device)
    
    # Process in batches
    all_costs = []
    batch_size = exp_config.batch_size
    
    print(f"\n🧠 Computing cognitive costs (batch_size={batch_size})...")
    for i in range(0, num_patches, batch_size):
        batch = patches[i:i+batch_size]
        costs = profiler(batch)
        all_costs.append(costs)
    
    patch_costs = torch.cat(all_costs)
    
    print(f"✓ Cognitive costs computed")
    print(f"   • Cost range:         [{patch_costs.min().item():.3f}, "
          f"{patch_costs.max().item():.3f}]")
    print(f"   • Mean cost:          {patch_costs.mean().item():.3f}")
    print(f"   • Std cost:           {patch_costs.std().item():.3f}")
    
    # =========================================================================
    # Stage 3: Layout-Aware Embedding
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 3: LAYOUT-AWARE EMBEDDING")
    print(f"="*80)
    
    embedder = ColPaliEmbedder(device=device)
    
    # For demo: use random embeddings (ColPali requires actual model)
    print(f"\n⚠️  Using random embeddings for demonstration")
    print(f"   (Install colpali-engine for real ColPali embeddings)")
    
    patch_embeddings = torch.randn(
        num_patches, exp_config.embedding_dim, device=device
    )
    patch_embeddings = F.normalize(patch_embeddings, p=2, dim=1)
    
    # Encode query
    query_text = "Find all technical diagrams and mathematical equations"
    print(f"\n📝 Query: '{query_text}'")
    
    query_embedding = torch.randn(1, exp_config.embedding_dim, device=device)
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    print(f"\n✓ Embeddings computed")
    print(f"   • Patch embeddings:   {patch_embeddings.shape}")
    print(f"   • Query embedding:    {query_embedding.shape}")
    
    # =========================================================================
    # Stage 4: Budgeted DPP Optimization
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 4: BUDGETED DPP OPTIMIZATION")
    print(f"="*80)
    
    optimizer = BudgetedDualDPP(dpp_config).to(device)
    
    selected_indices, total_cost = optimizer(
        patch_embeddings=patch_embeddings,
        patch_costs=patch_costs,
        query_embedding=query_embedding,
        budget=exp_config.budget,
        verbose=True
    )
    
    # =========================================================================
    # Stage 5: Analysis and Visualization
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 5: ANALYSIS & VISUALIZATION")
    print(f"="*80)
    
    # Compute relevance scores for analysis
    relevance_scores = torch.exp(
        (patch_embeddings @ query_embedding.T).squeeze(1) / dpp_config.temperature
    )
    
    # Print summary
    print_selection_summary(
        selected_indices=selected_indices,
        patch_embeddings=patch_embeddings,
        patch_costs=patch_costs,
        relevance_scores=relevance_scores,
        budget=exp_config.budget
    )
    
    # Create visualization
    print(f"\n📊 Generating visualization...")
    viz_path = os.path.join(
        os.path.dirname(__file__), '..', 'output', 'selection_analysis.png'
    )
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    
    visualize_selection(
        selected_indices=selected_indices,
        patch_embeddings=patch_embeddings,
        patch_costs=patch_costs,
        relevance_scores=relevance_scores,
        budget=exp_config.budget,
        save_path=viz_path
    )
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"✅ PIPELINE COMPLETE")
    print(f"="*80)
    
    print(f"\n🎉 Successfully demonstrated Budgeted Dual-DPP!")
    print(f"   • Selected {len(selected_indices)} patches from {num_patches}")
    print(f"   • Budget utilization: {total_cost/exp_config.budget:.1%}")
    print(f"   • Visualization saved: {viz_path}")
    
    print(f"\n" + "="*80)


if __name__ == "__main__":
    main()