#!/usr/bin/env python3
"""
Budgeted Dual-DPP: Main Pipeline for Real PDF Documents
========================================================

Usage:
    python main.py /path/to/document.pdf [--query "Your query"] [--budget 50.0]
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import DPPConfig, CognitiveProfiler, ColPaliEmbedder, BudgetedDualDPP, PDFHighlighter
from src.config import ExperimentConfig
from src.utils import print_selection_summary, visualize_selection, analyze_selection
from src.layout_aware_extractor import extract_layout_aware_patches, LayoutAwareHighlighter




def save_selected_patches(
    patches: torch.Tensor,
    selected_indices: List[int],
    output_dir: str,
    prefix: str = "selected_patch"
) -> None:
    """
    Save selected patches as images for visual inspection.
    
    Args:
        patches: All patches tensor (N, 3, H, W) in [0, 1]
        selected_indices: List of selected patch indices
        output_dir: Directory to save images
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving selected patches to: {output_dir}")
    
    for idx, patch_idx in enumerate(selected_indices):
        patch = patches[patch_idx]
        
        # Convert (3, H, W) -> (H, W, 3) and scale to [0, 255]
        patch_np = patch.permute(1, 2, 0).cpu().numpy()
        patch_np = (patch_np * 255).clip(0, 255).astype(np.uint8)
        patch_img = Image.fromarray(patch_np)
        
        # Save
        filename = f"{prefix}_{idx:03d}_original_{patch_idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        patch_img.save(filepath)
    
    print(f"   ✓ Saved {len(selected_indices)} patches")


def main():
    """Main pipeline execution."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Budgeted Dual-DPP for PDF Document Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # High quality (default)
    python main.py document.pdf
    
    # With custom query and highlighting
    python main.py document.pdf --query "Find all figures" --budget 75.0 --highlight-pdf
    
    # Fast mode (lower quality)
    python main.py document.pdf --dpi 150 --patch-size 224 --embedding-batch-size 8
    
    # Maximum quality (for detailed analysis)
    python main.py document.pdf --dpi 600 --patch-size 672 --embedding-batch-size 1
        """
    )
    
    parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to input PDF file'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        default="Find all key technical diagrams, system architectures, and experimental results",
        help='Query for retrieval (default: technical content)'
    )
    
    parser.add_argument(
        '--budget',
        type=float,
        default=50.0,
        help='Cognitive budget for selection (default: 50.0)'
    )
    
    parser.add_argument(
        '--patches-per-page',
        type=int,
        default=10,  # Increased from 12 for better control
        help='Number of patches per page (default: 10)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF rendering - higher = better quality (default: 300, use 150 for speed)'
    )
    
    parser.add_argument(
        '--patch-size',
        type=int,
        default=448,
        help='Size of extracted patches in pixels (default: 448, use 224 for speed)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature for relevance scoring (default: 0.1)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.6,
        help='Weight for entropy in cost (default: 0.6)'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=0.4,
        help='Weight for clutter in cost (default: 0.4)'
    )
    
    parser.add_argument(
        '--save-patches',
        action='store_true',
        help='Save selected patches as images'
    )
    
    parser.add_argument(
        '--highlight-pdf',
        action='store_true',
        help='Create highlighted PDF showing selected patches'
    )
    
    parser.add_argument(
        '--highlight-color',
        type=str,
        default='red',
        choices=['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan'],
        help='Color for patch highlights (default: red)'
    )
    
    parser.add_argument(
        '--comparison-pdf',
        action='store_true',
        help='Create side-by-side comparison PDF (original vs highlighted)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for cognitive profiling (default: 32)'
    )
    
    parser.add_argument(
        '--embedding-batch-size',
        type=int,
        default=4,
        help='Batch size for ColPali encoding (smaller = less memory, default: 4)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("="*80)
    print("Budgeted Dual-DPP: Real PDF Document Retrieval")
    print("="*80)
    
    # Validate PDF path
    if not os.path.exists(args.pdf_path):
        print(f"\n❌ Error: PDF not found at: {args.pdf_path}")
        sys.exit(1)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    print(f"\n⚙️  Configuration:")
    print(f"   • PDF:                {args.pdf_path}")
    print(f"   • Query:              {args.query}")
    print(f"   • Budget:             {args.budget}")
    print(f"   • Patches per page:   {args.patches_per_page}")
    print(f"   • DPI:                {args.dpi}")
    print(f"   • Patch size:         {args.patch_size}×{args.patch_size}")
    print(f"   • Temperature:        {args.temperature}")
    print(f"   • Cost weights:       α={args.alpha}, β={args.beta}")
    print(f"   • Device:             {device}")
    print(f"   • Batch sizes:        profiling={args.batch_size}, embedding={args.embedding_batch_size}")
    print(f"   • Output directory:   {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # =========================================================================
    # STAGE 1: PDF PATCH EXTRACTION (Layout-Aware)
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 1: LAYOUT-AWARE PATCH EXTRACTION")
    print(f"="*80)
    
    try:
        # Use layout-aware extraction with configurable quality
        patches, patch_metadata = extract_layout_aware_patches(
            args.pdf_path,
            dpi=args.dpi,
            patch_size=args.patch_size,
            target_patches_per_page=args.patches_per_page
        )
        patches = patches.to(device)
        
        print(f"\n✓ Patches respect document layout:")
        print(f"   • Aligned with text columns")
        print(f"   • Follow paragraph boundaries")
        print(f"   • No awkward text splitting")
        print(f"   • High resolution ({args.dpi} DPI)")
    except Exception as e:
        print(f"\n❌ Error extracting patches: {e}")
        sys.exit(1)
    
    # =========================================================================
    # STAGE 2: COGNITIVE PROFILING
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 2: COGNITIVE PROFILING")
    print(f"="*80)
    
    profiler = CognitiveProfiler(alpha=args.alpha, beta=args.beta).to(device)
    
    print(f"\n🧠 Computing cognitive costs (batch_size={args.batch_size})...")
    all_costs = []
    for i in range(0, len(patches), args.batch_size):
        batch = patches[i:i+args.batch_size]
        costs = profiler(batch)
        all_costs.append(costs)
    
    patch_costs = torch.cat(all_costs)
    
    print(f"\n✓ Cognitive costs computed")
    print(f"   • Cost range:         [{patch_costs.min().item():.3f}, "
          f"{patch_costs.max().item():.3f}]")
    print(f"   • Mean cost:          {patch_costs.mean().item():.3f}")
    print(f"   • Std cost:           {patch_costs.std().item():.3f}")
    
    # =========================================================================
    # STAGE 3: LAYOUT-AWARE EMBEDDING
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 3: LAYOUT-AWARE EMBEDDING")
    print(f"="*80)
    
    embedder = ColPaliEmbedder(device=device)
    
    # Encode patches in small batches to avoid OOM
    print(f"\n🔢 Encoding patches in batches (batch_size={args.embedding_batch_size})...")
    embedding_batch_size = args.embedding_batch_size
    all_embeddings = []
    
    for i in range(0, len(patches), embedding_batch_size):
        batch = patches[i:i+embedding_batch_size]
        batch_embeddings = embedder.encode_patches(batch)
        all_embeddings.append(batch_embeddings.cpu())  # Move to CPU to free GPU memory
        
        # Clear cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        if (i // embedding_batch_size) % 10 == 0:
            print(f"   Processed {i + len(batch)}/{len(patches)} patches...")
    
    patch_embeddings = torch.cat(all_embeddings).to(device)
    
    print(f"\n📝 Encoding query: '{args.query}'")
    query_embedding = embedder.encode_query(args.query)
    
    print(f"\n✓ Embeddings computed")
    print(f"   • Patch embeddings:   {patch_embeddings.shape}")
    print(f"   • Query embedding:    {query_embedding.shape}")
    
    # =========================================================================
    # STAGE 4: BUDGETED DPP OPTIMIZATION
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 4: BUDGETED DPP OPTIMIZATION")
    print(f"="*80)
    
    config = DPPConfig(
        temperature=args.temperature,
        epsilon=1e-6,
        alpha=args.alpha,
        beta=args.beta
    )
    
    optimizer = BudgetedDualDPP(config).to(device)
    
    selected_indices, total_cost = optimizer.optimize(
        patch_embeddings=patch_embeddings,
        patch_costs=patch_costs,
        query_embedding=query_embedding,
        budget=args.budget,
        verbose=True
    )
    
    # =========================================================================
    # STAGE 5: ANALYSIS & VISUALIZATION
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"STAGE 5: ANALYSIS & VISUALIZATION")
    print(f"="*80)
    
    # Compute relevance scores for analysis
    relevance_scores = torch.exp(
        (patch_embeddings @ query_embedding.T).squeeze(1) / config.temperature
    )
    
    # Print summary
    print_selection_summary(
        selected_indices=selected_indices,
        patch_embeddings=patch_embeddings,
        patch_costs=patch_costs,
        relevance_scores=relevance_scores,
        budget=args.budget
    )
    
    # Get detailed metrics
    metrics = analyze_selection(
        selected_indices=selected_indices,
        patch_embeddings=patch_embeddings,
        patch_costs=patch_costs,
        relevance_scores=relevance_scores,
        budget=args.budget
    )
    
    # Save metrics to JSON
    import json
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    
    # Convert numpy/torch types to Python types for JSON
    metrics_serializable = {
        k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
        for k, v in metrics.items()
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print(f"\n💾 Saved metrics: {metrics_file}")
    
    # Generate visualization
    print(f"\n📊 Generating visualization...")
    viz_path = os.path.join(args.output_dir, 'selection_analysis.png')
    
    visualize_selection(
        selected_indices=selected_indices,
        patch_embeddings=patch_embeddings,
        patch_costs=patch_costs,
        relevance_scores=relevance_scores,
        budget=args.budget,
        save_path=viz_path
    )
    
    # Save selected patches if requested
    if args.save_patches:
        patches_dir = os.path.join(args.output_dir, 'selected_patches')
        save_selected_patches(patches, selected_indices, patches_dir)
    
    # Create highlighted PDF if requested
    if args.highlight_pdf or args.comparison_pdf:
        print(f"\n" + "="*80)
        print(f"STAGE 6: PDF HIGHLIGHTING (Layout-Aware)")
        print(f"="*80)
        
        try:
            # Use layout-aware highlighter with exact bounding boxes
            highlighter = LayoutAwareHighlighter(
                pdf_path=args.pdf_path,
                dpi=args.dpi  # Match extraction DPI
            )
            
            # Get relevance scores for display
            relevance_values = relevance_scores[selected_indices].cpu().tolist()
            
            # Color mapping
            color_map = {
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'yellow': (255, 255, 0),
                'orange': (255, 165, 0),
                'purple': (128, 0, 128),
                'cyan': (0, 255, 255)
            }
            box_color = color_map.get(args.highlight_color, (255, 0, 0))
            
            # Create standard highlighted PDF
            if args.highlight_pdf:
                highlighted_pdf_path = os.path.join(
                    args.output_dir, 
                    'highlighted_document.pdf'
                )
                
                highlighter.highlight_patches(
                    selected_indices=selected_indices,
                    metadata=patch_metadata,
                    output_path=highlighted_pdf_path,
                    box_color=box_color,
                    box_width=4,
                    opacity=0.2,
                    show_labels=True,
                    scores=relevance_values
                )
                
                print(f"   ✓ Highlights respect document structure")
                print(f"   ✓ Bounding boxes align with content boundaries")
        
        except Exception as e:
            print(f"⚠️  Warning: PDF highlighting failed: {e}")
            print(f"   Continuing without highlighted PDF...")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n" + "="*80)
    print(f"✅ PIPELINE COMPLETE")
    print(f"="*80)
    
    print(f"\n🎉 Successfully processed: {args.pdf_path}")
    print(f"   • Total patches:      {len(patches)}")
    print(f"   • Selected patches:   {len(selected_indices)}")
    print(f"   • Budget utilization: {total_cost/args.budget*100:.1f}%")
    
    print(f"\n📁 Output files:")
    print(f"   • Visualization:      {viz_path}")
    print(f"   • Metrics:            {metrics_file}")
    if args.save_patches:
        print(f"   • Selected patches:   {patches_dir}/")
    if args.highlight_pdf:
        highlighted_pdf_path = os.path.join(args.output_dir, 'highlighted_document.pdf')
        if os.path.exists(highlighted_pdf_path):
            print(f"   • Highlighted PDF:    {highlighted_pdf_path}")
    if args.comparison_pdf:
        comparison_pdf_path = os.path.join(args.output_dir, 'comparison_document.pdf')
        if os.path.exists(comparison_pdf_path):
            print(f"   • Comparison PDF:     {comparison_pdf_path}")
    
    print(f"\n" + "="*80)


if __name__ == "__main__":
    main()