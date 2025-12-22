"""
ICB-Sum: Instruction-driven Cognitive Budget Summarization
=============================================================

A research-grade implementation of budgeted Determinantal Point Process (DPP)
optimization for visual patch selection in multimodal documents.

Target Venue: KDD 2025
"""

__version__ = "1.0.0"
__author__ = "Ved Umrajkar"

from .config import DPPConfig
from .cognitive_profiler import CognitiveProfiler
from .colpali_embedder import ColPaliEmbedder
from .dpp_optimizer import BudgetedDualDPP
from .pdf_highlighter import PDFHighlighter
from .layout_aware_extractor import LayoutAwarePatchExtractor

__all__ = [
    "DPPConfig",
    "CognitiveProfiler",
    "ColPaliEmbedder",
    "BudgetedDualDPP",
    "PDFHighlighter",
    "LayoutAwarePatchExtractor"
]