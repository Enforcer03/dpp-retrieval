"""
SciDuet processing module for DPP-retrieval pipeline.

This module provides:
- SciDuetConverter: Convert SciDuet samples to DocumentArtifact
- Pipeline execution: Run retrieval/selection stages
- I/O utilities: Save outputs, run evaluation, batch processing
"""

from .converter import SciDuetConverter, extract_sample_ids, lookup_requirements
from .pipeline import run_pipeline_on_document
from .io_utils import (
    save_outputs,
    process_sciduet_batch,
    load_requirements_metadata,
    collect_scores,
)

# Selection methods supported
METHODS = ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"]

# Mapping: method -> (context_filename, eval_bundle_filename)
METHOD_FILES = {
    "greedy": ("context.json", "eval_bundle.json"),
    "topk": ("context_topk.json", "eval_bundle_topk.json"),
    "greedy_cov": ("context_greedy_cov.json", "eval_bundle_greedy_cov.json"),
    "cost_norm": ("context_cost_norm.json", "eval_bundle_cost_norm.json"),
    "dpp": ("context_dpp.json", "eval_bundle_dpp.json"),
}

__all__ = [
    "METHODS",
    "METHOD_FILES",
    "SciDuetConverter",
    "extract_sample_ids",
    "lookup_requirements",
    "run_pipeline_on_document",
    "save_outputs",
    "process_sciduet_batch",
    "load_requirements_metadata",
    "collect_scores",
]