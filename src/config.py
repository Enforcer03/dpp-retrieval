"""
Configuration management for Budgeted Dual-DPP.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DPPConfig:
    """
    Configuration for Budgeted Dual-DPP optimization.
    
    Attributes:
        temperature: Temperature τ for relevance scoring (default: 0.1)
                    Lower values → sharper relevance distribution
        epsilon: Numerical stability constant for matrix operations (default: 1e-6)
        alpha: Weight for entropy component in cognitive cost (default: 0.6)
        beta: Weight for clutter component in cognitive cost (default: 0.4)
        device: Computation device ('cuda', 'cpu', or None for auto-detect)
        num_bins: Number of histogram bins for entropy computation (default: 256)
    """
    
    # Optimization parameters
    temperature: float = 0.1
    epsilon: float = 1e-6
    
    # Cognitive profiling parameters
    alpha: float = 0.6
    beta: float = 0.4
    num_bins: int = 256
    
    # System parameters
    device: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        
        if not (0 <= self.alpha <= 1):
            raise ValueError(f"alpha must be in [0,1], got {self.alpha}")
        
        if not (0 <= self.beta <= 1):
            raise ValueError(f"beta must be in [0,1], got {self.beta}")
        
        if abs(self.alpha + self.beta - 1.0) > 1e-6:
            import warnings
            warnings.warn(
                f"alpha + beta = {self.alpha + self.beta:.3f}, "
                f"consider normalizing to sum to 1.0"
            )
        
        if self.num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {self.num_bins}")


@dataclass
class ExperimentConfig:
    """
    Configuration for running experiments and demos.
    
    Attributes:
        num_pages: Number of document pages to simulate
        patches_per_page: Number of patches per page
        patch_size: Size of each patch (width/height in pixels)
        embedding_dim: Dimensionality of embeddings
        budget: Total cognitive budget for selection
        batch_size: Batch size for processing patches
        seed: Random seed for reproducibility
    """
    
    num_pages: int = 100
    patches_per_page: int = 10
    patch_size: int = 224
    embedding_dim: int = 128
    budget: float = 50.0
    batch_size: int = 100
    seed: int = 42