"""
Cognitive cost profiling for visual patches.

Computes information-theoretic and visual complexity metrics to estimate
the cognitive load required to process each patch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CognitiveProfiler(nn.Module):
    """
    Computes cognitive cost of visual patches based on information theory.
    
    Cost Model:
        c(x) = α · H(x) + β · C(x)
        
        where:
            H(x) = -Σ p(i) log p(i)  (Shannon entropy of pixel intensities)
            C(x) = ‖∇x‖              (visual clutter via gradient magnitude)
    
    Mathematical Intuition:
        - High entropy → more information → harder to process
        - High clutter (gradients) → more visual noise → harder to scan
    
    Reference:
        Shannon, C. E. (1948). A mathematical theory of communication.
        Rosenholtz, R. et al. (2007). Measuring visual clutter.
    """
    
    def __init__(
        self, 
        alpha: float = 0.6, 
        beta: float = 0.4, 
        num_bins: int = 256
    ):
        """
        Initialize cognitive profiler.
        
        Args:
            alpha: Weight for entropy component (information density)
            beta: Weight for clutter component (visual complexity)
            num_bins: Histogram bins for entropy estimation (default: 256 for uint8)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_bins = num_bins
        
        # Pre-register Sobel kernels for gradient computation
        # Using standard 3×3 Sobel operators
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=torch.float32
        )
        
        # Expand to (out_channels=1, in_channels=1, H, W) for conv2d
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))
    
    def _compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Shannon entropy H(X) = -Σ p(i) log p(i) for each patch.
        
        Args:
            x: Tensor of shape (B, C, H, W) with values in [0, 1]
        
        Returns:
            Entropy vector of shape (B,)
        
        Implementation Notes:
            - Uses torch.histc for efficient histogram computation
            - Handles log(0) by filtering out zero-count bins
            - Normalizes histogram to get probability distribution
        """
        B, C, H, W = x.shape
        entropies = []
        
        for i in range(B):
            # Flatten spatial dimensions, keep channels separate
            patch = x[i].flatten()  # (C*H*W,)
            
            # Compute histogram (bin counts)
            hist = torch.histc(patch, bins=self.num_bins, min=0.0, max=1.0)
            
            # Normalize to probability distribution
            hist = hist / hist.sum()
            
            # Filter out zero bins (avoid log(0))
            hist = hist[hist > 0]
            
            # Compute entropy: H = -Σ p log p
            entropy = -(hist * torch.log(hist)).sum()
            entropies.append(entropy)
        
        return torch.stack(entropies)
    
    def _compute_clutter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute visual clutter C(x) = ‖∇x‖ using Sobel gradient magnitude.
        
        Args:
            x: Tensor of shape (B, C, H, W) with values in [0, 1]
        
        Returns:
            Clutter vector of shape (B,)
        
        Mathematical Details:
            For each pixel (i,j):
                G_x = Sobel_x * I        (horizontal gradient)
                G_y = Sobel_y * I        (vertical gradient)
                G = √(G_x² + G_y²)       (gradient magnitude)
            
            Clutter = mean(G) across all pixels
        
        Reference:
            Sobel, I., & Feldman, G. (1968). 
            A 3x3 isotropic gradient operator for image processing.
        """
        B, C, H, W = x.shape
        
        # Convert to grayscale if RGB (simple averaging)
        if C == 3:
            gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            gray = x
        
        # Apply Sobel kernels via convolution
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)  # (B, 1, H, W)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)  # (B, 1, H, W)
        
        # Compute gradient magnitude: √(Gx² + Gy²)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Average across spatial dimensions
        clutter = gradient_magnitude.mean(dim=[1, 2, 3])  # (B,)
        
        return clutter
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total cognitive cost for a batch of patches.
        
        Args:
            x: Tensor of shape (B, C, H, W) with values in [0, 1]
        
        Returns:
            Cost vector of shape (B,) where cost[i] = α·H(x_i) + β·C(x_i)
        
        Example:
            >>> profiler = CognitiveProfiler(alpha=0.6, beta=0.4)
            >>> patches = torch.rand(32, 3, 224, 224)
            >>> costs = profiler(patches)
            >>> print(costs.shape)  # torch.Size([32])
        """
        entropy = self._compute_entropy(x)
        clutter = self._compute_clutter(x)
        
        # Linear combination of information and visual complexity
        cost = self.alpha * entropy + self.beta * clutter
        
        return cost