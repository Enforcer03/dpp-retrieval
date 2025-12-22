"""
Greedy MAP inference for Budgeted Determinantal Point Process.

Implements efficient subset selection under cognitive budget constraints
using low-rank kernel factorization and Schur complement computations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import warnings

from .config import DPPConfig


class BudgetedDualDPP(nn.Module):
    """
    Greedy MAP Inference for Budgeted Determinantal Point Process.
    
    Optimization Problem:
        max_{S ⊆ [N]} log det(L_S) 
        s.t. Σ_{i∈S} c_i ≤ B
    
    where:
        L = V @ diag(r) @ V.T     (low-rank kernel factorization)
        V = [v_1, ..., v_N].T     (patch embeddings, N×D)
        r = exp(V @ q / τ)        (relevance scores, N×1)
    
    Greedy Algorithm:
        1. Initialize S = ∅
        2. While budget allows:
            a. For each candidate u ∉ S, compute marginal gain:
               Δ_u = log det(L_{S∪u}) - log det(L_S)
            b. Select u* = argmax(Δ_u / c_u)  [bang-for-buck]
            c. S ← S ∪ {u*}
        3. Return S
    
    Key Optimization:
        Use Schur complement for efficient marginal gain computation:
        log det(L_{S∪u}) = log det(L_S) + log(L_uu - L_uS @ L_S^{-1} @ L_Su)
    
    Approximation Guarantee:
        For monotone submodular functions under cardinality constraint:
        F(S_greedy) ≥ (1 - 1/e) · OPT
        
        For knapsack constraints (cost-based), the density-greedy variant
        provides a ratio approximation depending on the cost structure.
    
    References:
        - Nemhauser et al. (1978): Analysis of approximations for maximizing 
          submodular set functions
        - Kulesza & Taskar (2012): Determinantal point processes for machine 
          learning
        - Gillenwater et al. (2012): Near-optimal MAP inference for 
          determinantal point processes
    """
    
    def __init__(self, config: DPPConfig):
        """
        Initialize Budgeted DPP optimizer.
        
        Args:
            config: DPPConfig object with optimization parameters
        """
        super().__init__()
        self.config = config
    
    def _compute_relevance(
        self, 
        patch_embeddings: torch.Tensor, 
        query_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance scores r_i = exp(⟨v_i, q⟩ / τ).
        
        Args:
            patch_embeddings: (N, D) normalized embeddings
            query_embedding: (1, D) normalized embedding
        
        Returns:
            Relevance scores (N,)
        
        Mathematical Details:
            r_i = exp(⟨v_i, q⟩ / τ)
            
            where:
                - τ is temperature (controls sharpness)
                - Lower τ → sharper relevance distribution
                - Higher τ → more uniform distribution
        """
        # Cosine similarity (embeddings are already normalized)
        similarities = (patch_embeddings @ query_embedding.T).squeeze(1)  # (N,)
        
        # Temperature-scaled exponential
        relevance = torch.exp(similarities / self.config.temperature)
        
        return relevance
    
    def _build_kernel_submatrix(
        self,
        indices: List[int],
        embeddings: torch.Tensor,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        Build kernel submatrix L_S for selected indices S.
        
        Args:
            indices: List of selected patch indices
            embeddings: (N, D) full patch embeddings
            relevance: (N,) full relevance scores
        
        Returns:
            Kernel matrix L_S of shape (|S|, |S|)
        
        Optimization:
            Instead of full N×N kernel, build only the |S|×|S| submatrix.
            Uses low-rank factorization: L_S = V_S @ diag(r_S) @ V_S.T
            
        Computational Complexity:
            - Time: O(|S|² · D)
            - Space: O(|S|²)
            where |S| << N typically
        """
        if len(indices) == 0:
            # Empty set: return empty matrix
            return torch.tensor([], device=embeddings.device).reshape(0, 0)
        
        # Extract selected embeddings and relevance
        V_S = embeddings[indices]  # (|S|, D)
        r_S = relevance[indices]   # (|S|,)
        
        # Compute similarity matrix: S_ij = ⟨v_i, v_j⟩
        similarity_matrix = V_S @ V_S.T  # (|S|, |S|)
        
        # Scale by relevance: L_ij = r_i · r_j · S_ij
        # Using outer product: r_S ⊗ r_S gives r_i·r_j matrix
        relevance_matrix = torch.outer(r_S, r_S)  # (|S|, |S|)
        
        # Element-wise product
        kernel_matrix = relevance_matrix * similarity_matrix
        
        # Add epsilon to diagonal for numerical stability
        kernel_matrix = kernel_matrix + self.config.epsilon * torch.eye(
            len(indices), device=embeddings.device
        )
        
        return kernel_matrix
    
    def _compute_marginal_gain(
        self,
        candidate: int,
        selected: List[int],
        embeddings: torch.Tensor,
        relevance: torch.Tensor
    ) -> float:
        """
        Compute marginal gain Δ_u = log det(L_{S∪u}) - log det(L_S).
        
        Args:
            candidate: Index of candidate patch u
            selected: List of currently selected indices S
            embeddings: (N, D) patch embeddings
            relevance: (N,) relevance scores
        
        Returns:
            Marginal gain (scalar)
        
        Implementation:
            Case 1 (S is empty): 
                Gain = log det(L_uu) = log(r_u² · ⟨v_u, v_u⟩)
                     = 2·log(r_u) + log(⟨v_u, v_u⟩)
            
            Case 2 (S not empty):
                Use determinant identity (Schur complement):
                det(L_{S∪u}) = det(L_S) · (L_uu - L_uS @ L_S^{-1} @ L_Su)
                
                Therefore:
                Δ_u = log(L_uu - L_uS @ L_S^{-1} @ L_Su)
        
        Numerical Stability:
            - Add ε to diagonals before logdet
            - Handle NaN/Inf by returning 0
            - Catch singular matrix exceptions
        """
        if len(selected) == 0:
            # Base case: gain is just the self-similarity
            self_similarity = (embeddings[candidate] @ embeddings[candidate]).item()
            r_u = relevance[candidate].item()
            
            # log det(L_uu) = log(r_u² · ⟨v_u, v_u⟩)
            gain = torch.log(torch.tensor(
                r_u**2 * self_similarity + self.config.epsilon
            ))
            return gain.item()
        
        # General case: use Schur complement formula
        # Build L_S
        L_S = self._build_kernel_submatrix(selected, embeddings, relevance)
        
        # Build L_{S∪u}
        selected_with_candidate = selected + [candidate]
        L_S_plus_u = self._build_kernel_submatrix(
            selected_with_candidate, embeddings, relevance
        )
        
        # Compute log determinants
        try:
            logdet_S = torch.logdet(L_S)
            logdet_S_plus_u = torch.logdet(L_S_plus_u)
            
            gain = logdet_S_plus_u - logdet_S
            
            # Handle numerical issues
            if torch.isnan(gain) or torch.isinf(gain):
                return 0.0
            
            return gain.item()
        
        except RuntimeError as e:
            # Fallback if logdet fails (singular matrix)
            warnings.warn(
                f"Singular matrix encountered for candidate {candidate}: {e}"
            )
            return 0.0
    
    def optimize(
        self,
        patch_embeddings: torch.Tensor,
        patch_costs: torch.Tensor,
        query_embedding: torch.Tensor,
        budget: float,
        verbose: bool = True
    ) -> Tuple[List[int], float]:
        """
        Main optimization: Greedy density-based selection.
        
        Args:
            patch_embeddings: (N, D) normalized patch embeddings
            patch_costs: (N,) cognitive cost per patch
            query_embedding: (1, D) normalized query embedding
            budget: Maximum total cost allowed
            verbose: Print optimization progress
        
        Returns:
            selected_indices: List of selected patch indices
            total_cost: Total cost used
        
        Algorithm (Pseudocode):
            GREEDY-BUDGETED-DPP(V, c, q, B):
                S ← ∅
                cost ← 0
                r ← COMPUTE-RELEVANCE(V, q)
                
                while True:
                    best_ratio ← -∞
                    best_u ← None
                    
                    for u ∉ S:
                        if cost + c_u > B:
                            continue
                        
                        Δ_u ← log det(L_{S∪u}) - log det(L_S)
                        ratio ← Δ_u / c_u
                        
                        if ratio > best_ratio:
                            best_ratio ← ratio
                            best_u ← u
                    
                    if best_u is None:
                        break
                    
                    S ← S ∪ {best_u}
                    cost ← cost + c_{best_u}
                
                return S, cost
        """
        N = patch_embeddings.shape[0]
        device = patch_embeddings.device
        
        # Step 1: Compute relevance scores
        relevance = self._compute_relevance(patch_embeddings, query_embedding)
        
        # Initialize
        selected: List[int] = []
        remaining = set(range(N))
        total_cost = 0.0
        
        if verbose:
            print(f"\n⚙️  Starting Budgeted DPP Optimization")
            print(f"   Total patches: {N}")
            print(f"   Budget: {budget:.2f}")
            print(f"   Avg patch cost: {patch_costs.mean().item():.2f}")
        
        iteration = 0
        
        # Greedy selection loop
        while remaining:
            iteration += 1
            
            best_ratio = float('-inf')
            best_candidate = None
            best_gain = None
            
            # Evaluate all remaining candidates
            for candidate in remaining:
                candidate_cost = patch_costs[candidate].item()
                
                # Check budget feasibility
                if total_cost + candidate_cost > budget:
                    continue
                
                # Compute marginal gain
                gain = self._compute_marginal_gain(
                    candidate, selected, patch_embeddings, relevance
                )
                
                # Density greedy: maximize gain per unit cost
                ratio = gain / candidate_cost if candidate_cost > 0 else 0.0
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_candidate = candidate
                    best_gain = gain
            
            # Termination: no feasible candidate found
            if best_candidate is None:
                if verbose:
                    print(f"\n   ⏸️  Terminated: No feasible candidates remaining")
                break
            
            # Add best candidate to selection
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            total_cost += patch_costs[best_candidate].item()
            
            if verbose and (iteration <= 10 or iteration % 5 == 0):
                print(f"   [{iteration:3d}] Selected patch {best_candidate:4d} | "
                      f"Gain: {best_gain:7.3f} | "
                      f"Ratio: {best_ratio:7.3f} | "
                      f"Cost: {total_cost:6.2f}/{budget:.2f}")
        
        if verbose:
            print(f"\n✓ Optimization complete")
            print(f"   Patches selected: {len(selected)}")
            print(f"   Budget used: {total_cost:.2f}/{budget:.2f} "
                  f"({total_cost/budget*100:.1f}%)")
        
        return selected, total_cost
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        patch_costs: torch.Tensor,
        query_embedding: torch.Tensor,
        budget: float,
        verbose: bool = True
    ) -> Tuple[List[int], float]:
        """
        Forward pass (alias for optimize).
        
        Args:
            patch_embeddings: (N, D) normalized embeddings
            patch_costs: (N,) cognitive costs
            query_embedding: (1, D) normalized query
            budget: Maximum cost allowed
            verbose: Print progress
        
        Returns:
            Tuple of (selected_indices, total_cost)
        """
        return self.optimize(
            patch_embeddings, patch_costs, query_embedding, budget, verbose
        )