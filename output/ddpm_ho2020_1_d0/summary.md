## Overview
- Hardware and architecture choices are driven by fitting large U-Net–style diffusion models into memory, achieving reasonable training/sampling throughput, and balancing model size vs. image resolution and dataset scale. [40de030796b3d32a][7e77adee2523ee5a][e22df3321d2941a1]

## Key Findings
- Compute platform:
  - All experiments used TPU v3-8, roughly comparable to 8 NVIDIA V100 GPUs. [e22df3321d2941a1]
- Model sizes and scaling:
  - CIFAR10 model: 35.7M parameters. [7e77adee2523ee5a]
  - LSUN and CelebA-HQ 256×256 models: 114M parameters. [7e77adee2523ee5a]
  - Larger LSUN Bedroom variant: ~256M parameters (increased filter count). [7e77adee2523ee5a]
- Architecture specifics:
  - Backbone: PixelCNN++-style U-Net built on a Wide ResNet. [7e77adee2523ee5a]
  - Normalization: group normalization instead of weight normalization (simpler implementation). [7e77adee2523ee5a]
  - Resolutions: 4 resolution levels for 32×32 (down to 4×4); 6 levels for 256×256. [7e77adee2523ee5a]
  - Two convolutional residual blocks per resolution; self-attention at 16×16 between conv blocks. [7e77adee2523ee5a]
  - Time conditioning: Transformer sinusoidal position embeddings added into each residual block. [7e77adee2523ee5a]
- Memory-driven hyperparameter choices:
  - Initial hyperparameters (e.g., network size) were chosen to fit within memory; most subsequent tuning focused on CIFAR10 sample quality and was then transferred to other datasets. [40de030796b3d32a]
- Batch sizes and EMA:
  - Batch size: 128 for CIFAR10; 64 for larger 256×256 images (no sweep). [19d09dc632f92a8d]
  - EMA decay on model parameters: 0.9999 (no sweep). [19d09dc632f92a8d]
- Training throughput and wall-clock:
  - CIFAR10 (32×32, 35.7M params, batch 128): 21 steps/s; 800k steps total; ~106 hours to completion. [e22df3321d2941a1]
  - CelebA-HQ/LSUN 256×256 (114M params, batch 64): 2.2 steps/s. [e22df3321d2941a1]
  - Training steps per dataset: CelebA-HQ 0.5M; LSUN Bedroom 2.4M; LSUN Cat 1.8M; LSUN Church 1.2M; larger LSUN Bedroom 1.15M. [e22df3321d2941a1]
- Sampling cost:
  - CIFAR10: batch of 256 images in 17 seconds. [e22df3321d2941a1]
  - CelebA-HQ/LSUN 256×256: batch of 128 images in 300 seconds. [e22df3321d2941a1]
- Diffusion length vs. cost:
  - Diffusion steps T=1000, shorter than image dimensionality; can be shortened for faster sampling or lengthened for expressiveness. [f930eaf67c8f4e97]
- Objective and parameterization tradeoffs:
  - Best CIFAR10 performance with ε-prediction and L_simple (‖ε̂−ε̂_θ‖²): IS 9.46±0.11, FID 3.17. [a90568670cdcc514]
  - Alternative μ̂-prediction or different objectives yield worse FID/IS or instability. [a90568670cdcc514]
- Reverse-process variance choice:
  - Use fixed isotropic Σ_θ = σ_t² I with σ_t² set to either β_t or \tilde{β}_t; both perform similarly and correspond to entropy bounds. [5db6a16cf1c632d5]

## Evidence
- Compute and runtime:
  - TPU v3-8 usage; step rates; training durations; sampling times; per-dataset step counts. [e22df3321d2941a1]
- Architecture:
  - PixelCNN++/U-Net/Wide ResNet backbone; group norm; resolution hierarchy; residual and attention blocks; time embeddings; parameter counts and scaling to 256M. [7e77adee2523ee5a]
- Hyperparameters:
  - Memory-constrained initial design; batch sizes; EMA decay; transfer of CIFAR10-tuned hyperparameters to other datasets. [40de030796b3d32a][19d09dc632f92a8d]
- Diffusion process:
  - T=1000; rationale that T can be adjusted for speed vs. expressiveness. [f930eaf67c8f4e97]
  - Fixed isotropic Σ_θ choices and their theoretical interpretation. [5db6a16cf1c632d5]
- Objective ablation:
  - Quantitative IS/FID for different parameterizations and losses, highlighting ε-prediction with L_simple as best. [a90568670cdcc514]

## Risks
- Hardware budget underestimation:
  - Training 100M–250M parameter models with long diffusion chains (T≈1000) is compute- and time-intensive; a single TPU v3-8 equivalent may still require multi-day runs per model. [7e77adee2523ee5a][e22df3321d2941a1]
- Memory constraints:
  - Architecture and batch sizes were explicitly constrained by memory; attempting larger models, higher resolutions, or larger batches on smaller GPUs may cause OOM without careful scaling (e.g., reduced channels, gradient checkpointing). [40de030796b3d32a][19d09dc632f92a8d][7e77adee2523ee5a]
- Sampling latency:
  - High sampling times (e.g., 300s for 128×256² images) may be unacceptable for interactive or production use unless T is reduced or hardware scaled out. [e22df3321d2941a1][f930eaf67c8f4e97]
- Objective/parameterization instability:
  - Some reverse-process parameterizations and objectives were unstable and produced poor samples, indicating sensitivity to these design choices. [a90568670cdcc514]
- Generalization of hyperparameters:
  - Hyperparameters tuned on CIFAR10 were transferred to other datasets; this may not be optimal for new domains or resolutions and could waste compute if blindly reused. [40de030796b3d32a]

## Next Steps
- Hardware budgeting:
  - For a CIFAR10-scale project, plan for at least one 8×V100-equivalent node for ~4–5 days per full training run (800k steps, batch 128). [e22df3321d2941a1]
  - For 256×256 models (≈114M params), budget significantly more time (multi-week) or more nodes, given 2.2 steps/s at batch 64 and up to 2.4M steps for LSUN Bedroom. [7e77adee2523ee5a][e22df3321d2941a1]
- Architecture selection:
  - Adopt the described U-Net/Wide-ResNet backbone with:
    - 4 resolution levels for 32×32, 6 for 256×256.
    - Two residual blocks per level and 16×16 self-attention.
    - Group normalization and sinusoidal time embeddings in each residual block. [7e77adee2523ee5a]
  - Choose parameter count (e.g., 35M vs. 114M vs. 256M) based on available memory and target resolution. [7e77adee2523ee5a]
- Hyperparameter baselines:
  - Start with batch size 128 (32×32) or 64 (256×256) and EMA decay 0.9999; adjust only if memory or convergence issues arise. [19d09dc632f92a8d]
- Objective and diffusion configuration:
  - Use ε-prediction with L_simple and fixed isotropic Σ_θ for stable, high-quality samples. [a90568670cdcc514][5db6a16cf1c632d5]
  - Begin with T≈1000, then experiment with smaller T to trade off sampling speed vs. quality once a baseline is established. [f930eaf67c8f4e97]
- Efficiency improvements:
  - If hardware is limited, consider:
    - Reducing channel counts to shrink parameter size.
    - Gradient checkpointing or mixed precision to fit larger models.
    - Early stopping or reduced training steps for exploratory runs before full-scale training. [40de030796b3d32a][7e77adee2523ee5a]