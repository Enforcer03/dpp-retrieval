## Overview
- The study benchmarks watermark robustness and inversion behavior of FLUX.1-dev (rectified flow, MM-DiT) against Stable Diffusion 2.1 (DDIM-based LDM) under matched configurations, focusing on separability of watermarked vs non-watermarked images and sensitivity to prompt guidance. [bf950cc5ac1a0413][98e0a1670aa761b8][a6b5d1ed98ed3e14]

## Key Findings
- FLUX.1-dev reconstructs latent noise more accurately than SD 2.1 on clean (non-attacked) images, but this advantage largely disappears under attacks; separability between watermarked and non-watermarked distributions collapses in attacked scenarios. [b4610aa875f73007]
- SD 2.1 (DDIM) maintains robust separation between watermarked and non-watermarked images via naive inversion, and its performance is largely insensitive to prompt guidance, unlike FLUX.1-dev. [98e0a1670aa761b8]
- FLUX.1-dev’s inversion quality is highly sensitive to prompt presence and accuracy; exact prompt guidance minimizes reconstruction error for clean images but does not help when images are attacked. [b4610aa875f73007]
- Architectural and training differences explain robustness gaps: FLUX’s rectified flow objective straightens the generative path for efficient sampling but discards information needed for invertibility, limiting watermark recovery even if higher-order solvers are used. [8767d43ace527e6c][8ba006aa21056da6][a6b5d1ed98ed3e14]
- Deep text–image entanglement in FLUX (MM-DiT + T5 encoder) makes generation more fundamentally prompt-dependent than SD 2.1’s UNet + CLIP cross-attention, further increasing inversion’s reliance on accurate prompts. [a6b5d1ed98ed3e14]
- Even with identical prompts, original vs reconstructed latents in FLUX yield perceptibly different images, indicating imperfect invertibility and potential limits for watermark-based provenance on FLUX. [4776a267e51de637][4374753e2f4379c4]

## Evidence
- Experimental parity: both FLUX.1-dev and SD 2.1 use 28 sampling steps for generation and inversion, guidance scale 3.5 (classifier-free), Euler ODE solver, and uniform timestep schedule between t=0 and t=1 to balance quality and efficiency and ensure fair comparison. [73aee49c9fb726ef][bf950cc5ac1a0413]
- Reproducibility controls: shared global random seed for initial latents, identical watermark key across tests, and uniform timestep schedule chosen because it significantly improves inversion quality. [0a27874556ade7d9]
- Clean vs attacked behavior: exact prompt guidance minimizes reconstruction error in both Fourier and spatial domains for clean images, but fails to aid reconstruction for attacked images; FLUX’s clean-image advantage over SD 2.1 vanishes under attacks. [b4610aa875f73007]
- SD 2.1 robustness: DDIM inversion yields strong, prompt-agnostic separation between watermarked and non-watermarked images. [98e0a1670aa761b8]
- FLUX architectural factors: MM-DiT with deeply entangled text–image features and T5 text encoder vs SD’s UNet + CLIP; rectified flow objective optimizes nearly linear transport for fast sampling but sacrifices invertibility and watermark recoverability. [8767d43ace527e6c][8ba006aa21056da6][a6b5d1ed98ed3e14]
- Qualitative gap: generated vs reconstructed FLUX images differ noticeably even under matched prompts, underscoring inversion limitations. [4776a267e51de637][4374753e2f4379c4]
- Watermark robustness is quantified via metrics and standardized attack scenarios such as the Waves benchmark, providing a structured robustness assessment context. [65a5c5af5466ce1d][6847523eb677e85f][3b99398f89f56cd4]

## Risks
- For FLUX.1-dev, watermark-based provenance and detection are fragile under realistic image attacks; reduced separability between watermarked and non-watermarked distributions can lead to high false negatives or ambiguous attribution. [b4610aa875f73007]
- Heavy dependence on accurate prompts for FLUX inversion introduces operational risk: missing, noisy, or adversarial prompts can severely degrade watermark recovery, unlike SD 2.1. [b4610aa875f73007][98e0a1670aa761b8][a6b5d1ed98ed3e14]
- Fundamental architectural constraints (rectified flow straight paths, information discard) limit the upside of purely numerical improvements (e.g., higher-order solvers), capping achievable robustness for FLUX-style models without deeper redesign. [8767d43ace527e6c][a6b5d1ed98ed3e14]
- Perceptible differences between original and reconstructed FLUX images, even with identical prompts, may undermine legal or forensic confidence in watermark-based evidence. [4776a267e51de637][4374753e2f4379c4]

## Next Steps
- For benchmarking:
  - Extend robustness evaluation using standardized attack suites like Waves to systematically compare FLUX.1-dev and SD 2.1 across diverse manipulations and intensities. [6847523eb677e85f][3b99398f89f56cd4]
  - Report explicit separability metrics (e.g., ROC/AUC for watermark detection) for clean vs attacked conditions to quantify the observed collapse in FLUX robustness. [65a5c5af5466ce1d][b4610aa875f73007]
- For model and method design:
  - Develop inversion techniques tailored to rectified flow models that compensate for information loss along straightened paths, potentially via auxiliary networks or modified training objectives. [05b9fd7dd2e748e6][a6b5d1ed98ed3e14]
  - Explore watermarking schemes that are less reliant on precise invertibility for FLUX (e.g., encoder-side or feature-space watermarks) and test their robustness under the same configuration used here. [05b9fd7dd2e748e6][b4610aa875f73007]
  - Investigate architectural or training modifications to FLUX that preserve more inversion-relevant information while retaining sampling efficiency, then re-benchmark against SD 2.1 under the same 28-step, Euler, uniform-schedule setup. [8767d43ace527e6c][73aee49c9fb726ef][a6b5d1ed98ed3e14]
- For deployment policy:
  - Prefer SD 2.1–like architectures for applications where robust watermark-based provenance is critical, while treating FLUX.1-dev as higher risk unless enhanced watermarking/inversion methods are in place. [98e0a1670aa761b8][b4610aa875f73007]