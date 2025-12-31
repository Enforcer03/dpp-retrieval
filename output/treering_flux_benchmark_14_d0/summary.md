## Overview
- FLUX.1-dev shows substantially weaker robustness of latent watermarks under common image attacks than Stable Diffusion 2.1 (DDIM), largely due to architectural and training differences that favor fast sampling over invertibility and watermark recoverability. [a6b5d1ed98ed3e14][f8b8eaed5124712d][bf00d0361b190a65]

## Key Findings
- **Baseline (no-attack) inversion quality**  
  - FLUX.1-dev achieves moderate latent reconstruction without attacks; prompt guidance measurably improves inversion (NMAE drops from 0.303→0.232; NMSE 0.106→0.063). [f8b8eaed5124712d]  
  - SD 2.1 (DDIM) maintains robust separation between watermarked and non-watermarked images even with naive inversion and is largely insensitive to prompt guidance, indicating stronger inherent invertibility. [98e0a1670aa761b8]
- **Attack robustness gap (blur/noise)**  
  - For FLUX.1-dev, blur and noise attacks catastrophically degrade watermark reconstruction: NMAE jumps to ~1.26–1.34 and NMSE to ~1.59–1.82, far above no-attack values, with minimal benefit from prompts. [f8b8eaed5124712d]  
  - In the Fourier domain, blur/noise attacks drastically increase distances and obscure ring patterns for FLUX.1-dev, while prior work shows SD 2.1’s latent watermark signals remain highly resilient under similar attacks. [fc27eed87276c60f][bf00d0361b190a65]
- **Prompt dependence vs. SD 2.1**  
  - FLUX.1-dev’s inversion quality is highly sensitive to prompt accuracy: no-prompt vs. correct-prompt vs. VLM-generated prompts show clear performance spread, confirming strong reliance on text conditioning for reconstruction. [f8b8eaed5124712d][5079eca6d92150ba]  
  - SD 2.1 (DDIM) shows consistent watermark separability regardless of prompt guidance, indicating that watermark recovery is less entangled with text conditioning. [98e0a1670aa761b8]
- **Architectural and training causes**  
  - FLUX uses a Multimodal Diffusion Transformer (MM DiT) with deeply entangled text–image features and a T5 encoder, versus SD’s UNet + CLIP cross-attention; this makes FLUX generation more fundamentally prompt-dependent. [a6b5d1ed98ed3e14]  
  - Rectified flow training in FLUX optimizes nearly straight trajectories between noise and data, enabling high-quality images in fewer steps (e.g., 28 steps with Euler integration) but discarding information critical for inversion and watermark recovery. [8767d43ace527e6c][40bc4484a922fb06][a6b5d1ed98ed3e14]
- **Benchmarking context**  
  - Robustness is evaluated via metrics like NMAE, NMSE, and Fourier-space distances, and via standardized attack suites such as Waves for watermark robustness. [65a5c5af5466ce1d][6847523eb677e85f][f8b8eaed5124712d][fc27eed87276c60f]  
  - Tree Ring Watermarking, previously validated on diffusion models like SD 2.1, has not been fully adapted to newer rectified-flow architectures like FLUX, where effectiveness is clearly reduced under attack. [1184c0487248f82e][bf00d0361b190a65]

## Evidence
- Quantitative watermark extraction metrics for FLUX.1-dev and SD 2.1 (NMAE, NMSE, average |ŵᵢ − wᵢ|) across no-attack, blur, and noise conditions, with/without prompts and with VLM prompts. [f8b8eaed5124712d]  
- Fourier-space distance distributions showing strong prompt-guided inversion in clean FLUX.1-dev cases and drastic distance increases under blur/noise attacks. [fc27eed87276c60f]  
- Qualitative and prior quantitative findings that DDIM (SD 2.1) maintains robust separation of watermarked vs. non-watermarked images under naive inversion and attacks. [98e0a1670aa761b8][bf00d0361b190a65]  
- Architectural description of FLUX (MM DiT, T5 encoder, rectified flow objective) and its implications for prompt dependence and poor invertibility. [a6b5d1ed98ed3e14][8767d43ace527e6c][40bc4484a922fb06]  
- Use of Qwen2-VL-2B to generate prompts for real-world “no original prompt” scenarios, enabling comparison of prompt-free vs. prompt-guided inversion. [5079eca6d92150ba][e9570c7d2a34af1f]  
- Use of Waves benchmark and Tree Ring Watermarking as the robustness and watermarking baselines. [3b99398f89f56cd4][1184c0487248f82e][6847523eb677e85f]

## Risks
- **Weaker provenance guarantees for FLUX-based systems**  
  - Under realistic perturbations (blur, noise), FLUX.1-dev’s watermark signals become hard to distinguish from background frequencies, undermining reliable detection and separability of watermarked vs. non-watermarked content. [bf00d0361b190a65][fc27eed87276c60f]
- **High sensitivity to prompt availability and accuracy**  
  - Inversion and watermark recovery for FLUX.1-dev degrade significantly without accurate prompts; VLM-generated prompts only partially mitigate this, posing challenges for forensic use where original prompts are unknown. [f8b8eaed5124712d][5079eca6d92150ba]
- **Architectural limitations not easily fixed by numerics**  
  - The straightened rectified-flow trajectories inherently discard inversion-relevant information; higher-order solvers may offer only incremental gains, limiting how much robustness can be recovered without rethinking the training objective. [a6b5d1ed98ed3e14]
- **Mismatch between existing watermark designs and new architectures**  
  - Tree Ring Watermarking, tuned for traditional diffusion (e.g., SD 2.1), is less effective on FLUX.1-dev, risking overestimation of watermark robustness if benchmarks are not architecture-aware. [1184c0487248f82e][bf00d0361b190a65]

## Next Steps
- **Benchmark systematically against SD 2.1 using Waves**  
  - Run side-by-side FLUX.1-dev vs. SD 2.1 evaluations across the full Waves attack suite (noise, blur, compression, cropping, etc.), reporting NMAE/NMSE and Fourier distances to quantify robustness gaps. [3b99398f89f56cd4][6847523eb677e85f][f8b8eaed5124712d]
- **Stratify by prompt condition**  
  - Evaluate FLUX.1-dev under: (1) original prompt, (2) Qwen2-VL-generated prompt, (3) no prompt, to measure how much robustness depends on prompt quality and to set operational expectations for forensic workflows. [5079eca6d92150ba][e9570c7d2a34af1f]
- **Explore architecture- and objective-aware watermarking**  
  - Design or adapt watermarking schemes specifically for rectified-flow and MM DiT architectures, potentially embedding signals earlier or more redundantly to survive the straightened trajectories. [a6b5d1ed98ed3e14][05b9fd7dd2e748e6]
- **Investigate modified training for better invertibility**  
  - Experiment with hybrid objectives that trade a small amount of sampling efficiency for improved invertibility (e.g., partial diffusion-style noise schedules or auxiliary inversion losses) and re-measure watermark robustness. [8767d43ace527e6c][40bc4484a922fb06][a6b5d1ed98ed3e14]
- **Develop evaluation and deployment guidelines**  
  - For systems built on FLUX.1-dev, document that watermark robustness is significantly weaker than SD 2.1 under attacks; recommend pairing FLUX with stronger external provenance mechanisms (e.g., cryptographic signatures) until architecture-aligned watermarking is mature. [1184c0487248f82e][bf00d0361b190a65]