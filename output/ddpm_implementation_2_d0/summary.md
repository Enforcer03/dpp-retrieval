## Overview
- Experiments were run on TPU v3-8 (compute similar to 8× V100 GPUs), with model sizes constrained initially by memory, then hyperparameters tuned mainly for CIFAR-10 and transferred to other datasets [40de030796b3d32a][e22df3321d2941a1].
- Training and sampling are computationally heavy, especially for higher-resolution datasets (CelebA-HQ/LSUN 256×256), which run an order of magnitude slower than CIFAR-10 [e22df3321d2941a1].

## Key Findings
- Hardware baseline: TPU v3-8 is sufficient to train CIFAR-10 and 256×256 models with the reported configurations; performance is roughly comparable to 8× V100 GPUs [e22df3321d2941a1].
- Throughput:
  - CIFAR-10: 21 training steps/s at batch size 128; 800k steps require ~106 hours; sampling 256 images takes 17 s [e22df3321d2941a1].
  - CelebA-HQ/LSUN 256×256: 2.2 training steps/s at batch size 64; sampling 128 images takes 300 s [e22df3321d2941a1].
- Training duration per dataset on TPU v3-8:
  - CIFAR-10: 800k steps (~4.4 days) [e22df3321d2941a1].
  - CelebA-HQ: 0.5M steps [e22df3321d2941a1].
  - LSUN Bedroom: 2.4M steps; larger variant: 1.15M steps [e22df3321d2941a1].
  - LSUN Cat: 1.8M steps; LSUN Church: 1.2M steps [e22df3321d2941a1].
- Architectural/algorithmic simplification: Forward-process variances βₜ are fixed constants; the approximate posterior has no learnable parameters, simplifying memory and compute (L_T is constant and ignored in training) [153dc7c5f73856d6].
- Hyperparameter strategy: Network size was first constrained to fit memory; most hyperparameter search focused on CIFAR-10 sample quality, then settings were reused for other datasets, reducing search cost on larger models [40de030796b3d32a].
- Sampling quality vs. compute: More reverse-process steps improve Inception Score and reduce FID up to 1,000 steps, implying a direct tradeoff between sampling time and quality [ac419d177969b0be].

## Evidence
- TPU and throughput details:
  - “We used TPU v3-8 (similar to 8 V100 GPUs) for all experiments. Our CIFAR model trains at 21 steps per second at batch size 128 (106 hours to train to completion at 800k steps), and sampling a batch of 256 images takes 17 seconds. Our CelebA-HQ/LSUN (256²) models train at 2.2 steps per second at batch size 64, and sampling a batch of 128 images takes 300 seconds. We trained on CelebA-HQ for 0.5M steps, LSUN Bedroom for 2.4M steps, LSUN Cat for 1.8M steps, and LSUN Church for 1.2M steps. The larger LSUN Bedroom model was trained for 1.15M steps.” [e22df3321d2941a1]
- Memory-constrained architecture and hyperparameter transfer:
  - “Apart from an initial choice of hyperparameters early on to make network size fit within memory constraints, we performed the majority of our hyperparameter search to optimize for CIFAR10 sample quality, then transferred the resulting settings over to the other datasets” [40de030796b3d32a].
- Fixed forward-process variances:
  - “We ignore the fact that the forward process variances β_t are learnable by reparameterization and instead fix them to constants (see Section 4 for details). Thus, in our implementation, the approximate posterior q has no learnable parameters, so L_T is a constant during training and can be ignored.” [153dc7c5f73856d6]
- Sampling quality vs. reverse steps:
  - Inception Score increases and FID decreases monotonically as reverse-process steps increase from 0 to 1,000, indicating better quality with more steps [ac419d177969b0be].

## Risks
- Compute and time cost:
  - Multi-day training runs per model on TPU v3-8; replicating or extending experiments on smaller hardware will significantly increase wall-clock time [e22df3321d2941a1].
- Memory constraints:
  - Initial architecture was explicitly sized to fit memory; scaling up model depth/width or resolution without more memory may fail or require aggressive gradient checkpointing and smaller batches [40de030796b3d32a].
- Sampling latency:
  - High-resolution models have very slow sampling (e.g., 300 s for 128 images), which may be impractical for interactive or large-scale generation [e22df3321d2941a1].
- Quality–compute tradeoff:
  - Better FID/IS requires more reverse steps, directly increasing sampling time; aggressive step reduction will degrade quality [ac419d177969b0be].

## Next Steps
- Hardware budgeting:
  - Plan for at least 8× high-end GPUs (or equivalent) to match TPU v3-8 behavior; if using fewer GPUs, scale expectations for training time proportionally.
  - Ensure sufficient GPU memory to host the CIFAR-10 and 256×256 architectures; start from the reported batch sizes (128 for CIFAR-10, 64 for 256×256) and adjust downward only if necessary.
- Architectural setup:
  - Begin with the published architecture sized to fit within a single 8-GPU node’s memory, mirroring the “initial choice of hyperparameters” approach [40de030796b3d32a].
  - Keep βₜ fixed (non-learned) initially to match the simpler training objective and avoid extra memory/compute overhead [153dc7c5f73856d6].
- Training schedule:
  - Allocate ~4–5 days of continuous training time per CIFAR-10 run at 800k steps; budget proportionally more for LSUN Bedroom (up to 2.4M steps) and other large datasets [e22df3321d2941a1].
  - Reuse CIFAR-10–tuned hyperparameters for other datasets to minimize additional search, following the original workflow [40de030796b3d32a].
- Sampling configuration:
  - Start with the full reverse-step schedule (up to 1,000 steps) for evaluation-quality samples; then experiment with reduced step counts to find an acceptable quality–latency tradeoff using FID/IS curves as guidance [ac419d177969b0be].
- Optimization and scaling:
  - If training time is a bottleneck, prioritize data/model parallelism across more devices or mixed-precision training, while monitoring memory headroom and throughput relative to the TPU v3-8 baseline [e22df3321d2941a1].