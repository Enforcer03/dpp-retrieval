## Overview
- Multimodal LLMs (MLLMs) have rapidly advanced from simple text–image systems to unified architectures handling text, images, audio, video, and even fMRI, with strong performance on understanding and generation tasks but persistent challenges in efficiency, controllability, data, and safety. [4d334ceccef6adf5][eca57f1482fcaa2e][154bd532c76a2233]

## Key Findings
- **Standardized architectural pattern**
  - Common pipeline: modality-specific encoders → multimodal fusion (linear or attention-based) → LLM → task-specific decoders for text, image, video, or audio outputs. [3b6be3edd259c18e][eca57f1482fcaa2e][e290af259e47ffad]
  - Universal encoders are emerging to handle diverse data types (audio, video, fMRI) into a shared feature sequence, simplifying integration and scaling. [eca57f1482fcaa2e]

- **Image tasks are the most mature**
  - Large family of LLaVA-style models (7B–80B) using frozen vision encoders + linear alignment to Vicuna LLMs, trained on standard web-scale caption datasets (COCO, CC3M, CC12M, SBU). [4e270d95aa6cca02]
  - Image understanding: strong capabilities in object/scene recognition, relational reasoning, and detailed captioning that surpass traditional vision-only methods. [615b9bb0b00f06c5][e4a52ad0e149962e]
  - Image generation: integration of LLMs with diffusion models (e.g., MM-Interleaved) enables high-quality, multi-image generation with reduced memory and better fine-grained detail. [c4511a9ae3ccabc3][405a3c69de7820dc]
  - Future focus: better architectures, more advanced fusion, larger and higher-quality datasets, improved transfer to downstream tasks, and addressing interpretability, fairness, and privacy. [298a03d35ceaa8fd][12afee02edb3bf64]

- **Two-stage training and frozen backbones dominate**
  - Typical recipe: pre-train alignment on large image–text (or multimodal) pairs with frozen encoders/LLM and a small trainable projection/Q-Former; then instruction-tune on curated, high-quality data.
  - MiniGPT-4: 20k pre-training steps on image–text pairs with only a linear projector trained (10 hours on 4×A100), followed by a brief high-quality fine-tune (~7 minutes on 1×A100) to fix coherence. [f509249fe42f7c0b][02e36a964efea70a]
  - InstructBLIP: BLIP-2 initialization, only Q-Former tuned in pre-training; then up to 60k instruction-tuning steps on 16×A100 over 1.5 days. [124d2d04a7c7e545]
  - X-InstructionBLIP: freezes feature-extraction Q-Former and top LLM; only interaction Q-Former + projection trained, yet outperforms others across 3D, audio, image, and silent video. [d377d457e535ab50]

- **Rapid expansion beyond images to audio and video**
  - Audio: models like Qwen-Audio-chat and Gemini train on dozens of datasets, >30 tasks, 8 languages, and varied audio types, using instruction tuning to unify heterogeneous labels and support speech recognition, translation, and general audio comprehension. [0a23e386024a84f1][be2712ee1077a158]
  - Video understanding and generation: Video-LLaMA, LLaVA-NeXT-Video, and NeXT-GPT use specialized Q-Formers and projection modules for temporal and audio-visual alignment; training often starts from image-caption alignment then instruction-tuning for video tasks. [a215da00386486ad][7fd88eb3f133cbc7][0aa52b555a560aa0]
  - Video generation MLLMs are emerging but still computationally heavy; research is pushing toward more efficient architectures and better multimodal fusion. [7d142118c248be78][7c57bb5b54d4bff7]

- **Multimodal fusion is a key performance lever**
  - Simple linear layers (e.g., Vitron, many LLaVA variants) are efficient but limited in capturing complex cross-modal interactions. [3b6be3edd259c18e][4e270d95aa6cca02]
  - Attention-based fusion (self- and cross-attention, as in LWM, LLaVA-NeXT-Video, Video-LLaMA-2) improves interaction quality at higher computational cost; innovation here is seen as a main driver of future gains. [3b6be3edd259c18e][0aa52b555a560aa0]
  - MM-Interleaved and MMFS show that better visual tokenization and synchronized feature extraction can reduce the number of visual tokens while preserving detail, improving efficiency for multi-image scenarios. [c4511a9ae3ccabc3][405a3c69de7820dc]

- **Strong transfer and adaptation mechanisms**
  - Universal instruction fine-tuning (MiniGPT-4, LLaVA) enables rapid adaptation to classification, detection, and image–text generation with modest task-specific data. [12afee02edb3bf64]
  - Parameter-efficient transfer: LLaMA-Adapter V2 uses small adapter modules with frozen backbones for fast adaptation; Yi-VL augments with external knowledge extraction to boost open-domain performance. [12afee02edb3bf64]
  - Image understanding has evolved through stages: traditional features → deep learning → multimodal cross-modal learning → reinforcement learning → integrated understanding + generation. [d38c0ec93c724b37]

- **Move from tool-orchestration to fully integrated multimodal I/O**
  - Early systems (Visual-ChatGPT, HuggingGPT, AudioGPT) treat the LLM as a planner calling external encoders/decoders, but suffer from dependence on external tools and limited controllability. [2bffbd74142e8ad7]
  - Newer models (NeXT-GPT, Vitron, EMO) integrate encoders/decoders via learned projection modules or directly train the LLM to emit decoder-ready features, enabling end-to-end multimodal input and output (e.g., talking-head video from image+audio). [2bffbd74142e8ad7][04f4dc3f87e0f4fa]

- **General-purpose vs specialized models is an open design tradeoff**
  - There is active debate between “big and comprehensive” generalist MLLMs and “small and specialized” domain models; evidence suggests smaller targeted models can outperform on narrow tasks, making balance between generalization and specialization a central research question. [9aa98ea88197007d]

## Evidence
- Systematic review notes SOTA coverage across natural language, vision, and audio tasks, while emphasizing remaining challenges. [4d334ceccef6adf5]
- Architectural description of encoders, fusion, and decoders, including universal encoders and multimodal output decoders. [eca57f1482fcaa2e][e290af259e47ffad][154bd532c76a2233]
- Detailed training recipes and compute for MiniGPT-4 and InstructBLIP. [f509249fe42f7c0b][124d2d04a7c7e545]
- LLaVA family table showing reliance on linear alignment and standard caption datasets across many parameter scales. [4e270d95aa6cca02]
- Reports of X-InstructionBLIP outperforming baselines across multiple modalities with limited trainable components. [d377d457e535ab50]
- Descriptions of MM-Interleaved/MMFS, Vitron, NeXT-GPT, EMO, and other integrated multimodal generation systems. [c4511a9ae3ccabc3][405a3c69de7820dc][2bffbd74142e8ad7][04f4dc3f87e0f4fa]
- Future directions for image and video MLLMs: architecture optimization, advanced fusion, large-scale datasets, interpretability, fairness, privacy, and external knowledge integration. [298a03d35ceaa8fd][7c57bb5b54d4bff7]

## Risks
- **Computational and data costs**
  - Attention-based fusion and large-scale video/audio training significantly increase compute and memory requirements, limiting accessibility and slowing iteration. [3b6be3edd259c18e][7c57bb5b54d4bff7]
  - Heavy reliance on large web-scale caption datasets (COCO, CC3M, CC12M, SBU) may embed dataset biases and limit domain coverage. [4e270d95aa6cca02][298a03d35ceaa8fd]

- **Controllability and dependency on components**
  - Tool-orchestration approaches depend heavily on external decoders’ quality and offer limited control over outputs; even integrated models can be hard to steer precisely across modalities. [2bffbd74142e8ad7][7c57bb5b54d4bff7]
  - Freezing large backbones while training small adapters/projections can constrain how deeply modalities are integrated, potentially capping performance on complex reasoning.

- **Safety, fairness, and privacy**
  - Review explicitly flags interpretability, fairness, and privacy as open challenges for image MLLMs, which generalize to broader multimodal settings (e.g., sensitive visual or audio data). [298a03d35ceaa8fd]
  - Multimodal data (faces, voices, medical images, fMRI) raises heightened privacy and misuse concerns if not carefully governed. [eca57f1482fcaa2e][298a03d35ceaa8fd]

- **Over-generalization vs specialization**
  - Pushing for ever-larger “general” MLLMs risks underperforming on specialized tasks compared with smaller, tuned models, and may waste compute without clear task benefits. [9aa98ea88197007d]

## Next Steps
- **Architecture and fusion choices**
  - For near-term systems, adopt frozen encoders + LLM with lightweight projection/Q-Former modules and two-stage training to control compute, following MiniGPT-4/InstructBLIP patterns. [f509249fe42f7c0b][124d2d04a7c7e545]
  - Experiment with hybrid fusion: start with linear alignment for efficiency, then selectively introduce cross-modal attention where task performance justifies the cost (e.g., fine-grained video reasoning). [3b6be3edd259c18e][0aa52b555a560aa0]

- **Task and modality prioritization**
  - If your use cases are image-centric (classification, VQA, captioning, editing), leverage LLaVA-style architectures and instruction-tuning; for multi-image generation, consider MM-Interleaved-like tokenization and diffusion backends. [4e270d95aa6cca02][c4511a9ae3ccabc3]
  - For audio and video, follow Qwen-Audio/Gemini/NeXT-GPT patterns: multi-dataset instruction tuning across tasks and languages, with explicit projection modules for temporal and audio-visual alignment. [0a23e386024a84f1][be2712ee1077a158][7fd88eb3f133cbc7]

- **Data and evaluation strategy**
  - Build or curate high-quality, instruction-style multimodal datasets (similar to MACAW-LLM, MiniGPT-4’s second stage) to improve coherence and alignment with human instructions. [f509249fe42f7c0b][be2712ee1077a158]
  - Complement generic caption datasets with domain-specific data and external knowledge sources (Yi-VL-style) for targeted applications. [12afee02edb3bf64]

- **Model scaling and specialization**
  - Decide early whether you need a generalist or specialized MLLM; for domain-specific deployments (e.g., medical imaging, autonomous driving), prioritize smaller specialized models with adapters and domain data. [9aa98ea88197007d][298a03d35ceaa8fd]
  - Use parameter-efficient methods (adapters, Q-Formers) to support rapid adaptation to new downstream tasks without retraining full models. [12afee02edb3bf64][d377d457e535ab50]

- **Governance and safety**
  - Incorporate interpretability tools, bias audits, and privacy-preserving practices (e.g., careful handling of faces, voices, medical images) into the development pipeline, aligning with the review’s emphasis on fairness and privacy. [298a03d35ceaa8fd]
  - For high-stakes applications, favor architectures with clearer control points (e.g., explicit decoders, modular projections) to enable constraint enforcement and user customization. [7c57bb5b54d4bff7][2bffbd74142e8ad7]