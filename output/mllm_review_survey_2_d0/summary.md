## Overview
- MLLM architectures differ mainly in fusion strategy (early vs late), connector complexity (linear vs attention/Q-Former/Transformers), and choice of LLM backbone (frozen vs partially tuned, big vs small), with strong implications for compute, flexibility, and domain performance. [12f84962dcc895b7][e966089ab3149cf5][3b6be3edd259c18e][9aa98ea88197007d]

## Key Findings
- **Connector design tradeoff (linear vs attention/Q-Former/Transformers)**  
  - Simple linear projection layers are lightweight and easy to train but capture limited cross-modal interactions. Used in models like Vitron and MiniGPT‑4’s vision→LLM bridge. [3b6be3edd259c18e][aef21bdb228b764b]  
  - Attention-based connectors (Q-Formers, Transformer projections, window-level Query Transformers) better model complex inter-modal relationships at higher compute cost; they are preferred when temporal structure or fine-grained alignment (video, audio) matters. [3b6be3edd259c18e][a215da00386486ad][f127cb58f4672b85][48a9938b61748612][f56bf7871dcd05c0]
- **Fusion strategy (early vs late) should match task needs**  
  - Early fusion (combining raw/low-level features) leverages tight cross-modal correlations but increases model complexity and compute. [12f84962dcc895b7]  
  - Late fusion (combining final modality-specific outputs) is simpler and suits tasks needing consolidated judgments rather than fine-grained cross-modal reasoning. [e966089ab3149cf5]
- **Backbone LLM: frozen vs tuned and size choice**  
  - Many systems freeze the main LLM and modality encoders, training only projection/connector layers for efficient multimodal alignment (Video‑LLaMA, NeXT‑GPT, MiniGPT‑4 stage 1). [a215da00386486ad][f509249fe42f7c0b][f56bf7871dcd05c0]  
  - Lightweight adaptation (LoRA) on top of a mostly frozen LLM improves cross-modal alignment and controllability with modest compute (SALMONN, NeXT‑GPT). [48a9938b61748612][f56bf7871dcd05c0]  
  - There is an explicit tension between “big & comprehensive” vs “small & specialized”; smaller targeted models can outperform large general ones in specific domains and are better for resource-constrained or real-time scenarios (e.g., Mini‑Gemini, LWM’s 1M‑parameter design). [9aa98ea88197007d][280dd300f503a0c9][17785d23908c28ab]
- **Modular, controller-style architectures aid tool use and extensibility**  
  - Architectures like HuggingGPT and the described audio system use an LLM as a central controller that: (1) normalizes modalities, (2) analyzes tasks, (3) routes to specialized base models, and (4) aggregates responses. This pattern scales well as new modalities/tools are added. [280dd300f503a0c9][db47d544b74506f7]
- **Universal encoders and discrete-unit pipelines improve standardization**  
  - Universal encoders map diverse inputs (audio, video, fMRI, physiological sequences) into a standardized feature sequence, simplifying downstream fusion and LLM integration. [eca57f1482fcaa2e][9e5855d0553e6f32][638a73543b3bd141]  
  - Discrete-unit pipelines (e.g., HuBERT + LLaMA + Hi‑Fi‑GAN) enable unified sequence modeling across modalities like speech, easing integration with text-centric LLMs. [750d54ec0980e159][0891db8b8185cee1]
- **Concrete performance/efficiency data for planning**  
  - MiniGPT‑4’s two-stage training shows that strong vision–language alignment can be achieved with modest resources:  
    - Stage 1: 20,000 steps, batch size 256, ~10 hours on 4×A100, training only the linear projection.  
    - Stage 2: high-quality fine-tuning, ~7 minutes on 1×A100, greatly improving linguistic coherence. [f509249fe42f7c0b]  
  - NeXT‑GPT achieves strong video understanding and generation with 7B/13B backbones by freezing encoders/decoders and training only Transformer-based projections plus LoRA, demonstrating a practical “small(er) but capable” pattern. [280dd300f503a0c9][f56bf7871dcd05c0]
- **Task-specific architecture patterns**  
  - **Vision–language**: MiniGPT‑4 uses a BLIP‑2 ViT+Q‑Former encoder feeding a Vicuna (LLaMA-based) decoder via a linear projection; modular design allows swapping LLMs. [02e36a964efea70a][aef21bdb228b764b]  
  - **Audio–text**: SALMONN uses dual encoders (Whisper for speech, BEATs for non-speech) plus a window-level Query Transformer into Vicuna, then LoRA tuning for alignment. [48a9938b61748612]  
  - **Video (audio+visual)–language**: Video‑LLaMA uses separate video and audio branches with Q‑Formers and linear projections into a frozen LLM; NeXT‑GPT uses Transformer projections and modality-switching instruction tuning for controllable video captioning/QA/generation. [f127cb58f4672b85][a215da00386486ad][e23ff1c504dc328b][f56bf7871dcd05c0]
- **Application-driven model choice**  
  - Models like Idefics2, Mini‑Gemini, Wiki‑LLaVA illustrate specialization: image generation, lightweight fast inference, and image–text semantic integration respectively. This supports choosing backbones/connectors based on target tasks and resource budgets. [17785d23908c28ab]  
  - MLLMs particularly benefit audio tasks (speech generation, sentiment, event detection, multimodal translation) by combining audio with text/vision, suggesting strong ROI for audio-aware connectors in assistants, smart homes, and media. [04088e6b6a00e2d1][9c53f6c635c4a605]

## Evidence
- MLLMs integrate text, images, video, audio, and physiological sequences to outperform single-modality systems across many tasks. [155c3af351b3bf48][df025fd336498968][78956211adc2960c][f58922a6a6db6540][04088e6b6a00e2d1]
- Fusion methods: linear layers vs attention mechanisms; attention captures richer inter-modal interactions but is more computationally expensive. [3b6be3edd259c18e]  
- Early vs late fusion definitions and use cases. [12f84962dcc895b7][e966089ab3149cf5]
- Representative architectures and design choices:  
  - HuggingGPT (LLM controller), EMO (specialized facial/audio modules), LWM (1M parameters, scalable), NeXT‑GPT (CLIP/ImageBind, VQGAN, 7B/13B). [280dd300f503a0c9]  
  - MiniGPT‑4: modular, GPT‑4-like multimodal model using Vicuna + BLIP‑2 encoder + linear projection; two-stage training with explicit compute/time stats. [02e36a964efea70a][aef21bdb228b764b][f509249fe42f7c0b]  
  - SALMONN: dual audio encoders + Query Transformer + Vicuna + LoRA. [48a9938b61748612]  
  - Video‑LLaMA: video and audio branches with Q‑Formers and linear mapping into frozen LLM; only Q‑Formers, position embeddings, and linear layers trained; no quantitative metrics reported. [f127cb58f4672b85][a215da00386486ad]  
  - NeXT‑GPT: frozen encoders/decoders/LLM, trainable Transformer projections, LoRA for instruction tuning; strong video captioning/QA/generation on MSRVT, MSVD‑QA, MSRVTT‑QA, NExTQA. [e23ff1c504dc328b][f56bf7871dcd05c0]
- Universal encoder and sequential-data encoders (1D‑CNN+LSTM) for standardized feature sequences across diverse modalities. [eca57f1482fcaa2e][9e5855d0553e6f32][638a73543b3bd141]
- Controller-style audio architecture: modality transformation → LLM module (ChatGPT) → task assignment to audio base model → response generation. [db47df544b74506f7]
- Specialized models and size debate: Idefics2, Mini‑Gemini, Wiki‑LLaVA; explicit discussion of “big vs small” MLLMs. [17785d23908c28ab][9aa98ea88197007d]
- Future video MLLM directions: temporal attention, lightweight designs for real-time, integration of external knowledge, higher-level reasoning. [343db583c5f4bc61]
- Survey context and comparative tables (e.g., audio MLLMs) for broader benchmarking. [6d10b691e667495c][bb6d3e47db084cec][743404687b49fe9a][b0702d20f0e3a033]

## Risks
- **Compute and latency overhead**  
  - Attention-heavy connectors (Q‑Formers, Transformers, temporal attention) and early fusion increase computational cost, potentially blocking real-time or edge deployment. [3b6be3edd259c18e][343db583c5f4bc61]  
  - Large, fully tuned backbones may be unnecessary for narrow domains and can waste resources compared to smaller specialized models. [9aa98ea88197007d]
- **Limited interpretability and trust**  
  - Multimodal fusion is often a “black box”; it is hard to attribute decisions to specific modalities, which is problematic in high-stakes domains like healthcare and finance. [8e134d3aefdd1d12][70a2d1dc0f1c2738]
- **Overfitting to benchmarks / weak evaluation**  
  - Some architectures (e.g., Video‑LLaMA) report qualitative results without quantitative metrics, making it risky to adopt them as backbones without additional evaluation. [a215da00386486ad]  
  - Heavy reliance on specific datasets (e.g., MSRVT, MSRVTT, NExTQA) may not generalize to your domain without targeted fine-tuning. [f56bf7871dcd05c0][b0702d20f0e3a033]
- **Complexity of multi-branch designs**  
  - Separate audio/visual branches with multiple encoders and Q‑Formers (Video‑LLaMA, SALMONN) increase engineering complexity and maintenance burden; misalignment between branches can degrade performance. [f127cb58f4672b85][48a9938b61748612]
- **Ethical and security concerns**  
  - Powerful MLLMs for audio/video generation and cross-modal translation can be misused (deepfakes, privacy violations); opacity exacerbates difficulty in detecting misuse and ensuring responsible deployment. [70a2d1dc0f1c2738][04088e6b6a00e2d1]

## Next Steps
- **Clarify target constraints and priorities**  
  - Decide acceptable latency and hardware budget (e.g., “≤1×A100” vs multi‑GPU) to choose between heavier attention-based connectors and simpler linear ones. Use MiniGPT‑4’s training profile as a reference for what is feasible on 1–4×A100. [f509249fe42f7c0b][3b6be3edd259c18e]
- **Select a backbone strategy**  
  - For general-purpose multimodal reasoning with moderate resources, favor a frozen mid-size LLaMA/Vicuna-like LLM with LoRA tuning, following NeXT‑GPT/SALMONN patterns. [48a9938b61748612][f56bf7871dcd05c0]  
  - For constrained or latency-critical environments, prioritize smaller specialized backbones (Mini‑Gemini/LWM-style) and late fusion, accepting narrower task coverage. [9aa98ea88197007d][17785d23908c28ab][280dd300f503a0c9]
- **Choose connector and fusion design per modality**  
  - Vision–language: start with a BLIP‑2-style encoder + linear projection into a frozen LLM (MiniGPT‑4 pattern); upgrade to attention/Q‑Former connectors only if you need finer cross-modal reasoning. [aef21bdb228b764b][3b6be3edd259c18e]  
  - Audio–text: adopt a dual-encoder + Query Transformer design if you need both speech and non-speech understanding (SALMONN pattern); otherwise, a single audio encoder + linear projection may suffice. [48a9938b61748612]  
  - Video: for strong temporal reasoning, use separate audio/visual branches with Q‑Formers or Transformer projections and consider temporal attention; if compute is tight, use late fusion of per-frame/per-segment features. [f127cb58f4672b85][343db583c5f4bc61][f56bf7871dcd05c0][e966089ab3149cf5]
- **Adopt a modular controller architecture**  
  - Implement an LLM-centric controller that: (1) normalizes modalities via encoders/universal encoder, (2) performs task analysis, (3) routes to specialized models/tools, and (4) aggregates outputs, following HuggingGPT and the audio system design. This eases future extension to new modalities. [280dd300f503a0c9][db47d544b74506f7][eca57f1482fcaa2e]
- **Plan evaluation and interpretability**  
  - Benchmark candidate architectures on domain-relevant datasets (or adapted versions of MSRVT/MSRVTT/NExTQA for video, and audio MLLM benchmarks) before committing. [6d10b691e667495c][f56bf7871dcd05c0][b0702d20f0e3a033]  
  - Incorporate basic interpretability tools (e.g., attention visualization per modality, ablation of modalities) to mitigate black-box risks, especially if targeting regulated domains. [8e134d3aefdd1d12]
- **Iterate with lightweight fine-tuning**  
  - Start with frozen encoders/LLM and train only projection/connector layers; then add LoRA-based instruction tuning for your tasks (modality switching, domain-specific instructions) as in NeXT‑GPT and SALMONN. [48a9938b61748612][f56bf7871dcd05c0]