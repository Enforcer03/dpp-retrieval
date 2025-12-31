## Overview
- Voicebox is a large-scale, non-autoregressive, text-guided speech infilling model trained on ~110K hours of audiobook speech (60K English, 50K multilingual in 6 languages), enabling zero-shot TTS, cross-lingual synthesis, denoising, and content editing via in-context learning rather than task-specific training. [806cea72b0cda9b8][edbe8b871feb758f][cd26777505f29aab]

## Key Findings
- **Editing & Denoising Capability**
  - Supports speech content editing by selectively resynthesizing only the frames corresponding to modified words/phones while preserving the rest of the original audio, using duration-aware masking and infilling. [6a6d78af8dd72a01][bfc84f074f6f7130]
  - Can perform speech denoising and transient noise removal as part of its general infilling/conditional generation capabilities. [bfc84f074f6f7130][806cea72b0cda9b8]
- **Cross-Lingual & Zero-Shot TTS**
  - Trained on multilingual audiobooks in six written languages and supports mono- and cross-lingual zero-shot TTS, including style transfer across languages. [edbe8b871feb758f][ff1b257a1da60f08][d026920d1e9adc59]
  - When paired with speech translation, enables cross-lingual zero-shot TTS so users can speak any language in their own voice. [12c19f9c07484725]
- **Quality, Intelligibility, and Style Transfer**
  - Outperforms YourTTS and other baselines on MOS and similarity metrics; transfers style (voice, speaking style, emotion, acoustic condition) more effectively than VALL-E, with +0.101/+0.108 SIM-r gains (cross-sentence/continuation) and +0.141 SIM-o vs raw audio on continuation. [a1acb2fee80cc7a3]
  - Achieves lower WER and higher similarity than VALL-E in zero-shot TTS (1.9% vs 5.9% WER; 0.681 vs 0.580 similarity) while being up to 20× faster. [806cea72b0cda9b8]
- **Diversity vs Accuracy Trade-offs**
  - Flow-matching sampling shows a trade-off between diversity (FSD) and intelligibility (WER): lower classifier guidance and higher NFE increase diversity but can slightly raise WER; low NFE yields less diverse but more ASR-friendly audio. [7607c9f8b2af6566][acb01355b3df89c5]
- **Robust Evaluation Metrics**
  - Introduces reproducible, public-model-based metrics (e.g., MS-MAE, MS-Corr, FSD) to replace non-comparable MOS-only evaluations, targeting correctness, intelligibility, and diversity. [48767a225c04882c][df0f6e884185d2c5][f9eb1f052a1ab253][f94f89dfae6dc95d]
- **Synthetic Speech Detection**
  - A binary classifier can reliably distinguish Voicebox-generated speech from real audio; detection accuracy reaches 1.0 vs original audio and up to 0.907 vs resynthesized audio at high mask ratios. [392e962194b3c42][c81762f38ba86854]

## Evidence
- **Model & Training**
  - Non-autoregressive flow-matching model trained to infill speech given audio context and text; trained on >50K hours of unfiltered, unenhanced speech, plus 60K hours English and 50K hours multilingual audiobooks. [806cea72b0cda9b8][edbe8b871feb758f]
  - Reduced ablation setup: English-only 1K-hour audiobook subset, 12-layer Transformer, 1024-d embeddings, 2048-d FFN, 8 heads, 150k steps, batch size 120k frames. [a77580c2c85fd261][27b529bd8ec11917]
- **Editing Mechanism**
  - Content editing formalism: construct new transcript and durations, copy durations for unchanged phones, set new phones to 0, sample missing durations and frames conditioned on context, then combine original and generated frames to form edited speech. [6a6d78af8dd72a01]
- **Cross-Lingual Evaluation Setup**
  - Cross-lingual tests use MLS test splits filtered by Whisper WER (<20%, or <30% for Polish/Portuguese) to avoid incomplete transcripts; some utterances removed due to alignment failures. [69b1da221bae920a]
  - For prompt-length studies, long samples (~15s) truncated to 4s at word boundaries; higher WERs attributed to ASR difficulty on incomplete sentences. [f65472bf5ec37b32]
- **Metrics & Benchmarks**
  - Critique of MOS and deterministic signal metrics (MCD, SNR/SDR) for generative speech; Voicebox proposes multi-sample metrics (MS-MAE, MS-Corr) and FSD using public models for reproducible comparison. [e47d9b8419c2340d][df0f6e884185d2c5][f9eb1f052a1ab253][f94f89dfae6dc95d][edbe8b871feb758f]
  - Subjective studies: Voicebox vs YourTTS and VALL-E show superior MOS and similarity, especially in style transfer. [a1acb2fee80cc7a3]
  - Flow-matching trade-offs: as NFE increases from 2 to 32, WER slightly increases (2.8→3.1) at low guidance (α=0), while FSD improves; stronger guidance stabilizes WER. [7607c9f8b2af6566][acb01355b3df89c5]
- **Detection & Safety**
  - Synthetic speech detector performance across mask ratios: perfect (1.000 accuracy/precision/recall) vs original audio; up to 0.907 accuracy, 0.881 precision, 0.942 recall vs resynthesized audio at 90% mask. [c81762f38ba86854]
  - Explicit acknowledgment of misuse risk for arbitrary voice-style generation and demonstration that detection is feasible. [392e962194b3c42]
- **Limitations**
  - Training data is read audiobooks; current models may not transfer well to conversational speech with casual style and non-verbal sounds; plan to scale to more diverse speech. [383135cc805c9893]
  - No disentangled control over individual style attributes (voice vs emotion vs acoustic condition); cannot independently mix attributes from different samples. [280beeac507466f7]

## Risks
- **Misuse for Impersonation and Deepfakes**
  - Ability to generate speech in arbitrary styles and voices, including cross-lingual, raises risk of impersonation, fraud, and misinformation despite detectability. [392e962194b3c42][12c19f9c07484725]
- **Limited Attribute Control**
  - Lack of disentangled control over voice, emotion, and acoustic conditions can hinder precise editing workflows and may cause unintended style changes when editing or transferring across languages. [280beeac507466f7]
- **Domain Generalization**
  - Models trained primarily on read audiobooks may degrade on conversational or noisy real-world speech, affecting editing, denoising, and cross-lingual performance in production settings. [383135cc805c9893]
- **Evaluation Biases**
  - MOS remains subjective and context-dependent; although new metrics are proposed, they rely on public ASR/feature models whose biases and domain limits can skew perceived intelligibility and diversity. [e47d9b8419c2340d][edbe8b871feb758f]

## Next Steps
- **For Editing & Denoising Deployment**
  - Implement the duration- and mask-based editing pipeline (copy unchanged phone durations/frames, zero and resample only new phones) to minimize artifacts in content edits and transient noise removal. [6a6d78af8dd72a01][bfc84f074f6f7130]
  - Tune NFE and classifier guidance to balance intelligibility (WER) vs diversity (FSD) for your target use case (e.g., prioritize lower NFE and higher guidance for ASR-friendly, clean edits). [7607c9f8b2af6566][acb01355b3df89c5]
- **For Cross-Lingual TTS**
  - Combine Voicebox with speech translation to enable “same-voice, different-language” applications; validate on MLS-like filtered test sets to ensure transcription quality. [12c19f9c07484725][69b1da221bae920a]
  - Benchmark cross-lingual performance per language and prompt length, accounting for higher WER on truncated/incomplete sentences. [f65472bf5ec37b32][ff1b257a1da60f08]
- **For Safety and Governance**
  - Integrate a synthetic speech detector similar to the one evaluated (targeting ≥0.9 accuracy vs resynthesized audio) into any production pipeline that exposes zero-shot or cross-lingual TTS. [c81762f38ba86854][392e962194b3c42]
  - Establish policy and access controls for arbitrary voice-style cloning, especially in cross-lingual scenarios.
- **For Research & Model Improvement**
  - Expand training data beyond audiobooks to conversational and noisy speech to improve robustness for real-world editing and cross-lingual use. [383135cc805c9893]
  - Investigate disentangled control mechanisms (prompting or text descriptions) to independently manipulate voice, emotion, and acoustic conditions during editing and cross-lingual generation. [280beeac507466f7]
  - Adopt and extend the proposed public-model-based metrics (MS-MAE, MS-Corr, FSD, WER) as standard evaluation tools for future speech generation and editing systems. [48767a225c04882c][df0f6e884185d2c5][f9eb1f052a1ab253][edbe8b871feb758f]