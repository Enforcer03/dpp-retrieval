## Overview
- Voicebox is a text-guided, multilingual speech infilling model enabling zero-shot TTS, cross-lingual style transfer, and content editing/denoising, trained primarily on read audiobooks in up to six languages with strong generalization but notable data and tooling constraints. [49fda3c64e5cde23][8bafd732ecaa2bdc][383135cc805c9893]

## Key Findings
- **Editing & infilling capability**
  - Voicebox performs text-guided speech infilling, predicting speech coherent with surrounding context and supporting content editing and transient noise removal as general “infilling” tasks. [bfc84f074f6f7130][3b0f0accc36f4dcd][12c19f9c07484725]
  - Unlike prior infilling models limited to ~1s segments due to deterministic assumptions, Voicebox’s CNP-based formulation can model arbitrary-length segments and rich variation. [3b0f0accc36f4dcd]
- **Zero-shot and cross-lingual TTS**
  - Enables zero-shot TTS and alignment-preserved style transfer, including cross-lingual zero-shot TTS where users can speak any language in their own voice. [d026920d1e9adc59][ff1b257a1da60f08][12c19f9c07484725]
  - Cross-lingual evaluation uses MLS-based test sets filtered by Whisper WER (<20%, or <30% for low-resource Polish/Portuguese) and MFA alignment success, yielding 36 language transfer directions from 3s prompts. [69b1da221bae920a][3c30f4acf734928f]
  - In cross-lingual transfer, longer prompts improve speaker similarity but increase WER, especially from English to non-English, due to English dominating (>90%) the training data. [eeda564ea53c848d][383135cc805c9893]
- **Training objectives and duration modeling**
  - Voicebox uses conditional flow-matching (FM) duration models; regression-based variants share architecture but differ in loss. [612c7314f424c7ac]
  - End-to-end metrics for duration variants (zero-shot cross-sentence, continuation, diverse generation) are reported separately for a reduced English-only 1K-hour setup (VB-En-1K). [a77580c2c85fd261][407c0dd71e4e14f3]
  - Standalone duration metrics include multi-sample MAE (MS-MAE), defined as masked absolute error per utterance normalized by average masked phoneme count. [e518edcc56a0a245][df0f6e884185d2c5]
- **Diversity, quality, and ASR utility**
  - Fréchet Speech Distance (FSD), adapted from FID using wav2vec 2.0 features, measures distribution-level similarity between generated and real speech, capturing both quality and diversity. [6674b14fe6498e17][1959113231b0f8e2]
  - Voicebox with FM duration models generates more diverse speech and yields ASR training data that reduces WER by >85% vs baselines, trailing real data by only 0.4% and 1.7% absolute. [9f78a2a0a03f290d]
- **Limitations of current evaluation metrics**
  - MOS is subjective and not comparable across studies; traditional signal-level metrics (MCD, SNR/SDR) assume deterministic outputs, which is ill-posed for stochastic generative models like Voicebox. [e47d9b8419c2340d]
- **Tooling and data constraints**
  - Voicebox depends on phonemizers and forced aligners for frame-level phonetic transcripts; many phonemizers are word-based and ignore cross-word context, harming pronunciation in context-dependent languages (e.g., French liaisons). [8a3f1ec1ecb12383]
  - Current models are trained on read audiobooks in up to six languages and may not transfer well to conversational speech with non-verbal sounds; scaling to more diverse data is planned. [383135cc805c9893]

## Evidence
- General model description and capabilities: [49fda3c64e5cde23][8bafd732ecaa2bdc][3b0f0accc36f4dcd]
- Applications: zero-shot TTS, cross-lingual TTS, content editing, denoising, data generation: [d026920d1e9adc59][ff1b257a1da60f08][12c19f9c07484725]
- Cross-lingual test construction and prompt setup: [69b1da221bae920a][3c30f4acf734928f]
- Prompt-length tradeoff (speaker similarity vs WER) and language imbalance: [eeda564ea53c848d][383135cc805c9893]
- Duration modeling configs and metrics (FM vs regression, MS-MAE, end-to-end metrics): [612c7314f424c7ac][e518edcc56a0a245][df0f6e884185d2c5][a77580c2c85fd261][407c0dd71e4e14f3]
- Diversity/quality metric (FSD) and ASR gains from synthetic data: [6674b14fe6498e17][1959113231b0f8e2][9f78a2a0a03f290d]
- Metric limitations (MOS, MCD, SNR/SDR): [e47d9b8419c2340d]
- Phonemizer/aligner dependence and future end-to-end direction: [8a3f1ec1ecb12383]
- Data domain and language coverage limitations: [383135cc805c9893]

## Risks
- **Cross-lingual accuracy degradation**
  - Longer prompts in English-to-non-English transfers increase WER despite better speaker similarity, risking mispronunciations or language mixing in deployment. [eeda564ea53c848d]
- **Domain mismatch**
  - Training on read audiobooks may cause failures on conversational, noisy, or highly expressive speech, including laughter and backchannels. [383135cc805c9893]
- **Toolchain brittleness**
  - Reliance on word-based phonemizers and forced aligners can mis-handle context-dependent pronunciation and limit language coverage; alignment failures already force test-set filtering. [8a3f1ec1ecb12383][69b1da221bae920a]
- **Evaluation blind spots**
  - Overreliance on MOS or deterministic signal metrics can misrepresent performance of stochastic, diverse generation; cross-study comparisons may be unreliable. [e47d9b8419c2340d]
- **Bias from data imbalance**
  - English-heavy training (>90%) biases the model toward English, especially as prompt length grows, potentially harming minority-language users. [eeda564ea53c848d][383135cc805c9893]

## Next Steps
- **Improve cross-lingual robustness**
  - Tune prompt-length strategies per language pair (e.g., cap English prompt length for En→non-En) and add balanced multilingual training data to reduce English dominance. [eeda564ea53c848d][383135cc805c9893]
- **Expand data diversity**
  - Incorporate conversational corpora (e.g., Common Voice, Switchboard-like datasets) and non-verbal events to close the domain gap from audiobooks. [383135cc805c9893][5d85e9dd8ce97e0e]
- **Reduce dependence on external phonemizers/aligners**
  - Develop end-to-end models that take raw text with punctuation and directly predict speech, eliminating separate phonemization and forced alignment while improving context-sensitive pronunciation and language coverage. [8a3f1ec1ecb12383]
- **Refine evaluation for editing and cross-lingual tasks**
  - Standardize use of FSD and multi-sample metrics (e.g., MS-MAE) alongside WER and task-specific subjective tests; avoid MOS-only reporting and deterministic metrics for stochastic outputs. [6674b14fe6498e17][1959113231b0f8e2][df0f6e884185d2c5][e47d9b8419c2340d]
- **Optimize duration models and NFE–quality tradeoffs**
  - Use conditional flow-matching duration models where diversity and ASR utility matter most, and explore NFE vs quality/latency tradeoffs guided by internal metrics and Figure 2-style analyses. [612c7314f424c7ac][7607c9f8b2af6566][9f78a2a0a03f290d]