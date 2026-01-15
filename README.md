# Instruction-Driven Complexity-Aware Budgeted Retrieval

Budget-constrained multimodal evidence extraction from scientific PDFs using cognitive cost modeling and multi-objective redundancy-aware selection.

## Overview

Traditional RAG systems treat all tokens equally when managing context budgets, leading to suboptimal selection when evidence units vary in information density. This work introduces **cognitive cost modeling** that leverages reasoning complexity to preferentially select information-dense units under budget constraints.

**Key Insight**: A 50-token complex derivation may contain more decision-relevant information than a 50-token descriptive paragraph. By estimating reasoning complexity via embedding models, we can assign lower effective costs to high-complexity units, biasing selection toward information-dense evidence.

Combined with flexible redundancy objectives (MMR, GraphCut, Facility Location) and multi-anchor query expansion, the system enables instruction-driven evidence extraction optimized for decision-making rather than mere coverage.

### Key Features

- **Cognitive Cost Profiling**: Embedding-based complexity estimation that reduces cost for reasoning-intensive content
- **Multi-Objective Selection**: 5 selection methods (greedy, cost-normalized, topk, coverage-aware, DPP) × 3 redundancy objectives (MMR, GraphCut, Facility Location)
- **Multi-Anchor Scoring**: LLM-generated diverse query aspects with attention-weighted aggregation
- **Layout-Aware Extraction**: Vision model-based parsing preserving document structure (tables, figures, text hierarchy)
- **Comparative Evaluation**: Automatic generation of metrics across all method-objective combinations

## Problem Formulation

Given query $q$ and PDF document $D = \{u_1, \ldots, u_n\}$, select evidence subset $\mathcal{S}^* \subseteq D$ maximizing:

$$
\mathcal{S}^* = \arg\max_{\mathcal{S}} \sum_{u \in \mathcal{S}} \text{gain}(u \mid \mathcal{S}, q)
$$

subject to: $\sum_{u \in \mathcal{S}} c(u) \leq B$

where $\text{gain}(u \mid \mathcal{S}, q)$ depends on the selection method (see table below).

## Core Components

### 1. Relevance Scoring

**Single Query**: $r(u, q) = \langle \mathbf{e}_u, \mathbf{e}_q \rangle$ (cosine similarity)

**Multi-Anchor** (optional): Generate $K$ diverse anchor queries, aggregate with temperature-scaled weights:

$$
r(u, q) = \sum_{k=1}^K w_k \langle \mathbf{e}_u, \mathbf{e}_{a_k} \rangle, \quad w_k = \frac{\exp(\langle \mathbf{e}_q, \mathbf{e}_{a_k} \rangle / \tau)}{\sum_j \exp(\langle \mathbf{e}_q, \mathbf{e}_{a_j} \rangle / \tau)}
$$

### 2. Cognitive Cost Modeling

**Simple Mode** (`cognitive_cost_mode: simple`):
$$
c(u) = \text{tokens}(u) + \gamma \cdot n_{\text{visual}}(u)
$$
where $n_{\text{visual}}(u)$ is computed as patch-based visual tokens: $\lceil w/p \rceil \times \lceil h/p \rceil$ with patch size $p=14$.

**Embedding Mode** (`cognitive_cost_mode: emb`, default):
$$
c(u) = \max\left(1, \text{tokens}(u) \cdot w_{\text{type}} - \delta \cdot \rho(u)\right) \cdot (1 - 0.15 \cdot \mathbb{1}_{\text{visual}}(u))
$$

where:
- $\rho(u) \in [0,1]$ is reasoning complexity via ModernBERT (higher complexity → **lower** cost → preferred for selection)
- $w_{\text{type}}$ are type-specific weights: text=1.0, table=1.15, equation=1.10, reference=3.90
- Visual units receive 15% cost discount (heuristic: images aid decision-making)

> **Note**: The embedding mode uses a simple visual indicator ($\mathbb{1}_{\text{visual}}$) rather than patch-based visual tokens. Full visual token integration ($\gamma \cdot n_{\text{visual}}$) is implemented in simple mode only.

### 3. Redundancy Penalties

Three objective functions control marginal gain computation:

| Objective | Redundancy Penalty | Formula |
|-----------|-------------------|---------|
| **MMR** | Max similarity | $\text{gain} = \alpha r(u) - \beta \max_{v \in \mathcal{S}} \text{sim}(u,v)$ |
| **GraphCut** | Sum similarity | $\text{gain} = \alpha r(u) - \beta \sum_{v \in \mathcal{S}} \text{sim}(u,v)$ |
| **Facility** | Representativeness | $\text{gain} = \alpha r(u) + \lambda \sum_j \max(0, \text{sim}(u,j) - \max_{v \in \mathcal{S}} \text{sim}(v,j)) - 0.5\beta \max_{v \in \mathcal{S}} \text{sim}(u,v)$ |

## Selection Methods

| Method | Description | Marginal Gain | Cost Normalization |
|--------|-------------|---------------|-------------------|
| **greedy** | Absolute gain maximization | Uses objective (MMR/GraphCut/Facility) | No |
| **cost_norm** | Cost-normalized greedy | Uses objective / $c(u)$ | Yes |
| **topk** | Relevance ranking baseline | Pure $r(u)$, no redundancy | N/A |
| **greedy_cov** | Coverage-aware greedy | $\alpha r(u) + \lambda \cdot \text{cov}(u) - \beta \cdot \text{red}(u)$ | No |
| **dpp** | Determinantal Point Process | $\text{orth}(\sqrt{q(u)} \cdot \mathbf{e}_u)$ (volume maximization) | No |

**Notes**:
- `greedy` (default) and `cost_norm` use the selected objective (`mmr`, `graphcut`, or `facility`)
- `topk` fills budget by relevance rank without redundancy control
- `greedy_cov` adds query term coverage bonus $\lambda \cdot \text{cov}(u)$ where $\text{cov}(u)$ counts new query terms
- `dpp` uses orthogonal projection for diversity (geometry-based, objective-independent)

## Pipeline

1. **Extraction**: Layout-aware PDF parsing (tables, figures, text blocks) via vision models
2. **Evidence Building**: Construct retrieval units with multimodal content
3. **Embedding**: Encode text/images with unified models (SigLIP, CLIP, or sentence-transformers)
4. **Retrieval**: RRF fusion of dense ranking + page-level ranking
   $$\text{RRF}(u) = \sum_{r \in \text{rankings}} \frac{1}{k + \text{rank}_r(u)}$$
5. **Selection**: Budget-constrained optimization (5 methods × 3 objectives)
6. **Generation**: LLM synthesis of selected evidence

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository_url>
cd dpp-retrieval

# Install dependencies
pip install numpy torch transformers sentence-transformers openai PyYAML pypdf pymupdf pillow tiktoken

# For embedding-based cost profiling (optional)
pip install accelerate
```

### Basic Usage

```bash
# Single PDF query
python main.py \
    --pdf paper.pdf \
    --query "What are the training hyperparameters?" \
    --config config/default_config.yaml

# With custom output directory
python main.py \
    --pdf paper.pdf \
    --query "Describe the model architecture" \
    --output_dir output/my_experiment

# Force recompute (bypass cache)
python main.py \
    --pdf paper.pdf \
    --query "What are the key findings?" \
    --recompute
```

### Batch Evaluation

```bash
# Run on SciDuet dataset
python run_sciduet.py \
    --split train \
    --config config/sciduet_config.yaml \
    --limit 100

# With evaluation
python run_sciduet.py \
    --split validation \
    --metadata data/sciduet_metadata.json \
    --eval_mode all \
    --eval_model gpt-5.1
```

## Configuration

Key hyperparameters in [`config/default_config.yaml`](config/default_config.yaml):

### Selection Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `budget_tokens` | $B$ | 2000 | Maximum context budget |
| `redundancy_beta` | $\beta$ | 0.1 | Redundancy penalty weight |
| `rel_weight` | $\alpha$ | 1.0 | Relevance weight |
| `coverage_weight` | $\lambda$ | 0.5 | Coverage bonus (greedy_cov only) |
| `cognitive_cost_mode` | - | `emb` | `simple` or `emb` |
| `alpha_visual` | $\gamma$ | 0.35 | Visual token penalty (simple mode) |

### Multi-Anchor Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `multi_anchor.enabled` | - | `true` | Enable multi-anchor scoring |
| `multi_anchor.anchor_count` | $K$ | 5 | Number of anchor queries |
| `multi_anchor.temperature` | $\tau$ | 1.0 | Softmax temperature for weights |

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_units_dense` | 150 | Candidate pool size |
| `unit_image_weight` | 1.15 | Image embedding weight |
| `table_boost` | 0.75 | Table relevance boost |
| `figure_boost` | 0.75 | Figure relevance boost |

## Output Structure

```
output/<query_id>/
├── summary.md                  # Generated answer
├── context.json               # Greedy selection
├── context_topk.json          # TopK baseline
├── context_greedy_cov.json    # Coverage variant
├── context_cost_norm.json     # Cost-normalized greedy
├── context_dpp.json           # DPP selection
├── diagnostics.png            # Selection visualization
└── highlighted.pdf            # Annotated source PDF
```

Each `context_*.json` contains:
- `selected`: Array of selected evidence units with metadata
- `retrieval.budgets`: Token budget usage
- `method`: Selection method used
- `multi_anchor`: Anchor queries (if enabled)

## Experimental Results Structure

The system generates comprehensive metrics for comparing selection methods:

**Per-sample metrics** (saved in each output directory):
- `n_selected`: Number of units selected
- `mean_relevance`: Average relevance score
- `mean_rel_per_token`: Relevance efficiency
- `budget_pct`: Budget utilization (%)
- `mean_offdiag_sim`: Pairwise redundancy
- `mean_novelty`: Diversity score

**Aggregated metrics** (batch runs):
- Mean ± std across all samples for each method
- Decision scores comparing methods
- Visualizations: spider charts, efficiency plots, diversity analysis

## Repository Structure

```
dpp-retrieval/
├── src/                          # Core modules
│   ├── optimizer.py             # Selection methods & objectives
│   ├── cognitive_profiler.py   # Cost modeling (simple & embedding-based)
│   ├── embedder.py              # Multimodal embedding
│   ├── indexing.py              # RRF retrieval
│   ├── multi_anchor.py          # Multi-anchor scoring
│   ├── evidence_builder.py     # Unit construction
│   ├── layout_aware_extractor.py # PDF parsing
│   └── diagnostics.py           # Visualization
├── sciduet/                     # SciDuet dataset integration
│   ├── pipeline.py              # Batch processing
│   └── io_utils.py              # Evaluation utilities
├── eval_engine/                 # Evaluation framework
├── config/                      # Configuration files
│   ├── default_config.yaml     # Single query config
│   └── sciduet_config.yaml     # Batch evaluation config
├── main.py                      # Single PDF entrypoint
└── run_sciduet.py              # Batch evaluation entrypoint
```

## Evaluation

The system supports evaluation with ground-truth requirements:

```bash
# With metadata containing requirements
python run_sciduet.py \
    --metadata data/sciduet_metadata.json \
    --eval_mode all \
    --eval_model gpt-5.1
```

Evaluation modes:
- `requirements`: Coverage of gold requirements
- `retrieval`: Quality of selected evidence
- `all`: Both evaluations

## Advanced Usage

### Objective Selection

By default, all methods use the **MMR** objective. To experiment with alternatives:

```python
# In your config or code
selector = BudgetedSelector(
    budget_tokens=2000,
    redundancy_beta=0.1,
    rel_weight=1.0,
    objective="facility",  # or "mmr", "graphcut"
    rep_weight=0.15,       # facility-location representativeness weight
    # ...
)
```

**When to use each objective**:
- **MMR**: Balanced, works well for most cases (default)
- **GraphCut**: Stronger diversity push, good for highly redundant documents
- **Facility**: Dataset-level representativeness, useful when coverage across many semantically distinct regions is critical

### Custom Embedding Models

The system supports flexible embedding backends:

```yaml
# config/default_config.yaml
embed:
  backend: sentence_transformer
  model_name: google/embeddinggemma-300m  # or clip-ViT-B-32, google/siglip-base-patch16-224
  device: cuda
  batch_size: 32
```

### Tuning Hyperparameters

Key parameters to tune:

1. **Budget size** (`budget_tokens`): Increase for more comprehensive extraction, decrease for focused selection
2. **Redundancy penalty** (`redundancy_beta`): Higher values → stronger diversity (typical range: 0.05-0.2)
3. **Multi-anchor count** (`anchor_count`): More anchors → broader coverage (3-7 recommended)
4. **Cost discount** (`max_discount_frac` in cognitive profiler): Controls complexity preference (0.3-0.5)

## Implementation Notes

- **Caching**: Extraction and anchor generation results are cached by default. Use `--recompute` to bypass.
- **GPU Acceleration**: Set `device: cuda` in config for embedding and cost profiling speedup.
- **Memory Management**: For large documents, reduce `batch_size` in embedding config.
- **Parallel Evaluation**: Batch processing uses ThreadPoolExecutor for parallel method evaluation.

## Troubleshooting

**Issue**: `CUDA out of memory` error
```bash
# Solution: Reduce batch size or switch to CPU
embed:
  device: cpu
  batch_size: 16  # reduce from 32
```

**Issue**: Extraction fails with timeout
```bash
# Solution: Increase timeout and reduce batch size
extract:
  timeout_s: 600  # increase from 360
  batch_size: 2   # reduce from 3
```

**Issue**: Missing tiktoken encoder
```bash
# Solution: Install tiktoken
pip install tiktoken
```

**Issue**: ModernBERT model download fails
```bash
# Solution: Use simple cost mode instead
selection:
  cognitive_cost_mode: simple
```

**Issue**: No GPU available but config uses CUDA
```bash
# Solution: Auto-detection will fallback to CPU, or explicitly set:
embed:
  device: cpu
selection:
  cognitive_cost_device: cpu
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{idbe2024,
  title={Instruction-Driven Complexity-Aware Budgeted Retrieval},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/dpp-retrieval}
}
```

---

**License**: MIT
