from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None
    _yaml_err = e


@dataclass(frozen=True)
class PathsConfig:
    work_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class AnchorConfig:
    enabled: bool
    max_chars: int


@dataclass(frozen=True)
class OcrConfig:
    enabled: bool
    engine: str
    lang: str
    min_conf: int
    psm: int
    on_pages: bool
    on_figures: bool


@dataclass(frozen=True)
class ExtractConfig:
    mode: str
    dpi: int
    keep_page_renders: bool
    vision_model: str
    timeout_s: int
    max_workers: int
    max_retries: int
    retry_wait_s: float
    batch_size: int
    max_side: int
    jpeg_quality: int
    use_layout_aware: bool
    max_text_chunk_tokens: int
    caption_search_px: int
    ocr: OcrConfig


@dataclass(frozen=True)
class EmbedConfig:
    """
    Configuration for embedding models.

    backend options:
      - "hf_clip": Use transformers CLIPModel
      - "hf_siglip": Use transformers AutoModel (for SigLIP)
      - "hf_auto": Auto-detect transformers model type
      - "sentence_transformer": Use sentence-transformers library (NEW)

    model_name:
      - For hf_* backends: HuggingFace model ID (e.g., "openai/clip-vit-base-patch32")
      - For sentence_transformer: SentenceTransformer model ID
        (e.g., "clip-ViT-B-32", "google/embeddinggemma-300m")

    Note: sentence_transformer backend auto-detects dual encoders
          (models with encode_query/encode_document methods)
    """
    backend: str
    model_name: str
    device: str
    batch_size: int
    dim: int  # 0 = auto-infer


@dataclass(frozen=True)
class RetrievalConfig:
    top_pages: int
    top_units_dense: int
    rrf_k: int
    unit_image_weight: float
    table_boost: float
    figure_boost: float
    min_table_candidates: int
    min_figure_candidates: int


@dataclass(frozen=True)
class SelectionConfig:
    budget_tokens: int
    alpha_visual: float
    redundancy_beta: float
    rel_weight: float
    coverage_weight: float
    min_gain: float
    stopwords: list[str]


@dataclass(frozen=True)
class OpenAIConfig:
    model: str
    temperature: float


@dataclass(frozen=False)
class SummarizationConfig:
    enabled: bool
    required_sections: list[str]
    max_chars_per_chunk: int


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str
    tags: list[str]


@dataclass(frozen=True)
class OutputConfig:
    save_context_json: bool
    save_highlighted_pdf: bool
    save_run_output_json: bool
    save_document_markdown: bool
    run_output_filename: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    file: Path


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    anchor: AnchorConfig
    extract: ExtractConfig
    embed: EmbedConfig
    retrieval: RetrievalConfig
    selection: SelectionConfig
    openai: OpenAIConfig
    summarization: SummarizationConfig
    wandb: WandbConfig
    output: OutputConfig
    logging: LoggingConfig


def _as_path(p: Any) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def load_config(path: str | Path) -> AppConfig:
    if yaml is None:  # pragma: no cover
        raise RuntimeError(f"PyYAML not available: {_yaml_err}")

    cfg_path = _as_path(path)
    data = yaml.safe_load(cfg_path.read_text())

    paths = data["paths"]
    anchor = data["anchor"]
    extract = data["extract"]
    embed = data["embed"]
    retrieval = data["retrieval"]
    selection = data["selection"]
    openai_cfg = data.get("openai", {}) or {}
    summ_cfg = data.get("summarization", {}) or {}
    wb = data.get("wandb", {}) or {}
    output = data["output"]
    logging = data["logging"]

    ocr = extract.get("ocr", {}) or {}

    return AppConfig(
        paths=PathsConfig(work_dir=_as_path(paths["work_dir"]), cache_dir=_as_path(paths["cache_dir"])),
        anchor=AnchorConfig(
            enabled=bool(anchor.get("enabled", True)),
            max_chars=int(anchor.get("max_chars", 1400)),
        ),
        extract=ExtractConfig(
            mode=str(extract.get("mode", "auto")),
            dpi=int(extract["dpi"]),
            keep_page_renders=bool(extract.get("keep_page_renders", True)),
            vision_model=str(extract.get("vision_model", "gpt-4o")),
            timeout_s=int(extract.get("timeout_s", 360)),
            max_workers=int(extract.get("max_workers", 2)),
            max_retries=int(extract.get("max_retries", 6)),
            retry_wait_s=float(extract.get("retry_wait_s", 300.0)),
            batch_size=int(extract.get("batch_size", 3)),
            max_side=int(extract.get("max_side", 1400)),
            jpeg_quality=int(extract.get("jpeg_quality", 85)),
            use_layout_aware=bool(extract.get("use_layout_aware", True)),
            max_text_chunk_tokens=int(extract.get("max_text_chunk_tokens", 2000)),
            caption_search_px=int(extract.get("caption_search_px", 100)),
            ocr=OcrConfig(
                enabled=bool(ocr.get("enabled", False)),
                engine=str(ocr.get("engine", "tesseract")),
                lang=str(ocr.get("lang", "eng")),
                min_conf=int(ocr.get("min_conf", 50)),
                psm=int(ocr.get("psm", 3)),
                on_pages=bool(ocr.get("on_pages", False)),
                on_figures=bool(ocr.get("on_figures", True)),
            ),
        ),
        embed=EmbedConfig(
            backend=str(embed["backend"]),
            model_name=str(embed["model_name"]),
            device=str(embed["device"]),
            batch_size=int(embed["batch_size"]),
            dim=int(embed.get("dim", 0)),
        ),
        retrieval=RetrievalConfig(
            top_pages=int(retrieval["top_pages"]),
            top_units_dense=int(retrieval["top_units_dense"]),
            rrf_k=int(retrieval["rrf_k"]),
            unit_image_weight=float(retrieval.get("unit_image_weight", 0.8)),
            table_boost=float(retrieval.get("table_boost", 0.25)),
            figure_boost=float(retrieval.get("figure_boost", 0.25)),
            min_table_candidates=int(retrieval.get("min_table_candidates", 0)),
            min_figure_candidates=int(retrieval.get("min_figure_candidates", 0)),
        ),
        selection=SelectionConfig(
            budget_tokens=int(selection["budget_tokens"]),
            alpha_visual=float(selection["alpha_visual"]),
            redundancy_beta=float(selection["redundancy_beta"]),
            rel_weight=float(selection["rel_weight"]),
            coverage_weight=float(selection["coverage_weight"]),
            min_gain=float(selection.get("min_gain", 0.0)),
            stopwords=list(selection.get("stopwords", [])),
        ),
        openai=OpenAIConfig(
            model=str(openai_cfg.get("model", "gpt-4o")),
            temperature=float(openai_cfg.get("temperature", 0.1)),
        ),
        summarization=SummarizationConfig(
            enabled=bool(summ_cfg.get("enabled", True)),
            required_sections=list(summ_cfg.get("required_sections", ["Overview", "Key Findings", "Evidence", "Risks", "Next Steps"])),
            max_chars_per_chunk=int(summ_cfg.get("max_chars_per_chunk", 1500)),
        ),
        wandb=WandbConfig(
            enabled=bool(wb.get("enabled", False)),
            project=str(wb.get("project", "icb-sum")),
            entity=str(wb.get("entity", "")),
            tags=list(wb.get("tags", [])),
        ),
        output=OutputConfig(
            save_context_json=bool(output.get("save_context_json", True)),
            save_highlighted_pdf=bool(output.get("save_highlighted_pdf", True)),
            save_run_output_json=bool(output.get("save_run_output_json", False)),
            save_document_markdown=bool(output.get("save_document_markdown", True)),
            run_output_filename=str(output.get("run_output_filename", "run_output.json")),
        ),
        logging=LoggingConfig(level=str(logging["level"]), file=_as_path(logging["file"])),
    )
