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
    keep_crops: bool
    caption_search_px: int
    max_text_chunk_tokens: int
    vision_batch_size: int
    ocr: OcrConfig


@dataclass(frozen=True)
class EmbedConfig:
    backend: str
    model_name: str
    device: str
    batch_size: int
    dim: int


@dataclass(frozen=True)
class RetrievalConfig:
    top_pages: int
    top_units_bm25: int
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
    enabled: bool
    model: str
    timeout_s: int
    temperature: float


@dataclass(frozen=True)
class SummarizationConfig:
    enabled: bool
    required_sections: list[str]
    max_chars_per_chunk: int


@dataclass(frozen=True)
class LLMExtractionConfig:
    enabled: bool
    model: str
    temperature: float
    wrap_unit_tags: bool
    max_units: int
    max_chars_per_input: int
    save_raw: bool


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str | None
    tags: list[str]


@dataclass(frozen=True)
class OutputConfig:
    save_context_json: bool
    save_highlighted_pdf: bool
    save_run_output_json: bool
    run_output_filename: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    file: Path


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    extract: ExtractConfig
    embed: EmbedConfig
    retrieval: RetrievalConfig
    selection: SelectionConfig
    openai: OpenAIConfig
    summarization: SummarizationConfig
    llm_extraction: LLMExtractionConfig
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
    extract = data["extract"]
    embed = data["embed"]
    retrieval = data["retrieval"]
    selection = data["selection"]
    openai_cfg = data.get("openai", {}) or {}
    summ_cfg = data.get("summarization", {}) or {}
    llm_ext = data.get("llm_extraction", {}) or {}
    wb = data.get("wandb", {}) or {}
    output = data["output"]
    logging = data["logging"]

    ocr = extract.get("ocr", {}) or {}

    return AppConfig(
        paths=PathsConfig(work_dir=_as_path(paths["work_dir"]), cache_dir=_as_path(paths["cache_dir"])),
        extract=ExtractConfig(
            mode=str(extract.get("mode", "auto")),
            dpi=int(extract["dpi"]),
            keep_page_renders=bool(extract["keep_page_renders"]),
            keep_crops=bool(extract["keep_crops"]),
            caption_search_px=int(extract["caption_search_px"]),
            max_text_chunk_tokens=int(extract["max_text_chunk_tokens"]),
            vision_batch_size=int(extract.get("vision_batch_size", 8)),
            ocr=OcrConfig(
                enabled=bool(ocr.get("enabled", False)),
                engine=str(ocr.get("engine", "tesseract")),
                lang=str(ocr.get("lang", "eng")),
                min_conf=int(ocr.get("min_conf", 45)),
                psm=int(ocr.get("psm", 6)),
                on_pages=bool(ocr.get("on_pages", True)),
                on_figures=bool(ocr.get("on_figures", True)),
            ),
        ),
        embed=EmbedConfig(
            backend=str(embed["backend"]),
            model_name=str(embed["model_name"]),
            device=str(embed["device"]),
            batch_size=int(embed["batch_size"]),
            dim=int(embed["dim"]),
        ),
        retrieval=RetrievalConfig(
            top_pages=int(retrieval["top_pages"]),
            top_units_bm25=int(retrieval["top_units_bm25"]),
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
            min_gain=float(selection["min_gain"]),
            stopwords=list(selection.get("stopwords", [])),
        ),
        openai=OpenAIConfig(
            enabled=bool(openai_cfg.get("enabled", False)),
            model=str(openai_cfg.get("model", "gpt-4.1-mini")),
            timeout_s=int(openai_cfg.get("timeout_s", 60)),
            temperature=float(openai_cfg.get("temperature", 0.1)),
        ),
        summarization=SummarizationConfig(
            enabled=bool(summ_cfg.get("enabled", True)),
            required_sections=list(summ_cfg.get("required_sections", ["Overview", "Key Findings", "Evidence", "Risks", "Next Steps"])),
            max_chars_per_chunk=int(summ_cfg.get("max_chars_per_chunk", 1200)),
        ),
        llm_extraction=LLMExtractionConfig(
            enabled=bool(llm_ext.get("enabled", False)),
            model=str(llm_ext.get("model", "gpt-4.1-mini")),
            temperature=float(llm_ext.get("temperature", 0.0)),
            wrap_unit_tags=bool(llm_ext.get("wrap_unit_tags", True)),
            max_units=int(llm_ext.get("max_units", 40)),
            max_chars_per_input=int(llm_ext.get("max_chars_per_input", 12000)),
            save_raw=bool(llm_ext.get("save_raw", True)),
        ),
        wandb=WandbConfig(
            enabled=bool(wb.get("enabled", False)),
            project=str(wb.get("project", "icb-sum")),
            entity=wb.get("entity") if isinstance(wb.get("entity"), str) else None,
            tags=list(wb.get("tags", [])) if isinstance(wb.get("tags"), list) else [],
        ),
        output=OutputConfig(
            save_context_json=bool(output.get("save_context_json", True)),
            save_highlighted_pdf=bool(output.get("save_highlighted_pdf", True)),
            save_run_output_json=bool(output.get("save_run_output_json", True)),
            run_output_filename=str(output.get("run_output_filename", "pipeline_output.json")),
        ),
        logging=LoggingConfig(level=str(logging["level"]), file=_as_path(logging["file"])),
    )
