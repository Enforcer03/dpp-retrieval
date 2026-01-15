"""ICB-Sum Pipeline Streamlit App - Information Constrained Budgeted Summarization"""

import streamlit as st
import numpy as np
import time
import re
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch
import fitz  # PyMuPDF

# Page config
st.set_page_config(
    page_title="ICB-Sum Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imports from src
from src.config import load_config
from src.utils import ensure_dir, stable_id, write_json
from src.openai_client import OpenAIChatClient
from src.anchor_summary import generate_anchor_query, generate_anchor_summary
from src.layout_aware_extractor import PdfLayoutExtractor
from src.evidence_builder import EvidenceBuilder
from src.cognitive_profiler import TokenCounter, SimpleCostProfiler, EmbeddingBasedCostProfiler
from src.embedder import MultiModalEmbedder
from src.indexing import CombinedRetriever
from src.multi_anchor import generate_multi_anchors, MultiAnchorScorer
from src.optimizer import BudgetedSelector
from src.diagnostics import RetrievalDiagnostics
from src.schema import DocumentArtifact, Element, PageArtifact

log = logging.getLogger(__name__)

# Check for Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

# -----------------------------------------------------------------------------
# Cached Resources
# -----------------------------------------------------------------------------
@st.cache_resource
def get_config():
    """Load and cache config."""
    return load_config("config/default_config.yaml")


@st.cache_resource
def get_embedder(device: str):
    """Load and cache the multimodal embedder."""
    return MultiModalEmbedder(
        backend="hf_siglip",
        model_name="google/siglip-base-patch16-224",
        device=device,
        batch_size=32,
        dim=0,
    )


@st.cache_resource
def get_cost_profiler(_token_counter, mode: str, model_name: str, device: str):
    """Load and cache cost profiler."""
    if mode == "emb":
        return EmbeddingBasedCostProfiler(
            token_counter=_token_counter,
            model_name=model_name,
            device=device,
        )
    return SimpleCostProfiler(token_counter=_token_counter, alpha_visual=0.35)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def build_unit_vectors(units, text_vecs, img_vecs, dim):
    unit_vecs = {}
    zero = np.zeros(dim, dtype=np.float32)
    for u in units:
        text_vec = text_vecs.get(u.id, zero)
        img_vec = img_vecs.get(u.id)
        combined = (text_vec + img_vec) / 2.0 if img_vec is not None else text_vec
        unit_vecs[u.id] = normalize(combined)
    return unit_vecs


def redundancy_metrics(sim: np.ndarray, dup_thresh: float = 0.90):
    n = sim.shape[0]
    if n <= 1:
        return {"n": n, "mean_offdiag": np.nan, "max_offdiag": np.nan, "dup_rate": np.nan}
    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    iu = np.triu_indices(n, k=1)
    return {
        "n": n,
        "mean_offdiag": float(np.nanmean(off)),
        "max_offdiag": float(np.nanmax(off)),
        "dup_rate": float(np.mean(sim[iu] >= dup_thresh)),
    }


# -----------------------------------------------------------------------------
# Local PDF Extraction (PyMuPDF + Tesseract)
# -----------------------------------------------------------------------------
def extract_local(pdf_path: Path, out_dir: Path, dpi: int = 150, ocr_figures: bool = False) -> DocumentArtifact:
    """Extract PDF using PyMuPDF locally (no API calls)."""
    images_dir = ensure_dir(out_dir / "images")
    doc_id = stable_id(str(pdf_path), str(pdf_path.stat().st_size))

    pdf_doc = fitz.open(str(pdf_path))
    num_pages = len(pdf_doc)

    pages = []
    elements = []

    for page_num in range(num_pages):
        page = pdf_doc[page_num]
        page_idx = page_num + 1

        rect = page.rect
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_path = images_dir / f"page_{page_idx:03d}.png"
        pix.save(str(img_path))
        render_w, render_h = pix.width, pix.height

        pages.append(PageArtifact(
            page=page_idx,
            width=render_w,
            height=render_h,
            page_image_path=str(img_path),
        ))

        # Page image element
        elements.append(Element(
            id=stable_id(doc_id, page_idx, "page_image"),
            page=page_idx,
            type="page_image",
            text="",
            bbox=(0, 0, render_w, render_h),
            image_path=str(img_path),
        ))

        # Extract text blocks
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block_idx, block in enumerate(blocks):
            if block["type"] == 0:  # Text
                lines = []
                for line in block.get("lines", []):
                    spans_text = [span.get("text", "") for span in line.get("spans", [])]
                    lines.append("".join(spans_text))
                text = "\n".join(lines).strip()

                if text:
                    bbox = block["bbox"]
                    scale_x, scale_y = render_w / rect.width, render_h / rect.height
                    scaled_bbox = (bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y)

                    # Classify type
                    elem_type = "text"
                    if len(text) < 100:
                        first_line = block.get("lines", [{}])[0]
                        first_span = first_line.get("spans", [{}])[0]
                        if first_span.get("size", 12) > 14 or first_span.get("flags", 0) & 16:
                            elem_type = "heading"
                    if re.match(r'^[\s]*[-•*]\s', text) or re.match(r'^[\s]*\d+[.)]\s', text):
                        elem_type = "list"
                    if '$' in text or '\\' in text:
                        elem_type = "equation"

                    elements.append(Element(
                        id=stable_id(doc_id, page_idx, elem_type, block_idx),
                        page=page_idx,
                        type=elem_type,
                        text=text,
                        bbox=scaled_bbox,
                        image_path=None,
                    ))

            elif block["type"] == 1:  # Image block (often empty, see XObject extraction below)
                pass

        # Extract images via XObjects (more reliable for figures)
        try:
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = pdf_doc.extract_image(xref)
                    if base_image:
                        img_data = base_image["image"]
                        img_ext = base_image.get("ext", "png")
                        fig_path = images_dir / f"page_{page_idx:03d}_xobj_{img_idx}.{img_ext}"
                        with open(fig_path, "wb") as f:
                            f.write(img_data)

                        # Get image placement on page
                        img_rects = page.get_image_rects(xref)
                        if img_rects:
                            bbox = img_rects[0]  # Use first placement
                            scale_x, scale_y = render_w / rect.width, render_h / rect.height
                            scaled_bbox = (bbox.x0 * scale_x, bbox.y0 * scale_y, bbox.x1 * scale_x, bbox.y1 * scale_y)
                        else:
                            # Fallback: use image dimensions centered
                            scaled_bbox = (0, 0, base_image.get("width", 100), base_image.get("height", 100))

                        # Skip very small images (likely icons/logos)
                        width = scaled_bbox[2] - scaled_bbox[0]
                        height = scaled_bbox[3] - scaled_bbox[1]
                        if width < 50 or height < 50:
                            continue

                        fig_text = ""
                        if ocr_figures and TESSERACT_AVAILABLE:
                            from PIL import Image
                            try:
                                fig_text = pytesseract.image_to_string(Image.open(fig_path)).strip()
                            except Exception:
                                pass

                        elements.append(Element(
                            id=stable_id(doc_id, page_idx, "figure", f"xobj_{img_idx}"),
                            page=page_idx,
                            type="figure",
                            text=fig_text,
                            bbox=scaled_bbox,
                            image_path=str(fig_path),
                        ))
                except Exception:
                    pass
        except Exception:
            pass

        # Try table detection
        try:
            if hasattr(page, "find_tables"):
                for tbl_idx, table in enumerate(page.find_tables()):
                    bbox = table.bbox
                    scale_x, scale_y = render_w / rect.width, render_h / rect.height
                    scaled_bbox = (bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y)
                    try:
                        md = table.to_pandas().to_markdown(index=False)
                    except Exception:
                        md = str(table.extract())
                    elements.append(Element(
                        id=stable_id(doc_id, page_idx, "table_text", tbl_idx),
                        page=page_idx,
                        type="table_text",
                        text=md,
                        bbox=scaled_bbox,
                        image_path=None,
                    ))
        except Exception:
            pass

    pdf_doc.close()

    return DocumentArtifact(doc_id=doc_id, pdf_path=str(pdf_path), pages=pages, elements=elements)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("ICB-Sum Pipeline")
st.sidebar.markdown("**Information Constrained Budgeted Summarization**")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
query = st.sidebar.text_area("Query", value="Summarize the key contributions of this paper", height=100)

st.sidebar.markdown("---")
st.sidebar.markdown("**Extraction Settings**")
extraction_mode = st.sidebar.selectbox(
    "Extraction Mode",
    ["local (PyMuPDF + Tesseract)", "api (Vision LLM)"],
    index=0,
    help="Local: Fast, no API cost. API: Better layout understanding but slower & costs tokens."
)
ocr_figures = st.sidebar.checkbox("OCR Figures (Tesseract)", value=TESSERACT_AVAILABLE, disabled=not TESSERACT_AVAILABLE)
extract_dpi = st.sidebar.slider("Render DPI", 72, 300, 150, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("**Selection Settings**")
budget_tokens = st.sidebar.slider("Budget (tokens)", 500, 5000, 2000, 100)
selection_method = st.sidebar.selectbox("Primary Selection Method", ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Optional Enhancements**")
enable_anchor = st.sidebar.checkbox("Enable Anchor Query", value=True)
enable_multi_anchor = st.sidebar.checkbox("Enable Multi-Anchor Scoring", value=True)

st.sidebar.markdown("---")
run_pipeline = st.sidebar.button("Run Pipeline", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
st.title("ICB-Sum: Budgeted PDF Summarization Pipeline")

if not uploaded_file:
    st.info("Upload a PDF and enter a query to get started.")
    st.stop()

if not run_pipeline and "results" not in st.session_state:
    st.info("Click **Run Pipeline** to process the document.")
    st.stop()

# Run pipeline
if run_pipeline:
    results = {}
    cfg = get_config()

    # Save uploaded file
    tmp_path = Path("temp_upload.pdf")
    tmp_path.write_bytes(uploaded_file.read())
    pdf_path = tmp_path.resolve()

    run_id = stable_id(pdf_path.name, str(pdf_path.stat().st_size))[:16]
    work_dir = ensure_dir(cfg.paths.work_dir / run_id)
    out_dir = ensure_dir(work_dir / "results")

    results["run_id"] = run_id
    results["pdf_name"] = uploaded_file.name
    results["query"] = query

    progress = st.progress(0, text="Initializing...")
    status = st.empty()

    # -------------------------------------------------------------------------
    # Stage 1: Anchor Query
    # -------------------------------------------------------------------------
    status.markdown("### Stage 1: Anchor Query Generation")
    progress.progress(5, text="Generating anchor query...")

    retrieval_query = query
    anchor_text = None

    if enable_anchor:
        client = OpenAIChatClient(timeout_s=cfg.extract.timeout_s)
        anchor_text, _ = generate_anchor_query(
            client=client,
            model=cfg.openai.model,
            temperature=cfg.openai.temperature,
            query=query,
            user_instruction=query,
            required_sections=cfg.summarization.required_sections,
            max_chars=cfg.anchor.max_chars,
        )
        retrieval_query = f"{query}\n\n{anchor_text}".strip()

    results["anchor_text"] = anchor_text
    results["retrieval_query"] = retrieval_query

    # -------------------------------------------------------------------------
    # Stage 2: PDF Extraction
    # -------------------------------------------------------------------------
    status.markdown("### Stage 2: PDF Extraction")
    use_local = extraction_mode.startswith("local")
    progress.progress(15, text=f"Extracting PDF ({'local' if use_local else 'API'})...")

    t0 = time.perf_counter()
    if use_local:
        doc = extract_local(pdf_path, out_dir, dpi=extract_dpi, ocr_figures=ocr_figures)
    else:
        extractor = PdfLayoutExtractor(cfg)
        doc = extractor.extract(pdf_path, out_dir=cfg.paths.work_dir)
    extract_time = time.perf_counter() - t0

    results["extraction_mode"] = "local" if use_local else "api"
    results["num_pages"] = len(doc.pages)
    results["num_elements"] = len(doc.elements)
    results["extract_time"] = extract_time

    # -------------------------------------------------------------------------
    # Stage 3: Evidence Units
    # -------------------------------------------------------------------------
    status.markdown("### Stage 3: Building Evidence Units")
    progress.progress(25, text="Building evidence units...")

    t0 = time.perf_counter()
    token_counter = TokenCounter()
    builder = EvidenceBuilder(
        max_text_chunk_tokens=cfg.extract.max_text_chunk_tokens,
        caption_search_px=cfg.extract.caption_search_px,
        stopwords=set(),
    )
    units = builder.build(doc, token_counter=token_counter)
    build_time = time.perf_counter() - t0

    results["num_units"] = len(units)
    results["build_time"] = build_time
    results["units"] = units

    # -------------------------------------------------------------------------
    # Stage 4: Embedding
    # -------------------------------------------------------------------------
    status.markdown("### Stage 4: Multimodal Embedding")
    progress.progress(35, text="Embedding text and images...")

    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = get_embedder(device)

    unit_texts = [u.retrieval_text or "" for u in units]
    text_res = embedder.embed_text([u.id for u in units], unit_texts)
    text_vecs = {i: v.astype(np.float32) for i, v in zip(text_res.ids, text_res.vecs)}

    img_units = [(u.id, u.image_paths[0]) for u in units if u.image_paths]
    img_vecs = {}
    if img_units:
        img_ids, img_paths = zip(*img_units)
        img_res = embedder.embed_images(list(img_ids), list(img_paths))
        img_vecs = {i: v.astype(np.float32) for i, v in zip(img_res.ids, img_res.vecs)}

    page_imgs = [(p.page, p.page_image_path) for p in doc.pages if p.page_image_path]
    page_vecs = {}
    if page_imgs:
        page_ids, img_paths = zip(*page_imgs)
        page_res = embedder.embed_images([str(i) for i in page_ids], list(img_paths))
        page_vecs = {int(i): v.astype(np.float32) for i, v in zip(page_res.ids, page_res.vecs)}

    dim = text_vecs[next(iter(text_vecs))].shape[0]
    unit_vecs = build_unit_vectors(units, text_vecs, img_vecs, dim)
    query_vec = normalize(embedder.embed_query(retrieval_query).astype(np.float32))
    embed_time = time.perf_counter() - t0

    results["embed_dim"] = dim
    results["num_img_units"] = len(img_vecs)
    results["embed_time"] = embed_time
    results["embedder"] = embedder
    results["unit_vecs"] = unit_vecs

    # -------------------------------------------------------------------------
    # Stage 5: Retrieval
    # -------------------------------------------------------------------------
    status.markdown("### Stage 5: Candidate Retrieval")
    progress.progress(50, text="Retrieving candidates...")

    t0 = time.perf_counter()
    retriever = CombinedRetriever(
        units=units,
        unit_vecs=unit_vecs,
        page_vecs=page_vecs,
        rrf_k=cfg.retrieval.rrf_k,
        table_boost=cfg.retrieval.table_boost,
        figure_boost=cfg.retrieval.figure_boost,
        min_table_candidates=cfg.retrieval.min_table_candidates,
        min_figure_candidates=cfg.retrieval.min_figure_candidates,
    )
    hits, top_pages = retriever.retrieve(
        query_vec=query_vec,
        top_pages=cfg.retrieval.top_pages,
        top_dense=cfg.retrieval.top_units_dense,
    )
    rel_scores = {h.unit_id: h.score for h in hits}
    candidates = [u for u in units if u.id in rel_scores]
    retrieve_time = time.perf_counter() - t0

    results["num_candidates"] = len(candidates)
    results["top_pages"] = top_pages
    results["retrieve_time"] = retrieve_time
    results["candidates"] = candidates

    # -------------------------------------------------------------------------
    # Stage 5.5: Multi-Anchor Scoring
    # -------------------------------------------------------------------------
    if enable_multi_anchor:
        status.markdown("### Stage 5.5: Multi-Anchor Scoring")
        progress.progress(60, text="Multi-anchor scoring...")

        t0 = time.perf_counter()
        client = OpenAIChatClient(timeout_s=cfg.extract.timeout_s)
        anchor_queries = generate_multi_anchors(
            client=client,
            model=cfg.openai.model,
            temperature=cfg.openai.temperature,
            query=query,
            user_instruction=query,
            anchor_count=cfg.multi_anchor.anchor_count,
            required_sections=cfg.summarization.required_sections,
        )

        candidate_ids = {h.unit_id for h in hits}
        candidate_vecs = {u.id: unit_vecs[u.id] for u in units if u.id in candidate_ids}

        scorer = MultiAnchorScorer(
            embedder=embedder,
            unit_vecs=candidate_vecs,
            temperature=cfg.multi_anchor.temperature,
        )
        rel_scores = scorer.compute_scores(anchor_queries)
        multi_anchor_time = time.perf_counter() - t0

        results["anchor_queries"] = anchor_queries
        results["multi_anchor_time"] = multi_anchor_time
    else:
        results["anchor_queries"] = None

    results["rel_scores"] = rel_scores

    # -------------------------------------------------------------------------
    # Stage 6: Selection
    # -------------------------------------------------------------------------
    status.markdown("### Stage 6: Budget-Constrained Selection")
    progress.progress(70, text="Selecting optimal units...")

    t0 = time.perf_counter()

    profiler = get_cost_profiler(
        token_counter,
        cfg.selection.cognitive_cost_mode,
        cfg.selection.cognitive_cost_hf_model,
        cfg.selection.cognitive_cost_device,
    )
    if cfg.selection.cognitive_cost_mode == "emb":
        costs = {u.id: profiler.cost(u.context_text, u.image_paths, unit_type=u.type).total for u in units}
    else:
        costs = {u.id: profiler.cost(u.context_text, u.image_paths).total for u in units}

    selector = BudgetedSelector(
        budget_tokens=budget_tokens,
        redundancy_beta=cfg.selection.redundancy_beta,
        rel_weight=cfg.selection.rel_weight,
        min_gain=cfg.selection.min_gain,
        coverage_weight=getattr(cfg.selection, "coverage_weight", 0.5),
        stopwords=set(getattr(cfg.selection, "stopwords", [])),
    )

    methods = ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"]
    selections = {}
    budgets_used = {}

    for m in methods:
        sel = selector.select_method(
            method=m,
            query=retrieval_query,
            coverage_query=query,
            candidates=candidates,
            rel_scores=rel_scores,
            costs=costs,
            vecs=unit_vecs,
        )
        selections[m] = sel
        budgets_used[m] = int(sum(s.cost for s in sel))

    select_time = time.perf_counter() - t0

    results["costs"] = costs
    results["selections"] = selections
    results["budgets_used"] = budgets_used
    results["select_time"] = select_time

    # -------------------------------------------------------------------------
    # Stage 7: Summary Generation
    # -------------------------------------------------------------------------
    status.markdown("### Stage 7: Summary Generation")
    progress.progress(85, text="Generating summary...")

    selected = selections[selection_method]
    t0 = time.perf_counter()

    selected_chunks = []
    for s in selected:
        u = s.unit
        selected_chunks.append({
            "chunk_id": u.id,
            "content": {"text": u.context_text, "image_paths": [str(p) for p in u.image_paths]},
            "metadata": {"unit_type": u.type, "cognitive_cost": int(s.cost), "importance_score": rel_scores.get(u.id)},
        })

    client = OpenAIChatClient(timeout_s=cfg.extract.timeout_s)
    summary_text, _ = generate_anchor_summary(
        client=client,
        model=cfg.openai.model,
        temperature=cfg.openai.temperature,
        user_instruction=query,
        required_sections=cfg.summarization.required_sections,
        selected_chunks=selected_chunks,
        max_chars_per_chunk=cfg.summarization.max_chars_per_chunk,
    )
    summary_time = time.perf_counter() - t0

    results["summary"] = summary_text
    results["summary_time"] = summary_time
    results["selected"] = selected

    progress.progress(100, text="Complete!")
    status.empty()

    st.session_state["results"] = results
    st.rerun()

# -----------------------------------------------------------------------------
# Display Results
# -----------------------------------------------------------------------------
if "results" in st.session_state:
    r = st.session_state["results"]

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pages", r["num_pages"])
    col2.metric("Elements", r["num_elements"])
    col3.metric("Evidence Units", r["num_units"])
    col4.metric("Selected", len(r["selected"]))

    st.divider()

    # Tabs for pipeline stages
    tabs = st.tabs([
        "Summary",
        "1. Anchor Query",
        "2. Extraction",
        "3. Evidence Units",
        "4. Embeddings",
        "5. Retrieval",
        "6. Selection",
        "Diagnostics",
    ])

    # Tab 0: Summary
    with tabs[0]:
        st.markdown("### Generated Summary")
        st.markdown(r["summary"])

    # Tab 1: Anchor Query
    with tabs[1]:
        st.markdown("### Anchor Query Generation")
        if r["anchor_text"]:
            st.success(f"Generated anchor query ({len(r['anchor_text'])} chars)")
            with st.expander("View Anchor Text"):
                st.text(r["anchor_text"])
            with st.expander("View Full Retrieval Query"):
                st.text(r["retrieval_query"])
        else:
            st.info("Anchor query generation was disabled")

    # Tab 2: Extraction
    with tabs[2]:
        st.markdown("### PDF Extraction")
        mode = r.get("extraction_mode", "unknown")
        st.info(f"**Mode:** `{mode}` {'(PyMuPDF + Tesseract)' if mode == 'local' else '(Vision LLM API)'}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pages", r["num_pages"])
        c2.metric("Elements", r["num_elements"])
        c3.metric("Time", f"{r['extract_time']:.1f}s")

    # Tab 3: Evidence Units
    with tabs[3]:
        st.markdown("### Evidence Unit Construction")
        c1, c2 = st.columns(2)
        c1.metric("Units Built", r["num_units"])
        c2.metric("Time", f"{r['build_time']:.1f}s")

        units = r["units"]
        type_counts = {}
        for u in units:
            type_counts[u.type] = type_counts.get(u.type, 0) + 1

        st.markdown("**Unit Types Distribution:**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(type_counts.keys(), type_counts.values(), color="#3498db")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.markdown("### All Extracted Figures & Tables with Similarity Scores")

        # Get all figure and table units
        visual_units = [u for u in units if u.type in ["figure", "table_text"]]

        if not visual_units:
            st.info("No figures or tables were extracted from this PDF.")
        else:
            # Compute similarity scores for all visual units
            rel_scores = r.get("rel_scores", {})
            unit_vecs = r.get("unit_vecs", {})
            embedder = r.get("embedder")
            anchor_queries = r.get("anchor_queries")

            # Compute scores for visual units that don't have scores yet
            visual_scores = {}
            for u in visual_units:
                if u.id in rel_scores:
                    visual_scores[u.id] = rel_scores[u.id]
                elif u.id in unit_vecs and embedder is not None:
                    # Compute score using query vector similarity
                    query_vec = normalize(embedder.embed_query(r["retrieval_query"]).astype(np.float32))
                    score = float(np.dot(unit_vecs[u.id], query_vec))
                    visual_scores[u.id] = score
                else:
                    visual_scores[u.id] = 0.0

            # If multi-anchor queries exist, compute per-anchor scores for visuals
            anchor_scores_per_unit = {}
            if anchor_queries and embedder is not None:
                st.markdown("**Multi-Anchor Similarity Breakdown:**")
                for u in visual_units:
                    if u.id in unit_vecs:
                        anchor_scores_per_unit[u.id] = []
                        for aq in anchor_queries:
                            aq_vec = normalize(embedder.embed_query(aq).astype(np.float32))
                            score = float(np.dot(unit_vecs[u.id], aq_vec))
                            anchor_scores_per_unit[u.id].append(score)

            # Filter by type
            visual_types = list(set(u.type for u in visual_units))
            visual_filter = st.multiselect("Filter visuals by type", options=visual_types, default=visual_types, key="visual_filter")

            # Sort options
            sort_by = st.selectbox("Sort by", ["Score (High to Low)", "Score (Low to High)", "Page Number"], key="visual_sort")

            filtered_visuals = [u for u in visual_units if u.type in visual_filter]

            # Sort
            if sort_by == "Score (High to Low)":
                filtered_visuals = sorted(filtered_visuals, key=lambda u: visual_scores.get(u.id, 0), reverse=True)
            elif sort_by == "Score (Low to High)":
                filtered_visuals = sorted(filtered_visuals, key=lambda u: visual_scores.get(u.id, 0))
            else:
                filtered_visuals = sorted(filtered_visuals, key=lambda u: u.page)

            # Check which are selected
            selected_ids = {s.unit.id for s in r["selected"]}

            st.write(f"Showing {len(filtered_visuals)} of {len(visual_units)} visual units")

            # Display in a grid
            cols_per_row = 3
            for i in range(0, len(filtered_visuals), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(filtered_visuals):
                        u = filtered_visuals[i + j]
                        score = visual_scores.get(u.id, 0.0)
                        is_selected = u.id in selected_ids

                        with col:
                            # Header with selection status
                            status = "✅ SELECTED" if is_selected else ""
                            st.markdown(f"**[{u.type.upper()}]** Page {u.page} {status}")
                            st.markdown(f"**Overall Score: {score:.4f}**")

                            # Show per-anchor scores if available
                            if u.id in anchor_scores_per_unit:
                                anchor_str = " | ".join([f"A{i+1}:{s:.3f}" for i, s in enumerate(anchor_scores_per_unit[u.id])])
                                st.caption(f"Anchors: {anchor_str}")

                            if u.image_paths:
                                for img_path in u.image_paths:
                                    if Path(img_path).exists():
                                        st.image(str(img_path), use_container_width=True)
                                    else:
                                        st.warning(f"Image not found")
                            else:
                                st.info("No image file")
                            # Show text preview for tables
                            if u.type == "table_text" and u.context_text:
                                with st.expander("View table text"):
                                    st.text(u.context_text[:500] + "..." if len(u.context_text) > 500 else u.context_text)

    # Tab 4: Embeddings
    with tabs[4]:
        st.markdown("### Multimodal Embedding")
        c1, c2, c3 = st.columns(3)
        c1.metric("Embedding Dim", r["embed_dim"])
        c2.metric("Image Units", r["num_img_units"])
        c3.metric("Time", f"{r['embed_time']:.1f}s")
        st.info("Using SigLIP unified text-image embedding space")

    # Tab 5: Retrieval
    with tabs[5]:
        st.markdown("### Candidate Retrieval")
        c1, c2, c3 = st.columns(3)
        c1.metric("Candidates", r["num_candidates"])
        c2.metric("Top Pages", len(r["top_pages"]))
        c3.metric("Time", f"{r['retrieve_time']:.1f}s")

        if r.get("anchor_queries"):
            st.markdown("**Multi-Anchor Queries:**")
            for i, aq in enumerate(r["anchor_queries"], 1):
                with st.expander(f"Anchor {i}"):
                    st.text(aq)

    # Tab 6: Selection
    with tabs[6]:
        st.markdown("### Budget-Constrained Selection")

        selections = r["selections"]
        budgets_used = r["budgets_used"]

        st.markdown("**Method Comparison:**")
        method_data = []
        for m in ["greedy", "topk", "greedy_cov", "cost_norm", "dpp"]:
            sel = selections[m]
            method_data.append({
                "Method": m,
                "Units": len(sel),
                "Budget Used": budgets_used[m],
                "Usage %": f"{100 * budgets_used[m] / budget_tokens:.1f}%",
            })

        st.dataframe(method_data, use_container_width=True, hide_index=True)

        st.markdown(f"**Primary Method: `{selection_method}`**")
        selected = r["selected"]
        st.write(f"Selected {len(selected)} units using {budgets_used[selection_method]}/{budget_tokens} tokens")

        st.divider()
        st.markdown("### Selected Units")

        # Filter options
        unit_types = list(set(s.unit.type for s in selected))
        type_filter = st.multiselect("Filter by type", options=unit_types, default=unit_types)

        filtered_selected = [s for s in selected if s.unit.type in type_filter]
        st.write(f"Showing {len(filtered_selected)} of {len(selected)} selected units")

        for idx, s in enumerate(filtered_selected):
            u = s.unit
            with st.expander(f"**{idx+1}. [{u.type.upper()}]** Page {u.page} | Cost: {int(s.cost)} tokens | Score: {r['rel_scores'].get(u.id, 0):.3f}"):
                col_text, col_img = st.columns([2, 1])

                with col_text:
                    st.markdown("**Content:**")
                    # Show text content (truncated if very long)
                    text_content = u.context_text or u.retrieval_text or ""
                    if len(text_content) > 1000:
                        st.text_area("Text", text_content[:1000] + "...", height=200, disabled=True, label_visibility="collapsed")
                    else:
                        st.text_area("Text", text_content if text_content else "(No text)", height=150, disabled=True, label_visibility="collapsed")

                    st.markdown(f"**Unit ID:** `{u.id[:16]}...`")
                    if u.bbox:
                        st.markdown(f"**Bbox:** ({u.bbox[0]:.0f}, {u.bbox[1]:.0f}, {u.bbox[2]:.0f}, {u.bbox[3]:.0f})")

                with col_img:
                    # Show images if available (figures and tables)
                    if u.image_paths:
                        st.markdown("**Visual:**")
                        for img_path in u.image_paths:
                            if Path(img_path).exists():
                                st.image(img_path, use_container_width=True)
                            else:
                                st.warning(f"Image not found: {img_path}")
                    elif u.type in ["figure", "table", "table_text"]:
                        st.info("No image available for this unit")

    # Tab 7: Diagnostics
    with tabs[7]:
        st.markdown("### Selection Diagnostics")

        selected = r["selected"]
        candidates = r["candidates"]
        units = r["units"]
        costs = r["costs"]
        rel_scores = r["rel_scores"]

        # Create diagnostics
        diag = RetrievalDiagnostics(
            all_units=units,
            candidates=candidates,
            selected=selected,
            costs=costs,
            rel_scores=rel_scores,
            budget_tokens=budget_tokens,
        )

        # 2x3 grid of plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        diag._plot_selection_space(axes[0, 0])
        diag._plot_cost_distribution(axes[0, 1])
        diag._plot_selection_pattern(axes[0, 2])
        diag._plot_budget_allocation(axes[1, 0])
        diag._plot_unit_type_distribution(axes[1, 1])
        diag._plot_efficiency_comparison(axes[1, 2])

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Diversity analysis
        st.markdown("### Selection Diversity")

        sorted_units = sorted(units, key=lambda u: (u.page, u.bbox[1], u.bbox[0]))
        selected_pages = set(s.unit.page for s in selected)
        total_pages = len(set(u.page for u in units))
        coverage = 100 * len(selected_pages) / total_pages if total_pages > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Pages Covered", f"{len(selected_pages)}/{total_pages}")
        c2.metric("Coverage", f"{coverage:.1f}%")
        c3.metric("Avg Units/Page", f"{len(selected) / max(len(selected_pages), 1):.1f}")

        # Similarity heatmap if we have the embedder
        if "embedder" in r and len(selected) > 1:
            st.markdown("### Similarity Analysis")

            embedder = r["embedder"]
            selected_ids = [s.unit.id for s in selected]
            selected_texts = [s.unit.context_text for s in selected]

            selected_res = embedder.embed_text(selected_ids, selected_texts)
            X = np.stack([v.astype(np.float32) for v in selected_res.vecs], axis=0)
            sim = cosine_similarity(X)

            metrics = redundancy_metrics(sim)

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Similarity", f"{metrics['mean_offdiag']:.3f}")
            c2.metric("Max Similarity", f"{metrics['max_offdiag']:.3f}")
            c3.metric("Dup Rate @0.9", f"{metrics['dup_rate']:.3f}")

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(sim, vmin=0, vmax=1, cmap="viridis")
            ax.set_title("Selected Units Similarity Heatmap")
            ax.set_xlabel("Unit Index")
            ax.set_ylabel("Unit Index")
            plt.colorbar(im, ax=ax, label="Cosine Similarity")
            st.pyplot(fig)
            plt.close()

    # Clear results button
    if st.sidebar.button("Clear Results", use_container_width=True):
        del st.session_state["results"]
        st.rerun()
