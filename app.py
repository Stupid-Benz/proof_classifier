"""Streamlit Proof-to-Manim Studio with hierarchical classification, NER and live logs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import contextlib
import io

import numpy as np
from PIL import Image

from classifier import HierarchicalProofClassifier
from feedback_store import FEEDBACK_PATH, count_feedback_records, save_reviewed_feedback
from image_latex_ocr import (
    LatexOCRInputError,
    append_latex_to_text,
    combine_latex_predictions,
    recognize_latex_from_bytes,
)
from manual_ner import (
    MANUAL_NER_PATH,
    count_manual_annotations,
    find_exact_occurrences,
    save_manual_annotation,
    validate_manual_entities,
)
from ner_alignment import align_parameters_to_spans, parameters_for_ner
from parameter_extractor import ParameterExtractor, highest_confidence_entity
from parameter_schema import (
    MANIM_FAMILIES,
    PARAMETER_FIELDS,
    SUBJECT_FAMILIES,
    TEMPLATE_FAMILY_MAP,
    build_text,
)
from template_runtime import (
    DEFAULT_SCENE_CLASSES,
    merge_parameter_suggestions,
    render_manim,
    template_path_for_family,
)
from page_latex_ocr import (
    PageOCRInputError,
    load_pix2text_model,
    recognize_page_image,
    recognize_pdf,
)

from argparse import Namespace
from pathlib import Path

import os

def detect_device() -> str:
    """Pick the fastest device available; honour P2V_DEVICE override."""
    override = os.environ.get("P2V_DEVICE", "auto").lower()
    if override in ("cpu", "cuda", "mps"):
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

DEVICE = detect_device()

def format_template_json(obj, indent=2, level=0):
    pad = " " * (indent * level)
    next_pad = " " * (indent * (level + 1))

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f'{next_pad}{json.dumps(k, ensure_ascii=False)}: {format_template_json(v, indent, level + 1)}'
            for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"

    if isinstance(obj, list):
        if not obj:
            return "[]"
        # Keep flat lists of numbers or strings on a single line
        if all(not isinstance(x, (dict, list)) for x in obj):
            return "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in obj) + "]"
        items = [f"{next_pad}{format_template_json(x, indent, level + 1)}" for x in obj]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"

    return json.dumps(obj, ensure_ascii=False)

st.set_page_config(page_title="Proof2Visual", page_icon="∑", layout="wide")

# --- Theme-Adaptive Shadcn-inspired Polish ---
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }
    /* Theme-aware container borders and backgrounds */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(128, 128, 128, 0.25) !important;
        border-radius: 0.5rem !important;
        background-color: var(--secondary-background-color) !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        border-radius: 0.5rem !important;
        background-color: var(--secondary-background-color) !important;
    }
    div[data-baseweb="textarea"] > div, div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-color: rgba(128, 128, 128, 0.3) !important;
        border-radius: 0.375rem !important;
    }
    button[kind="secondary"] {
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        border-radius: 0.375rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_badges(badge_info_list: list[tuple[str, str]], align_right: bool = False):
    """Renders robust theme-aware pill badges."""
    spans = []
    for label, variant in badge_info_list:
        if variant == "destructive":
            style = "background-color: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3);"
        elif variant == "secondary":
            style = "background-color: rgba(128, 128, 128, 0.15); color: var(--text-color); border: 1px solid rgba(128, 128, 128, 0.3);"
        elif variant == "outline":
            style = "background-color: transparent; color: var(--text-color); border: 1px solid rgba(128, 128, 128, 0.3);"
        else:
            style = "background-color: var(--text-color); color: var(--background-color); border: 1px solid var(--text-color);"
        spans.append(
            f'<span style="display: inline-flex; align-items: center; border-radius: 9999px; '
            f'padding: 0.125rem 0.625rem; font-size: 0.75rem; font-weight: 500; {style}">{label}</span>'
        )
    flex_align = "flex-end" if align_right else "flex-start"
    st.markdown(
        f'<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; justify-content: {flex_align};">{"".join(spans)}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, content: str, description: str):
    """Renders a clean, theme-aware metric card box."""
    st.markdown(
        f"""
        <div style="border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 0.5rem; padding: 1rem; background-color: var(--background-color); color: var(--text-color); height: 100%;">
            <div style="font-size: 0.875rem; font-weight: 500; opacity: 0.7; margin-bottom: 0.25rem;">{title}</div>
            <div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; line-height: 1.2;">{content}</div>
            <div style="font-size: 0.75rem; opacity: 0.6;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Header Bar ---
header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown("## ∑ Proof2Visual")
    st.caption("From Formal Proofs to Visual Intuition")
# with header_right:
#     st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
#     render_badges([("Math Studio", "secondary"), ("Manim Engine", "outline")], align_right=True)

PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "proof_classifier.joblib"
NER_MODEL_DIR = PROJECT_ROOT / "models" / "ner"


@st.cache_resource(show_spinner="Loading proof classifier…")
def load_classifier(path: str, device: str):
    clf = HierarchicalProofClassifier(path, device=device)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            clf.predict(
                "Prove that the sequence a_n = 1/n converges to 0.",
                "For every epsilon > 0, choose N > 1/epsilon. Then |1/n| < epsilon.",
            )
        except Exception as exc:
            print(f"Classifier warmup inference failed: {exc}")
    return clf


@st.cache_resource(show_spinner="Loading NER extractor…")
def load_extractor(path: str, device: str):
    ext = ParameterExtractor(path, device=device)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            warmup_family = next(iter(MANIM_FAMILIES))
            ext.extract(
                "Prove that the sequence a_n = 3/n converges to 0.",
                "For epsilon > 0, choose N > 3/epsilon. Then |3/n| < epsilon.",
                warmup_family,
                threshold=0.5,
            )
        except Exception as exc:
            print(f"Extractor warmup inference failed: {exc}")
    return ext


@st.cache_resource(show_spinner="Loading equation OCR (pix2tex)…")
def load_latex_ocr_model(device: str):
    from pix2tex.cli import LatexOCR

    # 指向仓库内预置的权重文件
    checkpoint_path = Path(__file__).parent / "models" / "pix2tex" / "weights.pth"
    resizer_path = Path(__file__).parent / "models" / "pix2tex" / "image_resizer.pth"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"pix2tex weights not found at {checkpoint_path}. "
            "Please commit weights.pth to your repository."
        )

    args = Namespace(
        config="settings/config.yaml",          # 使用默认配置
        checkpoint=str(checkpoint_path),        # 自定义权重路径
        no_cuda=(device == "cpu"),
        no_resize=not resizer_path.is_file(),
        temperature=0.25,
    )
    if resizer_path.is_file():
        args.image_resizer = str(resizer_path)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        model = LatexOCR(args)                  # 传入自定义 arguments
        if device != "cpu":
            try:
                model.model.to(device)
                model.args.device = device
                model.model.eval()
            except Exception as exc:
                print(f"pix2tex GPU move failed ({exc}); using default device.")

        try:
            model(Image.new("RGB", (256, 64), "white"))
        except Exception as exc:
            print(f"pix2tex warmup inference failed: {exc}")
    return model


@st.cache_resource(show_spinner="Loading page OCR (Pix2Text)…")
def load_page_ocr_model(device: str):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        model = load_pix2text_model(device=device, enable_table=False)
        # Pix2Text lazily loads layout, MFD, MFR, and text OCR on the first
        # recognize_page() call. Force them now so the first click is fast.
        try:
            model.recognize_page(Image.new("RGB", (512, 512), "white"))
        except Exception as exc:
            print(f"Pix2Text warmup inference failed: {exc}")
    return model


def clean_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    return text.replace("\r", "\n")


classifier = None
classifier_problem = None
if CLASSIFIER_PATH.exists():
    try:
        classifier = load_classifier(str(CLASSIFIER_PATH), DEVICE)
        expected_families = {family for families in SUBJECT_FAMILIES.values() for family in families}
        actual_families = set(classifier.bundle.get("proof_families", []))
        if actual_families and actual_families != expected_families:
            classifier_problem = (
                "The saved classifier has a different proof-family taxonomy. "
                f"Expected {sorted(expected_families)}, found {sorted(actual_families)}. "
                "Retrain with `python train_classifier.py --skip-cv`."
            )
            classifier = None
    except Exception as exc:
        classifier_problem = str(exc)

extractor = load_extractor(str(NER_MODEL_DIR), DEVICE)
# --- Model warmup: preload heavy OCR models so first clicks are instant ---
WARMUP_OCR = os.environ.get("P2V_WARMUP_OCR", "1") == "1"
ocr_pix2tex_ready = False
ocr_pix2text_ready = False
if WARMUP_OCR:
    try:
        load_latex_ocr_model(DEVICE)
        ocr_pix2tex_ready = True
    except Exception as exc:
        print(f"pix2tex warmup failed: {exc}")
    try:
        load_page_ocr_model(DEVICE)
        ocr_pix2text_ready = True
    except Exception as exc:
        print(f"Pix2Text warmup failed: {exc}")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Model status")
    status_badges = []
    if classifier is not None:
        status_badges.append(("Classifier: Loaded", "outline"))
    else:
        status_badges.append(("Classifier: Missing", "destructive"))

    if extractor.has_ner_model and not getattr(extractor, "checkpoint_problem", None):
        status_badges.append(("BIO NER: Loaded", "outline"))
    else:
        status_badges.append(("BIO NER: Offline", "secondary"))

    if WARMUP_OCR:
        status_badges.append(
            ("pix2tex: Warm" if ocr_pix2tex_ready else "pix2tex: Cold", "outline" if ocr_pix2tex_ready else "secondary")
        )
        status_badges.append(
            ("Pix2Text: Warm" if ocr_pix2text_ready else "Pix2Text: Cold", "outline" if ocr_pix2text_ready else "secondary")
        )

    device_label = {
        "cuda": "GPU (CUDA)",
        "mps": "GPU (MPS)",
        "cpu": "CPU",
    }.get(DEVICE, DEVICE.upper())
    status_badges.append((f"Device: {device_label}", "outline" if DEVICE == "cpu" else "secondary"))

    render_badges(status_badges)

    if classifier_problem:
        st.warning("Classifier could not be loaded: " + classifier_problem)
    elif classifier is None:
        st.info("Classifier not trained yet. Manual template-family selection is available.")

    if getattr(extractor, "checkpoint_problem", None):
        st.error("NER checkpoint issue: " + extractor.checkpoint_problem)
    elif not extractor.has_ner_model:
        st.info("NER model not trained yet. You can still enter template parameters manually.")
        st.code("python train_ner.py", language="powershell")

    st.divider()
    ner_threshold = st.slider("NER entity confidence threshold", 0.10, 0.95, 0.50, 0.05)
    quality_label = st.selectbox("Manim render quality", ["Low", "Medium", "High"], index=2)
    quality = {"Low (faster)": "low", "Medium": "medium", "High": "high"}[quality_label]
    timeout_seconds = st.number_input("Render timeout (seconds)", min_value=60, max_value=3600, value=900, step=60)
    st.caption("Low quality is recommended while testing. LaTeX and FFmpeg must be installed for Manim rendering.")

    st.divider()
    c_fb, c_ner = st.columns(2)
    c_fb.metric("Feedback", count_feedback_records())
    c_ner.metric("BIO Spans", count_manual_annotations())

# --- Equation Image OCR (pix2tex only; PaddleOCR temporarily disabled) ---
ocr_tab_eq, ocr_tab_page = st.tabs(["Equation OCR (pix2tex)", "Page OCR (Pix2Text)"])

with ocr_tab_eq:
    with st.expander("Equation Image OCR — pix2tex", expanded=False):
        st.write(
            "Upload a cropped mathematical expression or take a photo. "
            "pix2tex converts mathematical notation into editable LaTeX. "
            "For best results, crop closely around one equation; it is not designed "
            "to transcribe an entire page of prose."
        )

        uploaded_equations = st.file_uploader(
            "Upload equation image(s)",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="latex_ocr_uploads",
        )

        camera_equation = st.camera_input(
            "Or take a photo of an equation",
            key="latex_ocr_camera",
        )

        # Persist pasted images across Streamlit reruns.
        if "latex_ocr_pasted_files" not in st.session_state:
            st.session_state["latex_ocr_pasted_files"] = []

        pasted_equations = st.chat_input(
            "Paste an equation screenshot here (Ctrl+V)",
            accept_file="multiple",
            file_type=["png", "jpg", "jpeg", "webp"],
            key="latex_ocr_paste",
        )

        if pasted_equations is not None:
            new_files = getattr(pasted_equations, "files", []) or []
            if new_files:
                st.session_state["latex_ocr_pasted_files"] = [
                    (f.name or f"pasted_{i}.png", f.getvalue())
                    for i, f in enumerate(new_files, start=1)
                ]

        # Build a persistent list of (filename, image_bytes).
        image_sources = [
            (f.name, f.getvalue())
            for f in (uploaded_equations or [])
        ]

        if camera_equation is not None:
            image_sources.append(
                ("camera-photo.png", camera_equation.getvalue())
            )

        image_sources.extend(
            st.session_state["latex_ocr_pasted_files"]
        )

        if image_sources:
            st.success(f"{len(image_sources)} image(s) ready for recognition.")
            for filename, _ in image_sources:
                st.caption(f"📄 {filename}")
        else:
            st.info("Upload, paste or photograph an equation to begin.")

        if st.session_state["latex_ocr_pasted_files"]:
            if st.button("Clear pasted images", key="clear_pasted_ocr"):
                st.session_state["latex_ocr_pasted_files"] = []
                st.rerun()

        # Recognition
        if st.button(
            "Recognize image(s) as LaTeX",
            type="primary",
            disabled=not image_sources,
            key="run_latex_ocr",
            width='stretch',
        ):
            predictions = []
            failures = []

            with st.status(
                "Initializing pix2tex...",
                expanded=True,
            ) as status:
                try:
                    status.write("Step 1/2: Loading pix2tex model...")

                    progress = st.progress(
                        0.0,
                        text="Loading model..."
                    )

                    ocr_model = load_latex_ocr_model(DEVICE)

                    status.write("Model loaded successfully.")

                    total = len(image_sources)

                    for index, (filename, image_bytes) in enumerate(
                        image_sources
                    ):
                        status.write(
                            f"Recognizing image {index + 1}/{total}: {filename}"
                        )

                        try:
                            predicted_latex = recognize_latex_from_bytes(
                                image_bytes,
                                ocr_model,
                            )
                            predictions.append(
                                (filename, predicted_latex)
                            )

                            status.write(
                                f"Successfully recognized: {filename}"
                            )

                        except Exception as image_exc:
                            failures.append(
                                (filename, str(image_exc))
                            )
                            status.write(
                                f"Failed: {filename} — {image_exc}"
                            )

                        progress.progress(
                            (index + 1) / total,
                            text=f"Processed {index + 1}/{total} images"
                        )

                    if predictions:
                        st.session_state["latex_ocr_results"] = predictions
                        st.session_state["latex_ocr_output"] = (
                            combine_latex_predictions(predictions)
                        )

                        # Clear transient pasted files after successful processing.
                        st.session_state["latex_ocr_pasted_files"] = []

                        status.update(
                            label=(
                                f"Recognition completed: "
                                f"{len(predictions)}/{total} successful"
                            ),
                            state="complete",
                            expanded=True,
                        )

                        st.success(
                            "Recognition completed! "
                            "Review the generated LaTeX below."
                        )
                    else:
                        status.update(
                            label="Recognition failed.",
                            state="error",
                            expanded=True,
                        )

                        st.error(
                            "No image could be recognized. "
                            "Check the error messages above."
                        )

                    for failed_name, failure_message in failures:
                        st.warning(
                            f"{failed_name}: {failure_message}"
                        )

                except ImportError as exc:
                    status.update(
                        label="pix2tex dependency error.",
                        state="error",
                    )
                    st.error(
                        f"pix2tex dependency is missing: {exc}"
                    )

                except Exception as exc:
                    status.update(
                        label="OCR initialization failed.",
                        state="error",
                    )
                    st.error(
                        f"{type(exc).__name__}: {exc}"
                    )

        if st.session_state.get("latex_ocr_output"):
            st.text_area(
                "Review and edit recognized LaTeX",
                key="latex_ocr_output",
                height=180,
            )
            insert_target = st.radio(
                "Insert recognized LaTeX into",
                ["Proof", "Question / theorem statement"],
                horizontal=True,
                key="latex_ocr_insert_target",
            )
            if st.button(
                "Append recognized LaTeX",
                key="insert_latex_ocr",
                width='stretch',
            ):
                try:
                    target_key = (
                        "proof_input"
                        if insert_target == "Proof"
                        else "question_input"
                    )
                    existing = st.session_state.get(target_key, "")
                    st.session_state[target_key] = append_latex_to_text(
                        existing,
                        st.session_state.get("latex_ocr_output", ""),
                    )
                    st.success(
                        f"Recognized LaTeX appended to {insert_target.lower()}."
                    )
                except LatexOCRInputError as exc:
                    st.warning(str(exc))

# ---- New Page OCR tab ----
with ocr_tab_page:
    with st.container(border=True):
        st.markdown("#### Page OCR — Full page / PDF to Markdown")
        st.caption(
            "Upload full pages or PDFs, or paste screenshots. Pix2Text recognizes "
            "layout, text, tables, and math formulas, and returns Markdown with inline LaTeX. "
            "Best for full pages or whole documents; use the Equation OCR tab for a single "
            "cropped formula."
        )

        page_upload = st.file_uploader(
            "Upload page image(s) or PDF(s)",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "pdf"],
            accept_multiple_files=True,
            key="page_ocr_upload",
        )

        # Persist pasted screenshots across Streamlit reruns.
        if "page_ocr_pasted_files" not in st.session_state:
            st.session_state["page_ocr_pasted_files"] = []

        pasted_pages = st.chat_input(
            "Paste a page screenshot here (Ctrl+V)",
            accept_file="multiple",
            file_type=["png", "jpg", "jpeg", "webp"],
            key="page_ocr_paste",
        )

        if pasted_pages is not None:
            new_files = getattr(pasted_pages, "files", []) or []
            if new_files:
                st.session_state["page_ocr_pasted_files"] = [
                    (f.name or f"pasted_{i}.png", f.getvalue())
                    for i, f in enumerate(new_files, start=1)
                ]

        # Build a unified source list: uploaded files first, then pasted screenshots.
        page_sources: list[tuple[str, bytes, bool]] = [
            (f.name, f.getvalue(), (f.name or "").lower().endswith(".pdf"))
            for f in (page_upload or [])
        ]
        page_sources.extend(
            (name, data, False)
            for name, data in st.session_state["page_ocr_pasted_files"]
        )

        if page_sources:
            st.success(f"{len(page_sources)} source(s) ready for recognition.")
            for name, _, is_pdf in page_sources:
                icon = "📕" if is_pdf else "🖼️"
                st.caption(f"{icon} {name}")
        else:
            st.info("Upload or paste a page to begin.")

        if st.session_state["page_ocr_pasted_files"]:
            if st.button(
                "Clear pasted screenshots",
                key="clear_page_ocr_pasted",
            ):
                st.session_state["page_ocr_pasted_files"] = []
                st.rerun()

        page_numbers_input = st.text_input(
            "PDF page numbers (optional, comma-separated, 0-based; leave empty for all pages)",
            placeholder="e.g. 0,1,3",
            key="page_ocr_page_numbers",
        )

        if st.button(
            "Recognize page(s) as Markdown",
            type="primary",
            disabled=not page_sources,
            width='stretch',
            key="run_page_ocr",
        ):
            if not page_sources:
                st.warning("Please upload or paste a page first.")
            else:
                page_numbers = None
                raw_pages = page_numbers_input.strip()
                if raw_pages:
                    try:
                        page_numbers = [
                            int(x.strip())
                            for x in raw_pages.split(",")
                            if x.strip()
                        ]
                    except ValueError:
                        st.error(
                            "Page numbers must be integers separated by commas, "
                            "e.g. 0,1,2."
                        )
                        st.stop()

                results: list[tuple[str, str]] = []
                failures: list[tuple[str, str]] = []

                with st.status(
                    "Loading Pix2Text model…", expanded=True
                ) as status:
                    try:
                        status.write("Loading model…")
                        model = load_page_ocr_model(DEVICE)
                        status.write("Model loaded. Starting recognition.")

                        total = len(page_sources)
                        progress = st.progress(0.0, text="Starting…")

                        for index, (name, data, is_pdf) in enumerate(page_sources):
                            status.write(
                                f"Recognizing {index + 1}/{total}: {name}"
                            )
                            try:
                                if is_pdf:
                                    md = recognize_pdf(
                                        data,
                                        model,
                                        page_numbers=page_numbers,
                                    )
                                else:
                                    md = recognize_page_image(data, model)
                                results.append((name, md))
                                status.write(f"Done: {name}")
                            except (
                                PageOCRInputError,
                                LatexOCRInputError,
                            ) as exc:
                                failures.append((name, str(exc)))
                                status.write(
                                    f"Invalid input: {name} — {exc}"
                                )
                            except Exception as exc:
                                failures.append(
                                    (name, f"{type(exc).__name__}: {exc}")
                                )
                                status.write(f"Failed: {name} — {exc}")

                            progress.progress(
                                (index + 1) / total,
                                text=f"Processed {index + 1}/{total}",
                            )

                        if results:
                            combined = "\n\n---\n\n".join(
                                md for _, md in results
                            )
                            st.session_state["page_ocr_output"] = combined
                            # Clear transient pasted files after success,
                            # matching the pix2tex tab behaviour.
                            st.session_state["page_ocr_pasted_files"] = []
                            status.update(
                                label=(
                                    f"Recognition complete: "
                                    f"{len(results)}/{total} succeeded"
                                ),
                                state="complete",
                                expanded=True,
                            )
                            st.success(
                                "Recognition complete. Review and edit the "
                                "Markdown below."
                            )
                        else:
                            status.update(
                                label="Recognition failed",
                                state="error",
                                expanded=True,
                            )
                            st.error(
                                "No source could be recognized. See the "
                                "messages above."
                            )

                        for name, msg in failures:
                            st.warning(f"{name}: {msg}")

                    except ImportError as exc:
                        status.update(
                            label="Pix2Text dependency missing",
                            state="error",
                        )
                        st.error(
                            f"pix2text is not installed or failed to import: {exc}"
                        )
                    except Exception as exc:
                        status.update(
                            label="Recognition failed", state="error"
                        )
                        st.error(f"{type(exc).__name__}: {exc}")

        if st.session_state.get("page_ocr_output"):
            st.text_area(
                "Page OCR Markdown (editable)",
                key="page_ocr_output",
                height=320,
                help=(
                    "Pix2Text returns Markdown with LaTeX embedded inline. "
                    "You can edit it here before appending."
                ),
            )

            insert_target = st.radio(
                "Append recognized text to",
                ["Proof", "Question / theorem statement"],
                horizontal=True,
                key="page_ocr_insert_target",
            )

            col_append, col_clear = st.columns(2)
            with col_append:
                if st.button(
                    "Append to text box",
                    key="append_page_ocr",
                    width='stretch',
                    type="primary",
                ):
                    target_key = (
                        "proof_input"
                        if insert_target == "Proof"
                        else "question_input"
                    )
                    existing = st.session_state.get(target_key, "")
                    addition = st.session_state.get(
                        "page_ocr_output", ""
                    ).strip()
                    if not addition:
                        st.warning("There is nothing to append.")
                    elif addition in existing:
                        st.info(
                            "This content is already present in the target "
                            "text box; not appended again."
                        )
                    else:
                        st.session_state[target_key] = (
                            f"{existing.rstrip()}\n\n{addition}"
                            if existing.strip()
                            else addition
                        )
                        st.success(
                            f"Appended to {insert_target.lower()}."
                        )
            with col_clear:
                if st.button(
                    "Clear Page OCR result",
                    key="clear_page_ocr",
                    width='stretch',
                ):
                    st.session_state.pop("page_ocr_output", None)
                    st.rerun()

# --- Card 1: Theorem & Proof Input ---
with st.container(border=True):
    st.markdown("#### Theorem & Proof Input")
    question_col, proof_col = st.columns(2)
    with question_col:
        question = st.text_area(
            "Question / theorem statement",
            height=180,
            placeholder="e.g. Prove that a_n = 3/n converges to 0.",
            key="question_input",
        )
    with proof_col:
        proof = st.text_area(
            "Proof",
            height=180,
            placeholder="Paste the proof here, including mathematical notation if available…",
            key="proof_input",
        )

    # ctrl_col1, ctrl_col2 = st.columns([2.2, 1])
    # with ctrl_col1:
    #     manual_family = st.selectbox(
    #         "Manim template family (fallback if classifier is unavailable or unsupported)",
    #         list(MANIM_FAMILIES),
    #         key="manual_family",
    #     )
    # with ctrl_col2:
    #     st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    #     use_classifier = st.checkbox(
    #         "Use the trained classifier when available",
    #         value=classifier is not None,
    #         disabled=classifier is None,
    #     )

    # --- Sample inputs selector (place right before the Analyze button) ---
    SAMPLE_INPUTS = {
        "Select a sample input...": {"question": "", "proof": ""},
        "Sequence Convergence": {
            "question": "Prove that a_n = 3/n converges to 0.",
            "proof": (
                "Take any positive eps. Choose N as a natural number greater than 3/eps. Then for any n from N onwards, we have |3/n - 0| = 3/n ≤ 3/N < eps. Hence the sequence 3/n approaches 0."
            ),
        },
        "Integration": {
            "question": "Using a right-endpoint Riemann sum on the uniform partition of $[0,1]$, evaluate $\\int_0^1 \\sqrt{x}\\,dx$ and show that it equals $2/3$.",
            "proof": (
                "With right endpoints $k/n$ and width $1/n$, $S_n = (1/n)\\sum_{k=1}^n \\sqrt{k/n} = (1/n^{3/2})\\sum_{k=1}^n \\sqrt{k}$. Using the asymptotic $\\sum_{k=1}^n \\sqrt{k} \\sim \\frac{2}{3}n^{3/2}$ (from Faulhaber-type estimates), we get $S_n \\to 2/3$. Continuity of $\\sqrt{x}$ identifies the integral."
            ),
        },
        "Vector Operations": {
            "question": "Apply Gram-Schmidt to v1 = (1,2) and v2 = (2,1).",
            "proof": (
                "Set u1 equal to v1, which is (1,2). The inner product of v2 with u1 is 2 plus 2, which is 4, and the inner product of u1 with itself is 1 plus 4, which is 5. So the projection of v2 onto u1 is four fifths of (1,2), that is (4/5, 8/5). Subtracting this from v2 gives u2 equal to (6/5, -3/5). Checking orthogonality, the inner product of u1 and u2 is six fifths minus six fifths, which is 0."
            ),
        },
        "Linear Transformation": {
            "question": "Find the eigenvalues and eigenvectors of A = [[2,1],[1,2]].",
            "proof": (
                "The characteristic polynomial is $(2-\\lambda)^2 - 1 = (\\lambda-1)(\\lambda-3)$, so the eigenvalues are $1$ and $3$. For $\\lambda = 3$, $(A-3I)v = 0$ gives $-x+y = 0$, so $v = (1,1)$. For $\\lambda = 1$, $(A-I)v = 0$ gives $x+y = 0$, so $v = (1,-1)$. The eigenvectors are perpendicular, reflecting the spectral theorem for symmetric matrices."
            ),
        },
        "Curvature": {
            "question": "Compute the curvature of the parabola gamma(t) = (t, t^2) at t = 1.",
            "proof": (
                "The first derivatives are 1 and 2t, and the second derivatives are 0 and 2. The plane curvature formula gives two divided by the three halves power of one plus four t squared. At t equal to 1 this is two divided by five times the square root of five."
            ),
        },
    }

    def _load_sample_input():
        selected = st.session_state.get("sample_input_selector")
        if selected and selected != "Select a sample input...":
            sample = SAMPLE_INPUTS[selected]
            st.session_state["question_input"] = sample["question"]
            st.session_state["proof_input"] = sample["proof"]

    st.selectbox(
        "Load a sample theorem & proof",
        options=list(SAMPLE_INPUTS.keys()),
        key="sample_input_selector",
        on_change=_load_sample_input,
    )

    if st.button("Analyze proof and prepare Manim parameters", type="primary", width='stretch'):
        if not question.strip() and not proof.strip():
            st.warning("Enter at least a question or a proof.")
        else:
            try:
                classification = (
                    classifier.predict(question, proof) if classifier is not None else None
                )
                predicted_family = classification["proof_family"] if classification else None
                family = predicted_family if predicted_family in MANIM_FAMILIES else None
                extraction = extractor.extract(question, proof, family, threshold=ner_threshold)
                params = merge_parameter_suggestions(family, extraction.get("parameters", {}), question, proof)
                st.session_state["analysis"] = {
                    "question": question,
                    "proof": proof,
                    "family": family,
                    "classification": classification,
                    "unsupported_prediction": (
                        predicted_family if predicted_family and predicted_family not in MANIM_FAMILIES else None
                    ),
                    "extraction": extraction,
                }
                st.session_state["template_params_json"] = format_template_json(params, indent=2)
                st.session_state.pop("feedback_review_confirmed", None)
                st.session_state.pop("last_render", None)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

# --- Card 2: Classification, Extraction & Parameters ---
analysis = st.session_state.get("analysis")
if analysis:
    family = analysis["family"]
    extraction = analysis["extraction"]

    with st.container(border=True):
        st.markdown("#### 1. Classification & Parameter Extraction")

        best_entity = highest_confidence_entity(extraction.get("entities", []))
        ner_conf_str = f"{float(best_entity['confidence']):.1%}" if best_entity is not None else "N/A"
        ner_field_desc = (
            f"Field: {best_entity.get('field', 'parameter')}"
            if best_entity is not None
            else "No NER entities above threshold"
        )

        m1, m2 = st.columns(2)
        with m1:
            if analysis.get("classification"):
                res = analysis["classification"]
                render_metric_card(
                    title="Subject",
                    content=res["subject"],
                    description=f"Score {res['subject_score']:.1%}",
                )
            else:
                render_metric_card(
                    title="Subject",
                    content="Manual Selection",
                    description="Classifier disabled",
                )
        with m2:
            if analysis.get("classification"):
                res = analysis["classification"]
                render_metric_card(
                    title="Predicted Proof Family",
                    content=res["proof_family"],
                    description=f"Score {res['family_score']:.1%}",
                )
            else:
                render_metric_card(
                    title="Template Family",
                    content=family,
                    description="Manually selected",
                )
        # with m3:
        #     render_metric_card(
        #         title="Highest NER Confidence",
        #         content=ner_conf_str,
        #         description=f"{ner_field_desc} · {template_path_for_family(family).name}",
        #     )

        st.write("")

        if analysis.get("unsupported_prediction"):
            st.warning(
                f"The classifier predicted '{analysis['unsupported_prediction']}', but no Manim template is configured for that family. "
                f"Using your selected template family '{family}' instead."
            )

        if analysis.get("classification"):
            result = analysis["classification"]
            with st.expander("Classification probability distributions"):
                c_prob1, c_prob2 = st.columns(2)
                with c_prob1:
                    st.write("**Subject scores**")
                    st.dataframe(
                        pd.DataFrame(result["subject_probabilities"].items(), columns=["Subject", "Score"]),
                        hide_index=True,
                        width='stretch',
                    )
                with c_prob2:
                    st.write("**Family scores**")
                    st.dataframe(
                        pd.DataFrame(result["family_probabilities"].items(), columns=["Proof family", "Score"]),
                        hide_index=True,
                        width='stretch',
                    )

        if extraction.get("ner_enabled") and extraction.get("family_supported"):
            if extraction.get("entities"):
                entity_display = [
                    {key: value for key, value in entity.items() if key != "source"}
                    for entity in extraction["entities"]
                ]
                st.dataframe(pd.DataFrame(entity_display), hide_index=True, width='stretch')
            else:
                st.warning("NER returned no parameter spans above the selected confidence threshold.")
        elif not extraction.get("ner_enabled"):
            st.info("The NER model is not loaded. Verify the conservative text-based suggestions below.")

        with st.expander("View extracted NER parameter dictionary"):
            st.json(extraction.get("parameters", {}))

        allowed_manual_fields = list(PARAMETER_FIELDS.get(family, ()))
        with st.expander("Manual NER annotation", expanded=False):
            st.caption(
                "Select exact character spans in the combined question/proof text and assign the correct NER field. "
                "These manually verified spans are saved as direct BIO labels for the next NER training run."
            )
            if not allowed_manual_fields:
                st.info("The selected proof family does not currently have NER-supported fields.")
            else:
                manual_source_text = build_text(analysis.get("question", ""), analysis.get("proof", ""))
                manual_state_id = hashlib.sha1(
                    (family + "\0" + analysis.get("question", "") + "\0" + analysis.get("proof", "")).encode("utf-8")
                ).hexdigest()[:12]
                annotation_state_key = f"manual_ner_spans_{manual_state_id}"
                if annotation_state_key not in st.session_state:
                    st.session_state[annotation_state_key] = []

                st.code(manual_source_text, language="text", wrap_lines=True)
                phrase = st.text_input(
                    "Copy the exact entity text from above",
                    key=f"manual_span_phrase_{manual_state_id}",
                    placeholder="e.g. 1/n, (1/2)\\ln 2, (t, t^2)",
                )
                occurrences = find_exact_occurrences(manual_source_text, phrase)
                if phrase and not occurrences:
                    st.warning("That exact text was not found. Copy it directly from the source text above.")
                selected_occurrence = None
                if occurrences:

                    def _occurrence_label(index: int) -> str:
                        start, end = occurrences[index]
                        context = manual_source_text[max(0, start - 35) : min(len(manual_source_text), end + 35)]
                        context = context.replace("\n", " ↵ ")
                        return f"Occurrence {index + 1} · chars {start}:{end} · …{context}…"

                    selected_occurrence = st.selectbox(
                        "Choose the correct occurrence",
                        options=list(range(len(occurrences))),
                        format_func=_occurrence_label,
                        key=f"manual_span_occurrence_{manual_state_id}_{hashlib.md5(phrase.encode('utf-8')).hexdigest()[:8]}",
                    )
                selected_field = st.selectbox(
                    "NER field label",
                    allowed_manual_fields,
                    key=f"manual_span_field_{manual_state_id}",
                )
                if st.button(
                    "Add selected span",
                    disabled=not occurrences or selected_occurrence is None,
                    key=f"manual_span_add_{manual_state_id}",
                ):
                    start, end = occurrences[selected_occurrence]
                    candidate = {
                        "start": start,
                        "end": end,
                        "field": selected_field,
                        "text": manual_source_text[start:end],
                    }
                    existing_annotations = list(st.session_state[annotation_state_key])
                    if any(
                        item["start"] == start and item["end"] == end and item["field"] == selected_field
                        for item in existing_annotations
                    ):
                        st.info("This exact span and label have already been added.")
                    else:
                        try:
                            validated = validate_manual_entities(
                                manual_source_text, family, existing_annotations + [candidate]
                            )
                            st.session_state[annotation_state_key] = validated
                            st.success(f"Added {selected_field}: {candidate['text']}")
                        except ValueError as exc:
                            st.error(str(exc))

                current_annotations = st.session_state[annotation_state_key]
                if current_annotations:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Field": item["field"],
                                    "Text": item["text"],
                                    "Start": item["start"],
                                    "End": item["end"],
                                }
                                for item in current_annotations
                            ]
                        ),
                        hide_index=True,
                        width='stretch',
                    )
                    remove_index = st.selectbox(
                        "Remove an annotation",
                        options=list(range(len(current_annotations))),
                        format_func=lambda index: (
                            f"{current_annotations[index]['field']}: {current_annotations[index]['text']} "
                            f"({current_annotations[index]['start']}:{current_annotations[index]['end']})"
                        ),
                        key=f"manual_span_remove_select_{manual_state_id}",
                    )
                    if st.button("Remove selected annotation", key=f"manual_span_remove_{manual_state_id}"):
                        updated = list(current_annotations)
                        updated.pop(remove_index)
                        st.session_state[annotation_state_key] = updated
                        st.rerun()

                    confirm_manual = st.checkbox(
                        "I have checked every selected expression and its field label.",
                        key=f"manual_span_confirm_{manual_state_id}",
                    )
                    complete_manual = st.checkbox(
                        "Complete annotation: I have marked every occurrence of every supported NER field.",
                        value=False,
                        key=f"manual_span_complete_{manual_state_id}",
                    )
                    if st.button(
                        "Save manual BIO annotations for NER training",
                        disabled=not confirm_manual,
                        width='stretch',
                        key=f"manual_span_save_{manual_state_id}",
                    ):
                        try:
                            save_result = save_manual_annotation(
                                question=analysis.get("question", ""),
                                proof=analysis.get("proof", ""),
                                family=family,
                                entities=current_annotations,
                                complete=complete_manual,
                            )
                            if save_result["duplicate"]:
                                st.info(f"This exact annotation set is already saved ({save_result['record_id']}).")
                            else:
                                st.success(
                                    f"Saved {len(current_annotations)} manually labelled span(s) as {save_result['record_id']}."
                                )
                        except Exception as exc:
                            st.error(f"Could not save manual BIO annotations: {exc}")

    # --- Card 3: Template Parameters & Rendering ---
    with st.container(border=True):
        st.markdown("#### 2. Template Parameters & Render")
        st.caption(
            "Verify or edit the JSON inputs passed to the Manim template before generating the animation."
        )
        params_text = st.text_area(
            "Template input JSON",
            key="template_params_json",
            height=210,
            help="Edit values as JSON. Example curvature curve: [\"t\", \"t^2\"].",
        )

        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            with st.expander("Improve NER with feedback", expanded=False):
                st.caption("Save corrected JSON parameters as a weak-label training example.")
                try:
                    preview_template_params = json.loads(st.session_state.get("template_params_json", "{}"))
                    if not isinstance(preview_template_params, dict):
                        raise ValueError("Template parameters must be a JSON object.")
                    preview_targets = parameters_for_ner(preview_template_params, family)
                    preview_text = build_text(analysis.get("question", ""), analysis.get("proof", ""))
                    preview_spans = align_parameters_to_spans(preview_text, preview_targets, family)
                    if preview_spans:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Field": s["field"],
                                        "Matched": s["text"],
                                        "Span": f"{s['start']}:{s['end']}",
                                    }
                                    for s in preview_spans
                                ]
                            ),
                            hide_index=True,
                            width='stretch',
                        )
                    else:
                        st.warning("No training spans aligned.")
                except (json.JSONDecodeError, ValueError) as exc:
                    st.warning(f"Preview unavailable: {exc}")

                reviewed = st.checkbox(
                    "Confirm parameters are accurate.",
                    key="feedback_review_confirmed",
                )
                if st.button(
                    "Save reviewed example",
                    disabled=not reviewed,
                    width='stretch',
                    key="save_ner_feedback",
                ):
                    try:
                        reviewed_template_params = json.loads(st.session_state.get("template_params_json", "{}"))
                        save_result = save_reviewed_feedback(
                            question=analysis.get("question", ""),
                            proof=analysis.get("proof", ""),
                            family=family,
                            template_parameters=reviewed_template_params,
                            initial_predictions=extraction.get("parameters", {}),
                            classification=analysis.get("classification") or {},
                        )
                        if save_result["duplicate"]:
                            st.info(f"Already saved ({save_result['record_id']}).")
                        else:
                            st.success(f"Saved {save_result['record_id']}.")
                    except Exception as exc:
                        st.error(f"Could not save: {exc}")

        with col_exp2:
            internal_family = TEMPLATE_FAMILY_MAP[family]
            required_fields = {
                "Sequence Convergence": ["sequence", "limit", "bound (optional)"],
                "Integration": ["function", "interval", "sample_point", "integral_value"],
                "Vector": ["vectors"],
                "Matrix": ["matrix"],
                "Curvature": ["curve", "evaluation_point"],
            }[internal_family]
            with st.expander("Expected template schema"):
                st.write(required_fields)
                st.caption(f"Template file: `{template_path_for_family(family).name}`")

        with col_exp3:
            with st.expander("Advanced Manim settings"):
                scene_class_override = st.text_input(
                    "Base scene class override",
                    value="",
                    placeholder="Auto-select",
                    help="Examples: CurvatureCurve2D, CurvatureCurve3D, EigenDiagonal2D.",
                )

        if st.button("Generate Manim animation", type="primary", width='stretch'):
            try:
                params = json.loads(params_text)
                if not isinstance(params, dict):
                    raise ValueError("Template parameters must be a JSON object.")

                with st.container(border=True):
                    term_head_l, term_head_r = st.columns([3, 1])
                    with term_head_l:
                        st.markdown("**Live Manim Console Output**")
                    with term_head_r:
                        render_badges([("Rendering...", "default")], align_right=True)
                    with st.container(height=280):
                        log_placeholder = st.empty()
                        log_placeholder.code("Starting Manim…", language="text")
                    progress = st.progress(0.0, text="Preparing render…")

                def update_progress(value: float, message: str):
                    progress.progress(max(0.0, min(1.0, value)), text=message)

                def update_log(log_text: str):
                    cleaned = clean_ansi(log_text)
                    lines = cleaned.splitlines()
                    visible = "\n".join(lines[-220:])
                    log_placeholder.code(visible or "Waiting for Manim output…", language="text")

                report = render_manim(
                    family,
                    params,
                    scene_class=scene_class_override or None,
                    quality=quality,
                    timeout_seconds=int(timeout_seconds),
                    progress_callback=update_progress,
                    log_callback=update_log,
                )
                report["classification"] = analysis.get("classification")
                report["extraction"] = extraction
                st.session_state["last_render"] = report
                # if report["success"]:
                #     st.success("Manim rendering completed successfully.")
                # else:
                #     st.error("Manim did not produce a video. Review the console output below.")
            except json.JSONDecodeError as exc:
                st.error(f"The template parameters are not valid JSON: {exc}")
            except Exception as exc:
                st.error(f"Could not generate the Manim animation: {exc}")

# --- Card 4: Generated Animation Output ---
render = st.session_state.get("last_render")
if render:
    with st.container(border=True):
        st.markdown("#### 3. Generated Animation")
        video_col, report_col = st.columns([1.4, 1])
        with video_col:
            video_path = render.get("video_path")
            if video_path and Path(video_path).is_file():
                st.video(video_path)
            else:
                st.warning("No MP4 video was produced. Inspect the Manim console log below.")

        with report_col:
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", "Success" if render["success"] else "Failed")
            c2.metric("Duration", f"{render['elapsed_seconds']:.1f}s")
            c3.metric("Exit Code", render["return_code"])
            
            st.markdown(f"**Template:** `{render['template_file']}`")
            st.markdown(f"**Scene class:** `{render['scene_class']}`")
            st.caption(f"**Output:** `{render.get('video_path') or 'Not produced'}`")

            if render.get("mathematical_report"):
                with st.expander("Computed mathematical report"):
                    st.json(render["mathematical_report"])

            btn_dl_col1, btn_dl_col2 = st.columns(2)
            with btn_dl_col1:
                if video_path and Path(video_path).is_file():
                    with open(video_path, "rb") as video_file:
                        st.download_button(
                            "Download MP4",
                            data=video_file.read(),
                            file_name=f"{render['run_id']}.mp4",
                            mime="video/mp4",
                            type="primary",
                            width='stretch',
                        )
            with btn_dl_col2:
                st.download_button(
                    "Download Report",
                    data=json.dumps(
                        {key: value for key, value in render.items() if key != "log_text"},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    file_name="manim_render_report.json",
                    mime="application/json",
                    width='stretch',
                )

        with st.expander("Full Manim console log", expanded=not render["success"]):
            st.code(render.get("log_text", "No log captured."), language="text")
        with st.expander("Full render report"):
            st.json({key: value for key, value in render.items() if key != "log_text"})

# st.caption(
#     "The classifier retains all six proof families. Manim generation is currently implemented for five: "
#     "Sequence Convergence, Integration and Approximation, Vector Operations, Linear Transformations, "
#     "and Curves and Moving Frames. Manim output is saved in generated/<run-id>/."
# )