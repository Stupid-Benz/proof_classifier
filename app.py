"""Streamlit Proof-to-Manim Studio with hierarchical classification, NER and live logs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from classifier import HierarchicalProofClassifier
from document_ocr import (
    DocumentOCRInputError,
    append_document_text,
    create_pp_structure_pipeline,
    run_document_ocr,
)
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


@st.cache_resource
def load_classifier(path: str):
    return HierarchicalProofClassifier(path, device="cpu")


@st.cache_resource
def load_extractor(path: str):
    return ParameterExtractor(path, device="cpu")


@st.cache_resource(show_spinner=False)
def load_document_ocr_model():
    return create_pp_structure_pipeline(device="cpu")


@st.cache_resource(show_spinner=False)
def load_latex_ocr_model():
    from pix2tex.cli import LatexOCR

    return LatexOCR()


def clean_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)
    return text.replace("\r", "\n")


classifier = None
classifier_problem = None
if CLASSIFIER_PATH.exists():
    try:
        classifier = load_classifier(str(CLASSIFIER_PATH))
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

extractor = load_extractor(str(NER_MODEL_DIR))

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Model status")
    status_badges = []
    if classifier is not None:
        status_badges.append(("Classifier: Loaded", "default"))
    else:
        status_badges.append(("Classifier: Missing", "destructive"))

    if extractor.has_ner_model and not getattr(extractor, "checkpoint_problem", None):
        status_badges.append(("BIO NER: Loaded", "default"))
    else:
        status_badges.append(("BIO NER: Offline", "secondary"))

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

# --- Unified OCR Drawer ---
with st.expander("OCR & Document Import (Equations / Screenshots / PDFs)", expanded=False):
    ocr_tab1, ocr_tab2 = st.tabs(["Image Input — Equations to LaTeX", "Full-Page OCR — Complete Proof / PDF"])

    with ocr_tab1:
        st.write(
            "Upload cropped images of mathematical expressions or take a photo. "
            "LaTeX-OCR recognizes mathematical notation; it is not a full-page handwriting or prose OCR system."
        )
        uploaded_equations = st.file_uploader(
            "Upload equation image(s)",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="latex_ocr_uploads",
        )
        camera_equation = st.camera_input("Or take a photo of an equation", key="latex_ocr_camera")
        image_sources = list(uploaded_equations or [])
        if camera_equation is not None:
            image_sources.append(camera_equation)

        pasted_equations = st.chat_input(
            "Paste an equation screenshot here (Ctrl+V)",
            accept_file="multiple",
            file_type=["png", "jpg", "jpeg", "webp"],
            key="latex_ocr_paste",
        )
        if pasted_equations is not None:
            image_sources.extend(pasted_equations.files)

        if st.button(
            "Recognize image(s) as LaTeX",
            type="secondary",
            disabled=not image_sources,
            key="run_latex_ocr",
            use_container_width=True,
        ):
            try:
                with st.spinner("Loading LaTeX-OCR and recognizing images..."):
                    ocr_model = load_latex_ocr_model()
                    predictions, failures = [], []
                    for image_file in image_sources:
                        filename = getattr(image_file, "name", "camera-photo.png")
                        try:
                            predicted_latex = recognize_latex_from_bytes(image_file.getvalue(), ocr_model)
                            predictions.append((filename, predicted_latex))
                        except Exception as image_exc:
                            failures.append((filename, str(image_exc)))
                    if predictions:
                        st.session_state["latex_ocr_results"] = predictions
                        st.session_state["latex_ocr_output"] = combine_latex_predictions(predictions)
                    if not predictions and failures:
                        raise RuntimeError(
                            "No image could be recognized. "
                            + " | ".join(f"{name}: {message}" for name, message in failures)
                        )
                if predictions:
                    st.success(f"Recognized {len(predictions)} image(s). Review the LaTeX below.")
                for failed_name, failure_message in failures:
                    st.warning(f"Could not recognize {failed_name}: {failure_message}")
            except ImportError as exc:
                st.error(f"LaTeX-OCR is not installed in this environment. Details: {exc}")
            except (LatexOCRInputError, RuntimeError, OSError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not initialize LaTeX-OCR: {exc}")

        if st.session_state.get("latex_ocr_output"):
            st.text_area("Review and edit recognized LaTeX", key="latex_ocr_output", height=160)
            insert_target = st.radio(
                "Insert recognized LaTeX into",
                ["Proof", "Question / theorem statement"],
                horizontal=True,
                key="latex_ocr_insert_target",
            )
            if st.button("Append recognized LaTeX", key="insert_latex_ocr", use_container_width=True):
                try:
                    target_key = "proof_input" if insert_target == "Proof" else "question_input"
                    existing = st.session_state.get(target_key, "")
                    st.session_state[target_key] = append_latex_to_text(
                        existing, st.session_state.get("latex_ocr_output", "")
                    )
                    st.success(f"Recognized LaTeX appended to {insert_target.lower()}.")
                except LatexOCRInputError as exc:
                    st.warning(str(exc))

    with ocr_tab2:
        st.write(
            "Upload a PDF or a complete page image. PaddleOCR PP-StructureV3 detects page layout, "
            "recognizes ordinary text and formulas, and reconstructs the content in reading order."
        )
        pasted_document = st.chat_input(
            "Paste a screenshot here (Ctrl+V)",
            accept_file=True,
            file_type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            key="document_ocr_paste",
        )
        if pasted_document is not None and pasted_document.files:
            pasted_file = pasted_document.files[0]
            st.session_state["document_ocr_paste_bytes"] = pasted_file.getvalue()
            st.session_state["document_ocr_paste_name"] = pasted_file.name or "pasted_screenshot.png"

        pasted_bytes = st.session_state.get("document_ocr_paste_bytes")
        if pasted_bytes:
            st.image(pasted_bytes, caption="Pasted screenshot", use_container_width=True)
            if st.button("Remove pasted screenshot"):
                st.session_state.pop("document_ocr_paste_bytes", None)
                st.session_state.pop("document_ocr_paste_name", None)
                st.rerun()

        document_upload = st.file_uploader(
            "Upload a full-page PDF or image",
            type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=False,
            key="document_ocr_upload",
        )
        document_bytes = document_upload.getvalue() if document_upload is not None else pasted_bytes
        document_filename = (
            document_upload.name
            if document_upload is not None
            else st.session_state.get("document_ocr_paste_name", "pasted_screenshot.png")
        )

        if st.button(
            "Recognize complete document",
            disabled=document_bytes is None,
            type="secondary",
            use_container_width=True,
            key="run_document_ocr",
        ):
            try:
                with st.spinner("Recognizing document..."):
                    document_pipeline = load_document_ocr_model()
                    document_result = run_document_ocr(document_bytes, document_filename, document_pipeline)
                st.session_state["document_ocr_result"] = document_result
                st.session_state["document_ocr_text"] = document_result["text"]
                st.success("Document recognized successfully!")
            except Exception as exc:
                st.error(f"Full-page OCR failed: {exc}")

        document_result = st.session_state.get("document_ocr_result")
        if document_result:
            st.caption(
                f"Source: {document_result['filename']} · "
                f"Pages: {document_result['page_count']} · "
                f"Engine: {document_result['engine']}"
            )
            page_results = document_result.get("pages", [])
            if page_results:
                st.markdown("**Page-by-page OCR output**")
                for page in page_results:
                    st.markdown(f"**Page {page['page_number']}**")
                    st.markdown(page.get("text", "") or "*(No text detected on this page.)*")
                    if page is not page_results[-1]:
                        st.divider()
            st.text_area(
                "Review and edit reconstructed document text / LaTeX",
                key="document_ocr_text",
                height=260,
            )
            document_target = st.radio(
                "Append reviewed document text to",
                ["Proof", "Question / theorem statement"],
                horizontal=True,
                key="document_ocr_target",
            )
            if st.button("Append reviewed document to input", key="append_document_ocr", use_container_width=True):
                target_key = "proof_input" if document_target == "Proof" else "question_input"
                try:
                    st.session_state[target_key] = append_document_text(
                        st.session_state.get(target_key, ""),
                        st.session_state.get("document_ocr_text", ""),
                    )
                    st.success(f"Reviewed document text appended to {document_target.lower()}.")
                except DocumentOCRInputError as exc:
                    st.warning(str(exc))

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

    if st.button("Analyze proof and prepare Manim parameters", type="primary", use_container_width=True):
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
                        use_container_width=True,
                    )
                with c_prob2:
                    st.write("**Family scores**")
                    st.dataframe(
                        pd.DataFrame(result["family_probabilities"].items(), columns=["Proof family", "Score"]),
                        hide_index=True,
                        use_container_width=True,
                    )

        if extraction.get("ner_enabled") and extraction.get("family_supported"):
            if extraction.get("entities"):
                entity_display = [
                    {key: value for key, value in entity.items() if key != "source"}
                    for entity in extraction["entities"]
                ]
                st.dataframe(pd.DataFrame(entity_display), hide_index=True, use_container_width=True)
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
                        use_container_width=True,
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
                        use_container_width=True,
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
                            use_container_width=True,
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
                    use_container_width=True,
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

        if st.button("Generate Manim animation", type="primary", use_container_width=True):
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
                            use_container_width=True,
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
                    use_container_width=True,
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