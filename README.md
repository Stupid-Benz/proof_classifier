# Proof-to-Manim Studio

A small proof-to-animation project with a two-layer mathematical proof classifier, a BIO-NER parameter extractor, five Manim templates, symbolic preprocessing, and a Streamlit interface.

The application workflow is:

1. Enter a theorem/question and proof, use pix2tex for a cropped equation image, or use optional PP-StructureV3 full-page OCR for a complete page/PDF.
2. Review and edit any OCR output, then predict the mathematical subject and proof family or choose a template family manually.
3. Extract the supported parameter spans with the trained NER model; optionally add exact manual BIO annotations for difficult examples.
4. Prepare and edit the JSON inputs expected by the selected Manim template.
5. Render the animation. The application shows Manim's live console output while it runs, then displays the resulting MP4 and a symbolic report.

## 1. Supported proof families and templates

| Classifier proof family | Manim template file | Template scene class | Template inputs |
|---|---|---|---|
| Sequence Convergence | `templates/sequence_template.py` | `SequenceConvergenceTemplate` | `sequence`, `limit`, optional `bound` |
| Integration and Approximation | `templates/integration_template.py` | `IntegrationApproximationTemplate` | `function`, `interval`, `sample_point`, `integral_value` |
| Vector Operations | `templates/gram_schmidt_template.py` | `GramSchmidtTemplate` | `vectors` |
| Linear Transformations | `templates/matrix_eigen_template.py` | `EigenDiagonal2D` or `EigenDiagonal3D` | `matrix` |
| Curves and Moving Frames | `templates/curvature_template.py` | `CurvatureCurve2D` or `CurvatureCurve3D` | `curve`, `evaluation_point` |

The classifier still recognises **Function Properties**, but no Manim template is configured for it yet. If the classifier predicts that family, the app reports the prediction and uses the manually selected template family instead.

### NER fields versus template inputs

The NER vocabulary is intentionally the five-family vocabulary previously agreed:

- Sequence Convergence: `sequence`, `limit`, `bound`.
- Integration and Approximation: `function`, `interval`, `sample_point`, `integral_value`.
- Vector Operations: `vectors`.
- Linear Transformations: `matrix`.
- Curves and Moving Frames: `curve`, `evaluation_point`.

For Curves and Moving Frames, NER learns `curve` and `evaluation_point` directly. In the BIO vocabulary, the complete coordinate tuple is tagged as one entity with `B-curve`/`I-curve` tags; for example, the span `(t, t^2)` is returned under the JSON key `curve`. The app parses that span into the constructor list `['t', 't^2']` before passing it to `CurvatureCurve2D`. Three coordinate expressions select `CurvatureCurve3D`. Curvature is not an NER target: the symbolic preprocessing/template calculates it from the curve and evaluation point. A conservative text suggestion remains available only as a fallback when NER does not return a curve span.

The `vectors` field may contain multiple NER entity spans, one for each input vector. `math_preproc.py` accepts text spans such as `"(1, 2)"` as well as nested numeric lists. A matrix NER span such as `"[[4, 0], [0, 2]]"` is also accepted. For integration, NER labels `right` and `left` are converted to the template values `right_endpoint` and `left_endpoint`.

The template parameter JSON is editable before rendering. Check it carefully, especially when NER has low confidence or a proof contains several mathematical expressions.

The app displays individual extracted entities without their confidence values in section **1. Classification and parameter extraction**. Section **2. Template parameters** displays only one confidence metric: the highest confidence among the returned NER entities, together with the field that received it. If no NER entities are returned, the score is shown as unavailable. This is an entity-level confidence (mean token probability for that span), not a calibrated probability that the whole template parameter set is correct.

## 2. Project structure and file responsibilities

### Main application and model pipeline

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface. Accepts typed input, cropped equation images and full-page documents, runs classification/NER, supports manual BIO annotation, edits template parameters, launches Manim and displays live logs/video. |
| `image_latex_ocr.py` | Validates equation-image uploads, calls pix2tex, wraps results as editable LaTeX blocks, and appends them to the question/proof. |
| `document_ocr.py` | Validates full-page image/PDF uploads, runs an injected PP-StructureV3 pipeline, reconstructs page Markdown in reading order, and safely removes temporary upload files. PaddleOCR is imported lazily. |
| `manual_ner.py` | Finds exact phrase occurrences, validates character-offset annotations, saves manually labelled examples, and converts gold spans directly into BIO token labels. |
| `parameter_schema.py` | Shared classifier family labels, NER output-field definitions, and the separate Manim template-family/parameter contracts. |
| `classifier.py` | Loads the trained hierarchical classifier and predicts subject first, then proof family within that subject. |
| `embedding.py` | Loads the Sentence-Transformers MiniLM encoder used to embed question/proof text for the classifier. |
| `parameter_extractor.py` | Loads the trained token-classification model and converts predicted BIO spans into family-specific parameter entities. It has no rule-based inference fallback. |
| `ner_alignment.py` | Training-time weak-label alignment. Matches known values in each record's `parameters` object to spans in its question/proof and prepares BIO targets. |
| `data_loader.py` | Discovers and validates proof JSON files, including optional `manual_entities` character-span annotations. |
| `json_utils.py` | JSON serialization helpers for NumPy and other non-standard numeric values. |
| `validate_data.py` | Validates the JSON proof dataset and reports malformed or duplicate records. |
| `train_classifier.py` | Trains the two-layer MiniLM + logistic-regression classifier and writes the classifier model bundle. |
| `train_ner.py` | Combines weakly aligned BIO examples with direct user-labelled BIO spans and fine-tunes the MiniLM token-classification model. Manual feedback examples are training-only. |
| `train.py` | Optional orchestration command that validates the data, then runs classifier training followed by NER training. |

### Manim runtime and mathematical preprocessing

| File | Purpose |
|---|---|
| `template_runtime.py` | Maps classifier family labels to template files, prepares and validates inputs, chooses the correct scene class, computes a symbolic report, writes a parameterized scene wrapper, launches Manim, streams stdout/stderr, and saves render artifacts. |
| `math_preproc.py` | Shared SymPy-only mathematical functions: expression/vector/matrix parsing, Gram-Schmidt calculations, eigen-decomposition, curve geometry, and Riemann-sum calculations. It does not import Manim. |
| `templates/sequence_template.py` | Draws the sequence terms, proposed limit, epsilon band and an illustrative tail. |
| `templates/integration_template.py` | Draws the function graph and left/right/midpoint Riemann rectangles. |
| `templates/gram_schmidt_template.py` | Draws the two input vectors, projection and orthogonal component. |
| `templates/matrix_eigen_template.py` | Visualizes the matrix action and eigen-directions in 2D or 3D. |
| `templates/curvature_template.py` | Visualizes a planar or spatial parametrized curve and its Frenet/curvature information. |
| `templates/__init__.py` | Makes the template directory importable as a Python package when needed. |
| `templates/math_preproc.py` | Template-local copy of the mathematical preprocessing module, so the template files can also be used directly from that directory. The application runtime imports the project-root `math_preproc.py`. |
| `templates/README.md` | Notes on the standalone template interfaces and example usage. |

### Data, model outputs and tests

- `data/`: the JSON proof records. The loader combines the JSON files in this folder.
- `models/proof_classifier.joblib`: generated by `train_classifier.py`.
- `models/ner/`: generated by `train_ner.py` and contains the tokenizer/model checkpoint plus NER metadata.
- `generated/<run-id>/`: created for each Manim render. It contains `generated_scene.py`, `manim.log`, `report.json`, and a `media/` directory containing Manim's video/intermediate output.
- `feedback/ner_feedback.json`: corrected parameter feedback that is converted to BIO spans by the weak aligner.
- `feedback/manual_ner_annotations.json`: exact manually selected entity spans stored as character offsets and used directly as BIO labels.
- `tests/`: automated tests for data loading, weak/manual NER labels, both OCR paths and Manim input preparation.

Model files are not committed in this starter package. Train them locally before using automatic classification and NER extraction. If you are upgrading from an earlier package that trained the Differential Geometry NER field as `curvature`, retrain the NER model with `python train_ner.py`; the app detects and rejects an incompatible old NER checkpoint rather than silently using it. The Streamlit app can still be opened without trained models; it then allows manual family selection and manual template parameter entry.

## 3. Installation

Use Python 3.10–3.12 in a fresh virtual environment.

### Windows PowerShell

```powershell
cd proof_classifier_starter
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd proof_classifier_starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### System dependencies for Manim

Manim also needs system tools that are not installed by `pip`:

- **FFmpeg** for combining and encoding video files.
- A working LaTeX installation for `MathTex` (including `dvisvgm`). On Windows, install a LaTeX distribution such as MiKTeX and make sure its executables are on `PATH`.
- The usual compiler/system dependencies required by Manim on your operating system.

Check the command-line installation before launching the app:

```bash
manim --version
ffmpeg -version
```

If Manim reports a LaTeX or `dvisvgm` error, install/fix the LaTeX toolchain first. The live log pane will show the exact failing command or missing executable reported by Manim.

### Optional dependencies for full-page OCR

Full-page OCR uses PaddleOCR PP-StructureV3. It is deliberately **not** installed by the base `requirements.txt`, because PaddlePaddle wheels differ by operating system, CPU/GPU and CUDA version. First install the matching PaddlePaddle inference wheel using the [official PaddleOCR installation guide](https://www.paddleocr.ai/latest/en/quick_start.html). For example, the official CPU quick-start currently documents this PaddlePaddle 3.2.0 CPU installation command:

```bash
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

Use the current vendor instructions for your exact operating system and hardware rather than copying a GPU/CUDA command for a different platform. Then install the optional document parser and PDF validation dependencies:

```bash
pip install -r requirements-document-ocr.txt
```

The optional requirements use PaddleOCR's `doc-parser` dependency group and `pypdf`. On first use, PP-StructureV3 may download multiple model checkpoints. The application loads this pipeline only when you click **Recognize complete document**.

## 4. Validate data and run tests

From the project root:

```bash
python validate_data.py
python -m unittest discover -s tests -v
```

The tests do not require an actual Manim render. They verify the supported template mappings, parse the template's mathematical inputs, and test the NER alignment/data-loading logic.

## 5. Train the models

For a quick training run without the classifier's nested cross-validation:

```bash
python train_classifier.py --skip-cv
python train_ner.py
```

Or run both stages using the orchestrator:

```bash
python train.py --skip-cv
```

For a fuller classifier evaluation, omit `--skip-cv` and optionally specify the number of folds/repeats:

```bash
python train_classifier.py --cv-folds 5 --cv-repeats 3
```

The NER trainer uses automatically aligned weak labels from the `parameters` fields in the original dataset, plus any direct manual-span annotations saved from the app. A weak-label parameter value must be visibly represented in the question/proof for the aligner to create a span. Review the alignment report and manually inspect LaTeX-heavy examples; automatic weak labels are not a substitute for a manually annotated test set.

The requirements include `pix2tex==0.1.4` for image-to-LaTeX recognition. The app calls the package's Python interface directly; the optional PyQt desktop GUI extra is not needed because Streamlit provides the interface. On first OCR use, pix2tex may download its model checkpoints automatically, so allow network access for that initial model setup. The model can then be reused through Streamlit's resource cache.

## 6. Launch the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

1. Optionally expand **Image input — convert equation images to LaTeX**. Upload one or more cropped equation images, or take a photo with the camera input. Click **Recognize image(s) as LaTeX**, correct the editable OCR result, choose whether to append it to the Question or Proof field, and click **Append recognized LaTeX**.
2. Enter or edit the theorem/question and proof in the text areas. Image OCR is additive; it does not overwrite existing text.
3. Select a manual template family. Leave **Use the trained classifier when available** enabled to let the classifier override this choice when it predicts one of the five supported template families.
4. Click **Analyze proof and prepare Manim parameters**.
5. Inspect the classification and NER results.
6. Review/edit the JSON under **Template parameters**.
7. Click **Generate Manim animation**.

During rendering, the left panel reserves space for the video preview and the right panel shows the live Manim console output (including TeX writes, partial movie files, combining, and final render messages). When the render completes, the video appears in the preview section below. You can download the MP4 and JSON render report. The complete log is also saved in the corresponding `generated/<run-id>/manim.log` file.

## 7. Image input with LaTeX-OCR (pix2tex)

The application integrates the [LaTeX-OCR (pix2tex) GitHub project](https://github.com/lukas-blecher/LaTeX-OCR) using its Python API:

```python
from PIL import Image
from pix2tex.cli import LatexOCR

model = LatexOCR()
latex = model(Image.open("equation.png"))
```

In the app, the OCR section supports PNG, JPEG, WEBP, BMP and TIFF images, including a camera photo. Multiple images can be recognized in one request. Predictions are shown in an editable LaTeX text area and are inserted into the question or proof only when you click the append button. The app does not silently replace typed text or run the proof pipeline until you press Analyze.

**Important scope:** pix2tex is an image-to-LaTeX model intended for mathematical formulas/equations. Crop a single equation or a reasonably compact mathematical region for best results. It is not a general OCR engine for full-page prose, paragraphs, or an entire handwritten proof. Type/paste the surrounding explanation normally and use OCR for the mathematical expressions. Always review OCR output before analysis; the project's README notes that predictions can be wrong and recommends checking the result. Its model checkpoints are downloaded automatically on first use. See the [official GitHub README](https://github.com/lukas-blecher/LaTeX-OCR) for usage and model details.

Uploaded image bytes are processed in memory by the app and are not written into the project dataset or feedback store. Image validation limits uploads to 12 MB and 25 million pixels per image.

## 8. Full-page OCR with PaddleOCR PP-StructureV3

The optional **Full-page OCR — recognize a complete proof or document** panel accepts one PDF or one full-page image (PNG, JPEG, WEBP, BMP, or TIFF). The app validates the file, applies the configured upload/page limits, writes it to a temporary path, and calls PaddleOCR's `PPStructureV3.predict(input=...)`. The PDF is parsed page by page; the app uses the pipeline's `concatenate_markdown_pages()` helper when available and also keeps page-level output for review.

The pipeline is configured for English text and formula recognition. Its Markdown output can contain ordinary prose and LaTeX formula blocks. Review and edit the reconstructed output in the application, then append it to either the Question or Proof field. OCR never overwrites existing text automatically.

Current limits: one document per OCR run, maximum upload size 30 MB, maximum 25 PDF pages, and maximum 60 million pixels for a single image. Encrypted PDFs are not supported. Table/layout reconstruction and mathematical OCR can still make mistakes, so always proofread the result. PP-StructureV3 is a document-layout/OCR pipeline, not a guarantee of mathematically correct transcription. See the [official PP-StructureV3 usage guide](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.en.md) for API and model details.

## 9. Manual NER annotation (exact BIO labels)

The app provides an optional **Manual NER annotation** panel after proof analysis. It is intended for cases where weak alignment selects the wrong occurrence (for example, a value such as `1` appears in an interval endpoint and again as the integral result).

### How to annotate

1. Analyze the question/proof and expand **Annotate exact text spans**. The app displays the exact combined text used by NER (`Question: ...` followed by `Proof: ...`).
2. Copy an exact entity expression from that text into **Copy the exact entity text from above**.
3. If the same phrase occurs more than once, choose the intended occurrence using the contextual preview and character offsets.
4. Choose the correct NER field (for example, `integral_value`, `bound`, or `evaluation_point`) and click **Add selected span**.
5. Add all required spans, remove mistakes if needed, confirm the annotation set, and save it.

The application stores each span with `start`, `end`, `field`, and exact `text`. Offsets refer to the combined question/proof string, including the `Question:` and `Proof:` prefixes. Overlapping spans, invalid offsets, unsupported family labels, and text/offset mismatches are rejected. If you edit the question or proof, analyze again and recreate the annotations because their character offsets will have changed.

Manual annotations support two modes. **Partial annotation** is the default: only selected entity tokens contribute to the training loss, while all unselected tokens are masked with `-100` and are not incorrectly assumed to be non-entities. **Complete annotation** is an explicit checkbox: use it only after marking every occurrence of every supported NER field in the proof; then all remaining ordinary tokens are labelled `O`. Special tokens are ignored in both modes.

Manual examples are saved separately to:

```text
feedback/manual_ner_annotations.json
```

### How manual labels affect training

Run the normal NER training command:

```bash
python train_ner.py
```

`data_loader.py` preserves `manual_entities` and the complete/partial annotation flag. `train_ner.py` detects those records and generates BIO token labels directly from the manually reviewed character offsets. It does **not** run weak alignment for those manually annotated spans. These records are appended to the training partition only; the held-out test split remains drawn from the original `data/` dataset. The trainer reports the number of manual annotation records and spans by field.

Manual annotation can be partial: a proof may contribute one or several labels, and it does not need to include every field for its family. The span must fit inside the tokenizer window; if it is truncated by `--max-length`, training reports the issue and asks you to increase the limit or shorten the proof.

Use manual BIO annotation for ambiguous or repeatedly missed cases, while retaining weak alignment for the rest of your original dataset. This creates a hybrid training set: weakly supervised base records, reviewed parameter feedback, and exact manually annotated spans.

## 10. Example template parameters

### Sequence Convergence

```json
{
  "sequence": "3/n",
  "limit": "0",
  "bound": "3/n"
}
```

The template draws terms of `a_n`, a horizontal limit line and an illustrative epsilon band. Its chosen epsilon and displayed tail index are visual demonstration settings, not a proof-certified choice of `N`.

### Integration and Approximation

```json
{
  "function": "x^2",
  "interval": "[0, 1]",
  "sample_point": "right_endpoint",
  "integral_value": "1/3"
}
```

`sample_point` must be `right_endpoint`, `left_endpoint`, or `midpoint`.

### Vector Operations

```json
{
  "vectors": [[1, 2], [2, 1]]
}
```

Two-dimensional vectors are supported by the bundled Gram-Schmidt scene. String entries such as `["(1, 2)", "(2, 1)"]` are also accepted.

### Linear Transformations

```json
{
  "matrix": [[4, 0], [0, 2]]
}
```

The matrix scene supports square 2×2 and 3×3 matrices. The scene class is selected from the matrix dimension.

### Curves and Moving Frames

```json
{
  "curve": ["t", "t^2"],
  "evaluation_point": 1
}
```

Two coordinate functions select `CurvatureCurve2D`; three select `CurvatureCurve3D`. The curve's evaluated curvature is calculated by the symbolic preprocessing module and displayed in the animation/report.

## 11. Render artifacts and logs

Every render is isolated in a unique folder:

```text
generated/
└── 20261106_230000_abcdef/
    ├── generated_scene.py
    ├── manim.log
    ├── report.json
    └── media/
        ├── Tex/
        └── videos/
            └── generated_scene/
                └── 480p15/
                    └── RenderedScene.mp4
```

The actual video resolution folder depends on the selected Manim quality setting. `manim.log` is the complete stdout/stderr stream; the Streamlit log pane displays its most recent lines while rendering.

## 12. Notes and current limitations

- The classifier recognises six proof families, while NER and Manim support the five families listed above. Function Properties is intentionally excluded from NER and has no template.
- The bundled Gram-Schmidt visualization currently accepts 2D vectors and uses the first two vectors for its two-vector geometric scene.
- The curvature template requires `curve` (two or three coordinate expressions) and `evaluation_point`. NER predicts the complete coordinate tuple as one `curve` span; curvature is computed by the symbolic preprocessing/template and is not learned as an NER label. After changing from the older `curvature` label to `curve`, retrain the NER checkpoint with `python train_ner.py`.
- The matrix template is designed for square 2×2 and 3×3 matrices.
- The app does not claim that the proof is mathematically valid. It visualizes the supplied expressions and reports symbolic calculations that should be checked by the user.
- A successful Manim render requires the Python dependencies and OS-level FFmpeg/LaTeX tools to be installed in the same environment accessible to Streamlit.


## 13. NER output versus Manim parameter types

The NER model predicts text spans, so its raw extraction output is intentionally text-based. For example, it may return:

```json
{
  "vectors": ["(2,3,6)", "(3,-6,2)"],
  "matrix": ["[[0,2],[2,0]]"]
}
```

Before these values are shown in the editable Manim parameter panel or passed to a template, `template_runtime.py` converts them into Python structures:

```python
{
    "vectors": [[2, 3, 6], [3, -6, 2]],
    "matrix": [[0, 2], [2, 0]],
}
```

The adapter converts ordinary integer/decimal strings to Python numbers. Symbolic components such as `sqrt(2)` or `a` remain strings so that `math_preproc.py` can parse them symbolically. This conversion is post-processing; **it does not require retraining the NER model**. The Gram-Schmidt and matrix templates therefore receive the nested-list structure expected by their constructors.

## 14. Improving NER extraction quality

`ner_alignment.py` creates weak (pseudo) BIO labels from each record's `parameters` values and matching text in its `question`/`proof`:

1. `_value_candidates(value, field)` creates plausible written forms for a parameter value.
2. `align_parameters_to_spans(...)` searches the question/proof for those forms and selects a character span, using field-specific context rules where available.
3. `bio_labels_for_text(...)` converts the selected span into BIO token labels for training.

Adding aliases to `_value_candidates` is useful when the JSON target and proof use different but equivalent notation (for example, a target `1/n` written as `n^{-1}`). It does **not** create new examples, and an overly broad alias can create incorrect labels. Add candidates only when they are mathematically equivalent and can be located unambiguously. For the weaker fields (`bound`, `integral_value`, `evaluation_point`, and `function`), also inspect the generated spans and refine the context-selection logic in `_context_is_relevant`.

Adding more records can help if the records are genuinely varied and contain the target expression in the text. Doubling the current dataset is a reasonable experiment, but it is not a guarantee: repeated templates or ambiguous pseudo-labels may simply teach the same mistakes. Keep the test set fixed while comparing experiments, and report per-field precision/recall/F1.

The current trainer evaluates against automatically aligned weak labels, not independently human-annotated NER labels. For a trustworthy estimate of real extraction accuracy, manually annotate a small held-out set of complete entity spans and evaluate predictions against those annotations. To train directly on manually corrected spans, the training pipeline would also need an explicit gold-span/BIO input format; editing only the `parameters` values does not record which repeated occurrence in the proof is the correct one.

## 15. In-app feedback for improving NER

The Streamlit app includes a reviewer-feedback panel beneath the editable **Template input JSON**.

### How to use it

1. Enter a question and proof, then click **Analyze proof and prepare Manim parameters**.
2. Review the NER predictions and the generated template JSON.
3. Correct the JSON values where necessary. Keep the values in the form expected by the template (for example, vectors and matrices as nested JSON lists).
4. Expand **Improve the NER model with feedback**.
5. Tick **I have reviewed these parameters and confirm they are correct**.
6. Click **Save reviewed example for NER training**.

The app saves the question, proof, proof family, and only that family's NER-supported parameter fields to:

```text
feedback/ner_feedback.json
```

The initial model predictions are retained as feedback metadata for later inspection. Identical reviewed examples are deduplicated. The app also previews which corrected fields the current weak-label aligner can match in the question/proof. If a field cannot be aligned, the app warns you.

### How feedback affects training

Saving feedback does **not** update the model immediately. Once you have collected reviewed examples, run:

```bash
python train_ner.py
```

`train_ner.py` automatically reads the feedback file from `feedback/ner_feedback.json`. It uses fully aligned feedback examples in the **training partition only**. Feedback examples are deliberately excluded from the held-out test partition so that evaluation results remain based on the original dataset rather than examples used for learning.

A feedback record is skipped if any parameter field that you supplied cannot be aligned to a span in its question/proof. This is important: otherwise an unmatched correct expression could be treated as an `O` token and teach the model the wrong label. Fix the wording/notation or the alignment candidates first, then save the example again.

You can specify a different feedback directory or JSON file with:

```bash
python train_ner.py --feedback-data path/to/your_feedback.json
```

### What this feedback does—and does not do

- It learns from your corrected parameter values using the project's existing weak-label alignment process.
- Removing an incorrect predicted field from the reviewed JSON means that field is not treated as a positive target in that example.
- It does not immediately fine-tune the running model; retraining is required.
- For ambiguous repeated expressions, use the separate Manual NER annotation panel to select the exact occurrence and save direct BIO labels. Parameter-feedback examples still rely on weak alignment; manually annotated examples bypass it.
- The held-out metrics remain weak-label metrics. For a genuinely independent accuracy estimate, continue maintaining a small manually annotated test set.

`feedback_store.py` implements parameter-feedback validation and persistence; `manual_ner.py` implements exact span validation, persistence and BIO encoding. `app.py` provides both interfaces. `train_ner.py` adds aligned parameter-feedback examples and direct manual-span examples to training while preserving the original test split.
