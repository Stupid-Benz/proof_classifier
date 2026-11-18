# Proof-to-Manim parameter contracts

The app now supports five proof families. The former **Function Properties** family is removed from the classifier taxonomy and its dataset file is not included in the project data directory.

| Proof family | Template file | Parameters passed to template |
|---|---|---|
| Sequence Convergence | `sequence_template.py` | `sequence`, `limit`, `bound` |
| Integration | `integration_template.py` | `function`, `interval`, `sample_point`, `integral_value` |
| Vector | `gram_schmidt_template.py` | `vectors` |
| Matrix | `matrix_eigen_template.py` | `matrix` |
| Curvature | `curvature_template.py` | `curve`, `evaluation_point` |

## Sequence Convergence

DistilBERT performs token-level extraction of:

- `sequence`: general term, for example `3/n`.
- `limit`: target limit, for example `0`.
- `bound`: source-text span describing the choice of N, for example `Choose N ... greater than 3/eps`.

`bound` is optional for proof styles that do not explicitly choose an epsilon-N threshold. The template receives the extracted span as text; the user can edit it into a cleaner mathematical condition if desired. The fixed epsilon-N conditions (such as `epsilon > 0`, `N` being natural, and `n >= N`) are supplied by the selected template/proof style, not learned as NER entities.

## Integration

The template receives:

- `function`: integrand, e.g. `x^2`.
- `interval`: two endpoints, e.g. `[0, 1]`.
- `sample_point`: one of `left_endpoint`, `right_endpoint`, or `midpoint`.
- `integral_value`: expected exact value, e.g. `1/3`.

## Vector

The template receives `vectors`, a list of vectors, e.g. `[[1, 2], [2, 1]]`. The bundled starter animation currently visualizes two-dimensional Gram-Schmidt.

## Matrix

The template receives `matrix`, e.g. `[[4, 0], [0, 2]]`. `EigenDiagonal2D` is selected for a 2x2 matrix and `EigenDiagonal3D` for a 3x3 matrix. The uploaded template rejects matrices that are not diagonalizable over the reals.

## Curvature

The template receives `curve`, a list of coordinate expressions, and `evaluation_point`, e.g. `curve=["t", "t^2"]`, `evaluation_point=1`. The two-dimensional or three-dimensional scene is selected from the number of coordinate expressions.

## Rendering and report

The Streamlit app shows the parameter editor before rendering. When rendering finishes, it displays the MP4, render status, elapsed time, template/class name, command, console log, and a downloadable JSON report. Each run is saved under `generated/<run-id>/`.

`math_preproc.py` is kept separate from Manim and supplies SymPy parsing and geometry operations. It is copied into `templates/` as well so the uploaded `matrix_eigen_template.py` can import it directly.

## Template source status

The uploaded files available for this update were `math_preproc.py` and `matrix_eigen_template.py`. The project includes starter implementations under the other four required filenames so the rendering workflow can be exercised. Replace those starter files with the user's own versions when available; the app's template filenames and parameter contracts are centralized in `parameter_schema.py`.
