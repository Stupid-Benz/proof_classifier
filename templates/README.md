# Manim template modules

Each template is a reusable Manim scene class. You can use a template directly by subclassing it and passing the needed parameters in `super().__init__`, or let the Streamlit app create the wrapper automatically.

For example, the curvature template can be used directly like this:

```python
from manim import *
from curvature_template import CurvatureCurve2D

class A01(CurvatureCurve2D):
    def __init__(self, **kwargs):
        super().__init__(curve=["t", "t^2"], evaluation_point=1, **kwargs)
```

The application resolves the proof-family name to a template file and builds a small `RenderedScene` subclass with the editable JSON parameters. For Curves and Moving Frames, NER predicts the whole coordinate tuple as one `curve` entity (for example, `(t, t^2)`); the runtime parses that entity into `['t', 't^2']` before building the wrapper. This is the same constructor interface as the direct example above.

| File | Expected parameters | Default scene class |
|---|---|---|
| `sequence_convergence_template.py` | `sequence`, `limit`, optional `bound` | `SequenceConvergenceTemplate` |
| `integration_approximation_template.py` | `function`, `interval`, `sample_point`, `integral_value` | `IntegrationApproximationTemplate` |
| `gram_schmidt_template.py` | `vectors` | `GramSchmidtTemplate` |
| `eigen_diagonal_template.py` | `matrix` | `EigenDiagonal2D` or `EigenDiagonal3D` |
| `curvature_template.py` | `curve`, `evaluation_point` | `CurvatureCurve2D` or `CurvatureCurve3D` |

`template_runtime.py` selects `EigenDiagonal2D`/`EigenDiagonal3D` based on matrix dimension and `CurvatureCurve2D`/`CurvatureCurve3D` based on the number of coordinate functions. The app also offers an optional base-scene-class override under Advanced Manim settings.

`math_preproc.py` is included in this folder for standalone template use. The application uses the project-root `math_preproc.py`; keep the two copies synchronized if you edit the helper.
