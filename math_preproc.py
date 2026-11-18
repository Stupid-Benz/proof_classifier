"""
math_preproc.py — pure sympy. No manim imports.

Public API (vectors):
    parse_vector(vec)              -> sp.Matrix
    parse_vectors(vecs)            -> list[sp.Matrix]
    dot_product(u, v)              -> sp.Expr
    vector_norm(v)                 -> sp.Expr
    is_zero_vector(v, tol)         -> bool
    normalize_vector(v)            -> sp.Matrix
    gram_schmidt(v1, v2)           -> (u1, u2, proj, coef)
    vector_dimension(vecs)         -> int
    gs_summary(vectors, normalize) -> dict

Public API (previously added):
    normalize_expression, parse_expression, pick_index_symbol,
    safe_float, parse_interval, symbolic_riemann
"""

import re
import numpy as np
import sympy as sp


FUNCTIONS = ("sqrt", "sin", "cos", "tan", "arcsin", "arccos",
             "arctan", "sinh", "cosh", "tanh", "ln", "log", "exp", "abs")


# =====================================================================
#  General expression parsing
# =====================================================================

_FRAC_ARG = r"(\{[^{}]*\}|[A-Za-z0-9])"
_FRAC_PAT = re.compile(r"\\*\b[dt]?frac\s*" + _FRAC_ARG + r"\s*" + _FRAC_ARG)

def _frac_repl(m):
    a = m.group(1).strip("{}").strip()
    b = m.group(2).strip("{}").strip()
    return f"(({a})/({b}))"

def _sqrt_n_repl(m):
    return f"(({m.group(2)})**(1/({m.group(1)})))"

_SQRT_PAT  = re.compile(r"\\*\bsqrt\s*\{([^{}]*)\}")
_SQRTN_PAT = re.compile(r"\\*\bsqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}")

def _latex_to_python(s: str) -> str:
    """Translate common LaTeX math to Python/SymPy syntax."""
    if "\\" not in s and "{" not in s:
        return s                                   # nothing LaTeX-ish

    # NEW: collapse JSON double-escapes.  \\pi -> \pi ; \\\\sqrt -> \sqrt ;
    # and so on.  This runs first so every rule below only has to handle the
    # single-backslash form.
    s = re.sub(r"\\+", lambda _: "\\", s)

    # Sizing / spacing / currency.
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace("$", "")

    # Named symbols and operators (backslash required — bare "pi" etc. are
    # already valid SymPy tokens and must not be rewritten).
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\infty", "oo")
    s = s.replace(r"\cdot", "*")
    s = s.replace(r"\times", "*")

    # Function macros \sin, \cos, ... -> sin, cos, ...
    s = re.sub(
        r"\\?\b(sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|ln|log|exp|abs)\b",
        r"\1", s,
    )

    # \mathrm{...} / \text{...} wrappers -> contents.
    s = re.sub(r"\\?\b(?:mathrm|text)\s*\{([^{}]*)\}", r"\1", s)

    # Braced superscripts before frac/sqrt so groups have no nested braces.
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", s)

    # \sqrt[n]{...} -> (...)**(1/n)
    s = re.sub(r"\\*\bsqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}",
               r"((\2)**(1/(\1)))", s)

    # \sqrt{...} -> sqrt(...)  (innermost first, one-or-more backslashes)
    sqrt_re = re.compile(r"\\*\bsqrt\s*\{([^{}]*)\}")
    while sqrt_re.search(s):
        s = sqrt_re.sub(r"sqrt(\1)", s)

    # \frac / \dfrac / \tfrac -> ((...)/(...))  (innermost first)
    frac_re = re.compile(r"\\*\b[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    while frac_re.search(s):
        s = frac_re.sub(r"((\1)/(\2))", s)

    return s


_SUP_TO_ASCII = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

def _normalize_superscripts(s: str) -> str:
    """Convert Unicode superscripts to Python exponent syntax.

    't²'   -> 't**(2)'
    't³'   -> 't**(3)'
    't⁻¹'  -> 't**(-1)'
    '(t+1)²' -> '(t+1)**(2)'
    '10³'  -> '10**(3)'
    """
    return re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+",
        lambda m: "**(" + m.group(0).translate(_SUP_TO_ASCII) + ")",
        s,
    )


def normalize_expression(s) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().replace("π", "pi").replace("−", "-")
    s = _normalize_superscripts(s)              # ← NEW, before latex/^ handling
    s = _latex_to_python(s)
    s = s.replace("^", "**")
    s = re.sub(r"(?<=[a-zA-Z0-9\)])\s+(?=[a-zA-Z0-9\(])", "*", s).replace(" ", "")
    s = re.sub(r"(?<=[0-9])(?=[a-zA-Z])", "*", s)
    s = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", "*", s)
    s = re.sub(r"(?<=[a-zA-Z0-9\)])(?=\()", "*", s)
    for f in FUNCTIONS:
        s = s.replace(f + "*(", f + "(")
    return s


def parse_expression(s):
    if isinstance(s, sp.Basic):
        return s
    if isinstance(s, (int, float)):
        return sp.sympify(s)
    normalized = normalize_expression(s)

    n = sp.Symbol("n")
    local = {
        "n": n,
        "N": n,
        "pi": sp.pi,
        "E": sp.E,
        "I": sp.I,
    }

    try:
        return sp.sympify(normalized, locals=local)
    except (sp.SympifyError, SyntaxError) as exc:
        raise ValueError(
            f"Could not parse {s!r}; normalized form was {normalized!r}. "
            f"Underlying error: {exc}"
        ) from exc

_LATEX_MATRIX_ENVS = r"pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|matrix|smallmatrix"
_LATEX_MATRIX_RE = re.compile(
    rf"\\begin\{{({_LATEX_MATRIX_ENVS})\*?\}}(.*?)\\end\{{\1\*?\}}",
    re.S,
)

def _split_top_level_commas(text):
    parts, start, stack = [], 0, []
    pairs = {")": "(", "]": "[", "}": "{"}
    for i, ch in enumerate(text):
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
        elif ch == "," and not stack:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _parse_latex_matrix_rows(text):
    """Return list-of-rows of raw cell strings from a LaTeX matrix env, or None."""
    m = _LATEX_MATRIX_RE.search(text.strip())
    if not m:
        return None
    body = m.group(2).strip()
    if not body:
        return None
    rows = []
    for row_text in re.split(r"\\\\", body):
        row_text = row_text.strip()
        if not row_text:
            continue
        rows.append([cell.strip() for cell in row_text.split("&")])
    return rows or None


_DIAG_STRING_RE = re.compile(
    r"^(?:\\operatorname\s*\{\s*diag\s*\}|\\mathrm\s*\{\s*diag\s*\}|diag)\s*",
    re.I,
)

def _parse_diag_string(text):
    """``diag(a,b,c)`` / ``\\operatorname{diag}(a,b,c)`` → sp.Matrix, else None."""
    m = _DIAG_STRING_RE.match(text.strip())
    if not m:
        return None
    rest = re.sub(r"^\\left\s*", "", text[m.end():].lstrip())
    if not rest.startswith("("):
        return None
    depth, end = 0, None
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    inner = rest[1:end].strip()
    if inner.startswith("[") and inner.endswith("]"):
        candidate = inner[1:-1].strip()
        if "[" not in candidate and "]" not in candidate:
            inner = candidate
    parts = _split_top_level_commas(inner)
    if not parts:
        return None
    n = len(parts)
    mat = sp.zeros(n, n)
    for i, p in enumerate(parts):
        mat[i, i] = parse_expression(p)
    return mat

def pick_index_symbol(*exprs):
    free = set()
    for e in exprs:
        free |= e.free_symbols
    if sp.Symbol("n") in free:
        return sp.Symbol("n")
    return next(iter(free)) if free else sp.Symbol("n")


def safe_float(expr, sym, k, tol=1e-9):
    try:
        val = complex(expr.subs(sym, k))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if abs(val.imag) > tol or not np.isfinite(val.real):
        return None
    return float(val.real)


def parse_interval(s):
    if isinstance(s, (list, tuple)):
        return parse_expression(s[0]), parse_expression(s[1])
    m = re.match(r"[\[\(]\s*(.+?)\s*,\s*(.+?)\s*[\]\)]", str(s).strip())
    if not m:
        raise ValueError(f"Cannot parse interval {s!r}")
    return parse_expression(m.group(1)), parse_expression(m.group(2))


def symbolic_riemann(f, x_sym, a, b, sample_point):
    k = sp.Symbol("k", integer=True, positive=True)
    n = sp.Symbol("n", integer=True, positive=True)
    dx = (b - a) / n
    if sample_point == "right_endpoint":
        x_k = a + k * dx
    elif sample_point == "left_endpoint":
        x_k = a + (k - 1) * dx
    elif sample_point == "midpoint":
        x_k = a + (k - sp.Rational(1, 2)) * dx
    else:
        raise ValueError(f"Unknown sample_point {sample_point!r}")
    term = sp.simplify(f.subs(x_sym, x_k) * dx)
    try:
        S_n = sp.simplify(sp.summation(term, (k, 1, n)))
    except Exception:
        return None, n
    if S_n.has(sp.Sum):
        return None, n
    return S_n, n

# =====================================================================
#  Vector parsing & Gram–Schmidt
# =====================================================================

def _parse_vector_string(text):
    """LaTeX/env column or row, \\langle…\\rangle, or (a,b,…) → sp.Matrix, else None."""
    s = text.strip().replace(r"\left", "").replace(r"\right", "")
    latex_rows = _parse_latex_matrix_rows(s)
    if latex_rows is not None:
        if len(latex_rows) == 1:
            return sp.Matrix([parse_expression(c) for c in latex_rows[0]])
        if all(len(row) == 1 for row in latex_rows):
            return sp.Matrix([parse_expression(row[0]) for row in latex_rows])
        raise ValueError(f"Unsupported LaTeX vector shape: {latex_rows!r}")
    m = re.match(r"\\langle(.+?)\\rangle", s)
    if m:
        parts = _split_top_level_commas(m.group(1))
        return sp.Matrix([parse_expression(p) for p in parts]) if parts else None
    m = re.match(r"[\[\(](.+?)[\]\)]$", s)
    if m:
        parts = _split_top_level_commas(m.group(1))
        if len(parts) >= 2:
            return sp.Matrix([parse_expression(p) for p in parts])
    return None


def parse_vector(vec):
    """List/tuple/Matrix/string of components -> sp.Matrix column."""
    if isinstance(vec, sp.MatrixBase):
        return sp.Matrix(vec)
    if isinstance(vec, str):
        parsed = _parse_vector_string(vec)
        if parsed is not None:
            return parsed
    return sp.Matrix([parse_expression(c) for c in vec])


def parse_vectors(vecs):
    return [parse_vector(v) for v in vecs]


def dot_product(u, v):
    """Standard inner product <u, v> = u^T v."""
    u, v = sp.Matrix(u), sp.Matrix(v)
    if u.shape != v.shape:
        raise ValueError(f"Shape mismatch: {u.shape} vs {v.shape}")
    return sp.simplify((u.T * v)[0])


def vector_norm(v):
    return sp.sqrt(sp.simplify(dot_product(v, v)))


def is_zero_vector(v, tol=1e-12):
    for c in sp.Matrix(v):
        try:
            if abs(complex(c)) > tol:
                return False
        except (TypeError, ValueError):
            return False
    return True


def normalize_vector(v):
    n = vector_norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    return sp.Matrix([sp.simplify(c / n) for c in sp.Matrix(v)])


def vector_dimension(vecs):
    if not vecs:
        return 0
    return len(parse_vector(vecs[0]))


def gram_schmidt(v1, v2):
    """
    Two-vector Gram–Schmidt.

    Returns (u1, u2, proj, coef) where
        u1   = v1
        coef = <v2, u1> / <u1, u1>
        proj = coef * u1
        u2   = v2 - proj
    """
    u1 = sp.Matrix(v1)
    v2 = sp.Matrix(v2)
    ip_u1_u1 = dot_product(u1, u1)
    if ip_u1_u1 == 0:
        raise ValueError("v1 must be nonzero")
    ip_v2_u1 = dot_product(v2, u1)
    coef = sp.simplify(ip_v2_u1 / ip_u1_u1)
    proj = sp.simplify(coef * u1)
    u2 = sp.simplify(v2 - proj)
    return u1, u2, proj, coef


def gs_summary(vectors, normalize=False):
    """
    Gram–Schmidt summary.

    For exactly two vectors, the dict contains everything needed by the
    animation:
        v1, v2, u1, u2, ip_v2_u1, ip_u1_u1, coef, proj, ip_u1_u2, dimension
        (+ 'orthonormal' when normalize=True)

    For more than two, 'u' and 'v' are full lists; the 2-vector detail keys
    are omitted.
    """
    vs = parse_vectors(vectors)
    if len(vs) < 2:
        raise ValueError("Need at least two vectors")

    # --- 2-vector case: detailed output ---
    if len(vs) == 2:
        v1, v2 = vs
        u1, u2, proj, coef = gram_schmidt(v1, v2)
        result = {
            "v1": v1, "v2": v2,
            "u1": u1, "u2": u2,
            "ip_v2_u1": dot_product(v2, u1),
            "ip_u1_u1": dot_product(u1, u1),
            "coef": coef,
            "proj": proj,
            "ip_u1_u2": dot_product(u1, u2),
            "dimension": len(v1),
        }
        if normalize:
            result["orthonormal"] = [
                normalize_vector(u1) if not is_zero_vector(u1) else u1,
                normalize_vector(u2) if not is_zero_vector(u2) else u2,
            ]
        return result

    # --- N-vector case: iterative GS ---
    u_list = []
    for i, vi in enumerate(vs):
        ui = sp.Matrix(vi)
        for uj in u_list:
            denom = dot_product(uj, uj)
            if denom == 0:
                continue
            ui = ui - (dot_product(vi, uj) / denom) * uj
        u_list.append(sp.simplify(ui))

    result = {"v": vs, "u": u_list, "dimension": len(vs[0])}
    if normalize:
        result["orthonormal"] = [
            normalize_vector(ui) if not is_zero_vector(ui) else ui
            for ui in u_list
        ]
    return result

# =====================================================================
#  Matrix parsing & eigen-decomposition
# =====================================================================

def _num(c):
    """Coerce a Python number or string into an exact sp expression."""
    if isinstance(c, float):
        return sp.Rational(str(c))
    return parse_expression(c)


def parse_matrix(m):
    """List-of-rows / sp.MatrixBase / LaTeX / diag(...) -> sp.Matrix (exact where possible)."""
    if isinstance(m, sp.MatrixBase):
        return sp.Matrix(m)
    if isinstance(m, str):
        latex_rows = _parse_latex_matrix_rows(m)
        if latex_rows is not None:
            return sp.Matrix([[_num(c) for c in row] for row in latex_rows])
        diag = _parse_diag_string(m)
        if diag is not None:
            return diag
    return sp.Matrix([[_num(c) for c in row] for row in m])


def apply_matrix(M, v):
    """A @ v as sp.Matrix, fully simplified."""
    return sp.simplify(parse_matrix(M) * sp.Matrix(v))


def matrix_latex(M):
    return sp.latex(parse_matrix(M))


def nice_eigenvector(vec):
    """
    Canonicalise an eigenvector for display:
      - scale so the first nonzero entry is 1,
      - multiply by the lcm of the denominators to clear fractions.

    Irrational entries are left alone (lcm stays 1).
    """
    v = [sp.simplify(sp.sympify(x)) for x in vec]

    lead = next((x for x in v if sp.simplify(x) != 0), None)
    if lead is None:
        return sp.Matrix(v)

    if sp.simplify(lead).is_negative:
        v = [-x for x in v]
        lead = -lead

    v = [sp.simplify(x / lead) for x in v]            # first nonzero -> 1
    lcm = 1
    for x in v:
        d = sp.denom(x)
        if d.is_Integer:
            lcm = int(sp.ilcm(lcm, int(d)))
    return sp.Matrix([sp.simplify(x * lcm) for x in v])


def resolve_eigen(m, order="descending"):
    """
    SymPy is the source of truth; this canonicalises for display.

    Returns dict with keys:
        matrix, dimension,
        eigenvalues    (list, with multiplicity),
        eigenvectors   (list of sp.Matrix, aligned with eigenvalues),
        multiplicities (matching list),
        diagonalizable (bool),
        real_spectrum  (bool)

    `order`:
        'descending'     real eigenvalues, largest first (default)
        'magnitude_desc' |λ| largest first
    """
    M = parse_matrix(m)
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"resolve_eigen needs a square matrix, got {M.shape}")

    flat = []
    for val, mult, vecs in M.eigenvects():
        for v in vecs:
            flat.append((sp.simplify(val), mult, nice_eigenvector(v)))

    if order == "descending" and all(p[0].is_real for p in flat):
        flat.sort(key=lambda p: -float(p[0]))
    elif order == "magnitude_desc":
        flat.sort(key=lambda p: -abs(complex(p[0])))

    values  = [p[0] for p in flat]
    vectors = [p[2] for p in flat]

    return {
        "matrix":         M,
        "dimension":      M.shape[0],
        "eigenvalues":    values,
        "eigenvectors":   vectors,
        "multiplicities": [p[1] for p in flat],
        "diagonalizable": len(flat) == M.shape[0],
        "real_spectrum":  all(v.is_real for v in values),
    }


# =====================================================================
#  Parametric curve geometry (curvature, osculating circle)
# =====================================================================

def parse_curve(curve):
    """
    [x(t), y(t)] -> (x, y, t).

    The parameter symbol is always the real-assumption Symbol('t').
    parse_expression produces a plain Symbol('t') without assumptions,
    which SymPy treats as a *different* symbol — so we substitute to the
    real-assumption version, otherwise sp.diff(x, t) returns 0.
    """
    if not (isinstance(curve, (list, tuple)) and len(curve) == 2):
        raise ValueError("Curve must be [x(t), y(t)] — exactly two components")

    t_real  = sp.Symbol("t", real=True)
    t_plain = sp.Symbol("t")

    x = parse_expression(curve[0]).subs(t_plain, t_real)
    y = parse_expression(curve[1]).subs(t_plain, t_real)
    return x, y, t_real


def curve_geometry(curve, t0, interval=None):
    """
    Curvature geometry of the parametric curve (x(t), y(t)) at t = t0.

    Only `curve` and `t0` are required; `interval` defaults to
    [t0 - 2, t0 + 2] and is only used for display.

    Returns dict with keys:
        x, y, t, dx, dy, ddx, ddy
        speed_sq, numer
        kappa_expr, kappa_signed, kappa
        radius, point, tangent, normal, center
        t0, interval
    """
    x, y, t = parse_curve(curve)
    t0_expr = _refine_t0(t0)
    if not t0_expr.is_real:
        raise ValueError(f"evaluation_point must be real, got {t0_expr}")

    dx, dy   = sp.diff(x, t), sp.diff(y, t)
    ddx, ddy = sp.diff(x, t, 2), sp.diff(y, t, 2)

    speed_sq    = sp.simplify(dx**2 + dy**2)
    speed_sq_t0 = sp.simplify(speed_sq.subs(t, t0_expr))
    if speed_sq_t0 == 0:
        raise ValueError(f"Speed is zero at t = {t0_expr}: curvature undefined")

    numer    = sp.simplify(dx * ddy - dy * ddx)
    numer_t0 = sp.simplify(numer.subs(t, t0_expr))

    kappa_expr      = sp.simplify(sp.Abs(numer) / speed_sq ** sp.Rational(3, 2))
    kappa_signed_t0 = sp.simplify(numer_t0 / speed_sq_t0 ** sp.Rational(3, 2))
    kappa_t0        = sp.simplify(sp.Abs(kappa_signed_t0))

    if kappa_t0.free_symbols:
        raise ValueError(
            f"Curvature still symbolic at t = {t0_expr}: {kappa_t0}"
        )

    radius = sp.oo if kappa_t0 == 0 else sp.simplify(1 / kappa_t0)

    px    = sp.simplify(x.subs(t, t0_expr))
    py    = sp.simplify(y.subs(t, t0_expr))
    point = sp.Matrix([px, py])

    speed_t0 = sp.sqrt(speed_sq_t0)
    dx0 = sp.simplify(dx.subs(t, t0_expr))
    dy0 = sp.simplify(dy.subs(t, t0_expr))
    T   = sp.Matrix([sp.simplify(dx0 / speed_t0),
                     sp.simplify(dy0 / speed_t0)])

    sign = -1 if kappa_signed_t0.is_negative else 1
    N    = sp.Matrix([sp.simplify(-sign * T[1]),
                      sp.simplify( sign * T[0])])

    if kappa_t0 == 0 or not radius.is_real:
        center = None
    else:
        center = sp.Matrix([
            sp.simplify(point[0] + radius * N[0]),
            sp.simplify(point[1] + radius * N[1]),
        ])

    if interval is None:
        t0_f = float(t0_expr)
        interval = [t0_f - 2, t0_f + 2]

    return {
        "x": x, "y": y, "t": t,
        "dx": dx, "dy": dy, "ddx": ddx, "ddy": ddy,
        "speed_sq": speed_sq,
        "numer": numer,
        "kappa_expr": kappa_expr,
        "kappa_signed": kappa_signed_t0,
        "kappa": kappa_t0,
        "radius": radius,
        "point": point,
        "tangent": T,
        "normal": N,
        "center": center,
        "t0": t0_expr,
        "interval": interval,
    }

# =====================================================================
#  Parametric curve geometry — 3-D (curvature, torsion, Frenet frame)
# =====================================================================

def parse_curve_3d(curve):
    """
    [x(t), y(t), z(t)] -> (x, y, z, t).

    Same real-assumption `t` handling as parse_curve — the plain
    Symbol('t') from parse_expression is substituted to the real one.
    """
    if not (isinstance(curve, (list, tuple)) and len(curve) == 3):
        raise ValueError("3-D curve must be [x(t), y(t), z(t)]")

    t_real  = sp.Symbol("t", real=True)
    t_plain = sp.Symbol("t")

    x = parse_expression(curve[0]).subs(t_plain, t_real)
    y = parse_expression(curve[1]).subs(t_plain, t_real)
    z = parse_expression(curve[2]).subs(t_plain, t_real)
    return x, y, z, t_real


def _refine_t0(t0):
    """
    Convert t0 to a clean sympy expression.
    Floats that match pi, e, sqrt(2), sqrt(3) combinations are recovered.
    """
    if isinstance(t0, str):
        return parse_expression(t0)
    if isinstance(t0, sp.Basic):
        return t0
    t0_expr = sp.sympify(t0)
    if isinstance(t0_expr, sp.Float):
        refined = sp.nsimplify(t0_expr, [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3)])
        if abs(float(refined) - float(t0_expr)) < 1e-10:
            return refined
    return t0_expr


def curve_geometry_3d(curve, t0, interval=None):
    """
    Curvature geometry of the parametric curve (x(t), y(t), z(t)) at t = t0.

    Returns dict with keys:
        x, y, z, t
        dx, dy, dz, ddx, ddy, ddz
        velocity, acceleration        -- sp.Matrix (3-vectors)
        speed_sq, speed               -- |γ'|² and |γ'|
        cross                         -- γ' × γ''  (sp.Matrix)
        kappa_expr, kappa             -- κ(t) and κ(t0)  (non-negative)
        torsion_expr, torsion         -- τ(t) and τ(t0)
        radius                        -- 1 / κ(t0)   (sp.oo if κ = 0)
        point                         -- γ(t0) as sp.Matrix
        tangent, normal, binormal     -- T̂, N̂, B̂  (None when κ = 0)
        center                        -- osculating circle center (None when κ = 0)
        t0, interval
    """
    x, y, z, t = parse_curve_3d(curve)
    t0_expr = _refine_t0(t0)
    if not t0_expr.is_real:
        raise ValueError(f"evaluation_point must be real, got {t0_expr}")

    dx, dy, dz       = sp.diff(x, t), sp.diff(y, t), sp.diff(z, t)
    ddx, ddy, ddz    = sp.diff(x, t, 2), sp.diff(y, t, 2), sp.diff(z, t, 2)

    vel = sp.Matrix([dx, dy, dz])
    acc = sp.Matrix([ddx, ddy, ddz])

    speed_sq    = sp.simplify(vel.dot(vel))
    speed_sq_t0 = sp.simplify(speed_sq.subs(t, t0_expr))
    if speed_sq_t0 == 0:
        raise ValueError(f"Speed is zero at t = {t0_expr}: curvature undefined")

    speed_t0 = sp.sqrt(speed_sq_t0)

    cross       = sp.simplify(vel.cross(acc))
    cross_t0    = sp.simplify(cross.subs(t, t0_expr))
    cross_sq_t0 = sp.simplify(cross_t0.dot(cross_t0))

    kappa_expr = sp.simplify(sp.sqrt(cross.dot(cross)) / speed_sq ** sp.Rational(3, 2))
    kappa_t0   = sp.simplify(sp.sqrt(cross_sq_t0) / speed_t0 ** 3)

    if kappa_t0.free_symbols:
        raise ValueError(
            f"Curvature still symbolic at t = {t0_expr}: {kappa_t0}"
        )

    radius = sp.oo if kappa_t0 == 0 else sp.simplify(1 / kappa_t0)

    px    = sp.simplify(x.subs(t, t0_expr))
    py    = sp.simplify(y.subs(t, t0_expr))
    pz    = sp.simplify(z.subs(t, t0_expr))
    point = sp.Matrix([px, py, pz])

    vel_t0 = sp.simplify(vel.subs(t, t0_expr))
    T      = sp.simplify(vel_t0 / speed_t0)

    acc_t0    = sp.simplify(acc.subs(t, t0_expr))
    acc_perp  = sp.simplify(acc_t0 - acc_t0.dot(T) * T)
    acc_perp_norm = sp.simplify(sp.sqrt(acc_perp.dot(acc_perp)))

    if kappa_t0 == 0 or acc_perp_norm == 0:
        N      = None
        B      = None
        center = None
    else:
        N      = sp.simplify(acc_perp / acc_perp_norm)
        B      = sp.simplify(T.cross(N))
        center = sp.simplify(point + radius * N)

    # Torsion
    if cross_sq_t0 == 0:
        torsion_expr = sp.Integer(0)
        torsion_t0   = sp.Integer(0)
    else:
        dddx, dddy, dddz = sp.diff(x, t, 3), sp.diff(y, t, 3), sp.diff(z, t, 3)
        jerk = sp.Matrix([dddx, dddy, dddz])
        torsion_expr = sp.simplify(cross.dot(jerk) / cross.dot(cross))
        torsion_t0   = sp.simplify(torsion_expr.subs(t, t0_expr))
        if torsion_t0.free_symbols:
            torsion_t0 = sp.Integer(0)

    if interval is None:
        t0_f = float(t0_expr)
        interval = [t0_f - 2, t0_f + 2]

    return {
        "x": x, "y": y, "z": z, "t": t,
        "dx": dx, "dy": dy, "dz": dz,
        "ddx": ddx, "ddy": ddy, "ddz": ddz,
        "velocity": vel, "acceleration": acc,
        "speed_sq": speed_sq, "speed": speed_t0,
        "cross": cross,
        "kappa_expr": kappa_expr, "kappa": kappa_t0,
        "torsion_expr": torsion_expr, "torsion": torsion_t0,
        "radius": radius,
        "point": point,
        "tangent": T, "normal": N, "binormal": B,
        "center": center,
        "t0": t0_expr, "interval": interval,
    }