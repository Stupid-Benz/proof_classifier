"""
eigen_diagonal_template.py — eigenvalues / eigenvectors of a diagonalizable,
real-spectrum 2x2 or 3x3 matrix.

Two scenes in one module:

    EigenDiagonal2D(matrix=[[...], [...]])              — flat NumberPlane
    EigenDiagonal3D(matrix=[[...], [...], [...]])       — ThreeDAxes + ambient rotation

Both share:
  - resolve_eigen from math_preproc as the single source of truth
  - the same helper functions (_vec_tex, _matrix_tex, _step_for)
  - the same palette conventions:
        cool blue / teal / gold      → grid line families by direction
        warm red / orange / yellow   → eigenvector arrows (one per eigenpair)
        green                        → unit square / unit cube (pre-image)
        pink                         → image of the unit shape (post-image)
  - validity gates that raise ValueError at construction time for
    non-diagonalizable or complex-spectrum matrices
  - a ghost trail of the pre-grid that fades in after the transform
  - a formula panel showing  A·v_i = λ_i·v_i  per eigenpair

2-D reads with two-colour grid (TEAL verticals, BLUE horizontals).
3-D reads with three-colour grid (BLUE x-dir, TEAL y-dir, GOLD z-dir)
because the extra axis needs an extra tag to stay legible under shear.

Examples:

    EigenDiagonal2D(matrix=[[4, 0], [0, 2]])
    EigenDiagonal2D(matrix=[[2, 1], [1, 2]])
    EigenDiagonal2D(matrix=[[0, 1], [1, 0]])
    EigenDiagonal2D(matrix=[[3, 2], [1, 2]])
    EigenDiagonal2D(matrix=[[2, 0], [0, 2]])

    EigenDiagonal3D(matrix=[[4, 0, 0], [0, 2, 0], [0, 0, 1]])
    EigenDiagonal3D(matrix=[[2, 1, 0], [1, 2, 0], [0, 0, 3]])
    EigenDiagonal3D(matrix=[[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    EigenDiagonal3D(matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    EigenDiagonal3D(matrix=[[1, 0, 0], [0, -1, 0], [0, 0, 2]])

CLI:

    manim -pql eigen_diagonal_template.py EigenDiagonal2D
    manim -pql eigen_diagonal_template.py EigenDiagonal3D
"""

import numpy as np
import sympy as sp
from manim import *

from math_preproc import resolve_eigen


# =====================================================================
#  Shared helpers
# =====================================================================

def _vec_tex(v):
    return r"\left(" + ",".join(sp.latex(sp.simplify(c)) for c in v) + r"\right)"


def _matrix_tex(M):
    return sp.latex(sp.Matrix(M))


def _step_for(span):
    if span <= 5:   return 1
    if span <= 12:  return 2
    if span <= 30:  return 5
    return 10


# =====================================================================
#  Palette
# =====================================================================

# --- grid line families (cool, direction-coded) ---------------------
COL_X_2D  = TEAL_A       # 2-D: vertical lines, before
COL_Y_2D  = BLUE_A       # 2-D: horizontal lines, before
COL_X_2D_POST = TEAL_D   # 2-D: vertical lines, after
COL_Y_2D_POST = BLUE_D   # 2-D: horizontal lines, after

COL_X_3D  = BLUE_D       # 3-D: x-direction lines
COL_Y_3D  = TEAL_D       # 3-D: y-direction lines
COL_Z_3D  = GOLD_D       # 3-D: z-direction lines

# --- unit shape (green pre-image, pink post-image) ------------------
COL_SHAPE_PRE  = GREEN_D
COL_SHAPE_POST = PINK    # base PINK; no _D variant exists in Manim CE

# --- eigenvectors (warm) -------------------------------------------
COL_EIG_2D = [RED, ORANGE]              # 2 eigenvectors
COL_EIG_GD = [RED_E, GOLD_E]            # matching dashed guides (2-D)
COL_EIG_3D = [RED, ORANGE, YELLOW]      # 3 eigenvectors

# --- shared stroke tuning -------------------------------------------
GRID_WIDTH  = 2.0
GRID_ALPHA  = 0.95
GHOST_ALPHA_2D = 0.22
GHOST_ALPHA_3D = 0.18


# =====================================================================
#  Scene 1 — 2-D
# =====================================================================

class EigenDiagonal2D(Scene):
    """
    Flat 2-D view: NumberPlane + two-colour grid + unit square + eigenlines.

    Best for teaching:
      - symmetric  → orthogonal eigenvectors, pure stretch
      - reflection → eigenvalues ±1, colour families visibly swap direction
      - shear      → non-orthogonal eigenvectors, most dramatic visually
    """

    def __init__(self, matrix=None, **kwargs):
        super().__init__(**kwargs)
        raw = matrix if matrix is not None else [[4, 0], [0, 2]]
        data = resolve_eigen(raw, order="descending")

        if data["dimension"] != 2:
            raise ValueError("EigenDiagonal2D needs a 2x2 matrix")
        if not data["diagonalizable"]:
            raise ValueError("Not diagonalizable — use a different proof style")
        if not data["real_spectrum"]:
            raise ValueError("Complex spectrum — use a different proof style")

        self.data         = data
        self.M            = data["matrix"]
        self.eigenvalues  = data["eigenvalues"]
        self.eigenvectors = data["eigenvectors"]
        self.M_np         = np.array(self.M.tolist(), dtype=float)

    # ------------------------------------------------------------------
    def _bounds(self):
        corners = [sp.Matrix(p) for p in [(0, 0), (1, 0), (1, 1), (0, 1)]]
        images  = [self.M * c for c in corners]
        eig_ims = [self.M * v for v in self.eigenvectors]
        pts     = corners + images + list(self.eigenvectors) + eig_ims

        xs = [float(p[0]) for p in pts] + [0.0]
        ys = [float(p[1]) for p in pts] + [0.0]

        pad  = 0.6
        step = _step_for(max(max(xs) - min(xs), max(ys) - min(ys)))

        x_lo = float(np.floor((min(xs) - pad) / step) * step)
        x_hi = float(np.ceil ((max(xs) + pad) / step) * step)
        y_lo = float(np.floor((min(ys) - pad) / step) * step)
        y_hi = float(np.ceil ((max(ys) + pad) / step) * step)
        return (x_lo, x_hi, step), (y_lo, y_hi, step)

    # ------------------------------------------------------------------
    def _make_plane(self, xr, yr):
        x_lo, x_hi, x_step = xr
        y_lo, y_hi, y_step = yr
        x_span, y_span = x_hi - x_lo, y_hi - y_lo

        max_len = 6.2
        if x_span >= y_span:
            x_length, y_length = max_len, max_len * y_span / x_span
        else:
            x_length, y_length = max_len * x_span / y_span, max_len

        return NumberPlane(
            x_range=[x_lo, x_hi, x_step],
            y_range=[y_lo, y_hi, y_step],
            x_length=x_length, y_length=y_length,
            background_line_style={
                "stroke_opacity": 0.12, "stroke_width": 1,
            },
            axis_config={
                "stroke_width": 2,
                "stroke_opacity": 0.35,
                "include_numbers": True,
                "font_size": 16,
                "decimal_number_config": {"num_decimal_places": 0},
            },
        ).shift(LEFT * 3.6 + DOWN * 0.4)

    # ------------------------------------------------------------------
    def _grid_lines_split(self, plane, xr, yr, M_np=None):
        x_lo, x_hi, x_step = xr
        y_lo, y_hi, y_step = yr

        v_col = COL_X_2D      if M_np is None else COL_X_2D_POST
        h_col = COL_Y_2D      if M_np is None else COL_Y_2D_POST

        verticals = VGroup()
        for x in np.arange(x_lo, x_hi + x_step / 2, x_step):
            p1 = np.array([x, y_lo], dtype=float)
            p2 = np.array([x, y_hi], dtype=float)
            if M_np is not None:
                p1, p2 = M_np @ p1, M_np @ p2
            verticals.add(Line(
                plane.c2p(*p1), plane.c2p(*p2),
                color=v_col, stroke_width=GRID_WIDTH,
                stroke_opacity=GRID_ALPHA,
            ))

        horizontals = VGroup()
        for y in np.arange(y_lo, y_hi + y_step / 2, y_step):
            p1 = np.array([x_lo, y], dtype=float)
            p2 = np.array([x_hi, y], dtype=float)
            if M_np is not None:
                p1, p2 = M_np @ p1, M_np @ p2
            horizontals.add(Line(
                plane.c2p(*p1), plane.c2p(*p2),
                color=h_col, stroke_width=GRID_WIDTH,
                stroke_opacity=GRID_ALPHA,
            ))

        return verticals, horizontals

    # ------------------------------------------------------------------
    def _eigline(self, plane, v, xr, yr, color):
        x_lo, x_hi, _ = xr
        y_lo, y_hi, _ = yr
        vx, vy = float(v[0]), float(v[1])
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            return VGroup()

        candidates = []
        if abs(vx) > 1e-9:
            candidates += [abs(x_lo / vx), abs(x_hi / vx)]
        if abs(vy) > 1e-9:
            candidates += [abs(y_lo / vy), abs(y_hi / vy)]
        t = min(candidates) * 0.95

        return DashedLine(
            plane.c2p(vx * -t, vy * -t),
            plane.c2p(vx *  t, vy *  t),
            color=color, stroke_width=2,
            dash_length=0.12, stroke_opacity=0.55,
        )

    # ------------------------------------------------------------------
    def construct(self):
        xr, yr = self._bounds()
        M_np   = self.M_np

        title = MathTex(r"A = " + _matrix_tex(self.M), font_size=42)
        title.to_edge(UP, buff=0.3)

        ref_plane = self._make_plane(xr, yr)

        v_pre,  h_pre  = self._grid_lines_split(ref_plane, xr, yr, None)
        v_post, h_post = self._grid_lines_split(ref_plane, xr, yr, M_np)

        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        pre_square = Polygon(
            *[ref_plane.c2p(x, y) for x, y in corners],
            color=COL_SHAPE_PRE,
            fill_color=COL_SHAPE_PRE, fill_opacity=0.28,
            stroke_width=3,
        )
        post_square = Polygon(
            *[ref_plane.c2p(*(M_np @ np.array([x, y], dtype=float)))
              for x, y in corners],
            color=COL_SHAPE_POST,
            fill_color=COL_SHAPE_POST, fill_opacity=0.28,
            stroke_width=3,
        )

        pre_labels = VGroup(
            MathTex("1", font_size=26, color=COL_SHAPE_PRE).next_to(
                Line(ref_plane.c2p(0, 0), ref_plane.c2p(1, 0)),
                DOWN, buff=0.08),
            MathTex("1", font_size=26, color=COL_SHAPE_PRE).next_to(
                Line(ref_plane.c2p(1, 0), ref_plane.c2p(1, 1)),
                RIGHT, buff=0.08),
        )

        eig_lines = VGroup(*[
            self._eigline(ref_plane, v, xr, yr, COL_EIG_GD[i])
            for i, v in enumerate(self.eigenvectors)
        ])

        origin = ref_plane.c2p(0, 0)

        def make_arrow(end_xy, color):
            return Arrow(
                origin, ref_plane.c2p(*end_xy),
                buff=0, color=color, stroke_width=6, tip_length=0.2,
            )

        pre_arrows, post_arrows = [], []
        for i, (lam, v) in enumerate(zip(self.eigenvalues, self.eigenvectors)):
            pre_arrows.append(make_arrow(
                [float(v[0]), float(v[1])], COL_EIG_2D[i]))
            post_arrows.append(make_arrow(
                [float(lam * v[0]), float(lam * v[1])], COL_EIG_2D[i]))

        def make_label(arrow, tex, color):
            def updater(m):
                d = arrow.get_end() - arrow.get_start()
                n = np.linalg.norm(d)
                unit = d / n if n > 1e-6 else UR
                m.move_to(arrow.get_end() + 0.3 * unit)
            lab = MathTex(tex, color=color, font_size=28)
            updater(lab)
            lab.add_updater(updater)
            return lab

        labels = [
            make_label(pre_arrows[i], f"v_{i+1}", COL_EIG_2D[i])
            for i in range(len(pre_arrows))
        ]

        eqs = VGroup()
        for i, (lam, v) in enumerate(zip(self.eigenvalues, self.eigenvectors)):
            eqs.add(MathTex(
                rf"A\,{_vec_tex(v)} = {sp.latex(lam)}\,{_vec_tex(v)}",
                font_size=28, color=COL_EIG_2D[i],
            ))
        eqs.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        eqs.move_to(RIGHT * 3.5 + UP * 0.2)

        legend = VGroup(
            VGroup(
                Line(ORIGIN, RIGHT * 0.35, color=COL_X_2D, stroke_width=3),
                MathTex(r"x=\text{const}", font_size=20),
            ).arrange(RIGHT, buff=0.12),
            VGroup(
                Line(ORIGIN, RIGHT * 0.35, color=COL_Y_2D, stroke_width=3),
                MathTex(r"y=\text{const}", font_size=20),
            ).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        legend.to_corner(DR, buff=0.35)

        # --- animate ---------------------------------------------------
        self.play(Write(title))
        self.play(Create(ref_plane), run_time=0.8)

        self.play(LaggedStart(*[Create(l) for l in v_pre],
                              lag_ratio=0.04, run_time=0.7))
        self.play(LaggedStart(*[Create(l) for l in h_pre],
                              lag_ratio=0.04, run_time=0.7))
        self.play(FadeIn(legend))

        self.play(Create(eig_lines), run_time=0.6)
        self.play(Create(pre_square), FadeIn(pre_labels))
        self.play(
            *[GrowArrow(a) for a in pre_arrows],
            *[FadeIn(l)    for l in labels],
        )
        self.wait(0.4)

        self.play(
            Transform(v_pre, v_post),
            Transform(h_pre, h_post),
            Transform(pre_square, post_square),
            FadeOut(pre_labels),
            *[Transform(a, na) for a, na in zip(pre_arrows, post_arrows)],
            run_time=3,
        )

        v_ghost, h_ghost = self._grid_lines_split(ref_plane, xr, yr, None)
        v_ghost.set_stroke(opacity=GHOST_ALPHA_2D)
        h_ghost.set_stroke(opacity=GHOST_ALPHA_2D)
        self.play(FadeIn(v_ghost), FadeIn(h_ghost), run_time=0.8)
        self.wait(0.4)

        for eq in eqs:
            self.play(Write(eq))
            self.wait(0.2)

        self.wait(1.5)


# =====================================================================
#  Scene 2 — 3-D
# =====================================================================

class EigenDiagonal3D(ThreeDScene):
    """
    Three-dimensional view: ThreeDAxes + three-colour grid on the six faces
    of the bounding box + unit cube + eigenvector arrows.

    Grid is coloured by *direction* (not by plane):
        BLUE = x-direction lines
        TEAL = y-direction lines
        GOLD = z-direction lines

    Each direction family is drawn on all four box faces that contain it,
    so the full box reads as a closed parallelepiped rather than half-open.
    Set full_grid=False to fall back to the three origin-corner faces only
    (sparser, but the box will look "cut in half" from some angles).
    """

    def __init__(self, matrix=None, full_grid=True, **kwargs):
        super().__init__(**kwargs)
        raw = matrix if matrix is not None else [[4, 0, 0], [0, 2, 0], [0, 0, 1]]
        data = resolve_eigen(raw, order="descending")

        if data["dimension"] != 3:
            raise ValueError("EigenDiagonal3D needs a 3x3 matrix")
        if not data["diagonalizable"]:
            raise ValueError("Not diagonalizable — use a different proof style")
        if not data["real_spectrum"]:
            raise ValueError("Complex spectrum — use a different proof style")

        self.data         = data
        self.M            = data["matrix"]
        self.eigenvalues  = data["eigenvalues"]
        self.eigenvectors = data["eigenvectors"]
        self.M_np         = np.array(self.M.tolist(), dtype=float)
        self.full_grid    = bool(full_grid)

    # ------------------------------------------------------------------
    def _bounds(self):
        corners = [(x, y, z) for x in [0, 1] for y in [0, 1] for z in [0, 1]]
        images  = [self.M * sp.Matrix(c) for c in corners]
        eig_ims = [self.M * v for v in self.eigenvectors]
        pts     = corners + images + list(self.eigenvectors) + eig_ims

        xs = [float(p[0]) for p in pts] + [0.0]
        ys = [float(p[1]) for p in pts] + [0.0]
        zs = [float(p[2]) for p in pts] + [0.0]

        pad  = 0.6
        span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        step = _step_for(span)

        def rng(vals):
            lo = float(np.floor((min(vals) - pad) / step) * step)
            hi = float(np.ceil ((max(vals) + pad) / step) * step)
            return (lo, hi, step)

        return {"x": rng(xs), "y": rng(ys), "z": rng(zs)}

    # ------------------------------------------------------------------
    def _make_axes(self, b):
        x_lo, x_hi, x_s = b["x"]
        y_lo, y_hi, y_s = b["y"]
        z_lo, z_hi, z_s = b["z"]

        return ThreeDAxes(
            x_range=[x_lo, x_hi, x_s],
            y_range=[y_lo, y_hi, y_s],
            z_range=[z_lo, z_hi, z_s],
            x_length=6.0, y_length=6.0, z_length=6.0,
            axis_config={
                "stroke_width": 2,
                "stroke_opacity": 0.45,
                "include_numbers": False,
            },
        )

    # ------------------------------------------------------------------
    #  All four box faces per direction (full_grid=True) or just the
    #  two origin-corner faces (full_grid=False).
    # ------------------------------------------------------------------
    def _dir_lines(self, axes, b, direction, M_np=None, color=None):
        x_lo, x_hi, x_s = b["x"]
        y_lo, y_hi, y_s = b["y"]
        z_lo, z_hi, z_s = b["z"]
        full = self.full_grid

        def T(p):
            return p if M_np is None else M_np @ p

        def add(lines, p1, p2):
            lines.add(Line(
                axes.c2p(*T(p1)), axes.c2p(*T(p2)),
                color=color, stroke_width=GRID_WIDTH, stroke_opacity=GRID_ALPHA,
            ))

        lines = VGroup()

        if direction == "x":
            # faces z = z_lo and z = z_hi: x-lines at each y
            for y in np.arange(y_lo, y_hi + y_s / 2, y_s):
                add(lines,
                    np.array([x_lo, y, z_lo]), np.array([x_hi, y, z_lo]))
                if full:
                    add(lines,
                        np.array([x_lo, y, z_hi]), np.array([x_hi, y, z_hi]))
            # faces y = y_lo and y = y_hi: x-lines at each z
            for z in np.arange(z_lo, z_hi + z_s / 2, z_s):
                add(lines,
                    np.array([x_lo, y_lo, z]), np.array([x_hi, y_lo, z]))
                if full:
                    add(lines,
                        np.array([x_lo, y_hi, z]), np.array([x_hi, y_hi, z]))

        elif direction == "y":
            # faces z = z_lo and z = z_hi: y-lines at each x
            for x in np.arange(x_lo, x_hi + x_s / 2, x_s):
                add(lines,
                    np.array([x, y_lo, z_lo]), np.array([x, y_hi, z_lo]))
                if full:
                    add(lines,
                        np.array([x, y_lo, z_hi]), np.array([x, y_hi, z_hi]))
            # faces x = x_lo and x = x_hi: y-lines at each z
            for z in np.arange(z_lo, z_hi + z_s / 2, z_s):
                add(lines,
                    np.array([x_lo, y_lo, z]), np.array([x_lo, y_hi, z]))
                if full:
                    add(lines,
                        np.array([x_hi, y_lo, z]), np.array([x_hi, y_hi, z]))

        elif direction == "z":
            # faces y = y_lo and y = y_hi: z-lines at each x
            for x in np.arange(x_lo, x_hi + x_s / 2, x_s):
                add(lines,
                    np.array([x, y_lo, z_lo]), np.array([x, y_lo, z_hi]))
                if full:
                    add(lines,
                        np.array([x, y_hi, z_lo]), np.array([x, y_hi, z_hi]))
            # faces x = x_lo and x = x_hi: z-lines at each y
            for y in np.arange(y_lo, y_hi + y_s / 2, y_s):
                add(lines,
                    np.array([x_lo, y, z_lo]), np.array([x_lo, y, z_hi]))
                if full:
                    add(lines,
                        np.array([x_hi, y, z_lo]), np.array([x_hi, y, z_hi]))

        return lines

    # ------------------------------------------------------------------
    def _cube_edges(self, axes, M_np=None):
        corners = [(x, y, z) for x in [0, 1] for y in [0, 1] for z in [0, 1]]
        segments = []
        for i, c1 in enumerate(corners):
            for c2 in corners[i + 1:]:
                if sum(abs(a - b) for a, b in zip(c1, c2)) == 1:
                    p1 = np.array(c1, dtype=float)
                    p2 = np.array(c2, dtype=float)
                    if M_np is not None:
                        p1, p2 = M_np @ p1, M_np @ p2
                    segments.append((axes.c2p(*p1), axes.c2p(*p2)))
        return segments

    # ------------------------------------------------------------------
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-50 * DEGREES)

        bounds = self._bounds()
        axes   = self._make_axes(bounds)

        title = MathTex(r"A = " + _matrix_tex(self.M), font_size=38)
        title.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)

        # axis letter labels (billboarded)
        x_hi = bounds["x"][1]
        y_hi = bounds["y"][1]
        z_hi = bounds["z"][1]
        for end, letter in (
            (axes.c2p(x_hi, 0, 0), "x"),
            (axes.c2p(0, y_hi, 0), "y"),
            (axes.c2p(0, 0, z_hi), "z"),
        ):
            lbl = MathTex(letter, color=GREY, font_size=26).move_to(end)
            self.add_fixed_orientation_mobjects(lbl)

        # grid families — now covering all six box faces
        pre_x  = self._dir_lines(axes, bounds, "x", None,      COL_X_3D)
        pre_y  = self._dir_lines(axes, bounds, "y", None,      COL_Y_3D)
        pre_z  = self._dir_lines(axes, bounds, "z", None,      COL_Z_3D)
        post_x = self._dir_lines(axes, bounds, "x", self.M_np, COL_X_3D)
        post_y = self._dir_lines(axes, bounds, "y", self.M_np, COL_Y_3D)
        post_z = self._dir_lines(axes, bounds, "z", self.M_np, COL_Z_3D)

        # unit cube
        pre_edges = VGroup(*[
            Line(p1, p2, color=COL_SHAPE_PRE, stroke_width=3)
            for p1, p2 in self._cube_edges(axes, None)
        ])
        post_edges = VGroup(*[
            Line(p1, p2, color=COL_SHAPE_POST, stroke_width=3)
            for p1, p2 in self._cube_edges(axes, self.M_np)
        ])

        # eigenvector arrows
        origin = axes.c2p(0, 0, 0)
        pre_arrows, post_arrows = [], []
        for i, (lam, v) in enumerate(zip(self.eigenvalues, self.eigenvectors)):
            v_np  = np.array([float(c) for c in v])
            lam_f = float(lam)
            pre_arrows.append(Arrow3D(
                origin, axes.c2p(*v_np),
                color=COL_EIG_3D[i],
                thickness=0.025, base_radius=0.06, height=0.20,
            ))
            post_arrows.append(Arrow3D(
                origin, axes.c2p(*(lam_f * v_np)),
                color=COL_EIG_3D[i],
                thickness=0.025, base_radius=0.06, height=0.20,
            ))

        # eigenvector labels (billboarded, fixed orientation)
        for i, v in enumerate(self.eigenvectors):
            v_np = np.array([float(c) for c in v])
            norm = max(np.linalg.norm(v_np), 1e-9)
            pos = axes.c2p(*(v_np + 0.35 * v_np / norm))
            lbl = MathTex(f"v_{i+1}", color=COL_EIG_3D[i], font_size=26).move_to(pos)
            self.add_fixed_orientation_mobjects(lbl)

        # formula panel (fixed in frame)
        eqs = VGroup()
        for i, (lam, v) in enumerate(zip(self.eigenvalues, self.eigenvectors)):
            eqs.add(MathTex(
                rf"A\,{_vec_tex(v)} = {sp.latex(lam)}\,{_vec_tex(v)}",
                font_size=24, color=COL_EIG_3D[i],
            ))
        eqs.arrange(DOWN, aligned_edge=LEFT, buff=0.30)
        eqs.move_to(RIGHT * 4.2 + UP * 0.5)

        # legend (fixed in frame)
        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=COL_X_3D, stroke_width=3),
                   MathTex(r"x\text{-dir}", font_size=18)
                   ).arrange(RIGHT, buff=0.10),
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=COL_Y_3D, stroke_width=3),
                   MathTex(r"y\text{-dir}", font_size=18)
                   ).arrange(RIGHT, buff=0.10),
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=COL_Z_3D, stroke_width=3),
                   MathTex(r"z\text{-dir}", font_size=18)
                   ).arrange(RIGHT, buff=0.10),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.to_corner(DR, buff=0.30)

        # --- animate ---------------------------------------------------
        self.play(Write(title))
        self.play(Create(axes), run_time=1.0)

        self.play(
            LaggedStart(*[Create(l) for l in pre_x], lag_ratio=0.05, run_time=0.6),
            LaggedStart(*[Create(l) for l in pre_y], lag_ratio=0.05, run_time=0.6),
            LaggedStart(*[Create(l) for l in pre_z], lag_ratio=0.05, run_time=0.6),
        )

        self.add_fixed_in_frame_mobjects(legend)
        self.play(FadeIn(legend), run_time=0.5)

        self.play(*[Create(l) for l in pre_edges], run_time=0.8)
        self.play(
            *[Create(a) for a in pre_arrows],
            run_time=0.8,
        )
        self.wait(0.4)

        self.play(
            Transform(pre_x, post_x),
            Transform(pre_y, post_y),
            Transform(pre_z, post_z),
            Transform(pre_edges, post_edges),
            *[Transform(a, na) for a, na in zip(pre_arrows, post_arrows)],
            run_time=3.5,
        )

        ghost_x = self._dir_lines(axes, bounds, "x", None, COL_X_3D)
        ghost_y = self._dir_lines(axes, bounds, "y", None, COL_Y_3D)
        ghost_z = self._dir_lines(axes, bounds, "z", None, COL_Z_3D)
        for g in (ghost_x, ghost_y, ghost_z):
            g.set_stroke(opacity=GHOST_ALPHA_3D)
        self.play(
            FadeIn(ghost_x), FadeIn(ghost_y), FadeIn(ghost_z),
            run_time=0.8,
        )

        for eq in eqs:
            self.add_fixed_in_frame_mobjects(eq)
            self.play(FadeIn(eq), run_time=0.4)

        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        self.wait(1)