"""
gram_schmidt_template.py — Gram-Schmidt for two 2-D or two 3-D vectors.

Two scene classes:
    GramSchmidt2D  — flat NumberPlane, integer ticks, no camera motion.
    GramSchmidt3D  — ThreeDAxes, billboarded labels, optional auto-rotate.

All algebra lives in math_preproc.py (unchanged; dimension-agnostic).
"""

import numpy as np
import sympy as sp
from manim import *

from math_preproc import gs_summary


# ====================================================================
#  Shared helpers
# ====================================================================

def _vec_tex(v):
    return r"\left(" + ",".join(sp.latex(sp.simplify(c)) for c in v) + r"\right)"


def _axis_bounds(vals):
    """Return (lo, hi, step) — integer step, padded one tick either side."""
    lo = float(min(list(vals) + [0.0]))
    hi = float(max(list(vals) + [0.0]))
    span = max(hi - lo, 1.0)
    if span <= 5:
        step = 1
    elif span <= 12:
        step = 2
    elif span <= 30:
        step = 5
    else:
        step = 10
    lo_a = float(np.floor(lo / step) * step - step)
    hi_a = float(np.ceil(hi / step) * step + step)
    return lo_a, hi_a, step


def _build_formula_group(data):
    """Seven MathTex lines summarising the Gram-Schmidt step."""
    fs = 26
    u1, v2, proj, u2 = data["u1"], data["v2"], data["proj"], data["u2"]
    ip_v2_u1, ip_u1_u1 = data["ip_v2_u1"], data["ip_u1_u1"]
    coef, ip_u1_u2 = data["coef"], data["ip_u1_u2"]

    f1 = MathTex(rf"u_1 = v_1 = {_vec_tex(u1)}", font_size=fs)
    f2 = MathTex(rf"\langle v_2,u_1\rangle = {sp.latex(ip_v2_u1)}", font_size=fs)
    f3 = MathTex(rf"\langle u_1,u_1\rangle = {sp.latex(ip_u1_u1)}", font_size=fs)
    f4 = MathTex(
        r"\operatorname{proj}_{u_1}v_2 = "
        r"\frac{\langle v_2,u_1\rangle}{\langle u_1,u_1\rangle}u_1",
        font_size=fs,
    )
    if coef == 1:
        f5 = MathTex(rf"= u_1 = {_vec_tex(proj)}", font_size=fs)
    elif coef == 0:
        f5 = MathTex(r"= 0", font_size=fs)
    else:
        f5 = MathTex(rf"= {sp.latex(coef)}\,u_1 = {_vec_tex(proj)}", font_size=fs)
    f6 = MathTex(
        rf"u_2 = v_2-\operatorname{{proj}}_{{u_1}}v_2 = {_vec_tex(u2)}",
        font_size=fs,
    )
    f7 = MathTex(rf"\langle u_1,u_2\rangle = {sp.latex(ip_u1_u2)}", font_size=fs)

    group = VGroup(f1, f2, f3, f4, f5, f6, f7).arrange(
        DOWN, aligned_edge=LEFT, buff=0.26
    ).move_to(RIGHT * 3.55 + UP * 0.15)
    f5.shift(RIGHT * 0.45)
    return group


# ====================================================================
#  2-D scene
# ====================================================================

class GramSchmidt2D(Scene):

    def __init__(self, vectors=None, **kwargs):
        super().__init__(**kwargs)
        raw = list(vectors or [[1, 2], [2, 1]])
        if len(raw) != 2 or any(len(r) != 2 for r in raw):
            raise ValueError("GramSchmidt2D needs exactly two 2-D vectors")
        self.data = gs_summary(raw)
        for k in ("v1", "v2", "u1", "u2",
                  "ip_v2_u1", "ip_u1_u1", "coef", "proj", "ip_u1_u2"):
            setattr(self, k, self.data[k])

    # ------------------------------------------------------------------
    def _make_plane(self):
        pts = [self.v1, self.v2, self.u2, self.proj]
        x_lo, x_hi, x_step = _axis_bounds([float(p[0]) for p in pts])
        y_lo, y_hi, y_step = _axis_bounds([float(p[1]) for p in pts])
        return NumberPlane(
            x_range=[x_lo, x_hi, x_step],
            y_range=[y_lo, y_hi, y_step],
            x_length=6.0, y_length=4.8,
            background_line_style={"stroke_opacity": 0.30, "stroke_width": 1},
            axis_config={
                "stroke_width": 2,
                "include_numbers": True,
                "font_size": 18,
                "decimal_number_config": {"num_decimal_places": 0},
            },
        ).shift(LEFT * 3.6 + DOWN * 0.4)

    def _pt(self, plane, v):
        return plane.c2p(float(v[0]), float(v[1]))

    def _arrow(self, plane, v, color):
        return Arrow(
            plane.c2p(0, 0), self._pt(plane, v),
            buff=0, color=color, stroke_width=6, tip_length=0.2,
        )

    # ------------------------------------------------------------------
    def construct(self):
        title = MathTex(
            r"\text{Gram--Schmidt: }"
            rf"v_1={_vec_tex(self.v1)},\; v_2={_vec_tex(self.v2)}",
            font_size=40,
        ).to_edge(UP, buff=0.3)

        plane  = self._make_plane()
        origin = plane.c2p(0, 0)

        a_v1 = self._arrow(plane, self.v1, BLUE)
        a_v2 = self._arrow(plane, self.v2, ORANGE)
        l_v1 = MathTex("v_1", color=BLUE,   font_size=36).next_to(a_v1.get_end(), UL, buff=0.1)
        l_v2 = MathTex("v_2", color=ORANGE, font_size=36).next_to(a_v2.get_end(), UR, buff=0.1)

        self.play(Write(title))
        self.play(Create(plane), run_time=1.2)
        self.play(GrowArrow(a_v1), GrowArrow(a_v2), run_time=1.2)
        self.play(Write(l_v1), Write(l_v2))
        self.wait(0.3)

        f1, f2, f3, f4, f5, f6, f7 = _build_formula_group(self.data)

        # 1) u1 = v1
        self.play(Write(f1))
        self.play(Transform(
            l_v1,
            MathTex("u_1", color=BLUE, font_size=36).next_to(
                a_v1.get_end(), UL, buff=0.1
            ),
        ))
        self.wait(0.2)

        # 2) projection
        proj_dot  = Dot(self._pt(plane, self.proj), color=GREEN, radius=0.06)
        proj_line = DashedLine(origin, self._pt(plane, self.proj),
                               color=GREEN, stroke_width=3)
        err_line  = DashedLine(self._pt(plane, self.proj), self._pt(plane, self.v2),
                               color=RED, stroke_width=3)
        proj_lab  = MathTex(
            rf"{sp.latex(self.coef)}\,u_1", color=GREEN, font_size=30,
        ).next_to(self._pt(plane, self.proj), LEFT, buff=0.12)

        self.play(Write(f2), Write(f3))
        self.play(Create(proj_line), FadeIn(proj_dot))
        self.play(Write(f4), Write(f5))
        self.play(Write(proj_lab))
        self.play(Create(err_line))
        self.wait(0.2)

        # 3) u2 = v2 - proj
        a_u2 = self._arrow(plane, self.u2, RED)
        l_u2 = MathTex("u_2", color=RED, font_size=36).next_to(
            a_u2.get_end(), DR, buff=0.1
        )
        self.play(Write(f6))
        self.play(GrowArrow(a_u2), FadeIn(l_u2))
        self.wait(0.2)

        # 4) orthogonality — equation only, no marker
        self.play(Write(f7))
        self.wait(1.5)


# ====================================================================
#  3-D scene
# ====================================================================

class GramSchmidt3D(ThreeDScene):

    def __init__(self, vectors=None, **kwargs):
        super().__init__(**kwargs)
        raw = list(vectors or [[1, 2, 1], [2, 1, 3]])
        if len(raw) != 2 or any(len(r) != 3 for r in raw):
            raise ValueError("GramSchmidt3D needs exactly two 3-D vectors")
        self.data = gs_summary(raw)
        for k in ("v1", "v2", "u1", "u2",
                  "ip_v2_u1", "ip_u1_u1", "coef", "proj", "ip_u1_u2"):
            setattr(self, k, self.data[k])

    # ------------------------------------------------------------------
    def _make_axes(self):
        pts = [self.v1, self.v2, self.u2, self.proj]
        x_lo, x_hi, x_step = _axis_bounds([float(p[0]) for p in pts])
        y_lo, y_hi, y_step = _axis_bounds([float(p[1]) for p in pts])
        z_lo, z_hi, z_step = _axis_bounds([float(p[2]) for p in pts])
        self._bounds = {
            "x": (x_lo, x_hi), "y": (y_lo, y_hi), "z": (z_lo, z_hi),
        }
        return ThreeDAxes(
            x_range=[x_lo, x_hi, x_step],
            y_range=[y_lo, y_hi, y_step],
            z_range=[z_lo, z_hi, z_step],
            x_length=5.5, y_length=5.5, z_length=5.5,
            axis_config={"include_numbers": False},   # numbers off — see axis labels
        ).shift(LEFT * 2.5)

    def _pt(self, axes, v):
        return axes.c2p(*[float(c) for c in v])

    def _arrow(self, axes, v, color):
        return Arrow3D(
            axes.c2p(0, 0, 0), self._pt(axes, v),
            color=color, thickness=0.03,
            base_radius=0.09, height=0.22,
        )

    # ------------------------------------------------------------------
    #  Billboard helpers — anything we want to face the camera is passed
    #  through these.  Position is fixed in world coords; orientation is
    #  tied to the camera.
    # ------------------------------------------------------------------
    def _billboard(self, m):
        self.add_fixed_orientation_mobjects(m)
        return m

    def _label_at(self, axes, v, tex, color, font_size=30, offset=0.7):
        """MathTex placed just past the tip of `v`, along v's direction."""
        v_arr = np.array([float(c) for c in v])
        norm = np.linalg.norm(v_arr)
        if norm > 0:
            direction = v_arr / norm
        else:
            direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        pos = axes.c2p(*(v_arr + offset * direction))
        return self._billboard(
            MathTex(tex, color=color, font_size=font_size).move_to(pos)
        )

    # ------------------------------------------------------------------
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-55 * DEGREES)

        title = MathTex(
            r"\text{Gram--Schmidt: }"
            rf"v_1={_vec_tex(self.v1)},\; v_2={_vec_tex(self.v2)}",
            font_size=34,
        ).to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)

        axes = self._make_axes()

        # --- axis letter labels, billboarded --------------------------
        x_end = axes.c2p(self._bounds["x"][1], 0, 0)
        y_end = axes.c2p(0, self._bounds["y"][1], 0)
        z_end = axes.c2p(0, 0, self._bounds["z"][1])
        for end, letter in ((x_end, "x"), (y_end, "y"), (z_end, "z")):
            self._billboard(
                MathTex(letter, color=GREY, font_size=26).move_to(end)
            )

        a_v1 = self._arrow(axes, self.v1, BLUE)
        a_v2 = self._arrow(axes, self.v2, ORANGE)
        l_v1 = self._label_at(axes, self.v1, "v_1", BLUE)
        l_v2 = self._label_at(axes, self.v2, "v_2", ORANGE)

        self.play(Write(title))
        self.play(Create(axes), run_time=1.2)
        self.play(Create(a_v1), Create(a_v2), run_time=1.2)
        self.play(FadeIn(l_v1), FadeIn(l_v2))
        self.wait(0.3)

        formulas = _build_formula_group(self.data)
        for f in formulas:
            self.add_fixed_in_frame_mobjects(f)
        f1, f2, f3, f4, f5, f6, f7 = formulas

        # 1) u1 = v1
        self.play(Write(f1))
        l_u1 = self._label_at(axes, self.u1, "u_1", BLUE)
        self.play(FadeOut(l_v1), FadeIn(l_u1))
        self.wait(0.2)

        # 2) projection
        proj_dot = Sphere(radius=0.07, color=GREEN).move_to(self._pt(axes, self.proj))
        proj_line = DashedVMobject(
            Line(axes.c2p(0, 0, 0), self._pt(axes, self.proj),
                 color=GREEN, stroke_width=3),
            num_dashes=20,
        )
        err_line = DashedVMobject(
            Line(self._pt(axes, self.proj), self._pt(axes, self.v2),
                 color=RED, stroke_width=3),
            num_dashes=20,
        )
        proj_lab = self._label_at(
            axes, self.proj, rf"{sp.latex(self.coef)}\,u_1", GREEN, font_size=26,
        )

        self.play(Write(f2), Write(f3))
        self.play(Create(proj_line), FadeIn(proj_dot))
        self.play(Write(f4), Write(f5))
        self.play(FadeIn(proj_lab))
        self.play(Create(err_line))
        self.wait(0.2)

        # 3) u2
        a_u2 = self._arrow(axes, self.u2, RED)
        l_u2 = self._label_at(axes, self.u2, "u_2", RED)
        self.play(Write(f6))
        self.play(Create(a_u2), FadeIn(l_u2))
        self.wait(0.2)

        # 4) orthogonality — equation only
        self.play(Write(f7))
        self.wait(0.3)

        # slow rotation to show the 3-D picture
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(4)
        self.stop_ambient_camera_rotation()
