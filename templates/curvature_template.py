"""
curvature_template.py — curvature and osculating circle of a parametric
curve at a given parameter value.

Two scenes:

    CurvatureCurve2D(curve=["t", "t^2"], evaluation_point=1)
    CurvatureCurve3D(curve=["cos(t)", "sin(t)", "t"], evaluation_point=sp.pi/2)

Only `curve` and `evaluation_point` are required for either scene.
Everything else — parameter_interval, curvature, osculating radius,
tangent / normal / binormal, osculating centre, torsion (3-D only) —
is derived from those two inputs.

Subclassing patterns (both supported):

    # 1. Class attributes
    class A01(CurvatureCurve2D):
        curve = ["t", "t^2"]
        evaluation_point = 1

    # 2. __init__ override
    class A02(CurvatureCurve3D):
        def __init__(self, **kwargs):
            super().__init__(curve=["cos(t)", "sin(t)", "t"],
                             evaluation_point=sp.pi/2, **kwargs)
"""

import numpy as np
import sympy as sp
from manim import *

from math_preproc import curve_geometry, curve_geometry_3d


# =====================================================================
#  Palette (shared between scenes)
# =====================================================================

COL_CURVE   = YELLOW
COL_POINT   = YELLOW
COL_TANGENT = BLUE_D
COL_NORMAL  = RED_D
COL_BINORM  = PINK
COL_CIRCLE  = PINK

CURVE_WIDTH = 4


# =====================================================================
#  Helpers
# =====================================================================

def _vec_tex(components):
    return r"\left(" + ",".join(sp.latex(c) for c in components) + r"\right)"


def _nice_step(span):
    if span <= 20:  return 1
    if span <= 50:  return 2
    return 5


# =====================================================================
#  Scene 1 — 2-D
# =====================================================================

class CurvatureCurve2D(Scene):

    curve = None
    evaluation_point = None

    def __init__(self, curve=None, evaluation_point=None, **kwargs):
        super().__init__(**kwargs)
        if curve is None:
            curve = self.curve if self.curve is not None else ["t", "t^2"]
        if evaluation_point is None:
            evaluation_point = (
                self.evaluation_point
                if self.evaluation_point is not None
                else 1
            )
        self.data = curve_geometry(curve, evaluation_point)

    # ------------------------------------------------------------------
    def _curve_func(self, plane):
        x, y, t = self.data["x"], self.data["y"], self.data["t"]
        xf = sp.lambdify(t, x, "numpy")
        yf = sp.lambdify(t, y, "numpy")

        def f(u):
            try:
                xu, yu = float(xf(u)), float(yf(u))
                if not (np.isfinite(xu) and np.isfinite(yu)):
                    return plane.c2p(0, 0)
                return plane.c2p(xu, yu)
            except (TypeError, ValueError):
                return plane.c2p(0, 0)
        return f

    def _sample_curve(self, n=41):
        x, y, t = self.data["x"], self.data["y"], self.data["t"]
        t_lo, t_hi = self.data["interval"]
        pts = []
        for i in range(n):
            ti = t_lo + (t_hi - t_lo) * i / (n - 1)
            try:
                px, py = float(x.subs(t, ti)), float(y.subs(t, ti))
                if np.isfinite(px) and np.isfinite(py):
                    pts.append((px, py))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        return pts

    def _nice_range(self, lo, hi):
        span = max(hi - lo, 1.0)
        pad  = span * 0.08
        step = _nice_step(span + 2 * pad)
        return (float(np.floor((lo - pad) / step) * step),
                float(np.ceil ((hi + pad) / step) * step),
                step)

    def _compute_layout(self):
        data      = self.data
        curve_pts = self._sample_curve(41)
        xs_c = [p[0] for p in curve_pts]
        ys_c = [p[1] for p in curve_pts]
        curve_span = max(max(xs_c) - min(xs_c),
                         max(ys_c) - min(ys_c), 1.0)

        r_f = float(data["radius"]) if data["radius"] != sp.oo else float("inf")
        show_circle = (
            np.isfinite(r_f) and r_f > 0
            and data["center"] is not None
            and r_f <= curve_span
        )

        point_xy = (float(data["point"][0]), float(data["point"][1]))
        v_scale  = curve_span * 0.12
        T, N = data["tangent"], data["normal"]
        t_tip = (point_xy[0] + v_scale * float(T[0]),
                 point_xy[1] + v_scale * float(T[1]))
        n_tip = (point_xy[0] + v_scale * float(N[0]),
                 point_xy[1] + v_scale * float(N[1]))

        pts = list(curve_pts) + [point_xy, t_tip, n_tip]
        if show_circle:
            c = (float(data["center"][0]), float(data["center"][1]))
            pts.extend([(c[0] - r_f, c[1] - r_f), (c[0] + r_f, c[1] + r_f)])

        xr = self._nice_range(min(p[0] for p in pts), max(p[0] for p in pts))
        yr = self._nice_range(min(p[1] for p in pts), max(p[1] for p in pts))
        return xr, yr, show_circle, v_scale

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
            background_line_style={"stroke_opacity": 0.28, "stroke_width": 1},
            axis_config={
                "stroke_width": 2, "stroke_opacity": 0.45,
                "include_numbers": True, "font_size": 16,
                "decimal_number_config": {"num_decimal_places": 0},
            },
        ).shift(LEFT * 3.4 + DOWN * 0.4)

    def construct(self):
        data                       = self.data
        xr, yr, show_circ, v_scale = self._compute_layout()
        t0                         = data["t0"]
        plane                      = self._make_plane(xr, yr)

        title = MathTex(
            r"\text{Curvature of } \gamma(t) = "
            + _vec_tex([data["x"], data["y"]]),
            font_size=34,
        ).to_edge(UP, buff=0.3)

        f1 = MathTex(r"\gamma(t) = "
                     + _vec_tex([data["x"], data["y"]]), font_size=22)
        f2 = MathTex(r"\gamma'(t) = "
                     + _vec_tex([data["dx"], data["dy"]]),
                     font_size=22, color=COL_TANGENT)
        f3 = MathTex(r"\gamma''(t) = "
                     + _vec_tex([data["ddx"], data["ddy"]]),
                     font_size=22, color=COL_NORMAL)
        f4 = MathTex(r"\kappa(t) = " + sp.latex(data["kappa_expr"]),
                     font_size=22)
        sep = Line(LEFT * 1.4, RIGHT * 1.4, color=GREY, stroke_width=1)
        f5 = MathTex(rf"\kappa({sp.latex(t0)}) = {sp.latex(data['kappa'])}",
                     font_size=22, color=YELLOW)
        if data["radius"] == sp.oo:
            f6 = MathTex(rf"r({sp.latex(t0)}) = \infty",
                         font_size=22, color=COL_CIRCLE)
        else:
            f6 = MathTex(rf"r({sp.latex(t0)}) = {sp.latex(data['radius'])}",
                         font_size=22, color=COL_CIRCLE)
        panel = VGroup(f1, f2, f3, f4, sep, f5, f6).arrange(
            DOWN, aligned_edge=LEFT, buff=0.22
        ).move_to(RIGHT * 3.5 + UP * 0.3)

        curve = ParametricFunction(
            self._curve_func(plane),
            t_range=data["interval"],
            color=COL_CURVE, stroke_width=CURVE_WIDTH,
        )

        point_xy = (float(data["point"][0]), float(data["point"][1]))
        T, N = data["tangent"], data["normal"]
        t_tip = (point_xy[0] + v_scale * float(T[0]),
                 point_xy[1] + v_scale * float(T[1]))
        n_tip = (point_xy[0] + v_scale * float(N[0]),
                 point_xy[1] + v_scale * float(N[1]))

        pt_dot = Dot(plane.c2p(*point_xy), color=COL_POINT, radius=0.09)
        pt_label = MathTex(
            rf"\gamma({sp.latex(t0)}) = "
            + _vec_tex([data["point"][0], data["point"][1]]),
            font_size=20, color=COL_POINT,
        ).next_to(pt_dot, UR, buff=0.1)

        tan_arrow  = Arrow(plane.c2p(*point_xy), plane.c2p(*t_tip),
                           buff=0, color=COL_TANGENT,
                           stroke_width=5, tip_length=0.18)
        norm_arrow = Arrow(plane.c2p(*point_xy), plane.c2p(*n_tip),
                           buff=0, color=COL_NORMAL,
                           stroke_width=5, tip_length=0.18)
        tan_label  = MathTex(r"\hat{T}", color=COL_TANGENT, font_size=24
                             ).next_to(tan_arrow.get_end(), UR, buff=0.1)
        norm_label = MathTex(r"\hat{N}", color=COL_NORMAL, font_size=24
                             ).next_to(norm_arrow.get_end(), UL, buff=0.1)

        osc_circle = None
        osc_center = None
        if show_circ:
            c_xy = (float(data["center"][0]), float(data["center"][1]))
            r_f  = float(data["radius"])
            unit = np.linalg.norm(plane.c2p(1, 0) - plane.c2p(0, 0))
            osc_circle = Circle(
                radius=r_f * unit, color=COL_CIRCLE,
                stroke_width=3, stroke_opacity=0.85,
            ).move_to(plane.c2p(*c_xy))
            osc_center = Dot(plane.c2p(*c_xy),
                             color=COL_CIRCLE, radius=0.05)

        # --- animate ---------------------------------------------------
        self.play(Write(title))
        self.play(Write(f1), Write(f2), Write(f3), run_time=0.8)
        self.play(Create(plane), run_time=1.0)
        self.play(Create(curve), run_time=1.5)
        self.play(Write(f4), Write(sep))
        self.wait(0.3)

        self.play(FadeIn(pt_dot), Write(pt_label))
        self.play(GrowArrow(tan_arrow), GrowArrow(norm_arrow),
                  FadeIn(tan_label), FadeIn(norm_label))
        self.wait(0.4)

        if osc_circle is not None:
            self.play(Create(osc_circle), FadeIn(osc_center), run_time=1.5)
            self.wait(0.3)

        self.play(Write(f5))
        self.play(Write(f6))
        self.wait(2)


# =====================================================================
#  Scene 2 — 3-D
# =====================================================================

class CurvatureCurve3D(ThreeDScene):

    curve = None
    evaluation_point = None

    def __init__(self, curve=None, evaluation_point=None, **kwargs):
        super().__init__(**kwargs)
        if curve is None:
            curve = (self.curve if self.curve is not None
                     else ["cos(t)", "sin(t)", "t"])
        if evaluation_point is None:
            evaluation_point = (self.evaluation_point
                                if self.evaluation_point is not None
                                else sp.pi / 2)
        self.data = curve_geometry_3d(curve, evaluation_point)

    # ------------------------------------------------------------------
    def _curve_func(self, axes):
        x, y, z, t = (self.data["x"], self.data["y"],
                      self.data["z"], self.data["t"])
        xf = sp.lambdify(t, x, "numpy")
        yf = sp.lambdify(t, y, "numpy")
        zf = sp.lambdify(t, z, "numpy")

        def f(u):
            try:
                xu, yu, zu = float(xf(u)), float(yf(u)), float(zf(u))
                if not (np.isfinite(xu) and np.isfinite(yu) and np.isfinite(zu)):
                    return axes.c2p(0, 0, 0)
                return axes.c2p(xu, yu, zu)
            except (TypeError, ValueError):
                return axes.c2p(0, 0, 0)
        return f

    def _sample_curve(self, n=61):
        x, y, z, t = (self.data["x"], self.data["y"],
                      self.data["z"], self.data["t"])
        t_lo, t_hi = self.data["interval"]
        pts = []
        for i in range(n):
            ti = t_lo + (t_hi - t_lo) * i / (n - 1)
            try:
                px = float(x.subs(t, ti))
                py = float(y.subs(t, ti))
                pz = float(z.subs(t, ti))
                if np.isfinite(px) and np.isfinite(py) and np.isfinite(pz):
                    pts.append((px, py, pz))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        return pts

    def _range_3d(self, vals):
        span = max(vals) - min(vals)
        pad  = max(span * 0.12, 0.8)
        step = _nice_step(max(span + 2 * pad, 1.0))
        lo   = float(np.floor((min(vals) - pad) / step) * step)
        hi   = float(np.ceil ((max(vals) + pad) / step) * step)
        return (lo, hi, step)

    def _make_axes(self, curve_pts, extra_pts=()):
        all_pts = list(curve_pts) + list(extra_pts)
        xr = self._range_3d([p[0] for p in all_pts])
        yr = self._range_3d([p[1] for p in all_pts])
        zr = self._range_3d([p[2] for p in all_pts])
        # equal-length axes for correct 3-D proportions
        return ThreeDAxes(
            x_range=list(xr), y_range=list(yr), z_range=list(zr),
            x_length=5.5, y_length=5.5, z_length=5.5,
            axis_config={
                "stroke_width": 2, "stroke_opacity": 0.45,
                "include_numbers": False,
            },
        ), {"x": xr, "y": yr, "z": zr}

    def _np3(self, vec):
        return np.array([float(c) for c in vec])

    # ------------------------------------------------------------------
    def _circle_func(self, axes, center, radius, T, N):
        """
        Osculating circle:
            c(θ) = center + r · (−cos θ · N̂ + sin θ · T̂)
        At θ=0: c = center − r·N̂ = point  ✓
        Tangent at θ=0 = r·T̂  ✓
        """
        c_np = self._np3(center)
        T_np = self._np3(T)
        N_np = self._np3(N)
        r_f  = float(radius)

        def f(theta):
            p = c_np + r_f * (-np.cos(theta) * N_np + np.sin(theta) * T_np)
            return axes.c2p(*[float(x) for x in p])
        return f

    # ------------------------------------------------------------------
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-55 * DEGREES)

        data       = self.data
        t0         = data["t0"]
        curve_pts  = self._sample_curve(61)

        xs_c = [p[0] for p in curve_pts]
        ys_c = [p[1] for p in curve_pts]
        zs_c = [p[2] for p in curve_pts]
        curve_span = max(max(xs_c) - min(xs_c),
                         max(ys_c) - min(ys_c),
                         max(zs_c) - min(zs_c), 1.0)

        # --- decide what to draw --------------------------------------
        has_normal = data["normal"] is not None
        r          = data["radius"]
        r_f        = float(r) if r != sp.oo else float("inf")
        show_circle = (
            has_normal
            and np.isfinite(r_f) and r_f > 0
            and r_f <= curve_span * 1.4
        )

        point_np = self._np3(data["point"])
        T_np     = self._np3(data["tangent"])
        N_np     = self._np3(data["normal"]) if has_normal else None
        B_np     = self._np3(data["binormal"]) if has_normal else None

        v_scale = curve_span * 0.12
        t_tip = point_np + v_scale * T_np
        n_tip = point_np + v_scale * N_np if N_np is not None else None
        b_tip = point_np + v_scale * B_np if B_np is not None else None

        # --- axes -----------------------------------------------------
        extra_pts = [tuple(point_np), tuple(t_tip)]
        if n_tip is not None: extra_pts.append(tuple(n_tip))
        if b_tip is not None: extra_pts.append(tuple(b_tip))
        if show_circle:
            c_np = self._np3(data["center"])
            extra_pts.extend([
                tuple(c_np + r_f * np.array([1, 0, 0])),
                tuple(c_np - r_f * np.array([1, 0, 0])),
                tuple(c_np + r_f * np.array([0, 1, 0])),
                tuple(c_np - r_f * np.array([0, 1, 0])),
                tuple(c_np + r_f * np.array([0, 0, 1])),
                tuple(c_np - r_f * np.array([0, 0, 1])),
            ])

        axes, bounds = self._make_axes(curve_pts, extra_pts)

        # --- title (fixed in frame) -----------------------------------
        title = MathTex(
            r"\text{Curvature of } \gamma(t) = "
            + _vec_tex([data["x"], data["y"], data["z"]]),
            font_size=32,
        ).to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)

        # --- axis labels (billboarded) --------------------------------
        for key, letter in (("x", "x"), ("y", "y"), ("z", "z")):
            hi = bounds[key][1]
            if key == "x":
                end = axes.c2p(hi, 0, 0)
            elif key == "y":
                end = axes.c2p(0, hi, 0)
            else:
                end = axes.c2p(0, 0, hi)
            lbl = MathTex(letter, color=GREY, font_size=24).move_to(end)
            self.add_fixed_orientation_mobjects(lbl)

        # --- curve ----------------------------------------------------
        curve = ParametricFunction(
            self._curve_func(axes),
            t_range=data["interval"],
            color=COL_CURVE, stroke_width=CURVE_WIDTH,
        )

        # --- point ----------------------------------------------------
        pt_dot = Sphere(radius=0.07, color=COL_POINT
                        ).move_to(axes.c2p(*point_np))
        pt_label = MathTex(
            rf"\gamma({sp.latex(t0)})",
            font_size=22, color=COL_POINT,
        ).move_to(axes.c2p(*(point_np + 0.35 * T_np)))
        self.add_fixed_orientation_mobjects(pt_label)

        # --- Frenet frame arrows -------------------------------------
        def make_arrow(end_pt, color):
            return Arrow3D(
                axes.c2p(*point_np), axes.c2p(*end_pt),
                color=color, thickness=0.025,
                base_radius=0.06, height=0.20,
            )

        tan_arrow = make_arrow(t_tip, COL_TANGENT)
        tan_lbl = MathTex(r"\hat{T}", color=COL_TANGENT, font_size=24
                          ).move_to(axes.c2p(*(t_tip + 0.25 * T_np)))
        self.add_fixed_orientation_mobjects(tan_lbl)

        norm_arrow = norm_lbl = None
        if n_tip is not None:
            norm_arrow = make_arrow(n_tip, COL_NORMAL)
            norm_lbl = MathTex(r"\hat{N}", color=COL_NORMAL, font_size=24
                               ).move_to(axes.c2p(*(n_tip + 0.25 * N_np)))
            self.add_fixed_orientation_mobjects(norm_lbl)

        binorm_arrow = binorm_lbl = None
        if b_tip is not None:
            binorm_arrow = make_arrow(b_tip, COL_BINORM)
            binorm_lbl = MathTex(r"\hat{B}", color=COL_BINORM, font_size=24
                                 ).move_to(axes.c2p(*(b_tip + 0.25 * B_np)))
            self.add_fixed_orientation_mobjects(binorm_lbl)

        # --- osculating circle ---------------------------------------
        osc_circle = None
        if show_circle:
            osc_circle = ParametricFunction(
                self._circle_func(axes, data["center"], data["radius"],
                                  data["tangent"], data["normal"]),
                t_range=[0, TAU],
                color=COL_CIRCLE, stroke_width=3,
            )

        # --- formula panel (fixed in frame) --------------------------
        f1 = MathTex(r"\kappa(t) = " + sp.latex(data["kappa_expr"]),
                     font_size=22)
        f2 = MathTex(r"\tau(t) = " + sp.latex(data["torsion_expr"]),
                     font_size=22, color=COL_BINORM)
        sep = Line(LEFT * 1.4, RIGHT * 1.4, color=GREY, stroke_width=1)
        f3 = MathTex(rf"\kappa({sp.latex(t0)}) = {sp.latex(data['kappa'])}",
                     font_size=22, color=YELLOW)
        if data["radius"] == sp.oo:
            f4 = MathTex(rf"r({sp.latex(t0)}) = \infty",
                         font_size=22, color=COL_CIRCLE)
        else:
            f4 = MathTex(rf"r({sp.latex(t0)}) = {sp.latex(data['radius'])}",
                         font_size=22, color=COL_CIRCLE)
        f5 = MathTex(rf"\tau({sp.latex(t0)}) = {sp.latex(data['torsion'])}",
                     font_size=22, color=COL_BINORM)

        panel = VGroup(f1, f2, sep, f3, f4, f5).arrange(
            DOWN, aligned_edge=LEFT, buff=0.22
        ).move_to(RIGHT * 4.2 + UP * 0.4)

        # --- legend (fixed in frame) ---------------------------------
        def legend_row(color, tex):
            return VGroup(
                Line(ORIGIN, RIGHT * 0.3, color=color, stroke_width=3),
                MathTex(tex, font_size=20),
            ).arrange(RIGHT, buff=0.12)

        legend = VGroup(
            legend_row(COL_TANGENT, r"\hat{T}"),
            legend_row(COL_NORMAL,  r"\hat{N}"),
            legend_row(COL_BINORM,  r"\hat{B}"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        legend.to_corner(DR, buff=0.35)

        # =============================================================
        #  Animate
        # =============================================================
        self.play(Write(title))
        self.play(Create(axes), run_time=1.0)
        self.play(Create(curve), run_time=2.0)
        self.wait(0.4)

        self.play(FadeIn(pt_dot))
        self.add_fixed_orientation_mobjects(pt_label)
        self.play(FadeIn(pt_label))
        self.play(Create(tan_arrow))
        if norm_arrow is not None:
            self.play(Create(norm_arrow))
        if binorm_arrow is not None:
            self.play(Create(binorm_arrow))
        self.wait(0.3)

        self.add_fixed_in_frame_mobjects(legend)
        self.play(FadeIn(legend))

        if osc_circle is not None:
            self.play(Create(osc_circle), run_time=1.5)
            self.wait(0.3)

        for line in panel:
            self.add_fixed_in_frame_mobjects(line)
            self.play(FadeIn(line), run_time=0.4)

        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        self.wait(1)