"""
integration_approximation_template.py — Riemann-sum convergence, split-then-expand animation.

Inputs:
    function       : str    e.g. "x^2"
    interval       : str    e.g. "[0, 1]"
    sample_point   : str    "right_endpoint" | "left_endpoint" | "midpoint"
    integral_value : str/number

Optional:
    n_values       : list  default [2, 4, 8, 16, 32]
"""

import numpy as np
from manim import *
import sympy as sp

from math_preproc import (
    parse_expression, parse_interval, symbolic_riemann,
)


class IntegrationApproximationTemplate(Scene):

    def __init__(self, function="x^2", interval="[0, 1]", sample_point="right_endpoint",
                 integral_value="1/3", n_values=None, **kwargs):
        super().__init__(**kwargs)
        self.f = parse_expression(function)
        self.a, self.b = parse_interval(interval)
        self.sample_point = sample_point
        self.L = parse_expression(integral_value)
        self.n_values = list(n_values or [2, 4, 8, 16, 32])

        free = self.f.free_symbols
        self.x_sym = next(iter(free)) if free else sp.Symbol("x")

        self.S_n, self._n_sym = symbolic_riemann(
            self.f, self.x_sym, self.a, self.b, self.sample_point
        )

    # ------------------------------------------------------------------
    def _S_value(self, n):
        if self.S_n is None:
            return float("nan")
        try:
            return float(self.S_n.subs(self._n_sym, n))
        except Exception:
            return float("nan")

    def _partition_lines(self, axes, n):
        """n+1 dashed vertical lines at the subinterval boundaries."""
        a, b = float(self.a), float(self.b)
        dx = (b - a) / n
        y_lo = axes.y_range[0]
        y_hi = axes.y_range[1]
        lines = VGroup()
        for k_idx in range(n + 1):
            x = a + k_idx * dx
            lines.add(DashedLine(
                axes.c2p(x, y_lo), axes.c2p(x, y_hi),
                stroke_width=1, stroke_opacity=0.5, color=WHITE,
            ))
        return lines

    def _build_rect_components(self, axes, n):
        """
        Returns (full_rects, thin_rects, sample_dots, x_stars).

        thin_rects are zero-width columns at the sample x, height = f(x*).
        full_rects are the rectangles that fill each subinterval.
        """
        a, b = float(self.a), float(self.b)
        dx = (b - a) / n

        full_rects = VGroup()
        thin_rects = VGroup()
        sample_dots = VGroup()
        x_stars = []

        for k_idx in range(1, n + 1):
            if self.sample_point == "right_endpoint":
                x_star = a + k_idx * dx
                x_left = a + (k_idx - 1) * dx
                x_right = x_star
            elif self.sample_point == "left_endpoint":
                x_star = a + (k_idx - 1) * dx
                x_left = x_star
                x_right = a + k_idx * dx
            else:                                  # midpoint
                x_star = a + (k_idx - 0.5) * dx
                x_left = a + (k_idx - 1) * dx
                x_right = a + k_idx * dx

            y_star = float(self.f.subs(self.x_sym, x_star))
            x_stars.append(x_star)

            p_bot = axes.c2p(x_star, 0)
            p_top = axes.c2p(x_star, y_star)
            h = max(abs(p_top[1] - p_bot[1]), 0.01)

            thin = Rectangle(
                width=0.02, height=h,
                fill_color=BLUE, fill_opacity=0.35,
                stroke_color=BLUE, stroke_width=1,
            ).move_to([p_bot[0], (p_bot[1] + p_top[1]) / 2, 0])
            thin_rects.add(thin)

            p_l = axes.c2p(x_left, min(0, y_star))
            p_r = axes.c2p(x_right, max(0, y_star))
            full = Rectangle(
                width=max(abs(p_r[0] - p_l[0]), 0.01),
                height=max(abs(p_r[1] - p_l[1]), 0.01),
                fill_color=BLUE, fill_opacity=0.35,
                stroke_color=BLUE, stroke_width=1,
            ).move_to([(p_l[0] + p_r[0]) / 2, (p_l[1] + p_r[1]) / 2, 0])
            full_rects.add(full)

            sample_dots.add(
                Dot(axes.c2p(x_star, y_star), color=ORANGE, radius=0.07)
            )

        return full_rects, thin_rects, sample_dots, x_stars

    # ------------------------------------------------------------------
    def construct(self):
        try:
            L_float = float(self.L)
        except (TypeError, ValueError):
            raise ValueError(f"integral_value {self.L} is not numeric.")

        # ---- axes and curve ----
        samples = np.linspace(float(self.a), float(self.b), 200)
        vals = [float(self.f.subs(self.x_sym, t)) for t in samples]
        y_min, y_max = min(vals + [0]), max(vals + [0])
        span = max(y_max - y_min, 1e-6)
        pad = span * 0.15

        axes = Axes(
            x_range=[float(self.a), float(self.b),
                     (float(self.b) - float(self.a)) / 5],
            y_range=[y_min - pad * 0.3, y_max + pad, max(span / 5, 0.2)],
            axis_config={"include_numbers": True, "font_size": 18},
            x_length=8.5, y_length=5.5,
        ).to_edge(LEFT, buff=0.6).shift(DOWN * 0.3)

        curve = axes.plot(
            sp.Lambda(self.x_sym, self.f),
            x_range=[float(self.a), float(self.b)],
            color=YELLOW, stroke_width=3,
        )

        title = MathTex(
            rf"S_n \;\longrightarrow\; "
            rf"\int_{{{sp.latex(self.a)}}}^{{{sp.latex(self.b)}}} "
            rf"{sp.latex(self.f)}\,d{sp.latex(self.x_sym)}"
            rf" \;=\; {sp.latex(self.L)}"
        ).scale(0.8).to_edge(UP, buff=0.3)

        sample_note = Text(
            f"sampling: {self.sample_point.replace('_', ' ')}",
            font_size=20, color=ORANGE,
        ).to_corner(UL, buff=0.4).shift(DOWN * 0.6)

        self.play(Write(title), FadeIn(sample_note))
        self.play(Create(axes), Create(curve))

        # ---- right-side info panel ----
        PANEL_X, PANEL_TOP = 4.7, 2.0

        if self.S_n is not None:
            formula = MathTex(
                rf"S_n = {sp.latex(self.S_n)}", font_size=22
            ).move_to([PANEL_X, PANEL_TOP + 0.7, 0])
            self.play(Write(formula))

        info_n = MathTex(rf"n = {self.n_values[0]}", font_size=26)
        info_S = MathTex(
            rf"S_n = {self._S_value(self.n_values[0]):.4f}", font_size=26
        )
        info_L = MathTex(rf"L = {L_float:.4f}", font_size=26)
        info_err = MathTex(
            rf"|S_n - L| = "
            rf"{abs(self._S_value(self.n_values[0]) - L_float):.4f}",
            font_size=26,
        )
        panel = VGroup(info_n, info_S, info_L, info_err).arrange(
            DOWN, buff=0.35, aligned_edge=LEFT
        ).move_to([PANEL_X, PANEL_TOP - 0.8, 0])
        self.play(Write(panel))

        # ---- loop over n ----
        prev = None
        for i, n in enumerate(self.n_values):
            full_rects, thin_rects, sample_dots, x_stars = \
                self._build_rect_components(axes, n)
            part_lines = self._partition_lines(axes, n)

            # slower for the first n, faster for subsequent
            if i == 0:
                t_part, t_dot, t_thin, t_expand = 0.7, 0.5, 0.5, 1.2
            else:
                t_part, t_dot, t_thin, t_expand = 0.3, 0.2, 0.2, 0.6

            if prev is not None:
                self.play(
                    FadeOut(prev["part_lines"]),
                    FadeOut(prev["rects"]),
                    FadeOut(prev["dots"]),
                    run_time=0.3,
                )

            # 1. split: partition lines descend
            self.play(
                LaggedStart(
                    *[Create(l) for l in part_lines], lag_ratio=0.03,
                ),
                run_time=t_part,
            )

            # 2. sample dots appear on the curve
            self.play(
                LaggedStart(
                    *[GrowFromCenter(d) for d in sample_dots],
                    lag_ratio=0.05,
                ),
                run_time=t_dot,
            )

            # 3. thin columns rise from axis at each sample x
            self.play(
                LaggedStart(
                    *[GrowFromPoint(t, axes.c2p(xs, 0))
                      for t, xs in zip(thin_rects, x_stars)],
                    lag_ratio=0.03,
                ),
                run_time=t_thin,
            )

            # 4. thin columns expand into full rectangles
            self.play(
                LaggedStart(
                    *[Transform(t, r)
                      for t, r in zip(thin_rects, full_rects)],
                    lag_ratio=0.03,
                ),
                run_time=t_expand,
            )

            # update panel
            S_n = self._S_value(n)
            err = abs(S_n - L_float) if np.isfinite(S_n) else float("nan")
            new_n   = MathTex(rf"n = {n}", font_size=26).move_to(info_n)
            new_S   = MathTex(rf"S_n = {S_n:.4f}", font_size=26).move_to(info_S)
            new_err = MathTex(
                rf"|S_n - L| = {err:.4f}", font_size=26
            ).move_to(info_err)
            self.play(
                TransformMatchingTex(info_n, new_n),
                TransformMatchingTex(info_S, new_S),
                TransformMatchingTex(info_err, new_err),
                run_time=0.4,
            )
            info_n, info_S, info_err = new_n, new_S, new_err

            prev = dict(
                part_lines=part_lines,
                rects=thin_rects,   # note: after Transform, these ARE the rectangles
                dots=sample_dots,
            )
            self.wait(0.4)

        # closing
        # closing = MathTex(
        #     rf"S_n \;\to\; {sp.latex(self.L)} \quad (n \to \infty)",
        #     font_size=30,
        # ).next_to(axes, DOWN, buff=0.25)

        divider = Line(
            panel.get_left(), panel.get_right(),
            color=GREY, stroke_width=1, stroke_opacity=0.6,
        ).next_to(panel, DOWN, buff=0.3)

        closing = MathTex(
            rf"S_n \;\to\; {sp.latex(self.L)}",
            font_size=26,
        ).next_to(divider, DOWN, buff=0.25).align_to(panel, LEFT)
        self.play(Write(closing))
        self.wait(2)