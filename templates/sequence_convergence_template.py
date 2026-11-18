"""
sequence_convergence_template.py — manim only.

Inputs:
    sequence : str      e.g. "3/n", "(3n+2)/(n+1)", "1/(n+2)"
    limit    : number   e.g. 0, 3, "5/2"
    bound    : str      e.g. "1/n"   (used in the inequality chain)

Optional:
    epsilon_list : [float, ...]   default [0.5, 0.2, 0.1]
"""

import numpy as np
from manim import *
import sympy as sp

from math_preproc import parse_expression, pick_index_symbol, safe_float


class SequenceConvergenceTemplate(Scene):

    def __init__(self, sequence="3/n", limit=0, bound="3/n",
                 epsilon_list=None, **kwargs):
        super().__init__(**kwargs)
        self.a = parse_expression(sequence)
        self.L = parse_expression(limit)
        self.b = parse_expression(bound)
        self.n_sym = pick_index_symbol(self.a, self.b)
        self.epsilon_list = sorted(
            [float(e) for e in (epsilon_list or [0.5, 0.2, 0.1])],
            reverse=True,
        )

    def _bound_N(self, eps, max_search=100_000):
        """Proof's witness: smallest k with bound(k) < eps."""
        if not self.b.free_symbols:
            return 1
        for k in range(1, max_search + 1):
            v = safe_float(self.b, self.n_sym, k)
            if v is not None and v < eps:
                return k
        return 1

    def _in_band(self, k, L_float, eps):
        y = safe_float(self.a, self.n_sym, k)
        return y is not None and abs(y - L_float) < eps

    def construct(self):
        try:
            L_float = float(self.L)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"limit {sp.latex(self.L)} is not numeric."
            ) from e

        upper = self._bound_N(self.epsilon_list[-1])
        n_max = max(upper + 10, 25)
        max_eps = self.epsilon_list[0]

        a_vals = [safe_float(self.a, self.n_sym, k) for k in range(1, n_max + 1)]
        a_vals = [v for v in a_vals if v is not None] or [L_float]
        y_lo = min(min(a_vals), L_float - max_eps)
        y_hi = max(max(a_vals), L_float + max_eps)
        span = max(y_hi - y_lo, 1e-6)
        pad = max(span * 0.10, 0.2)

        # ---- axes ----
        axes = Axes(
            x_range=[0, n_max, max(1, n_max // 10)],
            y_range=[y_lo - pad, y_hi + pad, max(span / 5, 0.2)],
            axis_config={"include_numbers": True, "font_size": 18},
            x_length=10, y_length=5,
        ).to_edge(DOWN, buff=0.4)

        title = MathTex(
            rf"a_n \;=\; {sp.latex(self.a)}"
            rf"\;\;\longrightarrow\;\;"
            rf"L \;=\; {sp.latex(self.L)}"
        ).scale(0.85).to_edge(UP, buff=0.3)
        self.play(Write(title), Create(axes))

        # ---- dots ----
        stride = 1 if n_max <= 100 else max(1, n_max // 100)
        dot_map = {}
        dots = VGroup()
        for k in range(1, n_max + 1, stride):
            y = safe_float(self.a, self.n_sym, k)
            if y is not None:
                d = Dot(axes.c2p(k, y), color=BLUE, radius=0.05)
                dot_map[k] = d
                dots.add(d)
        if len(dots) > 0:
            self.play(
                LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.015),
                run_time=1.5,
            )

        # ---- limit line ----
        limit_line = DashedLine(
            axes.c2p(0, L_float), axes.c2p(n_max, L_float),
            color=GREEN, stroke_width=2,
        )
        limit_label = MathTex(
            rf"L = {sp.latex(self.L)}", font_size=26
        ).next_to(limit_line, RIGHT, buff=0.1).shift(DOWN * 0.15)
        self.play(Create(limit_line), Write(limit_label))

        # ============ LAYOUT CONSTANTS ============
        STRIP_Y      = 2.55
        EPS_X        = -4.9
        INEQ_RIGHT_X = 6.7
        N_LBL_Y      = axes.get_top()[1] + 0.35

        # ============ ε LOOP ============
        prev = None
        for i, eps in enumerate(self.epsilon_list):
            N = self._bound_N(eps)

            # --- band ---
            y_top = axes.c2p(0, L_float + eps)[1]
            y_bot = axes.c2p(0, L_float - eps)[1]
            band = Rectangle(
                width=axes.x_length,
                height=abs(y_top - y_bot),
                fill_color=YELLOW, fill_opacity=0.25, stroke_width=0,
            ).move_to(axes.c2p(n_max / 2, L_float))

            # --- ε label: fixed top-left ---
            band_lbl = MathTex(
                rf"\varepsilon = {eps:g}", font_size=26
            ).move_to([EPS_X, STRIP_Y, 0])

            # --- N line ---
            N_line = DashedLine(
                axes.c2p(N, y_lo - pad), axes.c2p(N, y_hi + pad),
                color=RED, stroke_width=2,
            )

            # --- N label: fixed Y, X follows the line ---
            N_x = axes.c2p(N, 0)[0]
            N_lbl = MathTex(
                rf"N = {N}", font_size=22
            ).move_to([N_x, N_LBL_Y, 0])

            # --- long inequality chain, right-anchored ---
            if abs(L_float) < 1e-12:
                inner_latex = sp.latex(self.a)
            else:
                inner_latex = rf"{sp.latex(self.a)} - {sp.latex(self.L)}"

            ineq = MathTex(
                rf"n \geq {N}:\;\; "
                rf"\left| {inner_latex} \right| "
                rf"\;\leq\; {sp.latex(self.b)} "
                rf"\;<\; {eps:g}",
                font_size=22,
            )
            if ineq.width > 7.4:
                ineq.scale_to_fit_width(7.4)
            ineq.align_to([INEQ_RIGHT_X, 0, 0], RIGHT)
            ineq.set_y(STRIP_Y)

            # ============ 1. band ============
            if i == 0:
                self.play(FadeIn(band), Write(band_lbl), run_time=0.7)
            else:
                self.play(
                    ReplacementTransform(prev["band"], band),
                    TransformMatchingTex(prev["band_lbl"], band_lbl),
                    run_time=0.6,
                )
            self.wait(0.3)

            # ============ 2. dots recolour ============
            recolour = [
                d.animate.set_color(
                    GOLD if self._in_band(k, L_float, eps) else BLUE
                )
                for k, d in sorted(dot_map.items())
                if d.get_color() != (
                    GOLD if self._in_band(k, L_float, eps) else BLUE
                )
            ]
            if recolour:
                self.play(
                    LaggedStart(*recolour, lag_ratio=0.01),
                    run_time=0.7,
                )
            self.wait(0.3)

            # ============ 3. N line + ineq ============
            if i == 0:
                self.play(
                    Create(N_line),
                    Write(N_lbl),
                    run_time=0.6,
                )
                self.play(Write(ineq), run_time=0.4)
            else:
                self.play(
                    ReplacementTransform(prev["N_line"], N_line),
                    TransformMatchingTex(prev["N_lbl"], N_lbl),
                    TransformMatchingTex(prev["ineq"], ineq),
                    run_time=0.7,
                )
            self.wait(0.4)

            prev = dict(
                band=band, band_lbl=band_lbl,
                N_line=N_line, N_lbl=N_lbl, ineq=ineq,
            )

        self.wait(2)