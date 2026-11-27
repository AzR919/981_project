"""
Symbolic checks (SymPy) of the algebra behind extra/analysis/theory/odds_law.md, extra/analysis/theory/late_time.md and
extra/analysis/theory/minimal_example.md.

Each test states the identity it checks. Asymptotic claims are checked on the exact solution of the reduced
model where one exists; claims about the full simulator are checked numerically elsewhere.
"""

import os as _os, sys as _sys  # extra/ layout: repo root, extra/, and both experiments/ folders on the path
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path[:0] = [_REPO, _os.path.join(_REPO, "experiments"), _os.path.join(_REPO, "extra"),
                 _os.path.join(_REPO, "extra", "experiments")]
import unittest

import sympy as sp


def softmax_vector(K, name):
    """Symbolic probabilities pi_1..pi_K with the last one fixed by the sum."""
    pis = list(sp.symbols(f"{name}1:{K}", positive=True))
    return pis + [1 - sum(pis)]


class FirstStepTests(unittest.TestCase):
    def test_cross_class_factor(self):
        # Linear softmax: grad_W p_x = p_x (e_{c_x} - pi_x) phi_x^T, so <grad p_y, grad p_x> = k(x, y) p_x p_y F with
        # F = (e_a - pi_x).(e_b - pi_y) = delta_ab - pi_y(a) - pi_x(b) + pi_x . pi_y.
        K = 4
        px, py = softmax_vector(K, "x"), softmax_vector(K, "y")
        for a in range(K):
            for b in range(K):
                ea = [1 if j == a else 0 for j in range(K)]
                eb = [1 if j == b else 0 for j in range(K)]
                lhs = sum((ea[j] - px[j]) * (eb[j] - py[j]) for j in range(K))
                rhs = (1 if a == b else 0) - py[a] - px[b] + sum(px[j] * py[j] for j in range(K))
                self.assertEqual(sp.expand(lhs - rhs), 0)
        # At a uniform start the factor is delta_ab - 1/K.
        u = [sp.Rational(1, K)] * K
        F = lambda a, b: (1 if a == b else 0) - u[a] - u[b] + sum(ui * ui for ui in u)  # noqa: E731
        self.assertEqual(F(0, 0), 1 - sp.Rational(1, K))
        self.assertEqual(F(0, 1), -sp.Rational(1, K))

    def test_push_ratio_block_form_and_odds_form(self):
        K, n, m = sp.symbols("K n m", positive=True)
        s_ba, s_same, s_diff, s = sp.symbols("S_BA S_same S_diff s", positive=True)
        d_b_a = -s * s_ba / K
        d_b_b = s * ((1 - 1 / K) * s_same - s_diff / K)
        R = sp.simplify(-d_b_a / d_b_b)
        self.assertEqual(sp.simplify(R - s_ba / ((K - 1) * s_same - s_diff)), 0)
        # Odds form with the kernel means (minority classes of equal size).
        k_ba, k_same, k_diff = sp.symbols("k_BA k_same k_diff", positive=True)
        n_a, n_b = m * n, (1 - m) * n
        subs = {s_ba: k_ba * n_b * n_a, s_same: k_same * n_b**2 / (K - 1), s_diff: k_diff * n_b**2 * (K - 2) / (K - 1)}
        c = (K - 1) * k_ba / ((K - 1) * k_same - (K - 2) * k_diff)
        self.assertEqual(sp.simplify(R.subs(subs) - c * m / (1 - m)), 0)
        # Threshold: R = 1 at m = 1/(1 + c); with class-blind features (all kernel means equal) c = K - 1 and m* = 1/K.
        self.assertEqual(sp.simplify(sp.solve(sp.Eq(c * m / (1 - m), 1), m)[0] - 1 / (1 + c)), 0)
        k = sp.symbols("k", positive=True)
        self.assertEqual(sp.simplify(c.subs({k_ba: k, k_same: k, k_diff: k})), K - 1)

    def test_junk_answers_two_classes(self):
        # Two classes with answers among K tokens, uniform start: cross factor -1/K, same-class factor 1 - 1/K,
        # so R = S_BA / ((K - 1) S_same) (T22, addendum 4).
        K, s_ba, s_same, s = sp.symbols("K S_BA S_same s", positive=True)
        R = sp.simplify((s * s_ba / K) / (s * (1 - 1 / K) * s_same))
        self.assertEqual(sp.simplify(R - s_ba / ((K - 1) * s_same)), 0)

    def test_tabular_monotonicity(self):
        # One prompt, softmax logits z, correct answer c, exact group-centred update direction on the logits
        # g_j = pi_j (delta_jc - p). First-order change of p: sum_j dp/dz_j g_j = p [p (1 - p)^2 + p sum_{j != c} pi_j^2] >= 0.
        K, c = 4, 0
        pi = softmax_vector(K, "q")
        p = pi[c]
        dp = [p * ((1 if j == c else 0) - pi[j]) for j in range(K)]
        g = [pi[j] * ((1 if j == c else 0) - p) for j in range(K)]
        lhs = sum(dp[j] * g[j] for j in range(K))
        rhs = p * (p * (1 - p) ** 2 + p * sum(pi[j] ** 2 for j in range(K) if j != c))
        self.assertEqual(sp.expand(lhs - rhs), 0)


class LateTimeTests(unittest.TestCase):
    def test_scalar_late_model_has_an_exact_solution(self):
        # dL/dtau = gamma/tau - c e^{-L}. With y = e^L: dy/dtau = (gamma/tau) y - c (linear), solved by
        # y(tau) = tau^gamma [y0 tau0^{-gamma} - c (tau^{1-gamma} - tau0^{1-gamma}) / (1 - gamma)].
        tau, tau0, y0, c, gamma = sp.symbols("tau tau0 y0 c gamma", positive=True)
        y = tau**gamma * (y0 * tau0**-gamma - c * (tau ** (1 - gamma) - tau0 ** (1 - gamma)) / (1 - gamma))
        self.assertEqual(sp.simplify(sp.diff(y, tau) - (gamma / tau * y - c)), 0)
        self.assertEqual(sp.simplify(y.subs(tau, tau0) - y0), 0)
        # L = log y satisfies the original equation.
        L = sp.log(y)
        self.assertEqual(sp.simplify(sp.diff(L, tau) - (gamma / tau - c * sp.exp(-L))), 0)

    def test_permanence_criterion(self):
        # For gamma > 1, y / tau^gamma tends to y0 tau0^{-gamma} - c tau0^{1-gamma} / (gamma - 1): the minority stays
        # absorbed (p ~ tau^{-gamma}) iff this limit is positive, i.e. iff it fell deep enough: y0 > c tau0 / (gamma - 1).
        tau, tau0, y0, c = sp.symbols("tau tau0 y0 c", positive=True)
        gamma = sp.Rational(3, 2)
        y = tau**gamma * (y0 * tau0**-gamma - c * (tau ** (1 - gamma) - tau0 ** (1 - gamma)) / (1 - gamma))
        limit = sp.limit(y / tau**gamma, tau, sp.oo)
        self.assertEqual(sp.simplify(limit - (y0 * tau0**-gamma - c * tau0 ** (1 - gamma) / (gamma - 1))), 0)
        # For gamma < 1 the bracket always reaches zero: recovery at a finite time tau*.
        g = sp.Rational(1, 2)
        ys = tau**g * (y0 * tau0**-g - c * (tau ** (1 - g) - tau0 ** (1 - g)) / (1 - g))
        self.assertEqual(sp.limit(ys, tau, sp.oo), -sp.oo)

    def test_pass_at_k_divides_the_late_exponent(self):
        # Majority margin under a rule whose weight vanishes like (1 - p)^{k-1}: dM/dtau = a e^{-k M}, so
        # M = log(k a tau + C) / k; the push on the minority is gamma times the majority's rate: dL/dtau = gamma dM/dtau.
        tau, a, C, gamma = sp.symbols("tau a C gamma", positive=True)
        for k in (1, 2, 4):
            M = sp.log(k * a * tau + C) / k
            self.assertEqual(sp.simplify(sp.diff(M, tau) - a * sp.exp(-k * M)), 0)
            push = gamma * sp.diff(M, tau)
            self.assertEqual(sp.limit(push * tau, tau, sp.oo), gamma / k)   # late exponent gamma / k

    def test_kl_restoring_terms_as_the_minority_vanishes(self):
        # Three answers; the minority prompt puts probability p on its correct answer b, q on c, the rest on the
        # majority's answer a. Logit gradient of each penalty on the correct answer's logit:
        # forward KL(ref || pi): ref_b - p, which tends to ref_b > 0 (non-vanishing pull);
        # reverse KL(pi || ref): -p (log(p / ref_b) - KL(pi || ref)), which tends to 0 like p log(1/p).
        p, q, rb, rc = sp.symbols("p q r_b r_c", positive=True)
        ra = 1 - rb - rc
        pi = {"a": 1 - p - q, "b": p, "c": q}
        ref = {"a": ra, "b": rb, "c": rc}
        forward = ref["b"] - pi["b"]
        self.assertEqual(sp.limit(forward, p, 0), rb)
        kl = sum(pi[j] * sp.log(pi[j] / ref[j]) for j in pi)
        reverse = -pi["b"] * (sp.log(pi["b"] / ref["b"]) - kl)
        self.assertEqual(sp.limit(reverse.subs(q, p), p, 0), 0)
        self.assertEqual(sp.limit(reverse.subs(q, p) / (p * sp.log(p)), p, 0), -1)


class MinimalExampleTests(unittest.TestCase):
    """The two-kind example (analysis/theory/minimal_example.md): majority features a, minority features b, K = 2,
    w = W_0 - W_1, margins u = w.a and v = w.b (the minority's deficit). Tail weights h_A = C_A e^{-alpha u},
    h_B = C_B e^{-beta v}; A2 = |a|^2, B2 = |b|^2, g = a.b, D = A2 B2 - g^2 = A2 |b_perp|^2 > 0."""

    al, be, m, CA, CB, A2, B2, g = sp.symbols("alpha beta m C_A C_B A2 B2 g", positive=True)

    def tail_system(self, X, Y):
        """dX/dt, dY/dt for X = e^{alpha u}, Y = e^{beta v} (homogeneous of degree 0)."""
        al, be, m, CA, CB, A2, B2, g = self.al, self.be, self.m, self.CA, self.CB, self.A2, self.B2, self.g
        return al * (m * CA * A2 - (1 - m) * CB * g * X / Y), be * (m * CA * g * Y / X - (1 - m) * CB * B2)

    def fixed_point(self):
        al, be, m, CA, CB, A2, B2, g = self.al, self.be, self.m, self.CA, self.CB, self.A2, self.B2, self.g
        D = A2 * B2 - g**2
        return al * be * m * CA * D / (be * B2 - al * g), al * be * (1 - m) * CB * D / (be * g - al * A2)

    def test_tail_system_from_the_margin_equations(self):
        # du/dt = m h_A A2 - (1 - m) h_B g, dv/dt = m h_A g - (1 - m) h_B B2; X = e^{alpha u}, Y = e^{beta v}.
        u, v = sp.symbols("u v", real=True)
        al, be, m, CA, CB, A2, B2, g = self.al, self.be, self.m, self.CA, self.CB, self.A2, self.B2, self.g
        hA, hB = CA * sp.exp(-al * u), CB * sp.exp(-be * v)
        du, dv = m * hA * A2 - (1 - m) * hB * g, m * hA * g - (1 - m) * hB * B2
        X, Y = sp.exp(al * u), sp.exp(be * v)
        fX, fY = self.tail_system(X, Y)
        self.assertEqual(sp.simplify(al * X * du - fX), 0)
        self.assertEqual(sp.simplify(be * Y * dv - fY), 0)

    def test_self_similar_solution_and_threshold(self):
        # X = xi t, Y = zeta t solves the tail system iff (xi, zeta) = F(xi, zeta); zeta* = beta (1 - m) C_B |b_perp|^2
        # / (beta gamma / alpha - 1) with gamma = g / A2, positive iff gamma > alpha / beta (and then xi* > 0 too).
        al, be, m, CB, A2, B2, g = self.al, self.be, self.m, self.CB, self.A2, self.B2, self.g
        xi, zeta = self.fixed_point()
        fX, fY = self.tail_system(xi, zeta)
        self.assertEqual(sp.simplify(fX - xi), 0)
        self.assertEqual(sp.simplify(fY - zeta), 0)
        t = sp.symbols("t", positive=True)
        fXt, fYt = self.tail_system(xi * t, zeta * t)
        self.assertEqual(sp.simplify(sp.diff(xi * t, t) - fXt), 0)
        gamma, b_perp2 = g / A2, B2 - g**2 / A2
        self.assertEqual(sp.simplify(zeta - be * (1 - m) * CB * b_perp2 / (be * gamma / al - 1)), 0)
        # With b = gamma a + b_perp (|a| = 1): beta B2 - alpha g > 0 whenever beta g > alpha A2, so xi* > 0 on the branch.
        G_, P_ = sp.symbols("gamma_ b_perp2", positive=True)
        expr = sp.expand((be * B2 - al * g).subs({A2: 1, g: G_, B2: G_**2 + P_}))
        self.assertEqual(sp.expand(expr - (G_ * (be * G_ - al) + be * P_)), 0)

    def test_fixed_point_is_a_saddle(self):
        # In log time s = log t with x = X / t, y = Y / t: dx/ds = F_1 - x, dy/ds = F_2 - y. Eigenvalues at (xi*, zeta*):
        # -1 (along the ray, a time shift) and (beta g - alpha A2)(beta B2 - alpha g) / (alpha beta D) > 0 on the branch.
        x, y = sp.symbols("x y", positive=True)
        al, be, A2, B2, g = self.al, self.be, self.A2, self.B2, self.g
        D = A2 * B2 - g**2
        F1, F2 = self.tail_system(x, y)
        xi, zeta = self.fixed_point()
        J = sp.Matrix([[sp.diff(F1, x) - 1, sp.diff(F1, y)], [sp.diff(F2, x), sp.diff(F2, y) - 1]]).subs({x: xi, y: zeta})
        lam2 = (be * g - al * A2) * (be * B2 - al * g) / (al * be * D)
        self.assertEqual(sp.simplify(J.trace() - (lam2 - 1)), 0)
        self.assertEqual(sp.simplify(J.det() + lam2), 0)
        self.assertEqual(sp.simplify(J * sp.Matrix([xi, zeta]) + sp.Matrix([xi, zeta])), sp.zeros(2, 1))

    def test_separatrix_is_the_ray(self):
        # rho = Y / X obeys d rho / dt = [m C_A (beta g - alpha A2) rho - (1 - m) C_B (beta B2 - alpha g)] / X: linear in
        # rho, so rho* = zeta*/xi* repels (when beta g > alpha A2) and the line beta v - alpha u = log rho* separates
        # "stays down" (rho grows) from "recovers" (rho falls).
        X, rho = sp.symbols("X rho", positive=True)
        al, be, m, CA, CB, A2, B2, g = self.al, self.be, self.m, self.CA, self.CB, self.A2, self.B2, self.g
        fX, fY = self.tail_system(X, rho * X)
        drho = (fY - rho * fX) / X
        want = (m * CA * (be * g - al * A2) * rho - (1 - m) * CB * (be * B2 - al * g)) / X
        self.assertEqual(sp.simplify(drho - want), 0)
        xi, zeta = self.fixed_point()
        self.assertEqual(sp.simplify(want.subs(rho, zeta / xi)), 0)

    def test_late_exponent_on_the_collapsed_branch(self):
        # Above the separatrix the minority's pull is negligible: X ~ alpha m C_A A2 t, so dv/dt = m C_A g / X =
        # (g / A2) / (alpha t) and v ~ (gamma / alpha) log t: minority success ~ t^{-gamma/alpha}, Y ~ t^{beta gamma / alpha}.
        t = sp.symbols("t", positive=True)
        al, m, CA, A2, g = self.al, self.m, self.CA, self.A2, self.g
        X = al * m * CA * A2 * t
        dv = m * CA * g / X
        self.assertEqual(sp.simplify(sp.limit(dv * t, t, sp.oo) - g / (A2 * al)), 0)

    def test_first_step_push_ratio(self):
        # At u = v = 0 both weights equal h(0); dv/dt > 0 (the minority falls) iff R = m g / ((1 - m) B2) > 1.
        m, g, B2, h0 = sp.symbols("m g B2 h0", positive=True)
        dv = h0 * (m * g - (1 - m) * B2)
        R = m * g / ((1 - m) * B2)
        self.assertEqual(sp.simplify(dv - h0 * (1 - m) * B2 * (R - 1)), 0)

    def test_class_label_interaction_factor(self):
        # phi(x, y) = e_y (x) phi(x), d(x, y) = (e_c - e_y) (x) phi(x): <d(x, y), d(x', y')> = <e_c - e_y, e_c' - e_y'> k(x, x').
        # Over wrong answers y != c, y' != c': same class gives 2 (K = 2) or 1 and 2 (K >= 3); across classes K = 2 gives only -2,
        # K >= 3 gives -2, -1 and 1 (and 0 for K >= 4).
        for K in (2, 3, 4):
            e = sp.eye(K)
            def factors(c, c2):  # noqa: E306
                return {((e[:, c] - e[:, y]).T * (e[:, c2] - e[:, y2]))[0] for y in range(K) for y2 in range(K)
                        if y != c and y2 != c2}
            self.assertEqual(factors(0, 0), {2} if K == 2 else {1, 2})
            self.assertEqual(factors(0, 1), {-2} if K == 2 else ({-2, -1, 1} if K == 3 else {-2, -1, 0, 1}))

    def test_softmax_linear_success_gradient(self):
        # grad p_x = p_x sum_y pi(y|x) d(x, y), with d(x, y) = phi(x, y*) - phi(x, y), for a symbolic 3-answer, 2-parameter case.
        th = sp.symbols("t1 t2", real=True)
        f = [sp.Matrix(sp.symbols(f"f{y}_1:3", real=True)) for y in range(3)]
        theta = sp.Matrix(th)
        z = [(theta.T * fy)[0] for fy in f]
        Z = sum(sp.exp(zy) for zy in z)
        pi = [sp.exp(zy) / Z for zy in z]
        p = pi[0]
        grad = sp.Matrix([sp.diff(p, t) for t in th])
        want = p * sum((pi[y] * (f[0] - f[y]) for y in range(3)), sp.zeros(2, 1))
        self.assertEqual(sp.simplify(grad - want), sp.zeros(2, 1))

    def test_reset_step_is_separable(self):
        # Inner problem of a reset (mean rule, constant weight c): maximise over the margin z of one prompt
        # c sigma(z) - beta KL(sigma(z) || sigma(z0)). Its stationarity condition is z - z0 = c / beta, whatever z0.
        z, z0, c, beta = sp.symbols("z z0 c beta", real=True)
        s, s0 = 1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-z0))
        kl = s * sp.log(s / s0) + (1 - s) * sp.log((1 - s) / (1 - s0))
        dJ = sp.diff(c * s - beta * kl, z)
        self.assertEqual(sp.simplify(dJ - s * (1 - s) * (c - beta * (z - z0))), 0)


if __name__ == "__main__":
    unittest.main()
