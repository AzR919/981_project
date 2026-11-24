"""
Symbolic checks (SymPy) of the algebra behind analysis/theory/odds_law.md and analysis/theory/late_time.md.

Each test states the identity it checks. Asymptotic claims are checked on the exact solution of the reduced
model where one exists; claims about the full simulator are checked numerically elsewhere.
"""

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


if __name__ == "__main__":
    unittest.main()
