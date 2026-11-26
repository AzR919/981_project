"""Curated knowledge for the paper graph: concepts, typed relations between papers, and
relations between papers and the project's claims.

Every relation carries a `basis`, so a graph built from it keeps an audit trail:
  READ     - checked directly in the PDF by the main session
  TRIAGE   - reported by a triage agent from the PDF, not independently checked
  INFERRED - derived by a triage agent (algebra or reasoning), not stated in the paper
  REVIEW   - established in a full review under reviews/deep/
  TEAM     - stated in the team's own milestone review
"""

# concept -> regex searched in the title, the one-line description and the triage "main claims"
CONCEPTS = {
    "expected-update identity (advantage gap)": r"advantage gap|scalar.{0,20}∇p|κ\(p\)|monotone (transform|rescaling)|W_G",
    "surrogate objective": r"surrogate|objective (of|induced)|arcsin|utility functional",
    "pass@k": r"pass@|pass-at-k|inference[- ]time objective|majority vot",
    "maximum likelihood / log p": r"maximum likelihood|likelihood objective|log p\b|rejection sampling|MaxRL",
    "KL regularization": r"\bKL\b|Kullback|divergence",
    "KL direction": r"forward[- ]KL|reverse[- ]KL|mass-covering|mode-seeking|KL\(π_ref|direction",
    "moving reference (resets, EMA, previous iterate)": r"reset|\bEMA\b|anchor|previous (iterate|policy)|mirror|π_t\b|moving",
    "fixed point / stationary policy": r"fixed point|stationar|optimal (target|polic)|optimum",
    "step-size threshold": r"step[- ]size|learning[- ]rate threshold|learning rate|critical batch",
    "collapse / instability": r"collaps|instabilit|unstable|fail",
    "prompt interference / shared parameters": r"interferen|shared[- ]param|gradient conflict|kernel|low-rank|cancel",
    "uninformative (zero-variance) groups": r"zero-variance|degenerate|homogeneous|all[- ]incorrect|all[- ]correct|advantage collapse|negative group|uniform group",
    "group size": r"group size|finite[- ]G|number of rollouts|more rollouts|\bG\b ?[=∈→]",
    "rollout budget allocation": r"allocat|budget",
    "baseline / variance reduction": r"baseline|variance reduction|control variate|shrinkage|gradient variance",
    "advantage normalization": r"normali[sz]|standard deviation|std-|z-score",
    "difficulty weighting": r"difficulty|hard prompts|easy prompts|medium-difficulty|reweight",
    "entropy / diversity collapse": r"entropy|diversity|mode collapse|support",
    "exploration at p ≈ 0": r"explor|p ?[≈~] ?0|bounded away from zero|log-barrier|escape",
    "softmax bandit theory": r"bandit",
    "coverage": r"coverage",
    "off-policy / staleness": r"off-policy|stale|asynchron|importance",
    "response length": r"length|horizon",
    "transformer learning theory": r"transformer|next-token|parity",
    "negative-sample reinforcement": r"negative (sample|reinforcement|gradient)|penali[sz]ing incorrect|incorrect samples",
    "scaling study": r"scaling|compute",
}

# (source id, relation, target id, note, basis)
RELATIONS = [
    ("2510.13651", "proves at finite group size", "PROJECT:H1",
     "§4 Eq. (4.1)–(4.2), p. 5: E[Z ∇log π] = κ(p)∇p for any weight linear in the reward given the leave-one-out count; κ is the project's W_G.", "READ"),
    ("2510.13651", "gives the finite-G surrogate of", "PROJECT:H2",
     "§6 and Fig. 3: finite-M GRPO surrogate h_{M,ε} for M = 2, 16, 256 against (2/π)arcsin√t.", "READ"),
    ("2510.23049", "agrees with", "2510.13651", "GRPO's surrogate is 2·arcsin√p in the infinite-group limit in both, as the team's milestone states.", "TEAM"),
    ("2502.18548", "states the stationary equation behind", "PROJECT:H4",
     "Eq. 11 (p. 9), for any group size and both KL directions, with P_G left as an expectation; binary cases written at G = 2 and G → ∞.", "TRIAGE"),
    ("2502.18548", "precedes and overlaps", "2503.06639", "Stationary policies of GRPO with a reference penalty, a month before Mroueh's fixed point.", "TRIAGE"),
    ("2510.20817", "gives the W ≡ 1 case of", "PROJECT:H4",
     "Optimal policies under KL(π_θ‖π_ref) (Eq. 2) and KL(π_ref‖π_θ) (Eq. 5), no group normalization.", "TRIAGE"),
    ("2506.09477", "gives the W ≡ 1 case of", "PROJECT:H4",
     "Both optima on a tabular softmax bandit (pp. 4–5) and which one sampled-gradient training reaches (Figs. 4–6).", "TRIAGE"),
    ("2506.09477", "agrees with", "2510.20817", "Same two optimal policies; GX-Chen cites Tang & Munos for GRPO's KL term acting as KL(π_ref‖π_θ).", "TRIAGE"),
    ("2512.21852", "agrees with", "2506.09477", "Both find that common KL estimators differentiate to the wrong gradient.", "TRIAGE"),
    ("2602.04417", "gives a step-size ceiling relevant to", "PROJECT:H4",
     "Lemma 4.2 (Eq. 9): αβλ_max < 1 + η for an EMA anchor; linearized model with frozen reward gradient.", "TRIAGE"),
    ("2505.24864", "empirical counterpart of", "2611.14467", "ProRL resets the reference policy by hand; Huang et al. analyse reference resets. Link made by the main session; neither paper was checked for citing the other.", "INFERRED"),
    ("2601.08521", "contains the finite-G expectations behind", "PROJECT:H1",
     "Eqs. 37, 112, 138: expected advantage for correct and incorrect samples conditional on a mixed group.", "TRIAGE"),
    ("2508.02833", "states the mean-baseline factor of", "PROJECT:H1", "Eq. 26 and §C.3: E[update] = (1 − 1/|G|)∇J/T for the empirical-mean baseline.", "TRIAGE"),
    ("2607.01490", "states the rescaling in", "PROJECT:H1", "Table 1 (p. 4): RLOO is G/(G−1) times mean-only; a 'scale axis' that rescales the learning rate.", "TRIAGE"),
    ("2602.21189", "defines the interference measure for", "PROJECT:H3",
     "Kernel κ(x,x′) = ⟨∇p(x), ∇p(x′)⟩ (Eq. 5) and agreement score (Eq. 7); one step, exact gradients; no finite G, effective step or absorption.", "TRIAGE"),
    ("2606.30789", "contradicts", "PROJECT:H2", "Prop. 10 (Eq. 12): the fixed point and mean trajectory are independent of G; thin evidence.", "TRIAGE"),
    ("2606.30789", "contradicts", "PROJECT:H4", "Same proposition: G only adds 1/G noise around a G-independent fixed point.", "TRIAGE"),
    ("2606.30789", "builds on", "2503.06639", "Local linearization around the fixed point imported from Mroueh.", "TRIAGE"),
    ("2610.18853", "reports collapse scaling relevant to", "PROJECT:H3", "Samples-to-collapse N_fail = C·√b/η (Eq. 7) under AdamW; a postulate, fitted exponent 0.64–0.72.", "TRIAGE"),
    ("2610.18853", "cites", "2607.01083", "For collapse time ∝ 1/η.", "TRIAGE"),
    ("2607.01083", "reports collapse scaling relevant to", "PROJECT:H3", "Largest stable learning rate ∝ 1/S; collapse time ∝ 1/η (abstract).", "READ"),
    ("2606.16154", "states a large-G form of", "PROJECT:H1", "Eq. 7 / Prop. 2: positive-only centred update ≈ (1 − q)∇q.", "TRIAGE"),
    ("0000.baudry", "qualifies", "PROJECT:H3",
     "Absorption-like failure in a tabular single-context bandit above a learning-rate threshold, driven by reward noise with no baseline.", "TRIAGE"),
    ("0000.baudry", "responds to", "2502.07141", "Mei et al. prove convergence for any constant learning rate; Baudry et al. show the regret can still be polynomial.", "TRIAGE"),
    ("2603.15001", "builds on", "0000.baudry", "Takes the hidden assumption (optimal arm's probability bounded away from zero) from Baudry et al.", "TRIAGE"),
    ("2603.15001", "builds on", "2502.07141", "Adds a log-barrier to the stochastic gradient bandit analysed by Mei et al.", "TRIAGE"),
    ("2602.02710", "same rule under binary rewards as", "2611.05942", "A_i ∝ r_i/K − 1/G with all-wrong groups dropped; W = (1 − (1−p)^(G−1))/p in both.", "INFERRED"),
    ("2611.05942", "uses the same name for a different quantity than", "PROJECT:H1", "'Advantage gap' there is the within-group A⁺ − A⁻ at a realized count, not W_G(p).", "TRIAGE"),
    ("2611.05942", "has a heuristic precedent for", "PROJECT:H4", "Log-odds shift κ = (r⁺ − r⁻)/β (Eq. 2) with per-group gaps; no fixed-point equation in p.", "TRIAGE"),
    ("2605.30154", "restricts the identity of", "2510.13651", "Theorem 2.1 is the same identity for rules with zero or constant weight on failures; found Davis & Recht's finite-M result through it.", "TRIAGE"),
    ("2505.15201", "gives an unbiased estimator for the pass@k objective of", "2510.23049", "Eq. 8, unbiased for any G ≥ k; relating it to RLOO_k is the triage agent's reading.", "INFERRED"),
    ("2503.19595", "precedes", "2505.15201", "Earlier policy-gradient formulation of k-sample objectives (March against May 2025).", "READ"),
    ("2605.21125", "states the informative fraction of", "PROJECT:H5", "App. F: probability of a homogeneous group is p^G + (1−p)^G; ACR is its measured batch average.", "TRIAGE"),
    ("2602.05548", "states the difficulty weighting in", "PROJECT:H3", "Theorem 2: Σ|A_i| = 2G√(p̂(1−p̂)) per group.", "TRIAGE"),
    ("2602.01601", "gives gradient-variance formulas for", "PROJECT:H5", "Prop. 4.2–4.3: variance ∝ p(1−p) with (n−1)/n² for Dr. GRPO and 1/(n−1) for RLOO.", "TRIAGE"),
    ("2510.13786", "LLM-scale evidence for", "PROJECT:H5", "G ∈ {8, 16, 24, 32} at fixed total batch leaves the fitted curves essentially unchanged.", "TRIAGE"),
    ("2507.14843", "restates the W ≡ 1 fixed point of", "2503.06639", "App. F: exponential tilt π_β ∝ q·e^{βR}.", "TRIAGE"),
    ("2505.22257", "follow-up to", "2503.06639", "Improvement lower bounds for on- and off-policy GRPO; no fixed point, no finite G.", "TRIAGE"),
]

# Papers given a full review after triage (see reviews/TIERS.md).
DEEP = {"2510.13651", "2502.18548", "2601.08521", "2602.21189", "2510.20817", "2506.09477", "2602.04417",
        "2602.02710", "2611.05942", "2610.18853", "2607.01083", "2605.21125", "2602.05548", "0000.baudry", "2603.15001"}

PROJECT_NODES = {
    "PROJECT:H1": "H1 — the identity E[ĝ] = W_G(p)∇p holds at finite G; rules whose W_G differ by a constant match after rescaling the step",
    "PROJECT:H2": "H2 — group size changes GRPO's surrogate, from p at G = 2 toward 2·arcsin√p",
    "PROJECT:H3": "H3 — stability is set by one effective step (η/B)·W_G(p)·p(1−p) and by interference between prompts; failure is absorption at p ≈ 0",
    "PROJECT:H4": "H4 — SGD with a KL penalty settles at the finite-G fixed point; KL raises the stability threshold",
    "PROJECT:H5": "H5 — group size under a fixed budget, the informative fraction φ, and escape from p ≈ 0",
}
