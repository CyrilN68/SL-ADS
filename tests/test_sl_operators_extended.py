"""tests/test_sl_operators_extended.py

Extended unit tests for the Subjective Logic operators that are NOT
already covered by ``test_fusion_wbf_canonical.py``:

  * Cumulative Belief Fusion (CBF, Jøsang 2016 Eq. 12.14-15)
  * Trust discount (Jøsang 2016 Def. 14.6) — opinion-level
  * Contextual discount (3-class projection-shift) — opinion-level
  * Temporal ageing (λ_dyn = λ_base × (1 − K_eff)^γ) — evidence-level
  * Conflict degree K (Eq. 12.4 belief_mass mode) — evidence-level
  * Evidence-to-opinion bijection (Def. 3.9)

Phase H — added 2026-04-29 to satisfy USENIX SecAE "Artifacts
Functional" §C ("appropriate evidence of verification and
validation") and ACM "Artifacts Evaluated — Reusable" badge criteria.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# conftest.py at the project root adds src/ to sys.path; defensive
# belt-and-braces for direct invocations.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sl_ads.core.subjective_logic import (  # noqa: E402
    MultinomialOpinion,
    fusion_abf,
    fusion_bcf,
    fusion_cbf,
    fusion_ccf,
    fusion_by_mode,
    fusion_evidence_average_n_sources,
    fusion_maxbf,
    fusion_minbf,
    apply_trust_discount,
    apply_contextual_discount,
    evidence_to_opinion,
    temporal_ageing_fusion,
    temporal_adaptive_ageing,
    compute_conflict_degree,
)


# ────────────────────────────────────────────────────────────────────────
# Fixtures (opinions and evidence vectors)
# ────────────────────────────────────────────────────────────────────────
W = 3.0  # SL_PARAM_K (per CONFIG)
A_UNIF = np.full(3, 1 / 3)


@pytest.fixture
def vacuous():
    return MultinomialOpinion(beliefs=np.zeros(3), u=1.0, a=A_UNIF.copy())


@pytest.fixture
def moderate_safe():
    return MultinomialOpinion(beliefs=np.array([0.6, 0.1, 0.1]), u=0.2, a=A_UNIF.copy())


@pytest.fixture
def moderate_attack():
    return MultinomialOpinion(beliefs=np.array([0.1, 0.1, 0.6]), u=0.2, a=A_UNIF.copy())


@pytest.fixture
def r_zero():
    return np.zeros(3)


@pytest.fixture
def r_attack():
    return np.array([0.0, 0.0, 5.0])


@pytest.fixture
def r_safe():
    return np.array([5.0, 0.0, 0.0])


# ────────────────────────────────────────────────────────────────────────
# CBF (Cumulative Belief Fusion)
# ────────────────────────────────────────────────────────────────────────
class TestCBF:
    def test_cbf_with_vacuous_returns_other(self, moderate_attack, vacuous):
        out = fusion_cbf(moderate_attack, vacuous)
        np.testing.assert_allclose(out.b, moderate_attack.b, atol=1e-9)
        assert abs(out.u - moderate_attack.u) < 1e-9

    def test_cbf_symmetric_in_args(self, moderate_safe, moderate_attack):
        """CBF is commutative (Jøsang Th. 12.4)."""
        out_ab = fusion_cbf(moderate_safe, moderate_attack)
        out_ba = fusion_cbf(moderate_attack, moderate_safe)
        np.testing.assert_allclose(out_ab.b, out_ba.b, atol=1e-9)
        assert abs(out_ab.u - out_ba.u) < 1e-9

    def test_cbf_preserves_bijection(self, moderate_safe, moderate_attack):
        out = fusion_cbf(moderate_safe, moderate_attack)
        assert abs(out.b.sum() + out.u - 1.0) < 1e-9

    def test_cbf_amplifies_concordant_evidence(self, moderate_attack):
        """Two concordant attack opinions → resulting b_atk ≥ either input."""
        out = fusion_cbf(moderate_attack, moderate_attack)
        assert out.b[2] >= moderate_attack.b[2] - 1e-9


# ────────────────────────────────────────────────────────────────────────
# Trust discount (Jøsang 2016 Def. 14.6) — opinion-level
# ────────────────────────────────────────────────────────────────────────
class TestAdditionalFusionOperators:
    def _assert_valid(self, op):
        assert abs(float(op.b.sum()) + float(op.u) - 1.0) < 1e-9
        assert np.all(np.isfinite(op.b))
        assert np.isfinite(op.u)
        assert np.all(op.b >= -1e-12)
        assert op.u >= -1e-12
        assert abs(float(op.a.sum()) - 1.0) < 1e-9

    def test_abf_is_idempotent_for_duplicate_source(self, moderate_attack):
        out = fusion_abf(moderate_attack, moderate_attack)
        np.testing.assert_allclose(out.b, moderate_attack.b, atol=1e-9)
        assert abs(out.u - moderate_attack.u) < 1e-9
        self._assert_valid(out)

    def test_abf_with_vacuous_increases_uncertainty(self, moderate_safe, vacuous):
        out = fusion_abf(moderate_safe, vacuous)
        self._assert_valid(out)
        assert out.u >= moderate_safe.u - 1e-9

    def test_bcf_reinforces_concordant_belief(self, moderate_attack):
        out = fusion_bcf(moderate_attack, moderate_attack)
        self._assert_valid(out)
        assert out.b[2] >= moderate_attack.b[2] - 1e-9

    def test_bcf_zadeh_style_conflict_collapses_to_tiny_middle_class(self):
        op_safe = MultinomialOpinion([0.99, 0.01, 0.0], 0.0, A_UNIF.copy())
        op_atk = MultinomialOpinion([0.0, 0.01, 0.99], 0.0, A_UNIF.copy())
        out = fusion_bcf(op_safe, op_atk)
        self._assert_valid(out)
        assert out.b[1] > 0.999

    def test_ccf_is_idempotent_for_duplicate_source(self, moderate_safe):
        out = fusion_ccf(moderate_safe, moderate_safe)
        np.testing.assert_allclose(out.b, moderate_safe.b, atol=1e-9)
        assert abs(out.u - moderate_safe.u) < 1e-9

    def test_minbf_is_conservative_and_valid(self, moderate_safe, moderate_attack):
        out = fusion_minbf(moderate_safe, moderate_attack)
        self._assert_valid(out)
        assert out.b[2] <= min(moderate_safe.b[2], moderate_attack.b[2]) + 1e-9

    def test_maxbf_is_aggressive_and_valid(self, moderate_safe, moderate_attack):
        out = fusion_maxbf(moderate_safe, moderate_attack)
        self._assert_valid(out)
        assert out.b[2] >= min(moderate_safe.b[2], moderate_attack.b[2]) - 1e-9
        assert out.u <= max(moderate_safe.u, moderate_attack.u) + 1e-9

    def test_hierarchical_evidence_average_differs_from_wbf_when_confidence_differs(self):
        op_high_conf_safe = MultinomialOpinion([0.89, 0.05, 0.01], 0.05, A_UNIF.copy())
        op_low_conf_atk = MultinomialOpinion([0.05, 0.10, 0.35], 0.50, A_UNIF.copy())
        hierarchical = fusion_evidence_average_n_sources([op_high_conf_safe, op_low_conf_atk], W=W)
        wbf = fusion_by_mode([op_high_conf_safe, op_low_conf_atk], mode="wbf", W=W)
        self._assert_valid(hierarchical)
        self._assert_valid(wbf)
        assert abs(hierarchical.b[2] - wbf.b[2]) > 1e-3

    @pytest.mark.parametrize("mode", ["wbf", "abf", "hierarchical", "cbf", "bcf", "ccf", "minbf", "maxbf"])
    def test_dispatcher_modes_return_valid_opinion(self, mode, moderate_safe, moderate_attack):
        out = fusion_by_mode([moderate_safe, moderate_attack], mode=mode, W=W)
        self._assert_valid(out)


class TestTrustDiscount:
    def test_trust_one_is_identity(self, moderate_attack):
        out = apply_trust_discount(moderate_attack, t=1.0)
        np.testing.assert_allclose(out.b, moderate_attack.b, atol=1e-9)
        assert abs(out.u - moderate_attack.u) < 1e-9

    def test_trust_zero_is_vacuous(self, moderate_attack):
        out = apply_trust_discount(moderate_attack, t=0.0)
        np.testing.assert_allclose(out.b, np.zeros(3), atol=1e-9)
        assert abs(out.u - 1.0) < 1e-9

    def test_trust_monotone(self, moderate_attack):
        out_high = apply_trust_discount(moderate_attack, t=0.9)
        out_low  = apply_trust_discount(moderate_attack, t=0.3)
        assert out_high.b[2] >= out_low.b[2] - 1e-9
        assert out_high.u   <= out_low.u   + 1e-9

    def test_trust_preserves_bijection(self, moderate_attack):
        for trust in (0.0, 0.25, 0.5, 0.75, 1.0):
            out = apply_trust_discount(moderate_attack, t=trust)
            assert abs(out.b.sum() + out.u - 1.0) < 1e-9, f"bijection broken @t={trust}"


# ────────────────────────────────────────────────────────────────────────
# Contextual discount — opinion-level
# ────────────────────────────────────────────────────────────────────────
class TestContextualDiscount:
    def test_alpha_one_is_identity(self, moderate_attack):
        out = apply_contextual_discount(moderate_attack, alpha=[1.0, 1.0, 1.0])
        np.testing.assert_allclose(out.b, moderate_attack.b, atol=1e-9)
        assert abs(out.u - moderate_attack.u) < 1e-9

    def test_alpha_attack_zero_collapses_attack_belief(self, moderate_attack):
        """alpha[2]=0 → b_atk transferred into u."""
        out = apply_contextual_discount(moderate_attack, alpha=[1.0, 1.0, 0.0])
        assert out.b[2] < 1e-9
        # u must absorb the discounted belief
        assert out.u >= moderate_attack.u + moderate_attack.b[2] - 1e-9

    def test_preserves_bijection(self, moderate_attack):
        for av in [(1.0, 1.0, 1.0), (1.0, 1.0, 0.5), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)]:
            out = apply_contextual_discount(moderate_attack, alpha=list(av))
            assert abs(out.b.sum() + out.u - 1.0) < 1e-9, f"bijection broken @alpha={av}"


# ────────────────────────────────────────────────────────────────────────
# Temporal ageing — evidence-level
# ────────────────────────────────────────────────────────────────────────
class TestTemporalAgeing:
    def test_lambda_one_keeps_history(self, r_attack):
        """λ=1 → no decay : r_acc stays full-weighted."""
        r_acc = np.array([2.0, 0.0, 4.0])
        out = temporal_ageing_fusion(r_acc, r_attack, lam=1.0)
        # λ=1 ⇒ out = r_acc + r_current (all history preserved)
        np.testing.assert_allclose(out, r_acc + r_attack, atol=1e-9)

    def test_lambda_zero_is_purely_current(self, r_attack):
        """λ=0 → full decay : history forgotten, only r_current remains."""
        r_acc = np.array([2.0, 0.0, 4.0])
        out = temporal_ageing_fusion(r_acc, r_attack, lam=0.0)
        np.testing.assert_allclose(out, r_attack, atol=1e-9)

    def test_intermediate_lambda_blends(self, r_attack):
        r_acc = np.array([2.0, 0.0, 4.0])
        out_high = temporal_ageing_fusion(r_acc, r_attack, lam=0.85)
        out_low  = temporal_ageing_fusion(r_acc, r_attack, lam=0.30)
        # Higher λ retains more accumulated history, so attack-class evidence
        # at index 2 is at least as large.
        assert out_high[2] >= out_low[2] - 1e-9

    def test_adaptive_ageing_returns_evidence_and_metadata(self, r_attack):
        """temporal_adaptive_ageing returns ``(r_aged, K, lam_dyn)``.

        We assert: r_aged is a finite non-negative ndarray of the same
        shape as the inputs, K ∈ [0, 1], λ_dyn ∈ [0, 1]."""
        r_acc = np.array([2.0, 0.0, 4.0])
        out = temporal_adaptive_ageing(
            r_accumulated=r_acc, r_current=r_attack,
            lam_base=0.85, W=W,
        )
        assert isinstance(out, tuple) and len(out) == 3
        r_aged, K, lam_dyn = out
        assert isinstance(r_aged, np.ndarray)
        assert r_aged.shape == r_attack.shape
        assert np.all(np.isfinite(r_aged))
        assert np.all(r_aged >= 0.0)
        assert 0.0 <= float(K) <= 1.0 + 1e-9
        assert 0.0 <= float(lam_dyn) <= 1.0 + 1e-9


# ────────────────────────────────────────────────────────────────────────
# Conflict degree (evidence-level)
# ────────────────────────────────────────────────────────────────────────
class TestConflictDegree:
    def test_conflict_zero_when_concordant(self, r_attack):
        """K(r, r) = 0 (no conflict between identical evidence)."""
        K = compute_conflict_degree(r_attack, r_attack, W=W)
        assert K < 1e-9, f"expected K≈0 for identical r, got {K}"

    def test_conflict_high_when_opposed(self, r_safe, r_attack):
        """Conflict between safe-evidence and attack-evidence > 0."""
        K = compute_conflict_degree(r_safe, r_attack, W=W)
        assert K > 0.0

    def test_conflict_zero_with_zero_evidence(self, r_attack, r_zero):
        """K(r, 0) = 0: zero evidence has no belief mass to conflict with."""
        K = compute_conflict_degree(r_attack, r_zero, W=W)
        assert K < 1e-9

    def test_conflict_in_unit_interval(self, r_safe, r_attack):
        for a, b in [(r_safe, r_attack), (r_attack, r_safe)]:
            K = compute_conflict_degree(a, b, W=W)
            assert 0.0 <= K <= 1.0 + 1e-9


# ────────────────────────────────────────────────────────────────────────
# Evidence-to-opinion bijection (Def. 3.9)
# ────────────────────────────────────────────────────────────────────────
class TestEvidenceToOpinion:
    def test_zero_evidence_is_vacuous(self):
        op = evidence_to_opinion(np.zeros(3), W=W, a=A_UNIF.copy())
        np.testing.assert_allclose(op.b, np.zeros(3), atol=1e-9)
        assert abs(op.u - 1.0) < 1e-9

    def test_dogmatic_evidence_collapses_uncertainty(self):
        r = np.array([0.0, 0.0, 1e6])
        op = evidence_to_opinion(r, W=W, a=A_UNIF.copy())
        assert op.u < 1e-3
        assert op.b[2] > 1.0 - 1e-3

    def test_evidence_to_opinion_bijection(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            r = np.abs(rng.normal(size=3) * 5.0)
            op = evidence_to_opinion(r, W=W, a=A_UNIF.copy())
            assert abs(op.b.sum() + op.u - 1.0) < 1e-9
