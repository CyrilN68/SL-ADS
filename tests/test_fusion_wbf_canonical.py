"""
test_fusion_wbf_canonical.py — Unit tests for PATCH M-01 / F01
================================================================
Validates :func:`sl_formulas_v2.fusion_wbf_canonical_two` — the literal
opinion-space reproduction of Jøsang 2016 Eq. 12.22-12.24 (2-source
Weighted Belief Fusion).

Properties tested
-----------------
1. Bijection constraint Σb^⋄ + u^⋄ = 1 holds over random opinions.
2. Base rate bijection Σa^⋄ = 1 holds over random opinions.
3. Symmetry : f(A, B) = f(B, A) over random opinions.
4. Idempotence on self-fusion : f(A, A) = A.
5. Dogmatic Case II : u_A = u_B = 0 yields u^⋄ = 0 with averaged beliefs.
6. One-dogmatic limit : as u_A → 0 with u_B > 0, fusion tends toward A
   (dogmatic source dominates).
7. Continuity at Case I ↔ Case II boundary.
8. Alias :func:`fusion_evidence_average_confidence_weighted` is indeed a
   reference to the same function as :func:`fusion_wbf_n_sources`.

Run
---
    python tests/test_fusion_wbf_canonical.py

Exits 0 on success, non-zero on any failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Phase H: conftest.py at the project root prepends ``src/`` and the
# project root to sys.path so ``sl_ads.*`` is importable from tests.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from sl_ads.core.subjective_logic import (  # noqa: E402  Phase H
    MultinomialOpinion,
    fusion_evidence_average_confidence_weighted,
    fusion_wbf_canonical_two,
    fusion_wbf_n_sources,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _random_opinion(rng: np.random.Generator,
                    force_dogmatic: bool = False) -> MultinomialOpinion:
    """Sample a valid random ternary opinion."""
    if force_dogmatic:
        # Dogmatic : u = 0, b is a probability vector on the simplex.
        raw = rng.random(3)
        b = raw / raw.sum()
        u = 0.0
    else:
        # General : u in (0, 1], b + u = 1.
        u = float(rng.uniform(1e-3, 0.99))
        raw = rng.random(3)
        b = (raw / raw.sum()) * (1.0 - u)
    # Base rate (also on simplex, non-uniform for extra rigor).
    raw_a = rng.random(3) + 1e-6
    a = raw_a / raw_a.sum()
    return MultinomialOpinion(b, u, a)


def _assert_valid(op: MultinomialOpinion, label: str) -> None:
    s_b = float(np.sum(op.b))
    s_a = float(np.sum(op.a))
    total = s_b + op.u
    assert np.isclose(total, 1.0, atol=1e-9), (
        f"[{label}] Σb + u = {total!r} (expected 1.0) ; "
        f"b={op.b.tolist()}, u={op.u}"
    )
    assert np.isclose(s_a, 1.0, atol=1e-9), (
        f"[{label}] Σa = {s_a!r} (expected 1.0) ; a={op.a.tolist()}"
    )
    assert np.all(op.b >= -1e-12), f"[{label}] negative belief component: b={op.b}"
    assert op.u >= -1e-12, f"[{label}] negative uncertainty: u={op.u}"
    assert np.all(op.a >= -1e-12), f"[{label}] negative base rate: a={op.a}"


def _assert_same_opinion(op1: MultinomialOpinion,
                          op2: MultinomialOpinion,
                          label: str,
                          atol: float = 1e-9) -> None:
    assert np.allclose(op1.b, op2.b, atol=atol), (
        f"[{label}] b differs : {op1.b.tolist()} vs {op2.b.tolist()}"
    )
    assert np.isclose(op1.u, op2.u, atol=atol), (
        f"[{label}] u differs : {op1.u} vs {op2.u}"
    )
    assert np.allclose(op1.a, op2.a, atol=atol), (
        f"[{label}] a differs : {op1.a.tolist()} vs {op2.a.tolist()}"
    )


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────
def test_bijection_random() -> None:
    """Σb + u = 1 holds on 500 random opinion pairs."""
    rng = np.random.default_rng(seed=20260424)
    for i in range(500):
        A = _random_opinion(rng)
        B = _random_opinion(rng)
        F = fusion_wbf_canonical_two(A, B)
        _assert_valid(F, f"bijection_random[i={i}]")
    print("[OK] test_bijection_random : 500 random pairs, sum(b)+u=1 and sum(a)=1 all valid")


def test_symmetry() -> None:
    """f(A, B) == f(B, A) over 200 random opinion pairs."""
    rng = np.random.default_rng(seed=20260424 + 1)
    for i in range(200):
        A = _random_opinion(rng)
        B = _random_opinion(rng)
        F_AB = fusion_wbf_canonical_two(A, B)
        F_BA = fusion_wbf_canonical_two(B, A)
        _assert_same_opinion(F_AB, F_BA, f"symmetry[i={i}]", atol=1e-9)
    print("[OK] test_symmetry : 200 random pairs, f(A,B) = f(B,A)")


def test_idempotence_self_fusion() -> None:
    """f(A, A) should recover A (up to floating-point)."""
    rng = np.random.default_rng(seed=20260424 + 2)
    for i in range(100):
        A = _random_opinion(rng)
        F = fusion_wbf_canonical_two(A, A)
        _assert_same_opinion(F, A, f"idempotence_self[i={i}]", atol=1e-9)
    print("[OK] test_idempotence_self_fusion : 100 random self-fusions recover input")


def test_dogmatic_case() -> None:
    """u_A = u_B = 0 yields u_fused = 0 and b_fused = 0.5·b_A + 0.5·b_B."""
    rng = np.random.default_rng(seed=20260424 + 3)
    for i in range(100):
        A = _random_opinion(rng, force_dogmatic=True)
        B = _random_opinion(rng, force_dogmatic=True)
        F = fusion_wbf_canonical_two(A, B)
        _assert_valid(F, f"dogmatic[i={i}]")
        assert np.isclose(F.u, 0.0, atol=1e-12), (
            f"[dogmatic {i}] expected u_fused = 0, got {F.u}"
        )
        expected_b = 0.5 * A.b + 0.5 * B.b
        assert np.allclose(F.b, expected_b, atol=1e-12), (
            f"[dogmatic {i}] b_fused != 0.5 b_A + 0.5 b_B ; "
            f"got {F.b}, expected {expected_b}"
        )
    print("[OK] test_dogmatic_case : 100 dogmatic pairs, u=0 and b averaged symmetrically")


def test_continuity_dogmatic_limit() -> None:
    """As u_A → 0 with u_B fixed > 0, b_fused → b_A (dogmatic dominates).

    Jøsang Eq. 12.22 in the limit u_A → 0:
        b^⋄(x) → [c_A · u_B · b_A(x) + c_B · u_A · b_B(x)] / (c_A · u_B)
              = b_A(x) + (c_B · u_A / (c_A · u_B)) · b_B(x)
    With u_A → 0 and c_A → 1, the second term vanishes : b^⋄ → b_A.
    """
    rng = np.random.default_rng(seed=20260424 + 4)
    for i in range(20):
        # A non-dogmatic, B non-dogmatic.
        A_full = _random_opinion(rng)
        B = _random_opinion(rng)
        # Push A progressively toward dogmatism.
        for eps in (1e-2, 1e-4, 1e-6, 1e-8):
            # Rescale A: u becomes eps, b stays proportional.
            b_dir = A_full.b / (np.sum(A_full.b) + 1e-12)
            A_near_dog = MultinomialOpinion(b_dir * (1.0 - eps), eps, A_full.a.copy())
            F = fusion_wbf_canonical_two(A_near_dog, B)
            _assert_valid(F, f"continuity[i={i}, eps={eps}]")
        # At the smallest eps, b_fused must be very close to b of A_near_dog.
        err = float(np.max(np.abs(F.b - A_near_dog.b)))
        assert err < 1e-3, (
            f"[continuity i={i}] expected b_fused ≈ b_A at eps=1e-8, "
            f"max|Δb| = {err:.3e}"
        )
    print("[OK] test_continuity_dogmatic_limit : u_A -> 0  =>  b_fused -> b_A")


def test_alias_identity() -> None:
    """fusion_evidence_average_confidence_weighted IS fusion_wbf_n_sources."""
    assert fusion_evidence_average_confidence_weighted is fusion_wbf_n_sources, (
        "Alias fusion_evidence_average_confidence_weighted is not identity with "
        "fusion_wbf_n_sources"
    )
    print("[OK] test_alias_identity : alias is the same function object")


def test_canonical_vs_evidence_space_consistency() -> None:
    """On 2 non-dogmatic sources with no external weights, canonical WBF
    and evidence-space WBF should agree closely (they are algebraically
    consistent via the bijection — floating-point agreement is expected
    to 1e-6 or better for typical opinions).
    """
    rng = np.random.default_rng(seed=20260424 + 5)
    n_checked = 0
    max_diff_b = 0.0
    max_diff_u = 0.0
    for i in range(200):
        A = _random_opinion(rng)
        B = _random_opinion(rng)
        # Skip edge cases where one source is near-vacuous (u > 0.99)
        # because the evidence-space form handles those numerically
        # differently.
        if A.u > 0.98 or B.u > 0.98:
            continue
        F_canon = fusion_wbf_canonical_two(A, B)
        F_evid = fusion_wbf_n_sources([A, B])
        diff_b = float(np.max(np.abs(F_canon.b - F_evid.b)))
        diff_u = abs(F_canon.u - F_evid.u)
        max_diff_b = max(max_diff_b, diff_b)
        max_diff_u = max(max_diff_u, diff_u)
        n_checked += 1
    # Algebraic identity : the two forms should agree exactly up to FP noise.
    # Evidence-space WBF uses c_i weights on evidence vectors, which via the
    # bijection gives the same result as the opinion-space Case I formula
    # up to typical FP error (~1e-10).  Empirically we allow 1e-6 for
    # robustness.
    assert max_diff_b < 1e-6, (
        f"canonical vs evidence-space WBF : max|Δb| = {max_diff_b:.3e} "
        f"(expected < 1e-6 for algebraic consistency)"
    )
    assert max_diff_u < 1e-6, (
        f"canonical vs evidence-space WBF : max|Δu| = {max_diff_u:.3e} "
        f"(expected < 1e-6)"
    )
    print(f"[OK] test_canonical_vs_evidence_space_consistency : "
          f"n={n_checked}, max|delta_b|={max_diff_b:.2e}, max|delta_u|={max_diff_u:.2e}")


def test_asymmetric_confidence() -> None:
    """If c_A >> c_B (i.e. u_A << u_B), fusion is pulled toward A."""
    # Concrete example : A has u=0.01, B has u=0.5 ; belief directions differ.
    A = MultinomialOpinion([0.98, 0.01, 0.0], 0.01, [1/3, 1/3, 1/3])
    B = MultinomialOpinion([0.0, 0.1, 0.4], 0.5, [1/3, 1/3, 1/3])
    F = fusion_wbf_canonical_two(A, B)
    _assert_valid(F, "asymmetric_confidence")
    # With c_A = 0.99, c_B = 0.5, u_A = 0.01, u_B = 0.5 :
    #   D = 0.99 * 0.5 + 0.5 * 0.01 = 0.495 + 0.005 = 0.500
    #   weight on b_A : c_A u_B / D = 0.495 / 0.500 = 0.99
    #   weight on b_B : c_B u_A / D = 0.005 / 0.500 = 0.01
    # Expected : b_fused ≈ 0.99 · b_A + 0.01 · b_B = [0.9702, 0.0109, 0.004]
    expected_b = (0.99 * A.b + 0.01 * B.b) / (0.99 + 0.01)
    assert np.allclose(F.b, expected_b, atol=1e-6), (
        f"asymmetric: expected {expected_b}, got {F.b}"
    )
    # u_fused = (c_A + c_B) · u_A · u_B / D = 1.49 · 0.005 / 0.500 = 0.0149
    expected_u = (0.99 + 0.5) * 0.01 * 0.5 / 0.500
    assert np.isclose(F.u, expected_u, atol=1e-6), (
        f"asymmetric: expected u={expected_u}, got {F.u}"
    )
    print(f"[OK] test_asymmetric_confidence : b_fused pulled toward high-confidence "
          f"A (weight 0.99), u_fused = {F.u:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────
def _run_all() -> int:
    tests = [
        test_bijection_random,
        test_symmetry,
        test_idempotence_self_fusion,
        test_dogmatic_case,
        test_continuity_dogmatic_limit,
        test_alias_identity,
        test_canonical_vs_evidence_space_consistency,
        test_asymmetric_confidence,
    ]
    n_fail = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            n_fail += 1
            print(f"[FAIL] {t.__name__} : {exc}")
        except Exception as exc:
            n_fail += 1
            print(f"[ERROR] {t.__name__} : {type(exc).__name__}: {exc}")
    print("")
    if n_fail == 0:
        print(f"ALL {len(tests)} TESTS PASSED")
        return 0
    print(f"{n_fail} / {len(tests)} TEST(S) FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
