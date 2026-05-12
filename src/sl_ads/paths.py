"""Helpers centralisant les noms de version et chemins artefacts.

Usage:
    from sl_ads.paths import (get_version_names, get_results_dir,
                              get_model_path, get_decision_threshold)
"""
import json
import os
from pathlib import Path


_SIDECAR_SENSITIVE_KNOBS = {
    "fusion_mode_at_calibration": "INTER_METHOD_FUSION",
    "wbf_weight_mode": "WBF_WEIGHT_MODE",
    "lambda_decay": "LAMBDA_DECAY",
    "balance_ratio": "BALANCE_RATIO",
    "cd_alpha_attack": "CD_ALPHA_ATTACK",
    # PATCH H2 — calendar-aware EVT.  When the calibration ran with
    # ``CALENDAR_EVT_ENABLED=True`` the sidecar exposes a
    # ``calendar_evt_signature`` field (the regime function's versioned
    # signature, see ``sl_ads.calendar.regime.REGIME_FN_SIGNATURE``).
    # A runtime/calibration mismatch on this field is treated like any
    # other sensitive-knob drift: ``RuntimeError("[A1.9] ...")`` unless
    # the explicit ablation bypass is set.  When the sidecar predates
    # H2 (no ``calendar_evt_signature`` present) the field is reported
    # as ``missing`` and the legacy single-threshold path is used —
    # there is no hard failure for backward compatibility with old
    # PKLs.
    "calendar_evt_signature": "CALENDAR_EVT_SIGNATURE_RUNTIME",
}


def get_version_names(config: dict) -> tuple[str, str]:
    version = config.get("VERSION_NAME", "trained_models_v9_v6_v4s")
    version_modif = config.get("VERSION_NAME_MODIF", f"{version}_attacks")
    return version, version_modif


def _prefix(up_levels: int) -> str:
    if up_levels <= 0:
        return "."
    return os.path.join(*([".."] * up_levels))


def get_results_dir(config: dict, up_levels: int = 1) -> str:
    version, _ = get_version_names(config)
    default_dir = os.path.join(_prefix(up_levels), "results", f"resultats_{version}")
    eval_cfg = config.get("EVAL", {})
    return eval_cfg.get("RESULTS_DIR", config.get("RESULTS_DIR", default_dir))


def get_model_path(config: dict, up_levels: int = 1) -> str:
    version, _ = get_version_names(config)
    return os.path.join(_prefix(up_levels), f"trained_models_{version}.pkl")


def _clean_fusion_mode_for_path(fusion_mode) -> str:
    return str(fusion_mode).strip().lower().replace("-", "_")


def get_threshold_sidecar_path(config: dict,
                               up_levels: int = 1,
                               fusion_mode: str | None = None) -> str:
    """Retourne le chemin du fichier JSON sidecar de seuils (léger, ~100 octets).

    ``fusion_mode=None`` preserves the historical generic sidecar name.  Passing
    a mode returns the mode-specific sidecar, e.g.
    ``trained_models_X_threshold_abf.json``.
    """
    model_path = Path(get_model_path(config, up_levels=up_levels))
    # Utiliser pathlib pour éviter le bug de str.replace qui modifie .pkl dans
    # le chemin complet (ex: "backup.pkl.bak" → "backup_threshold.json.bak").
    suffix = "_threshold"
    if fusion_mode is not None:
        suffix += f"_{_clean_fusion_mode_for_path(fusion_mode)}"
    return str(model_path.parent / (model_path.stem + f"{suffix}.json"))


def resolve_threshold_sidecar_path(config: dict, up_levels: int = 1) -> str:
    """Return the mode-specific threshold sidecar when available.

    This lets a user switch only ``CONFIG['INTER_METHOD_FUSION']`` between
    ``abf`` and ``wbf`` while still loading the matching calibrated threshold.
    The legacy generic sidecar remains the fallback for old model packages.
    """
    mode = config.get("INTER_METHOD_FUSION")
    if mode:
        mode_path = get_threshold_sidecar_path(
            config,
            up_levels=up_levels,
            fusion_mode=str(mode),
        )
        if os.path.exists(mode_path):
            return mode_path
    return get_threshold_sidecar_path(config, up_levels=up_levels)


def _sidecar_cfg_value(config: dict, cfg_key: str):
    """Return the runtime value comparable to the threshold sidecar."""
    if cfg_key == "CD_ALPHA_ATTACK":
        # Historical RedeRio configs omit CD_ALPHA_ATTACK.  The training
        # sidecar writer uses 1.0 as the effective no-discount default.
        raw = config.get(cfg_key, 1.0)
        return 1.0 if raw in (None, "auto") else raw
    if cfg_key == "CALENDAR_EVT_SIGNATURE_RUNTIME":
        # PATCH H2 — the runtime equivalent is not a CONFIG field; it is
        # the imported ``REGIME_FN_SIGNATURE`` constant when calendar-aware
        # EVT is enabled, or ``None`` otherwise.  We import lazily to keep
        # ``paths.py`` light and to avoid circular imports.
        if not config.get("CALENDAR_EVT_ENABLED", False):
            return None
        try:
            from sl_ads.calendar.regime import REGIME_FN_SIGNATURE  # noqa: WPS433
            return REGIME_FN_SIGNATURE
        except ImportError:  # pragma: no cover — defensive
            return None
    return config.get(cfg_key)


def _values_match(runtime_value, sidecar_value, *, allow_null_match: bool = False) -> bool:
    """Compare sidecar/runtime values with numeric tolerance when possible."""
    if runtime_value is None or sidecar_value is None:
        return allow_null_match and runtime_value is None and sidecar_value is None
    try:
        return abs(float(runtime_value) - float(sidecar_value)) <= 1e-9
    except (TypeError, ValueError):
        return str(runtime_value).strip().lower() == str(sidecar_value).strip().lower()


def validate_threshold_sidecar_config(
        config: dict,
        up_levels: int = 1,
        sidecar_data: dict | None = None,
        strict: bool = True,
) -> dict:
    """Validate that calibration-sensitive runtime knobs match the sidecar.

    The decision threshold is only an operational FPR guarantee when the
    deployed fusion/discount configuration matches the one used at
    calibration.  Older/minimal sidecars may not contain the sensitive
    fields; those are reported as ``missing`` and are not hard failures so
    legacy tests and artifacts remain readable.
    """
    sidecar_path = resolve_threshold_sidecar_path(config, up_levels=up_levels)
    data = sidecar_data
    if data is None:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    checked = {}
    missing = []
    mismatches = {}
    for sidecar_key, cfg_key in _SIDECAR_SENSITIVE_KNOBS.items():
        if sidecar_key not in data:
            missing.append(sidecar_key)
            continue
        runtime_value = _sidecar_cfg_value(config, cfg_key)
        sidecar_value = data.get(sidecar_key)
        checked[sidecar_key] = {
            "config_key": cfg_key,
            "runtime": runtime_value,
            "sidecar": sidecar_value,
        }
        if not _values_match(
            runtime_value,
            sidecar_value,
            allow_null_match=(sidecar_key == "calendar_evt_signature"),
        ):
            mismatches[sidecar_key] = checked[sidecar_key]

    result = {
        "sidecar_path": sidecar_path,
        "checked": checked,
        "missing": missing,
        "mismatches": mismatches,
        "ok": not mismatches,
    }
    if strict and mismatches:
        details = "; ".join(
            f"{meta['config_key']} runtime={meta['runtime']!r} "
            f"!= sidecar[{key}]={meta['sidecar']!r}"
            for key, meta in mismatches.items()
        )
        raise RuntimeError(
            "[A1.9] Threshold sidecar/config mismatch on calibration-sensitive "
            f"knobs for {sidecar_path}: {details}. "
            "Recalibrate the decision threshold on the deployed pipeline or "
            "restore the calibrated config before running evaluation/inference."
        )
    return result


def get_decision_threshold(config: dict, up_levels: int = 1) -> float:
    """
    Retourne DECISION_THRESHOLD en priorité depuis le JSON sidecar (auto-calibré
    à l'entraînement), ou depuis CONFIG['EVAL']['DECISION_THRESHOLD'] en fallback.

    Le sidecar est écrit par train_v10.py après auto-calibration EVT/FPR.
    Variable cible : proj_atk = b_atk + a_atk·u  (Jøsang Eq. 3.23).
    Ref : Ali et al. TISSEC 2013 ; Sun et al. ICML 2024.
    PATCH TASK-43 (audit_codex MIN-01, 2026-04-27): docstring updated
    to reference the active training entrypoint.
    """
    sidecar = resolve_threshold_sidecar_path(config, up_levels=up_levels)
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            allow_ablation_mismatch = os.environ.get(
                "SL_ALLOW_THRESHOLD_FUSION_MISMATCH_FOR_ABLATION", ""
            ).strip().lower() in ("1", "true", "yes")
            status = validate_threshold_sidecar_config(
                config,
                up_levels=up_levels,
                sidecar_data=data,
                strict=not allow_ablation_mismatch,
            )
            if allow_ablation_mismatch and not status.get("ok", False):
                print(
                    "[WARN][A1.9] Threshold sidecar/config mismatch allowed "
                    "for explicit fusion ablation only; using the calibrated "
                    "threshold as a fixed reference, not as a headline FPR "
                    "guarantee."
                )
            thr = data.get("decision_threshold")
            if thr is not None:
                return float(thr)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"[WARN] paths.py: lecture sidecar '{sidecar}' échouée ({e})"
                  " — fallback sur DECISION_THRESHOLD config.")
    return float(config.get("EVAL", {}).get("DECISION_THRESHOLD", 0.20))


def get_decision_variable(config: dict, up_levels: int = 1) -> str:
    """
    Retourne la variable de décision calibrée à l'entraînement : 'b_atk' ou 'proj_atk'.
    Lit depuis le JSON sidecar (écrit par train_v10.py).
    Fallback : 'proj_atk' pour rétrocompatibilité avec les artefacts anciens.
    PATCH TASK-43 (audit_codex MIN-01, 2026-04-27): docstring updated
    to reference the active training entrypoint.
    """
    sidecar = resolve_threshold_sidecar_path(config, up_levels=up_levels)
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            var = data.get("decision_variable")
            if var in ("b_atk", "proj_atk"):
                return var
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"[WARN] paths.py: lecture sidecar '{sidecar}' échouée ({e})"
                  " — fallback sur 'proj_atk'.")
    return "proj_atk"


def get_detection_col(config: dict, up_levels: int = 1) -> str:
    """
    Retourne le nom de colonne à utiliser pour la détection dans le CSV de compute_opinions_v3.
    Dépend de decision_variable : 'FINAL_SYSTEM_CBF_b_atk' ou 'FINAL_SYSTEM_CBF_proj_atk'.

    PATCH TASK-44 (audit_codex MAJ-09, 2026-04-27).  The column prefix
    ``FINAL_SYSTEM_CBF`` is HISTORICAL: regardless of the value chosen
    for ``CONFIG['INTER_METHOD_FUSION']`` (wbf | abf | cbf | bcf | ccf |
    minbf | maxbf | hierarchical),
    ``compute_opinions_v3.py`` writes the fused output under this prefix
    for backward-compatibility with 31 downstream consumers.  The actual
    fusion mode is recorded in ``fusion_mode_at_compute_opinions.json``
    next to the CSV; use ``get_fusion_mode_for_run`` to read it.

    Use the alias ``get_detection_col_fused`` (same return value) when
    you want the new naming convention to leak into your code.
    """
    return f"FINAL_SYSTEM_CBF_{get_decision_variable(config, up_levels=up_levels)}"


def get_detection_col_fused(config: dict, up_levels: int = 1) -> str:
    """
    PATCH TASK-44 (audit_codex MAJ-09, 2026-04-27) — forward-compatible
    alias for ``get_detection_col``.  Returns the same column name (still
    prefixed ``FINAL_SYSTEM_CBF`` because the artifact format hasn't
    been migrated yet — see ``docs/maj09_naming_migration.md``) but
    signals at the call site that the *output* is the inter-method
    fused opinion regardless of fusion mode.
    """
    return get_detection_col(config, up_levels=up_levels)


def get_fusion_mode_for_run(output_dir: str) -> dict | None:
    """
    PATCH TASK-44 (audit_codex MAJ-09, 2026-04-27).  Read the fusion
    metadata sidecar that ``compute_opinions_v3.py`` writes next to its
    CSV outputs.  Returns ``None`` if the sidecar is absent (older runs
    pre-2026-04-27) or unparseable.

    Recommended usage:

        meta = get_fusion_mode_for_run(OUTPUT_DIR)
        fusion_mode = (meta or {}).get('actual_fusion_mode', 'cbf')
        # 'wbf' | 'abf' | 'cbf' | 'bcf' | 'ccf' | 'minbf' | 'maxbf' | 'hierarchical'
    """
    sidecar = os.path.join(output_dir, "fusion_mode_at_compute_opinions.json")
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN][MAJ-09] paths.get_fusion_mode_for_run: "
              f"failed to read {sidecar} ({e}).")
        return None


def validate_full_sl_ads_defaults(config: dict) -> dict:
    """Retourne l'état des invariants Full SL-ADS attendus."""
    return {
        "lambda_0_85": config.get("LAMBDA_DECAY") == 0.85,
        "k_3_0": config.get("SL_PARAM_K") == 3.0,
        "decision_threshold_sidecar": os.path.exists(resolve_threshold_sidecar_path(config)),
        "wbf_mode_uniform": config.get("WBF_WEIGHT_MODE") == "uniform",
        "c3_uniform_disabled": config.get("C3_WEIGHT_MODE") != "uniform",
    }
