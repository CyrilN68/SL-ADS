# SL-ADS Audit Documentation Index

**Status date:** 2026-05-12  
**Scope:** documentation map for `current_version/docs/`.

Start here, then follow the canonical documents below. Historical reviews are
preserved under `archive/` but are not the current audit state.

## Recommended Reading Order

1. [AUDIT_CURRENT_STATUS.md](AUDIT_CURRENT_STATUS.md) - current verdict,
   residual risks, and WBF vs ABF decision.
2. [DOCS_GOVERNANCE.md](DOCS_GOVERNANCE.md) - rules for updating, archiving,
   and avoiding duplicate claims.
3. [audit/audit_verification_tracker.md](audit/audit_verification_tracker.md)
   - finding-by-finding evidence trail.
4. [scientific_deconstruction/ASSUMPTIONS.md](scientific_deconstruction/ASSUMPTIONS.md)
   - formal assumptions and residual scientific risks.
5. [scientific_deconstruction/METHODS.md](scientific_deconstruction/METHODS.md)
   - method inventory and operator definitions.
6. [honest_limitations.md](honest_limitations.md) - paper-facing limitations.
7. [REPRODUCIBILITY_CHECKLIST.md](REPRODUCIBILITY_CHECKLIST.md) and
   [ARTIFACT_APPENDIX.md](ARTIFACT_APPENDIX.md) - reproducibility and artifact
   packaging.

## Active Top-Level Documents

| File | Role |
|---|---|
| [AUDIT_CURRENT_STATUS.md](AUDIT_CURRENT_STATUS.md) | Canonical current audit state. |
| [DOCS_GOVERNANCE.md](DOCS_GOVERNANCE.md) | Maintenance and archiving rules. |
| [honest_limitations.md](honest_limitations.md) | Publication-ready limitations section. |
| [REPRODUCIBILITY_CHECKLIST.md](REPRODUCIBILITY_CHECKLIST.md) | Reproducibility checklist for artifact review. |
| [ARTIFACT_APPENDIX.md](ARTIFACT_APPENDIX.md) | Artifact appendix template and verification notes. |

## Active Audit Evidence

| File | Role |
|---|---|
| [audit/audit_verification_tracker.md](audit/audit_verification_tracker.md) | Master tracker for audit findings, status, and verification commands. |
| [audit/trust_discount_r2_analysis.md](audit/trust_discount_r2_analysis.md) | RedeRio trust-discount pathology evidence. |
| [audit/wu_keogh_self_assessment.md](audit/wu_keogh_self_assessment.md) | Self-assessment against Wu and Keogh time-series benchmark pitfalls. |

## Active Scientific Reviews

| File | Role |
|---|---|
| [review/FUSION_OPERATOR_ABLATION_20260506.md](review/FUSION_OPERATOR_ABLATION_20260506.md) | Historical fusion-operator diagnostic explaining why WBF remains default; not the source of final headline metrics. |
| [review/M10_sbn_architecture_analysis.md](review/M10_sbn_architecture_analysis.md) | Qualifier terminology and architecture analysis. |
| [review/PUBLICATION_TABLES.md](review/PUBLICATION_TABLES.md) | Current paper-facing table inputs from complete run `2e12261d55a8f975`. |
| [review/calendar_evt_design.md](review/calendar_evt_design.md) | Phase B H2 design + post-mortem for calendar-aware EVT thresholds. Audit-grade opt-in shipped 2026-05-07; current complete-run root cause is correlation-level, not solved by per-metric regime EVT alone. |
| [review/regime_fpr_root_cause_analysis.md](review/regime_fpr_root_cause_analysis.md) | Complete-run regime-FPR values and TASK-58 root-cause verdict (`H_correlation`). Exploratory alpha-sweeps are future work only; no alpha value is shipped in the current paper. |

## Formal Scientific Deconstruction

| File | Role |
|---|---|
| [scientific_deconstruction/ASSUMPTIONS.md](scientific_deconstruction/ASSUMPTIONS.md) | Assumptions, failure modes, and mitigations. |
| [scientific_deconstruction/METHODS.md](scientific_deconstruction/METHODS.md) | Formal methods and operator inventory. |
| [scientific_deconstruction/PIPELINE_LOGIC.md](scientific_deconstruction/PIPELINE_LOGIC.md) | End-to-end data and inference logic. |
| [scientific_deconstruction/THEORY_GRAPH.md](scientific_deconstruction/THEORY_GRAPH.md) | Graph of theory-to-code dependencies. |
| [scientific_deconstruction/REFERENCES.md](scientific_deconstruction/REFERENCES.md) | Citation and patch traceability. |

## Archive

Retired material lives under [archive/](archive/). The 2026-05-07 cleanup moves
superseded Phase E/F/G module reviews and reconciliation drafts into
`archive/2026-05-07_audit_cleanup/` after consolidating their actionable
findings into the active tracker and current-status documents.

The 2026-05-11 public-release cleanup moved the Phase H renaming log and the
dated 2026-05-06 hardening report into
`archive/2026-05-11_public_release_cleanup/`. Their still-relevant findings are
tracked in `AUDIT_CURRENT_STATUS.md`, `audit/audit_verification_tracker.md`,
and `scientific_deconstruction/`.

Archived files are preserved for traceability. They should not be used as the
current scientific state unless an active document explicitly cites them.
