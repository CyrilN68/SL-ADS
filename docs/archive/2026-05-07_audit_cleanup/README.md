# Archive Index - 2026-05-07 Audit Cleanup

**Archive date:** 2026-05-07  
**Reason:** consolidate the audit corpus so active documents describe the
current scientific state without duplicating older Phase E/F/G drafts.

No document was deleted. The files below were moved here after their actionable
items were consolidated into:

- `docs/AUDIT_CURRENT_STATUS.md`
- `docs/audit/audit_verification_tracker.md`
- `docs/scientific_deconstruction/*.md`
- `docs/honest_limitations.md`
- `docs/REPRODUCIBILITY_CHECKLIST.md`
- `docs/ARTIFACT_APPENDIX.md`

## Archived Files

| Original path | Current replacement | Reason |
|---|---|---|
| `docs/audit/pipeline_reconciliation_20260425.md` | tracker TASK-19 + `AUDIT_CURRENT_STATUS.md` | Historical rerun report; no longer current pipeline state. |
| `docs/audit/reviewer_target_calibration.md` | `AUDIT_CURRENT_STATUS.md` open-work table | Venue-risk note; useful strategic content preserved as current open work. |
| `docs/audit/scientific_audit_reconciliation_20260425.md` | tracker rows TASK-20..46 + current status | Phase F/G closeout; row-level evidence now lives in tracker. |
| `docs/review/AUDIT_SCIENTIFIQUE_PIPELINE.md` | `honest_limitations.md` + scientific deconstruction docs | Early exhaustive French audit; findings consolidated. |
| `docs/review/CHECKLIST_RAPPORT_TECHNIQUE_PIPELINE.md` | `DOCS_GOVERNANCE.md` + `REPRODUCIBILITY_CHECKLIST.md` | Report checklist superseded by formal governance/checklist. |
| `docs/review/CONSOLIDATED_AUDIT_REVIEW.md` | `AUDIT_CURRENT_STATUS.md` + tracker | Large synthesis superseded by canonical current status. |
| `docs/review/HYPOTHESES_ET_MENACES_VALIDITE.md` | `scientific_deconstruction/ASSUMPTIONS.md` | Threat catalog consolidated into formal assumptions. |
| `docs/review/review_compute_evidence_v2.md` | `scientific_deconstruction/METHODS.md` | Module review superseded by package-level method inventory. |
| `docs/review/review_compute_opinions_v3.md` | `scientific_deconstruction/METHODS.md` and `PIPELINE_LOGIC.md` | Legacy module review superseded by fusion/operator docs. |
| `docs/review/review_evaluate_injection_v2.md` | `scientific_deconstruction/METHODS.md` + tracker | Legacy module review; evaluation risks preserved. |
| `docs/review/review_evaluate_qualify_sbn.md` | `M10_sbn_architecture_analysis.md` + `honest_limitations.md` | Qualifier-evaluation review superseded by current terminology/limits. |
| `docs/review/review_inject_at_evidence_level.md` | `scientific_deconstruction/METHODS.md` + tracker | Injection design consolidated. |
| `docs/review/review_qualify_anomaly_sbn.md` | `M10_sbn_architecture_analysis.md` + `honest_limitations.md` | Latest old qualifier module review; useful warnings consolidated. |
| `docs/review/review_qualify_anomaly_sbn_v1.md` | `M10_sbn_architecture_analysis.md` | Superseded qualifier draft. |
| `docs/review/review_qualify_anomaly_sbn_v2.md` | `M10_sbn_architecture_analysis.md` | Superseded qualifier draft. |
| `docs/review/SCIENTIFIC_AUDIT.md` | tracker + `AUDIT_CURRENT_STATUS.md` | Original full scientific audit; findings closed, disclosed, or deferred. |
| `docs/review/SCIENTIFIC_HARDENING_20260504.md` | `docs/review/SCIENTIFIC_HARDENING_20260506.md` | Superseded by later hardening pass. |
| `docs/review/PUBLICATION_TABLES.md` | new `docs/review/PUBLICATION_TABLES.md` | Large legacy table draft archived as `PUBLICATION_TABLES_legacy_20260507.md`; active file now contains only current canonical inputs. |
| `docs/review/cold_start_options.md` | `docs/honest_limitations.md` §5.3.11 (corrected post-mortem) | Phase A research note exploring 7 cold-start mitigation options; the post-mortem inspection of the actual proj_atk trace at NETWORK_OUTAGE_NOV17 (peak=0.117, threshold=0.129) showed the SL detector reacts correctly within one window and the 0/4 stat is a calibration-boundary effect, not a cold-start defect. All 7 options were closed as REJECTED; only the §5.3.11 documentation was corrected. |

## Rules For Using This Archive

- Treat these reports as historical evidence, not current status.
- If a claim here conflicts with an active document, the active document wins.
- If a future reviewer finds an unresolved issue here that is absent from the
  tracker, add it to `docs/audit/audit_verification_tracker.md` before citing
  the archived report.
