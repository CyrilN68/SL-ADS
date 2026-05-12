# Audit Documentation Governance - SL-ADS

**Status date:** 2026-05-07  
**Scope:** rules for maintaining `current_version/docs/`.

This document defines where audit information belongs, when an old report can
be archived, and how to prevent duplicate or contradictory scientific claims.

## 1. Source-of-Truth Order

When two documents disagree, use this order:

1. `docs/AUDIT_CURRENT_STATUS.md` - current scientific and audit state.
2. `docs/audit/audit_verification_tracker.md` - finding-level evidence and
   verification commands.
3. `docs/scientific_deconstruction/*.md` - formal assumptions, methods,
   pipeline logic, theory graph, and reference discipline.
4. `docs/honest_limitations.md`, `docs/REPRODUCIBILITY_CHECKLIST.md`,
   `docs/ARTIFACT_APPENDIX.md` - paper-facing material.
5. `docs/review/*` - active review notes that still inform the current paper.
6. `docs/archive/*` - historical trace only.

Archived material is never a current claim unless an active source-of-truth file
explicitly cites it.

## 2. Where New Information Goes

| New item | Required destination |
|---|---|
| New audit finding | Add a row to `audit/audit_verification_tracker.md`. |
| New resolved finding | Update the tracker row, verification command, and date. |
| New headline scientific result | Update `AUDIT_CURRENT_STATUS.md` and the relevant review/method file. |
| New limitation | Update `honest_limitations.md` and link the evidence in the tracker. |
| New reproducibility requirement | Update `REPRODUCIBILITY_CHECKLIST.md`. |
| New paper table or metric | Update `review/PUBLICATION_TABLES.md` or create a dated active review note. |
| New fusion/operator change | Update `scientific_deconstruction/METHODS.md`, `PIPELINE_LOGIC.md`, `THEORY_GRAPH.md`, and `REFERENCES.md`. |
| New command that creates artifacts | Record output paths and expected checks in the tracker. |

Do not leave an important result only in a chat transcript, console output, or
ad hoc scratch file.

## 3. Archiving Rules

A document can move to `docs/archive/YYYY-MM-DD_audit_cleanup/` only when all
three conditions hold:

1. Its useful unresolved findings have been copied into the tracker or a
   current active document.
2. Its current scientific conclusions have been superseded by
   `AUDIT_CURRENT_STATUS.md` or a newer active review.
3. The archive index records original path, reason for archiving, and current
   replacement.

Archived files should not be edited except to fix broken archive-local links or
encoding damage. Prefer adding a note in the archive index over mutating old
reports.

## 4. Duplicate Policy

Duplicates are allowed only when they serve different audiences:

- `AUDIT_CURRENT_STATUS.md` gives the short current verdict.
- `audit/audit_verification_tracker.md` gives row-level verification.
- `scientific_deconstruction/*.md` gives formal reasoning.
- Paper-facing files give publication wording.

If two documents repeat the same claim, the claim must use the same status,
date, metric values, and artifact path. If the old copy is no longer maintained,
archive it.

## 5. Status Vocabulary

Use the same status words everywhere:

- `RESOLVED`: implemented or documented, with a verification path.
- `IN_PROGRESS`: active work exists but the finding is not closed.
- `DEFERRED`: scientifically valid but out of current scope.
- `NEEDED`: required before submission, not yet done.
- `REJECTED`: considered and deliberately not adopted.
- `PENDING_DATA`: blocked on external data or manual provenance.

Avoid informal labels such as "mostly fixed" in tables. Put nuance in the note
column.

## 6. Minimum Evidence Standard

Every serious claim should include:

- exact date;
- config knobs that matter;
- command or script used;
- artifact path;
- primary metric values;
- residual limitation.

For comparative claims, calibrate each compared mode under its own valid
configuration unless the purpose is explicitly a stress test.

## 7. Link Discipline

Use relative paths inside documentation. Before a cleanup is considered done,
search active docs for filenames that were moved to archive and update those
links.

Recommended checks:

```powershell
Get-ChildItem .\current_version\docs -Recurse -File -Filter *.md |
  Select-String -Pattern 'SCIENTIFIC_AUDIT|CONSOLIDATED_AUDIT_REVIEW|review_compute|20260425'
```

Keep generated results under `current_version/results/` or
`current_version/outputs/`, and cite the exact dated subdirectory.

## 8. README Duties

`docs/README.md` is an index, not an audit report. Update it whenever:

- a file becomes canonical;
- a file is archived;
- a new dated review becomes active;
- the recommended reading order changes.

Do not put metric conclusions only in `README.md`; link to the document that
owns the conclusion.
