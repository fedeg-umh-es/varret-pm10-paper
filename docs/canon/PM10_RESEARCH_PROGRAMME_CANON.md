# PM10_RESEARCH_PROGRAMME_CANON

**Version:** `2.0`
**Date:** 2026-08-06
**Authority:** `docs/decision_log/2026-08-06-p2-portfolio-realignment.md`
**Supersedes:** programme canon v1.x (P3 active front, P2 controlled incubation)

---

## 0. Provenance note

As with `P2_PROJECT_CANON.md`, the v1.x programme canon was **not versioned in
this repository** — no such file exists on any branch or in any commit. This
document installs the programme canon in the repository for the first time, at
v2.0. The v1.x state it replaces is recorded on the authority of the decision
that mandated this realignment and is not independently verifiable here.

---

## 1. Portfolio state

```text
PORTFOLIO_REALIGNED

PRIMARY_FRONT:
P2 — Finite Linear Memory Skill Decomposition

P2_EXPERIMENT_STATUS:
GO

P2_MANUSCRIPT_STATUS:
NO-GO PENDING EMPIRICAL GATE

SECONDARY_FRONT:
NONE

P3:
HOLD — NEXT IN QUEUE

P4:
NO-GO / ARCHIVED

CROSS-DOMAIN H*:
WAIT PROTECTED
```

---

## 2. Front-by-front

### P2 — Finite Linear Memory Skill Decomposition — **ACTIVE, SOLE PRIMARY**

Canon: `docs/canon/P2_PROJECT_CANON.md` v2.0.
Experiment authorised. Manuscript blocked behind the `P2_PAPER_GO` gate.

This reverses the v1.x classification of P2 as `INCUBATION / NO-GO AS STANDALONE
PAPER` and the v1.x instruction that P2 must not interrupt P3.

### P3 — **HOLD, NEXT IN QUEUE**

P3 is no longer the active front. Its evidence base in this repository
(`outputs/p3_05_*` … `outputs/p3_12r_*`, Paper A / EMS submission artefacts) is
**frozen, not retracted**. No P3 execution proceeds while P2 holds the primary
slot. P3 resumes when P2 either clears its gate or is itself put on hold.

Nothing in this version invalidates any P3 result already published or submitted.

### P4 — **NO-GO / ARCHIVED**

Closed. The prior `GO_TO_WRITING` verdict recorded in commit `4a49b08`
(`audit(p4): set final verdict GO_TO_WRITING`) is **superseded** by this
version. P4 artefacts remain in `audit/` as historical record and are not
deleted. No further P4 work is authorised.

### Cross-domain H* paper — **WAIT PROTECTED**

Protected wait. Not cancelled, not scheduled. It may not be opened as a front,
and it may not be quietly absorbed into P2's scope. `H*` remains an operational
criterion tied to relative usefulness against a benchmark — never a theoretical
predictability bound.

---

## 3. Consolidation front

**None.** v2.0 removes the consolidation front. Consolidation work that was
already completed (P3-11R evidence reconciliation, P3-12R artefact
consolidation) stands as finished record.

---

## 4. Governance rules in force

1. **Canonisation precedes execution.** No new scientific execution may begin
   before the corresponding canon entry is committed with version, date and
   justification.
2. **Single primary front.** Exactly one primary front at a time. v2.0: P2.
3. **`docs/canon/` is the source of truth.** Documents outside it that describe
   project scope or status are superseded nomenclature.
4. **Superseded nomenclature.** Documents assigning *Predictability Bound* to P1
   and *Operational Meteorology* to P2 must not be used as a source of truth.
5. **Historical artefacts are not rewritten.** Superseded status is recorded
   here, not by editing past outputs.
6. **Manuscripts are not touched by experiment work.** Execution under a canon
   entry may not modify `paper_a.tex`, `paper_a_ems.tex`, or any manuscript.

---

## 5. Change log

| Version | Date | Change |
|---|---|---|
| 2.0 | 2026-08-06 | Portfolio realigned. P2 promoted to sole primary front with `EXPERIMENT GO / PAPER NO-GO PENDING GATE`. P3 to HOLD. P4 to NO-GO/ARCHIVED. Consolidation front removed. Cross-domain H* set to WAIT PROTECTED. Programme canon versioned in-repository for the first time. |
| 1.x | — | P3 active front; P2 controlled incubation. Not versioned in this repository; recorded on the authority of the v2.0 decision. |
