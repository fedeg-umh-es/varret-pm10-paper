# PM10 Research Programme Canon

Version: 1.0
Last updated: 2026-08-06
Status: CANONICAL PORTFOLIO ALLOCATION
Decision ID: `2026-08-06-p2-finite-linear-memory-skill-decomposition`

---

## 1. Purpose

This file records the portfolio-level allocation of the PM10 research
programme. It governs which scientific front is active, which are held, and
which are archived. It does not restate the scientific specification of any
individual project; that lives in each project canon.

For P2 the scientific source of truth is `P2_PROJECT_CANON.md` v2.0, with
`P2_CLAIM_CONTRACT.md` and `P2_PAIRED_DECOMPOSITION_CONTRACT.md` as its
subordinate contracts.

---

## 2. Portfolio decision

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

P2 v2.0 is the only active scientific front. The former allocation
`P2 = INCUBATION / NO-GO AS STANDALONE PAPER`, and the instruction that P2
must not interrupt P3, are superseded.

---

## 3. Project status table

| Project | Status | Write access | Notes |
|---|---|---|---|
| P1 | Methodological backbone | Not modified by this decision | Rigorous-evaluation and H* backbone. |
| P2 — Finite Linear Memory Skill Decomposition | PRIMARY FRONT, experiment GO | Active | Manuscript remains NO-GO pending the empirical gate. |
| P3 | HOLD — next in queue | Not modified by this decision | Held, not cancelled. |
| P4 — Ghost Skill / Dynamic Fidelity | NO-GO / ARCHIVED | Read-only | Must not be executed, modified, branched or silently imported as a code dependency. |
| P5 | Not active | Not modified by this decision | — |
| Cross-domain H* | WAIT PROTECTED | Not modified by this decision | Protected; no cross-domain claims from P2 evidence. |

---

## 4. Dominant scientific question of the active front

> What part of the MSE reduction relative to persistence can be reproduced by
> AR(1) dependence and finite linear memory AR(p), and what part remains as
> model-specific residual gain, by station and forecast horizon?

Canonical reference ladder:

```text
P -> AR(1) -> AR(p) -> M
```

---

## 5. Programme-level invariants binding on P2 execution

1. Lag-order sensitivity set `p in {7, 14, 21}`; no order is universally optimal.
2. AR(1) and AR(p) are **direct**, horizon-specific linear projections; AR(1)
   is the `p = 1` case of the same projection. Recursive iteration of a
   one-step AR(p) is not the primary reference.
3. Attribution is performed in MSE. MSE and RMSE must not be mixed without an
   explicit and correct transformation.
4. The principal comparison uses the **global common paired support** shared by
   every compared method and every lag order, so that the verification sample
   does not move when `p` moves.
5. Per-model decomposition curves are reported individually. A test-set
   best-model envelope is not a legitimate primary result.
6. Uncertainty comes from a moving-block bootstrap over **origin vectors**;
   the same sampled blocks are applied to every method and horizon.
7. Daily series are reindexed to the complete daily calendar; missing values
   stay `NaN` and are never dropped before lag construction.
8. Means, autocovariances and projection coefficients are estimated
   train-only.
9. All ten gate conditions in `P2_PROJECT_CANON.md` §15 must be supported by
   traceable artefacts before any manuscript work.
10. `NON_TRIVIAL_INTERPRETATION_FOUND` is a human scientific judgement and can
    never be set automatically.

---

## 6. Superseded nomenclature

The following historical labels are **SUPERSEDED NOMENCLATURE** and are not
sources of truth:

| Superseded label | Current canonical label |
|---|---|
| "P1 — Predictability Bound" | P2 — Finite Linear Memory Skill Decomposition |
| "P2 — Operational Meteorology" | Not an active front; operational meteorology is out of P2 scope |
| "P2 — Predictability Bound" (v1.0, 2026-07-27) | P2 v2.0, see `P2_PROJECT_CANON_v1.0_SUPERSEDED_20260727.md` |
| "universal linear predictability ceiling" | finite-memory optimal linear reference / AR(p)–Yule–Walker linear predictability reference |
| test-set "best model" envelope | prohibited as a primary result (oracle selection) |

---

## 7. Boundaries

P2 must not absorb operational meteorology availability, Ghost Skill, variance
retention as a central object, Aurora evaluation, or cross-domain H* claims.

Three-station daily PM10 evidence is bounded evidence. It must not be
generalised to PM10 forecasting as a whole.
