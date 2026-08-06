# PM10 Research Decision Log

Status: CANONICAL LOG — append-only
Last updated: 2026-08-06

Entries are append-only. Superseded entries are marked, never deleted.

---

## 2026-07-27 — P2 v1.0 incubation allocation

Decision ID: `2026-07-27-p2-predictability-bound-incubation`

- P2 ("Predictability Bound") placed in `INCUBATION / NO-GO AS STANDALONE PAPER`.
- P2 instructed not to interrupt the active P3 manuscript.
- A test-set best-model envelope was authorised as the comparison target.
- Initial daily lag order fixed at `p = 14`.

Status: **SUPERSEDED on 2026-08-06.** Retained for provenance in
`P2_PROJECT_CANON_v1.0_SUPERSEDED_20260727.md`.

---

## 2026-08-06 — Portfolio realignment and P2 v2.0

Decision ID: `2026-08-06-p2-finite-linear-memory-skill-decomposition`

```text
PORTFOLIO_REALIGNED

PRIMARY_FRONT:  P2 — Finite Linear Memory Skill Decomposition
P2_EXPERIMENT_STATUS:   GO
P2_MANUSCRIPT_STATUS:   NO-GO PENDING EMPIRICAL GATE
SECONDARY_FRONT:        NONE
P3:                     HOLD — NEXT IN QUEUE
P4:                     NO-GO / ARCHIVED
CROSS-DOMAIN H*:        WAIT PROTECTED
```

Scientific changes relative to v1.0:

1. The object is **attribution** of persistence-relative MSE reduction along
   the ladder `P -> AR(1) -> AR(p) -> M`, not a universal predictability
   ceiling.
2. The lag order becomes a mandatory sensitivity set `p in {7, 14, 21}`.
3. AR(1) is redefined as the `p = 1` case of the same **direct** horizon-specific
   linear projection, `gamma(h)/gamma(0)`, not `rho(1)^h`.
4. The historical hybrid construction is demoted to `diagnostic only`.
5. The test-set best-model envelope is **prohibited** as a primary result.
6. Uncertainty is a moving-block bootstrap over origin vectors with blocks
   shared across all methods.
7. The manuscript gate acquires ten conditions, of which
   `NON_TRIVIAL_INTERPRETATION_FOUND` requires human scientific review.

Supporting documents created: `P2_PROJECT_CANON.md` v2.0,
`P2_CLAIM_CONTRACT.md` v1.0, `P2_PAIRED_DECOMPOSITION_CONTRACT.md` v1.0.

---

## 2026-08-06 — P2 paired decomposition: repository resolution (execution note)

Decision ID: `2026-08-06-p2-execution-repository-resolution`

**Context.** The P2 implementation superprompt names a canonical execution
repository at
`.../03_Investigacion/repos/P2_Predictability_Bound`, with
`/Users/fede/repos/pm10-predictability-bound` as a historical alternative, and
classifies `varret-pm10-paper` as an **external, read-only historical
producer** to be consulted only for provenance.

**Observed state.** The execution environment provisioned for this task is a
remote container containing exactly one git repository,
`fedeg-umh-es/varret-pm10-paper`, checked out on the branch
`claude/p2-finite-linear-memory-gbx3mt` designated for this P2 work. Neither
`P2_Predictability_Bound` nor `pm10-predictability-bound` exists in the
environment, and no filesystem path from the superprompt is reachable.

**Resolution taken.** The P2 implementation was executed **inside**
`varret-pm10-paper`, under a dedicated `src/p2_decomposition/` namespace, with
outputs confined to `outputs/p2_paired_decomposition/`,
`inputs/`, `docs/canon/` and `reports/`.

**Consequences recorded honestly:**

- The row-level PM10 prediction artefacts required by P2 are **native to this
  repository** (`outputs/metrics/predictions*.csv`, produced by this repo's own
  pipeline). Under the superprompt's model they would have been classified
  `VERIFIED_EXTERNAL_IMMUTABLE_INPUT` and copied into
  `inputs/external_predictions/`. Because the producer repository *is* the
  execution repository, they are classified `VERIFIED_LOCAL_P2` and used in
  place, with SHA-256, size, producer repository and producer commit recorded
  in `inputs/P2_INPUT_PROVENANCE.json`.
- No manuscript artefact was read for content, modified, or produced. No
  `.tex`, `.bib`, PDF or Overleaf package was touched.
- P4 was not reachable, not read, not executed and not modified.
- No code was imported from P4. The P2 package has no dependency on the
  pre-existing `src/` packages of this repository.

**Deviation flagged for human decision.** If the intended home of P2 is a
separate `P2_Predictability_Bound` repository, the deliverable of this task
(`src/p2_decomposition/`, `scripts/run_p2_paired_decomposition.py`,
`config/p2_paired_decomposition.yaml`, `tests/test_p2_*.py`,
`docs/canon/`, `inputs/`, `outputs/p2_paired_decomposition/`,
`reports/P2_PAIRED_DECOMPOSITION_REPORT.md`) is self-contained and can be moved
there wholesale; the input provenance manifest then needs its
`provenance_status` re-issued as `VERIFIED_EXTERNAL_IMMUTABLE_INPUT`.

---

## 2026-08-06 — P2 paired decomposition: SARIMA comparability (execution note)

Decision ID: `2026-08-06-p2-sarima-comparability`

`P2_PAIRED_DECOMPOSITION_CONTRACT.md` §7 places `sarima` in the principal
global intersection **only when genuinely comparable**. The available row-level
artefacts show that it is not:

- `sarima` is absent for Elche entirely;
- for Valencia Vivers and Zarra EMEP it was generated with `--origin-step 14`,
  so it covers roughly one origin in fourteen (about 150–175 cases per horizon
  against about 1,450 for the other methods).

Placing `sarima` inside the principal intersection would therefore silently
reduce the verification sample for `ridge_direct` and `hgb_direct` by an order
of magnitude — precisely the kind of undisclosed support change the contract
forbids.

**Resolution.** Two disclosed support types are produced, never merged:

- `GLOBAL_COMMON` — principal: `P, AR1, AR7, AR14, AR21, ridge_direct, hgb_direct`;
- `GLOBAL_COMMON_WITH_SARIMA` — secondary: the same intersection additionally
  requiring `sarima`, reported with its own explicit case counts, for the two
  stations where `sarima` exists.

Both are labelled in every output row via the `support_type` column. Neither is
substituted for the other.
