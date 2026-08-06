# Decision Log Entry

**Decision ID:** `2026-08-06-p2-portfolio-realignment`
**Task:** Canonise the portfolio realignment promoting P2 to sole primary front,
and specify the paired AR(p) decomposition contract
**Author:** Claude (Claude Code session), authorised by fedeg@umh.es
**Documents produced:** `docs/canon/P2_PROJECT_CANON.md` v2.0,
`docs/canon/PM10_RESEARCH_PROGRAMME_CANON.md` v2.0,
`docs/canon/P2_CLAIM_CONTRACT.md` v1.0,
`docs/canon/P2_PAIRED_DECOMPOSITION_CONTRACT.md` v1.0,
`docs/superprompts/P2_CODEX_SUPERPROMPT.md`

---

## Question

Should P2 be promoted from controlled incubation to sole primary front, with a
redefined dominant question — decomposing persistence-relative skill into
one-lag dependence, additional finite linear memory, and model-specific residual
gain — and if so, under what binding specification?

## Available evidence

**Governance state.** The canon documents this decision was said to contradict
(`P2_PROJECT_CANON.md` with `INCUBATION / NO-GO AS STANDALONE PAPER`,
`PM10_RESEARCH_PROGRAMME_CANON.md` with P3 as active front) **do not exist in
this repository**. `git log --all --diff-filter=A --name-only` over the full
history returns no such files on any branch or commit, and no file in the
working tree references them. They existed only outside version control.

**Repository capability.** Verified directly:

- Three stations with rolling-origin prediction tables in the required paired
  schema (`dataset, model, fold, origin_date, horizon, date, y_true, y_pred`):
  `outputs/metrics/predictions.csv` (26 001 rows),
  `predictions_valencia_vivers.csv` (52 028), `predictions_zarra_emep.csv`
  (52 136).
- Generator `scripts/01_generate_e1_rr_lags_only_predictions.py` already
  implements full-calendar reindexing without imputation, train-only fitting per
  origin, and persistence as mandatory baseline — the substrate the AR references
  must match.
- Calendar missingness on the daily inputs: `pm10_daily.csv` 19.6 %,
  `pm10_valencia_vivers.csv` 8.3 %, `pm10_zarra_emep.csv` 4.0 %.

**Specification defect found.** SARIMA predictions exist at 0 of 8 667 origins
for `e1_rr_daily`, 1 098 of 10 186 (10.8 %) for `valencia_vivers`, and 1 196 of
10 188 (11.7 %) for `zarra_emep` — it was generated on a subsampled origin grid.
The proposed rule "global intersection across the three orders of `p` **and all
methods**" is therefore not implementable: it would collapse the primary paired
support by roughly 90 % and annihilate it entirely for `e1_rr_daily`.

**Bibliographic basis.** The gap assessment and reference list supporting it were
supplied by external search tooling. They were **not** verified against sources
in this repository, and `references.bib` was not modified.

## Decision

**`PORTFOLIO_REALIGNED — P2 PRIMARY, EXPERIMENT GO / PAPER NO-GO PENDING GATE`**

Adopted as specified, with one specification correction and one governance
correction:

1. **Specification correction (§4.5 of the P2 canon).** The primary paired
   support is defined over a **core method set** —
   `persistence, AR(1), AR(7), AR(14), AR(21), ridge_direct, hgb_direct` — and
   SARIMA is decomposed separately on its own labelled support with its own
   `n_paired`. Without this, the contract is unexecutable on the existing
   evidence.

2. **Governance correction.** Because the v1.x canon was never versioned here,
   these documents do not amend a prior canonical state; they **install the canon
   in the repository for the first time at v2.0**, recording the superseded v1.x
   state on the authority of this decision rather than on verifiable evidence.
   From this entry onward, `docs/canon/` is the single source of truth.

All other mandated adjustments are adopted verbatim: AR(1) as the `p = 1` case of
the same direct projection (not `ρ̂(1)^h`); AR(p) as a direct per-horizon
projection (not an iterated one-step model); per-model primary curves with no
oracle envelope; `p ∈ {7, 14, 21}` with primary and per-`p` supports; moving-block
bootstrap over origin vectors with shared blocks; absolute MSE differences
primary and normalised fractions secondary, untruncated and unclipped.

Additionally superseded: the P4 verdict `GO_TO_WRITING` recorded in commit
`4a49b08` is replaced by `NO-GO / ARCHIVED`.

## Claims allowed

Binding list: `docs/canon/P2_CLAIM_CONTRACT.md`. Summary — the principal
methodological claim (C0) and the interpretive claim (C3) are allowed now; the
bibliographic gap claim (C1) is allowed **only** in its corpus-scoped form
("within the reviewed corpus of 20 works"), never as a statement about the
literature.

## Claims prohibited

That residual gain demonstrates non-linearity or exogenous information (C4), and
its variants: calling `Δ_res` "non-linear skill"; reading `Δ_res > 0` as physical
predictability; presenting `L_ARp` as a bound or limit; presenting `H*` as a
theoretical predictability bound. The empirical claim C2 (station- and
horizon-dependence of the components) may not be written until produced by the
tables with bootstrap uncertainty.

## Required next evidence

1. Execute `docs/superprompts/P2_CODEX_SUPERPROMPT.md` to produce
   `outputs/p2_paired_decomposition/` in full.
2. Clear the ten `P2_PAPER_GO` conditions, each with a traceable artefact.
3. Resolve the `VERIFY_AUTHORSHIP` flag on arXiv 2603.20315 (C5) — self-citation
   versus external antecedent changes how it must be cited.
4. Run and archive the systematic Scopus / Web of Science search that would let
   the C1 scope qualifier be widened. Not a blocker for the scoped form.
5. Verify the externally supplied bibliographic records before any of them enter
   `references.bib`.

## Effect on manuscript

**None.** No manuscript was touched. `paper_a.tex`, `paper_a_ems.tex` and
`references.bib` are unmodified, and the execution contract explicitly forbids
modifying them. P3 evidence under `outputs/p3_*` and `audit/` is frozen as
historical record, not retracted — nothing published or submitted is invalidated
by this entry.
