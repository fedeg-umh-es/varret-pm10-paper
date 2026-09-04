# P4 Numerical Invariance Hash Contract Decision

## Decision ID

2026-08-01-p4-numerical-invariance-hash-contract

## Status

ACCEPTED

## Context

The previous documentary closeout referred to an expected SHA-256 value
`d8a4cc5dcc24ebc7fdb942a597bbd9378e3a551c5d436544415a74e002b56e1f`.

No executable script, configuration field, manifest entry or complete
documentary specification was found that defined the text range,
tokenization, normalization, serialization, encoding or source commit used to
produce that value.

Independent verification confirmed that the numeric-token sequences in the
base and documentary-closeout versions of `paper_a.tex` are identical.

## Decision

The undocumented historical hash `d8a4cc…` is retired as an obsolete and
non-reproducible documentary control.

It is replaced by an explicit numerical-invariance contract:

- file: `paper_a.tex`;
- start marker: `\begin{abstract}`;
- end marker, excluded: `\section*{Data and Code Availability}`;
- tokenizer: Python regex `[+-]?(?:\d+\.?\d*|\.\d+)`;
- token order: textual order;
- serialization: comma-separated tokens;
- encoding: UTF-8;
- digest: SHA-256;
- base commit: `f57f076078760af8a88bd87815fdf94ab0064fa3`;
- documentary closeout commit:
  `390685f1f1312954ee67513f3e0db11b2670e7f9`;
- validation-correction commit:
  `11f8c88e5e3860eee75a333c83382127c360a930`;
- token count: 187;
- resulting digest:
  `cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3`.

## Interpretation

The contract verifies invariance of the numeric-token sequence between the
specified manuscript versions.

It does not validate the scientific correctness of individual values, replace
the provenance manifest, or imply that the manuscript text is otherwise
identical.

This is a **documentary contract correction**, not a scientific result
correction or an empirical result change.

## Evidence

- provenance manifest: 22 entries, zero missing files, zero SHA-256 mismatches
  and zero size mismatches;
- protected scientific artefacts unchanged;
- numeric-token sequences identical across the base, documentary-closeout and
  validation-correction commits;
- historical expected hash not reproducible from any documented contract.

## Consequences

- the previous `EXPECTED_HASH_CONTRACT_FAIL` is resolved by an explicit
  canonical contract correction;
- no empirical result is changed;
- no manuscript number is changed;
- no code, data, table, figure or result artefact is modified;
- P4 is classified as `P4_DOCUMENTARY_CLOSEOUT_VALIDATED`;
- P3 remains deferred until P2 is audited as closed or explicitly released.

## Prohibited interpretation

This decision must not be described as recovering the original `d8a4cc…`
contract. The original contract remains unknown.

The new contract is a documented replacement, not a retrospective reproduction
of the historical hash and not a silent hash replacement.

## Canonical links

- Canon: [[P4_PROJECT_CANON]]
- Repository contract:
  `docs/p4_numerical_invariance_contract.md` in the canonical P4 repository
