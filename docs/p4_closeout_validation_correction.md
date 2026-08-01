# P4 Closeout Validation Correction

This correction records the canonical relocation of the P4 repository and the
independent re-verification of the documentary closeout. It resolves the
contradiction between the declared expected numeric-token hash and the hash
actually reproduced from `paper_a.tex`. No code, data, empirical result, figure,
table, or manuscript numeric value was modified.

## Repository
- Canonical path: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`
- Backup path: `/Users/fede/repos/varret-pm10-paper` (preserved, not deleted)
- Branch: `codex/p4-documentary-closeout`
- Base SHA: `f57f076078760af8a88bd87815fdf94ab0064fa3`
- Previous closeout SHA: `390685f1f1312954ee67513f3e0db11b2670e7f9`

## Relocation
- Source tree hash: `e195612a6da12a6835b4bcac69c8b5b0e390c298`
- Target tree hash: `e195612a6da12a6835b4bcac69c8b5b0e390c298`
- Integrity result: `RELOCATION_INTEGRITY_OK` (identical commit, identical tree,
  `git fsck --full` clean, base and closeout objects present as commits)
- External remote: `origin → https://github.com/fedeg-umh-es/varret-pm10-paper`
- Local backup remote: `source-local → /Users/fede/repos/varret-pm10-paper`
- Note: the canonical clone already existed from a prior interrupted run
  (`--no-hardlinks`, `origin` restored, `source-local` added). Its branch, HEAD
  SHA, and remotes match the source exactly; the only working-tree difference
  was the in-progress documentary edit to the closeout report. No destructive
  operation (`reset --hard`, `clean`, `rebase`, `merge`, `force-push`) was used;
  no `push` or PR was performed.

## Provenance manifest
- Entries: 22
- Missing: 0
- SHA-256 mismatches: 0
- Size mismatches: 0
- Verdict: `PROVENANCE_MANIFEST_PASS`

## Relative numerical invariance
- Source commit: `f57f076078760af8a88bd87815fdf94ab0064fa3` (base)
- Target commit: `390685f1f1312954ee67513f3e0db11b2670e7f9` (closeout)
- Text range: Abstract → Data and Code Availability in `paper_a.tex`
- Tokenization: regex `[+-]?(?:\d+\.?\d*|\.\d+)`, comma-joined
- Token count: 187 (base) = 187 (closeout); sequences byte-for-byte identical
- Base hash: `cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3`
- Final hash: `cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3`
- Verdict: `RELATIVE_NUMERICAL_INVARIANCE_PASS`

## Expected-hash contract
- Declared expected hash: `d8a4cc5dcc24ebc7fdb942a597bbd9378e3a551c5d436544415a74e002b56e1f`
- Contract source: prose only — the prior committed closeout text and the P4
  task specification. No script, configuration, or manifest field in the
  repository computes it; tokenizer, range, normalization, separator, and
  computing commit are undocumented.
- Reproduced hash: `cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3`
- Match: No
- Verdict: `EXPECTED_HASH_CONTRACT_FAIL`
- Explanation: The documented tokenizer was applied to both the base and
  closeout `paper_a.tex` blobs across four text ranges and four separators
  (sixteen combinations). None reproduce `d8a4cc…`. The comma-joined
  abstract→Data-and-Code range reproduces `cfd238…` consistently for both
  commits. The declared `d8a4cc…` therefore appears to be a stale or
  differently-scoped value with no traceable executable contract. In compliance
  with the closeout rules, the observed hash was **not** substituted for the
  declared expected hash to manufacture a pass.

## Scientific artefacts
- Outputs changed: none (`outputs/reproduction/`, `outputs/tables/`,
  `outputs/figures/` identical base→closeout)
- Scripts changed: none (`run_paper_a_empirical.py`, `render_paper_a_results.py`,
  `recover_madrid_pm10.py`, `src/data/madrid_hourly.py`,
  `src/plotting/plot_master_figure.py` identical base→closeout)
- Data changed: none
- Verdict: `SCIENTIFIC_ARTIFACTS_UNCHANGED`
- Note: `paper_a.tex` was reworded in the pre-existing closeout commit (prose
  and methodological disclaimers only). Its numeric-token sequence is invariant,
  as proven above. This is recorded history, not a change introduced by this
  correction.

## Corrected P4 status
- `P4_CLOSEOUT_BLOCKED_BY_HASH_CONTRACT`
- The prior report contradicted itself (Section 1 recorded the blocked status
  while Section 9 declared "P4: DOCUMENTARILY CLOSED"). Section 9 has been
  corrected to the blocked verdict, and the expected-hash section now records
  the reproduction scope, the contract-source gap, and the explicit
  no-substitution rule.

## P3 status
- DEFERRED. `P3 remains deferred until P4 is validated and P2 has been closed or
  explicitly cleared.`

## Residual issues
- The expected numeric-token hash contract is unresolved: `d8a4cc…` has no
  reproducible definition in the repository. This is the sole item blocking P4.
- `paper_a.pdf` was not recompiled (no LaTeX toolchain available); it does not
  yet reflect the closeout wording edits. Packaging-only, not empirical.
- Manuscript prose was softened in the closeout commit (e.g. "operational
  usefulness" → "event-detection fidelity"; SARIMA order no longer claimed as
  "recovered from the upstream configuration"). Numbers are invariant, but a
  scientific reviewer should confirm the softened claims are acceptable.

## Final verdict
- `P4_CLOSEOUT_BLOCKED_BY_HASH_CONTRACT`
