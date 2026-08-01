# P4 reproducible numerical-invariance contract

## 1. Objective

This documentary control verifies that the ordered sequence of numeric tokens
in the scientific portion of `paper_a.tex` is invariant between the canonical
producer/base commit and the documentary closeout. It also checks the subsequent
validation-correction commit as an additional control.

## 2. Scope

This contract is a **documentary contract correction**. It does not validate
the scientific correctness of individual numbers, compare non-numeric prose,
replace the provenance manifest, or regenerate any result.

The historical expected hash
`d8a4cc5dcc24ebc7fdb942a597bbd9378e3a551c5d436544415a74e002b56e1f`
has no documented computational contract and is retired as an obsolete
documentary control. It is preserved here only for provenance. This document
does not claim to recover or reproduce that original contract.

## 3. File

`paper_a.tex`

## 4. Commits

- canonical producer/base:
  `f57f076078760af8a88bd87815fdf94ab0064fa3`;
- documentary closeout:
  `390685f1f1312954ee67513f3e0db11b2670e7f9`;
- validation correction, additional control:
  `11f8c88e5e3860eee75a333c83382127c360a930`.

The invariance decision is based on the base and documentary-closeout commits.
The validation-correction commit is included to confirm that it did not modify
the manuscript numeric-token sequence.

## 5. Text markers and range

- start marker, included: literal `\begin{abstract}`;
- end marker, excluded: literal `\section*{Data and Code Availability}`.

For each Git blob, select from the first character of the start marker through
the character immediately before the end marker. Missing markers, reversed
order or decoding failure are contract failures; markers must not be adapted
silently.

## 6. Tokenizer

Python regular expression:

```python
r"[+-]?(?:\d+\.?\d*|\.\d+)"
```

Tokens retain their matched spelling and their left-to-right textual order. No
numeric conversion, whitespace normalization, rounding or deduplication is
performed.

## 7. Serialization

```python
",".join(tokens)
```

The separator is one ASCII comma with no spaces. There is no leading comma,
trailing comma or terminal newline in the hashed serialization.

## 8. Encoding

UTF-8. The Git blob is decoded as UTF-8. The comma-joined token string is
encoded as UTF-8 before hashing.

## 9. Hash algorithm

SHA-256, emitted as a 64-character lowercase hexadecimal digest.

## 10. Expected token count

187 tokens for every specified commit.

## 11. Expected digest

```text
cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3
```

## 12. Complete reproduction script

The script may be copied to any temporary location and executed unchanged. It
accepts an optional repository path; otherwise it uses the canonical P4 path.
It reads Git blobs only and does not modify the repository.

```python
import argparse
import hashlib
import re
import subprocess
from pathlib import Path

CANONICAL_REPO = Path(
    "/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/"
    "03_Investigacion/repos/varret-pm10-paper"
)

COMMITS = {
    "base": "f57f076078760af8a88bd87815fdf94ab0064fa3",
    "documentary_closeout": "390685f1f1312954ee67513f3e0db11b2670e7f9",
    "validation_correction": "11f8c88e5e3860eee75a333c83382127c360a930",
}

EXPECTED_TOKEN_COUNT = 187
EXPECTED_HASH = (
    "cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3"
)
PATTERN = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)")
START_MARKER = r"\begin{abstract}"
END_MARKER = r"\section*{Data and Code Availability}"


def extract(repo, label, commit):
    source = subprocess.check_output(
        ["git", "show", f"{commit}:paper_a.tex"], cwd=repo
    ).decode("utf-8")

    start = source.find(START_MARKER)
    end = source.find(END_MARKER)
    if start < 0:
        raise RuntimeError(f"{label}: missing start marker")
    if end < 0:
        raise RuntimeError(f"{label}: missing end marker")
    if end <= start:
        raise RuntimeError(f"{label}: invalid marker order")

    tokens = PATTERN.findall(source[start:end])
    digest = hashlib.sha256(
        ",".join(tokens).encode("utf-8")
    ).hexdigest()
    return tokens, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "repo", nargs="?", type=Path, default=CANONICAL_REPO
    )
    args = parser.parse_args()
    repo = args.repo.resolve()

    results = {
        label: extract(repo, label, commit)
        for label, commit in COMMITS.items()
    }

    base_tokens = results["base"][0]
    passed = True
    for label, (tokens, digest) in results.items():
        same_tokens = tokens == base_tokens
        count_ok = len(tokens) == EXPECTED_TOKEN_COUNT
        hash_ok = digest == EXPECTED_HASH
        print(
            f"{label}: tokens={len(tokens)} sha256={digest} "
            f"same_tokens_as_base={same_tokens}"
        )
        passed = passed and same_tokens and count_ok and hash_ok

    print(f"EXPECTED_TOKEN_COUNT={EXPECTED_TOKEN_COUNT}")
    print(f"EXPECTED_HASH={EXPECTED_HASH}")
    print(f"CONTRACT_REPRODUCED={passed}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

Example after copying the block to a temporary file:

```bash
python3 /tmp/verify_p4_numerical_invariance.py
```

or with an explicit repository:

```bash
python3 /tmp/verify_p4_numerical_invariance.py \
  "/path/to/varret-pm10-paper"
```

No independent executable script is added to the repository by this decision.

## 13. Reproduced result

```text
base: tokens=187 sha256=cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3
documentary_closeout: tokens=187 sha256=cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3
validation_correction: tokens=187 sha256=cfd238e5832e0764f168c4633e2755c7f5ea2432cfd809c3b58ebb00da7dbbc3
documentary_closeout: same_tokens_as_base=True
validation_correction: same_tokens_as_base=True
CONTRACT_REPRODUCED=True
```

## 14. Limitations

- Numeric-token invariance does not prove scientific correctness.
- Non-numeric prose may differ between commits.
- The selected range excludes Data and Code Availability and all following
  material.
- Regex matching is lexical: numbers embedded in commands, citations or labels
  inside the selected range are tokens if they match the expression.
- The original method that yielded `d8a4cc…` remains unknown.

## 15. Relationship to the provenance manifest

`docs/p4_canonical_provenance_manifest.json` remains unchanged. Its 22 entries
cover data, the software environment, producer scripts and empirical artifacts.
This numerical-invariance contract complements that manifest by checking the
manuscript numeric-token sequence; it neither replaces nor modifies artifact
hashes.

At contract adoption, independent verification found zero missing files, zero
SHA-256 mismatches and zero size mismatches across all 22 manifest entries.

## 16. Canonical decision

Decision ID: `2026-08-01-p4-numerical-invariance-hash-contract`.

Canonical vault note:
`P4_Ghost_Skill_Dynamic_Fidelity/P4_Numerical_Invariance_Hash_Contract_Decision.md`.

The decision classifies the change as a documentary contract correction and
sets the final P4 documentary status to `P4_DOCUMENTARY_CLOSEOUT_VALIDATED`.
