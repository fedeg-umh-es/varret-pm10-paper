"""Hashing, provenance manifests and output metadata stamping.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` sections 1 and 14. Every
generated artefact carries the code commit, configuration hash, input-manifest
hash and generation timestamp that produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

__all__ = [
    "PROVENANCE_STATUSES",
    "ArtefactProvenance",
    "RunStamp",
    "git_commit",
    "sha256_bytes",
    "sha256_file",
    "stamp_frame",
    "utc_now",
    "write_json",
]

PROVENANCE_STATUSES: tuple[str, ...] = (
    "VERIFIED_LOCAL_P2",
    "VERIFIED_EXTERNAL_IMMUTABLE_INPUT",
    "MISSING",
    "AMBIGUOUS_PROVENANCE",
    "INCOMPATIBLE_SCHEMA",
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's raw bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(repo_root: str | Path = ".") -> str:
    """Current ``HEAD`` SHA, or ``"UNKNOWN"`` when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "UNKNOWN"


@dataclass(frozen=True)
class ArtefactProvenance:
    """One entry of ``inputs/P2_INPUT_PROVENANCE.json``."""

    logical_role: str
    station_scope: str
    model_scope: str
    source_path: str
    copied_path: str | None
    producer_repository: str
    producer_commit: str
    sha256: str
    size_bytes: int
    schema_summary: str
    provenance_status: str
    allowed_use: str

    def __post_init__(self) -> None:
        if self.provenance_status not in PROVENANCE_STATUSES:
            raise ValueError(
                f"provenance_status {self.provenance_status!r} not in {PROVENANCE_STATUSES}"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RunStamp:
    """Run-level metadata stamped onto every generated artefact."""

    code_commit: str
    config_sha256: str
    input_manifest_sha256: str
    generated_at_utc: str
    decision_id: str
    canon_version: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def stamp_frame(frame: pd.DataFrame, stamp: RunStamp) -> pd.DataFrame:
    """Append the run stamp columns to a results frame."""
    out = frame.copy()
    for name, value in stamp.as_dict().items():
        out[name] = value
    return out


def write_json(path: str | Path, payload: object) -> str:
    """Write ``payload`` as indented JSON and return the file's SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))
