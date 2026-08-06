"""P2 — Finite Linear Memory Skill Decomposition.

Paired decomposition of persistence-relative MSE reduction along the canonical
reference ladder::

    P -> AR(1) -> AR(p) -> M

Canon
-----
``docs/canon/P2_PROJECT_CANON.md`` (v2.0) is the scientific source of truth.
``docs/canon/P2_PAIRED_DECOMPOSITION_CONTRACT.md`` (v1.0) is the implementation
contract this package implements. Nothing in this package may relax either
document; where a contract clause could not be satisfied mechanically the
deviation is recorded in ``docs/canon/PM10_RESEARCH_DECISION_LOG.md``.

Hard invariants enforced by this package
----------------------------------------
1. Daily series are reindexed to the complete daily calendar; missing days stay
   ``NaN`` and are never dropped before lag construction.
2. ``mu_hat`` and ``gamma_hat(k)`` are estimated train-only, from
   calendar-aligned observed pairs at the true lag ``k``.
3. AR(1) is the ``p = 1`` case of the same *direct* horizon-specific
   projection used for AR(p). Recursive iteration is not implemented as a
   primary reference.
4. ``Gamma_p`` is never regularised, pseudo-inverted or repaired. Invalid
   systems fail closed and are recorded as invalid.
5. All compared losses live on a validated one-to-one paired support.
6. The additive identity ``Delta_total == Delta_AR1 + Delta_mem + Delta_res``
   is checked numerically wherever components are produced.
7. Model selection is a declared fixed list; selecting by evaluated test loss
   raises.
"""

from __future__ import annotations

__all__ = [
    "autocovariance",
    "bootstrap",
    "calendar",
    "decomposition",
    "diagnostics",
    "gate",
    "linear_references",
    "pairing",
    "provenance",
    "synthetic",
]

__version__ = "1.0.0"
DECISION_ID = "2026-08-06-p2-finite-linear-memory-skill-decomposition"
CANON_VERSION = "2.0"
