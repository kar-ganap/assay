"""Stage-0 (Crawl) code — Phase 0.1's calibration harness and toy task families.

**This code is disposable by design.** Phase 0.1's non-goals include performance, generality and
reusability; Phase 0.2 rebuilds the same task properly under the ``verifiers`` spec. It lives under
``src/`` only so ``make check`` covers it — not because it is library surface.

Deliberately *not* generalised against ``assay.screen``'s ``p_hack@64``. The two are the same
measurement *shape* (sample k from the base policy, apply an admission band) over different objects
(task difficulty here, exploit reachability there). Phase 1.4 should factor the primitive out with
**two** concrete instances in hand rather than one guess.
"""

from __future__ import annotations

__all__ = ["advantage", "calibrate", "rewards", "sampling", "tasks"]
