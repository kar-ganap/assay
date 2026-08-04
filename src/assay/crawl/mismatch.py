"""M2 — is a fast sampler safe for our estimator?

**Why this file exists.** Our loop samples with ``policy.generate()`` and differentiates
``policy.logprobs()``. Today those are one HF forward pass, so the sampling and scoring
distributions are identical *by construction*, and ``config.py`` leans on that in writing:

    with a single epoch the importance ratio is identically 1, so clipping is a no-op no matter
    what epsilon says. **Rung 4 of the ladder is cut for exactly this reason.**

vLLM makes sampler and scorer different *implementations* — different kernels, different attention
paths, different accumulation order — so ``pi_HF(y) / pi_vLLM(y) != 1``. Three things stop holding
at once, and the third is the one that bites:

1. The estimator is no longer unbiased by construction; it needs an importance weight.
2. Rung 4's cut loses its justification **exactly when clipping starts to matter** — and
   ``clipping_is_active`` would still return ``False``, because it gates on ``epochs_per_batch > 1``.
   The guard and the hazard are keyed on different things.
3. Any reproduction run on vLLM tests a *modified* loop, so its result is not about the loop we
   verified in Phase 0.1.

``prime-rl`` logs ``Max Off-Policy`` on every training line, so the field treats this as real rather
than theoretical.

**The measurement.** Sample a batch with vLLM; score the *same token ids* with our existing
``policy.logprobs()`` — never the same *text*, since round-tripping through the tokenizer can change
the ids and would measure re-tokenization instead of the samplers. Then report the **per-token**
discrepancy

    delta_t = log pi_HF(y_t) - log pi_vLLM(y_t)

which is length-independent, and the sequence ratio it implies at the lengths we operate at.

**Why the per-token distribution rather than the sequence ratio directly.** The sequence number is
a statement about our ``max_new_tokens``; the per-token one is a statement about the samplers, and
transfers to every later phase that changes the length. The catch is that extrapolating from one to
the other assumes the per-token errors are independent — so ``independence_ratio`` measures that
assumption instead of making it (see ``mismatch_statistics``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

__all__ = [
    "LENGTHS",
    "OPERATING_LENGTH",
    "RATIO_BAND",
    "RIG_BROKEN_MEAN_ABS",
    "implied_sequence_ratio",
    "mismatch_statistics",
    "mismatch_verdict",
    "token_discrepancies",
]

#: Pre-registered in ``docs/phases/phase-0.3-r0-plan.md`` §M2, before the measurement was built.
#: Inside the band, adopt vLLM; outside, either re-enable rung 4 with genuine importance weights
#: (un-cut on *earned* evidence) or keep HF ``generate`` and re-scope what runs on it.
RATIO_BAND = (0.9, 1.1)

#: Reported at all three so the length-sensitivity is visible rather than implied.
LENGTHS = (64, 512, 1024)

#: The verdict is taken at one **pinned** length, not at whichever of ``LENGTHS`` looks best.
#: 512 is the screen's ``max_new_tokens`` and the operating point for anything Countdown-shaped.
OPERATING_LENGTH = 512

#: Above this mean |delta| the instrument, not vLLM, is the likeliest explanation. Two independent
#: implementations of the same forward pass disagree at the level of float accumulation order —
#: order 1e-3 nats/token, not 1. Half a nat per token is a 65% per-token probability disagreement,
#: which is what scoring *different tokens* looks like.
RIG_BROKEN_MEAN_ABS = 0.5


def token_discrepancies(
    hf_logprobs: Sequence[Sequence[float]], vllm_logprobs: Sequence[Sequence[float]]
) -> list[float]:
    """Flatten two aligned per-token log-prob sequences into ``delta_t = hf - vllm``.

    Both arguments are per-sequence lists of per-token log-probs **for the same token ids in the
    same order**. Alignment is the caller's job and is checked, not assumed: a ragged pair means the
    two scorers saw different sequences, which is the single failure mode that would masquerade as a
    large-but-real mismatch. Zipping to the shorter one would report the worst possible bug as a
    small number, so it raises instead.
    """
    if len(hf_logprobs) != len(vllm_logprobs):
        raise ValueError(
            f"different numbers of sequences: {len(hf_logprobs)} HF vs {len(vllm_logprobs)} vLLM — "
            "the two scorers did not see the same batch"
        )
    deltas: list[float] = []
    for i, (h, v) in enumerate(zip(hf_logprobs, vllm_logprobs, strict=True)):
        if len(h) != len(v):
            raise ValueError(
                f"sequence {i} has different length under the two scorers: {len(h)} vs {len(v)} — "
                "the token ids were not aligned, so any discrepancy is the harness, not vLLM"
            )
        deltas.extend(float(a) - float(b) for a, b in zip(h, v, strict=True))
    return deltas


def mismatch_statistics(
    hf_logprobs: Sequence[Sequence[float]], vllm_logprobs: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """The per-token discrepancy distribution, plus what it implies about extrapolating it.

    ``independence_ratio`` is the load-bearing one and is easy to skip. Extrapolating a per-token
    spread to a 1024-token sequence assumes the errors are independent, giving ``sigma * sqrt(L)``.
    If instead each *sequence* is biased — the same numerical regime for its whole length — the sum
    grows like ``L`` and the extrapolation understates the drift badly. So rather than assume
    independence, measure it:

        independence_ratio = Var(sum over a sequence) / (mean_length * Var(per token))

    which is ~1 for independent errors and grows with the length for correlated ones. It is reported
    beside the ratio so a comfortable per-token number cannot be read as a comfortable sequence
    number when it is not.
    """
    per_seq_sums = [math.fsum(float(a) - float(b) for a, b in zip(h, v, strict=True))
                    for h, v in zip(hf_logprobs, vllm_logprobs, strict=True)]
    deltas = token_discrepancies(hf_logprobs, vllm_logprobs)
    if not deltas:
        raise ValueError("no completion tokens to compare — nothing to measure")

    n = len(deltas)
    mean = math.fsum(deltas) / n
    var = math.fsum((d - mean) ** 2 for d in deltas) / max(1, n - 1)
    ordered = sorted(deltas)
    lengths = [len(h) for h in hf_logprobs]
    mean_length = math.fsum(lengths) / len(lengths)

    # Var over sequences, against what independence would predict for that mean length.
    if len(per_seq_sums) > 1 and var > 0.0 and mean_length > 0.0:
        seq_mean = math.fsum(per_seq_sums) / len(per_seq_sums)
        seq_var = math.fsum((s - seq_mean) ** 2 for s in per_seq_sums) / (len(per_seq_sums) - 1)
        independence_ratio = seq_var / (mean_length * var)
    else:
        independence_ratio = float("nan")

    finite = [d for d in deltas if math.isfinite(d)]
    return {
        "n_tokens": n,
        "n_sequences": len(hf_logprobs),
        "mean_length": mean_length,
        "mean": mean,
        "std": math.sqrt(var),
        "mean_abs": math.fsum(abs(d) for d in finite) / len(finite) if finite else float("nan"),
        "median": _quantile(ordered, 0.5),
        "p01": _quantile(ordered, 0.01),
        "p99": _quantile(ordered, 0.99),
        "max_abs": max((abs(d) for d in finite), default=float("nan")),
        # The field's metric (`prime-rl` logs it every line): the largest deviation of the *ratio*
        # from 1. Asymmetric on purpose — exp(+d) leaves 1 faster than exp(-d) approaches 0.
        "max_off_policy": max((abs(math.exp(d) - 1.0) for d in finite), default=float("nan")),
        "independence_ratio": independence_ratio,
        "n_non_finite": n - len(finite),
    }


def implied_sequence_ratio(stats: dict[str, Any], length: int) -> dict[str, float]:
    """The sequence-level ratio ``pi_HF(y)/pi_vLLM(y)`` implied at ``length`` tokens.

    The sequence log-ratio is the sum of ``length`` per-token discrepancies, so under independence
    it has mean ``L*mu`` and standard deviation ``sigma*sqrt(L)``. Reported as a median and a
    +/-1 sigma interval, because reporting the median alone would call a centred-but-wide
    distribution negligible: zero mean drift and individual sequences far from 1 is precisely the
    situation clipping exists to contain.

    Check ``stats["independence_ratio"]`` before trusting this at lengths far above those measured.
    """
    mu, sigma = stats["mean"], stats["std"]
    centre = length * mu
    spread = sigma * math.sqrt(length)
    return {
        "median": _safe_exp(centre),
        "lo": _safe_exp(centre - spread),
        "hi": _safe_exp(centre + spread),
        "log_median": centre,
        "log_sigma": spread,
    }


def mismatch_verdict(
    stats: dict[str, Any],
    *,
    operating_length: int = OPERATING_LENGTH,
    band: tuple[float, float] = RATIO_BAND,
) -> dict[str, Any]:
    """M2's gate. **Band pre-registered in the phase plan before the measurement was built.**

    ==================  =====================================================================
    verdict             condition
    ==================  =====================================================================
    ``rig_broken``      any non-finite discrepancy, **or** ``mean_abs`` above
                        ``RIG_BROKEN_MEAN_ABS``. Checked *first*: misaligned token ids produce
                        enormous structureless deltas, and calling that ``not_free`` would
                        report a harness bug as a finding about vLLM. Same shape as the screen's
                        ``parse_fail`` branch — separate "the thing under test is bad" from
                        "the instrument is broken", and check the instrument first.
    ``negligible``      the **whole** +/-1 sigma interval at ``operating_length`` lies inside the
                        band. Adopt vLLM and record the number.
    ``not_free``        anything else. Either re-enable rung 4 with genuine importance weights —
                        un-cutting it on *earned* evidence — or keep HF ``generate``.
    ==================  =====================================================================

    The interval, not the median, is tested against the band. A centred median with a wide spread
    means the average sequence is fine and individual sequences are not, which is a real hazard for
    a per-sequence estimator and would otherwise pass.
    """
    by_length = {length: implied_sequence_ratio(stats, length) for length in LENGTHS}
    if operating_length not in by_length:
        by_length[operating_length] = implied_sequence_ratio(stats, operating_length)
    at_length = by_length[operating_length]

    lo_band, hi_band = band
    broken = (
        stats["n_non_finite"] > 0
        or not math.isfinite(stats["mean_abs"])
        or stats["mean_abs"] > RIG_BROKEN_MEAN_ABS
    )
    inside = lo_band <= at_length["lo"] and at_length["hi"] <= hi_band

    if broken:
        verdict = "rig_broken"
    elif inside:
        verdict = "negligible"
    else:
        verdict = "not_free"

    return {
        "verdict": verdict,
        "band": band,
        "operating_length": operating_length,
        "at_length": at_length,
        "by_length": by_length,
        "rig_broken_threshold": RIG_BROKEN_MEAN_ABS,
        "stats": stats,
    }


def _safe_exp(x: float) -> float:
    """``exp`` that saturates instead of raising.

    A rig-broken input — misaligned token ids giving multi-nat discrepancies — extrapolates to
    ``exp(512 * 2.3)``, which raises ``OverflowError``. Raising there would crash *before*
    ``mismatch_verdict`` could classify the run as ``rig_broken``, so the branch written to catch
    exactly that case would never be reached. Saturating to infinity keeps the report honest (the
    implied ratio really is astronomical) and lets the verdict do its job.
    """
    if not math.isfinite(x):
        return float("nan") if math.isnan(x) else (math.inf if x > 0 else 0.0)
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not ordered:
        return float("nan")
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
