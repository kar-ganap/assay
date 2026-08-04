# Compute options — what the remaining science can actually run on

> Researched 2026-08-04, after `scripts/cost_model.py` put the remaining science at **$147 / $549 /
> $1,654** (low / mid / high) against a **~$17** Modal balance. Prompted by another agent suggesting
> Verda / RunPod / Vast; **no prior provider research existed** — `tasks/spend.md` recorded only
> Modal, Tinker, Prime and the Anthropic API.
>
> **Verification status is marked per row.** ⬤ = read from the vendor's own pricing page on the date
> shown. ☐ = secondary source (comparison blog), **not** load-bearing until checked first-hand —
> `lessons.md` #1.

## 1. The framing: price shopping cannot close this gap

The mid estimate is **32× the balance**. The cheapest credible provider is **~2–5× cheaper than
Modal**, not 30×. So the three levers, ranked by how much they actually move:

| lever | size | status |
|---|---|---|
| **Free / credit programs** | **up to the whole gap** | Prime's free queue alone is worth ~$424 of the mid estimate (line 2.1). Currently a single point of failure. |
| **Scope** | ~2–4× | Grid size (8 vs 12), seed count, 1.5B vs 2B — all pre-registered choices with money attached. |
| **Price per hour** | ~2–5× | Real, and the smallest of the three. |

**Do not let the third lever's visibility crowd out the first two.** A provider table is easy to
produce and feels like progress; it closes maybe a fifth of the gap.

## 2. What our workload actually needs

Constraints from measured runs, not from a spec sheet:

- **VRAM 24–48 GB.** Phase 0.1 peaked **13.5–14.5 GB** training 1B at 64 tokens (LoRA + KL
  reference). M1/M3 peaked **5.9 / 9.8 GB** for 1.5B / 3B inference at 512 tokens. At ~1.7B with
  ~1200-token episodes, 24 GB is tight and 48 GB is comfortable. **80 GB is not needed.**
- **Run length 0.7 h (measured, 1B/64tok) to ~17 h** (mid scenario, 1.7B/1200tok).
- **Single GPU.** Nothing in the plan needs multi-node.
- **No mid-run checkpointing.** See §4 — this is the finding that reshapes the price table.

## 3. Providers

### ⬤ RunPod — `runpod.io/pricing`, read 2026-08-04

| GPU | VRAM | Community | Secure |
|---|---|---|---|
| RTX A5000 | 24 GB | **$0.16** | $0.27 |
| RTX 3090 | 24 GB | $0.22 | $0.46 |
| RTX 4090 | 24 GB | $0.34 | $0.69 |
| L4 | 24 GB | $0.44 | **$0.39** |
| **RTX A6000** | **48 GB** | **$0.33** | $0.49 |
| A40 | 48 GB | $0.35 | $0.44 |
| L40S | 48 GB | $0.79 | $0.99 |
| A100 PCIe | 80 GB | $1.19 | $1.39 |

**The stand-out for us is RTX A6000 at $0.33/h Community — 48 GB for less than Modal's 24 GB L4
($0.799).** That is **2.4× cheaper with 2× the memory**, which removes the tier-escalation risk the
cost model had to assume. A5000 at $0.16 is 5× cheaper than Modal L4 if 24 GB suffices.

### ☐ Vast.ai — secondary sources only, **unverified**

Peer-to-peer marketplace; hosts set prices. Reported RTX 4090 **$0.29–0.50/h**, typical ~$0.39.
Interruptible is the cheap tier. Needs a first-hand check before use.

### ☐ Verda (formerly DataCrunch) — partially verified

Finnish, founded 2018, 100% renewable. Reported **from $0.17/GPU-h**, 13 GPU types including
A100 40/80 GB, H100, H200, B200. Hourly billing, no minimum; reserved terms discounted.
**Prices not read first-hand — check `verda.com/pricing` before committing.**

### Already in use

| | rate | note |
|---|---|---|
| **Prime Sprints free queue** | **$0** | ⬤ Proven: 3 hosted GRPO runs in Phase 0.2, one to eval 0.9980. Requires a PUBLIC env + CI SUCCESS + the sprint tag. |
| Modal L4 / A100-40 | $0.799 / $2.099 | ⬤ verified 2026-08-01 |
| Tinker Qwen3-8B | $0.40/M tok | ☐ flagged for re-verification since 2026-07-11 |

## 4. The catch that reprices the cheap tier

**Interruptible / spot is where the headline savings are, and we cannot use it today.**

Our loop has **no mid-run checkpointing** — verified 2026-08-04 by inspection: `loop.py` and
`modal_app.py` persist artifacts *at completion*, and `runlog.write_manifest` writes only at start.
The comment in `modal_app.py` is explicit that persistence exists so a *dropped client* does not lose
results; nothing survives the *container* dying.

So a preempted spot instance loses the entire run. And the exposure scales with the thing the cost
model says is already dominant: at 0.74 h/run preemption is a nuisance; at 17 h/run it is close to a
certainty over a 12-run grid.

**Consequence:** cheap interruptible pricing carries an engineering prerequisite — checkpoint every N
steps plus resume-from-checkpoint. That is a real piece of work on `loop.py`, which is the user's
file under §7. **Worth doing only if the grid runs long**, which is exactly what Phase 1.1 will tell
us. Until then, price on-demand rates, not spot.

## 5. Credit programs — the lever that can actually close the gap

Leads only; **none applied to, none verified for eligibility.**

| programme | who | note |
|---|---|---|
| **NSF ACCESS** | US-based researchers | Broadest free-compute path; no existing NSF grant required. Maximize-allocation window was **2026-06-15 → 07-31**, awards from 10-01. **That window has closed** — check for the next. |
| **NAIRR Pilot** | US | Cloud credits + HPC + datasets; rolling allocations at `nairrpilot.org/opportunities/allocations`. |
| **Nebius Research Grants** | academic-leaning | Accepting for the 2026–27 academic year, reviews through summer 2026. |
| **fal Research Grants** | open — **no degree required**, open-source focus | Best fit for an independent portfolio project. |
| **CloudRift AI Grant** | open — indie devs, students, hobbyists | Prioritises open-source tools, papers, educational content. |
| **Thunder Compute's roundup** | — | Aggregates ~15 programmes; a starting index, not a source. |

**Why this project is unusually well-positioned.** Most grant programmes want open-source output and
public artifacts. `assay` already ships a PUBLIC Hub environment (`gkartik/assay-add3digit`), commits
its raw data, and is pre-registered — and its *negative* results (R0's retirement, M2's `not_free`)
are exactly the reproducibility-flavoured contribution these programmes say they prioritise. The
free-tier requirement that felt like a tax in Phase 0.2 (publish before the paper) is the same
property that qualifies it here.

## 6. Recommendation

1. **Do not switch providers yet.** The cost model's dominant uncertainty is `bisect`'s episode
   length, unmeasurable until Phase 1.1. Optimising $/h against a 3× band is premature.
2. **Phase 0.4 (R1) stays where it is** — free queue, ~$0, and $17 covers it if not.
3. **At Phase 1.1, measure a real `bisect` episode first.** That single number collapses an $875 span
   and turns this page from speculation into a decision.
4. **Then move paid work to RunPod Community RTX A6000 ($0.33/h, 48 GB)** if the numbers hold —
   2.4× cheaper than Modal L4 with 2× the memory. Verify first-hand; prices move.
5. **Apply to fal and CloudRift now**, in parallel — both are open to independents, both weight
   open-source output, and neither blocks on Phase 1.1. NSF ACCESS's window has closed; check the
   next one.
6. **Treat Prime's free queue as a single point of failure** and say so in the paper's limitations.
   The sprint that introduced it closed ~2026-06-20 with no announced successor (`CLAUDE.md` §15),
   and ~$424 of the mid estimate rests on it.

## Sources

- [RunPod pricing](https://www.runpod.io/pricing) — ⬤ read 2026-08-04
- [Verda](https://verda.com/pricing) · [Verda products](https://verda.com/products) — ☐
- [RunPod vs Vast.ai 2026 (Spheron)](https://www.spheron.network/blog/runpod-vs-vastai-2026/) · [Vast.ai vs RunPod pricing (Medium)](https://medium.com/@velinxs/vast-ai-vs-runpod-pricing-in-2026-which-gpu-cloud-is-cheaper-bd4104aa591b) · [Vast.ai RTX 4090 price (SynpixCloud)](https://www.synpixcloud.com/blog/vast-ai-vs-runpod-rtx-4090-pricing) — ☐
- [Free cloud GPU credits 2026 (Thunder Compute)](https://www.thundercompute.com/blog/free-cloud-gpu-credits) · [AI compute grants guide (Granted AI)](https://grantedai.com/blog/ai-compute-grants-gpu-credits-guide) — ☐
- [Nebius research grants](https://nebius.com/nebius-research-grants) · [CloudRift AI Grant](https://www.cloudrift.ai/ai-grant) · [GPU-Grants list](https://github.com/eric-prog/GPU-Grants) — ☐
