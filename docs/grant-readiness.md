# Grant readiness — what a strong compute proposal needs, and what we are missing

> Written 2026-08-04, after `scripts/cost_model.py` put the remaining science at ~$147–1,654 against
> a ~$17 balance. Targets: **fal Research Grants** and **CloudRift AI Grant** (both open to
> independents, no degree requirement, both weighting open-source output), plus NAIRR/Nebius and the
> next NSF ACCESS window. Full provider and programme list: `docs/compute-options.md`.

## 1. The reviewer asks two questions. We can answer one of them very well.

**"Will this person use GPUs well?"** — Our evidence here is unusually strong, and it is the part
most applications cannot substantiate at all.

**"Will something valuable come out?"** — **Untested.** H1 (does `assay_score` predict the post-GRPO
proxy–true gap?) has no evidence. Everything shipped so far is Crawl: infrastructure and de-risking.

The proposal must lead with the first and be candid about the second, framing the ask as *funding the
experiment that could falsify the hypothesis* — which is precisely what these programmes say they
want to fund, and is a stronger position than promising a result.

## 2. Assets — what we can point at today

| asset | why a reviewer cares |
|---|---|
| **A PUBLIC Hub environment** — `gkartik/assay-add3digit` v0.1.1, CI SUCCESS | Not a promise of open-source output; an existing one. |
| **An independent trainer trained on it** — `prime-rl`, eval **0.9980**, step-1 eval **0.5879** landing inside our own measured band (0.571 ± 0.019) | Third-party validation that the artifact works. Rare in an application. |
| **GRPO written from scratch + 7 ablations**, n=3 seeds on headline arms, 4 breakage signatures | Demonstrates the mechanics are understood, not wrapped. |
| **Pre-registration with bands and both failure branches**, written before every run | Distinguishes us from applicants who will report whatever they find. |
| **Three shipped negative results** — R0 retired, M2 `not_free`, M3's inert axis | See §3. This is the strongest single item. |
| **`scripts/cost_model.py`** — the compute ask, bottom-up from measured runs, with a sensitivity table | See §4. |
| **20-page technical tutorial** (`docs/tutorial/reinforce-to-grpo.tex`) | Matches CloudRift's stated "educational content" priority verbatim. |
| **350 tests, ruff + mypy --strict, committed raw data, reproducible figures** | Engineering credibility. |

### Findings that are genuinely interesting to a reviewer, not just to us

- **A screening statistic that transferred out-of-sample on first use.** `dead = p⁸+(1−p)⁸` predicted
  **0.851** before any GPU was booked; measured **0.845** — different model, task family and scale.
- **`prime-rl`'s headline `Reward` is computed over a filtered subset**, so it excludes the policy's
  successes with a bias that *grows as the policy improves*. Verified by integrality, not inferred.
  This is the project's own thesis appearing inside a production RL stack.
- **Length normalisation breaks `E[∇log π] = 0`** — reached by measurement, two independent lines
  toward Dr. GRPO.
- **The importance ratio under vLLM stays unbiased in expectation and becomes useless per sequence**
  (`μ = −σ²/2`, agreeing to 12%), independently corroborated by `verl`'s shipped `rollout_correction`
  parameters.

## 3. The unusual thing to lead with: pre-registered nulls

Most applications promise positive results. Ours can point at **three pre-registered questions that
returned "no", each with the band written down first, each shipped rather than buried**:

- **R0 retired** — and not for budget. The reproduction target publishes no number, so the ledger's
  required delta had nothing to compute against.
- **M2 `not_free`** — which *un-cut* a rung of our own ladder rather than confirming a convenient
  design decision.
- **M3** — where the pre-registered negative branch was confirmed **for the wrong reason**, and we
  said so.

Cost of learning all three: **$2.21**, against a $10 line, with $7.79 returned unspent.

That is the single most credible signal available that credits would not be burned on a foregone
conclusion. **Quantify it in the application**: dollars spent per decision retired.

## 4. The compute ask should be the sensitivity table, not a number

Most applications state a GPU-hour figure with no derivation. Ours can state:

- a **measured anchor** — one 200-step GRPO run at 1B/64tok/L4 = 0.74 h = $0.59;
- a **range** — $147 / $549 / $1,654 (low/mid/high);
- **which assumption the range is made of** — `bisect_tokens_per_episode`, span **$875**;
- and **what we will measure first to collapse it** — one real `bisect` episode at Phase 1.1.

**Ask for a staged allocation against that**: a small tranche to measure the episode and tighten the
estimate, then the confirmatory tranche. That is a more fundable shape than a lump ask, and it is
honest about a 3× band rather than hiding it behind false precision.

## 5. Gaps — ordered by proposal value per unit of effort

### Tier 1 — do before applying

1. ~~**Make the repository public.**~~ **DONE 2026-08-04** — `github.com/kar-ganap/assay` is public.
   §6's decision is closed; its reasoning is kept below as the record of why.

2. **Run R1 (Phase 0.4). ~$0–2.** *In progress — `docs/phases/phase-0.4-r1-plan.md`.* **Stronger
   than assessed here.** Prime publishes a reproducible **steps-to-saturation** table, so R1 carries
   a pre-registered *prediction* (R1-P) that a hack's step-0 base rate predicts its onset — **H2 in
   miniature, on someone else's published curve, for ~$0.** That is the thesis evidence §1 says the
   case is missing, obtainable before we own an environment. *The highest-value experiment available for the proposal*, because
   **it de-risks the thing the grant would fund.** R1 asks whether a 1B model reward-hacks inside a
   100-step budget. If yes, the Run-stage experiment is demonstrably feasible and the ask becomes
   "fund a design we have shown works at small scale." If no, we have a problem credits cannot
   fix — and we would want to know that before asking for them, not after.

   > **⚠ SUPERSEDED 2026-08-06 — R1 has run and this is no longer the ask.** R1-P came back
   > **unresolved**: the 95% interval for the discriminating pair is [−7.84, +9.85] steps, which
   > excludes Prime's −26-step ordering effect and does not exclude zero. Worse for this pitch, R1
   > ran the *easiest* instance of the mechanism — same model class forecasting itself, same
   > environment, same exploit, no capability gap — so the null bounds H2's difficulty from below
   > rather than evidencing it. **What R1 does buy** is a decisive reachability result (15/15
   > saturate at 1B for `$0`) and a genuinely novel negative: the admission screen's own lower bound
   > is disconfirmed, with a measured 42–68% false-negative rate on reachable exploits. The ask
   > changes from *"validate the screen"* to **"fix and validate the screen"** — which is a smaller
   > claim and a more honest one. Rewrite §1 and §5 before this document is sent anywhere.

3. **A short public writeup of Crawl.** Working title: *"Three pre-registered screens, $2.21, and
   what they killed."* Publishable today, demonstrates output, and matches CloudRift's "research
   papers / educational content" criterion. The retros are already 80% of the raw material.

### Tier 2 — needed for a *strong* proposal, no GPU required

4. **Phase 0.5 literature gate.** `docs/related-work.md` is stamped **UNVERIFIED — every row from a
   single LLM-assisted pass**. A reviewer will ask *"how is this not the fuzzing-verifiers paper?"*
   and today the answer is machine-generated. This is free, and it is the difference between a claim
   and a defensible one. Given this project has already had two scaffold claims falsified by
   measurement, shipping an unverified differentiation table into a funding application is the
   highest-embarrassment risk on the list.

5. **A `bisect` design sketch** — enough to make the substrate concrete in the proposal. Not the
   implementation.

### Explicitly NOT blockers — do not wait for these

- The paper. H1 results. A built `bisect`. Those are what the credits buy; waiting for them inverts
  the purpose of the application.

## 6. The one decision that is genuinely the user's: going public

**For:**
- Both programmes weight open-source output; this is the cheapest large improvement available.
- **A public, git-timestamped pre-registration establishes priority.** That is protection, not
  exposure — it is the standard argument for pre-registration and it applies here.
- **The marginal scoop risk is smaller than it looks.** Phase 0.2's free tier already required a
  PUBLIC environment whose README states the hypotheses. The thesis is *already partly public*.
- What is public today is **de-risking work**, not the contested claim. `bisect` — the substrate that
  would actually be scooped — does not exist yet.

**Against:**
- `CLAUDE.md` §15: *"the window is narrow and closing"*; if the field publishes the zero-step
  prediction question before Run lands, the fallback is the η leg. Publishing the framing accelerates
  anyone inclined to take it.

**Recommendation: go public. — ADOPTED 2026-08-04.** The priority-by-timestamp argument is strong, the hypotheses are
already out via the Hub README, and a funding application that cannot show its work is materially
weaker. If the concern bites, the middle path is publishing the repo while keeping
`docs/conceptual.md`'s sharpest framing in a private branch until Run lands — but that is a fiddly
half-measure and it would show.

## 7. Weaknesses to state rather than let a reviewer find

- **Solo and unaffiliated.** Mitigated only by shipped artifacts, which is why §5's Tier 1 matters.
- **The central hypothesis is untested.** Say it plainly and frame the ask around falsification.
- **The substrate does not exist.** Say that the first measured `bisect` episode is the first
  milestone, and that the cost model already names it as the dominant uncertainty.
- **Prime's free queue is a single point of failure** — ~$424 of the mid estimate rests on it, and the
  sprint that introduced it closed ~2026-06-20 with no successor. A credit award is partly a hedge
  against exactly this, which is worth saying out loud.

## 8. Suggested sequence

```
now      -> decide on going public (§6)
week 1   -> R1 (~$0-2)  +  Phase 0.5 literature gate (free)   [parallel]
week 2   -> Crawl writeup, public
then     -> apply to fal + CloudRift with R1's result, a verified differentiation
            table, a public repo, and a staged compute ask
```

R1 and the literature gate are both already on the plan as Phase 0.4 and 0.5. **Nothing here asks the
project to deviate from its own roadmap** — it asks that the application wait for the two things the
roadmap produces next, because both materially change what can be claimed.
