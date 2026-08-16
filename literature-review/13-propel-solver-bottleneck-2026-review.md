# Review: Breaking the Solver Bottleneck — Training Task Generators at the Learnable Frontier (PROPEL)

**Paper:** Lorenz Wolf¹, Connor Watts², Roger Creus Castanyer¹, Geoffrey Bradway¹, Maxwill Lin¹, Augustine N. Mavor-Parker¹, Matthew Daborn-Sargent¹. **¹Vmax, ²Goodfire AI.** arXiv:2606.18284v1 [cs.LG], 2026-06-10 (dated June 18, 2026). *(Reading-list #13 — clusters K2 "closest prior art / scoop" + K3 "the battery's axes".)*
**Reviewed:** 2026-08-06
**Reviewer:** Claude (structured review for interactive discussion)
**PDF:** `https://arxiv.org/pdf/2606.18284v1` · **ID ⬤-verified 2026-08-06** (title/authors/affiliation/date read from the PDF itself).
**Added under the recursive novelty-perimeter rule** — surfaced 2026-08-06 while researching Vmax. Not on the original 12-paper list; appended rather than renumbered.

> **This is the closest methodological neighbour `assay` has, and it was not on the reading list.** PROPEL trains a cheap probe to predict what an expensive solver rollout would say, then uses it as the RL reward for a *task generator*. That is `assay`'s bet — a cheap forecaster standing in for an expensive training outcome — executed first, in a neighbouring problem. **It does not scoop `assay`, and it materially strengthens `assay`'s premise, but it takes the A3 axis outright and forces H3 from a hypothesis into a differentiation necessity.** Full analysis in **Connection to Project**.

> **Scope of this review (read first).** Drafted 2026-08-06 by an out-of-project session doing
> interview preparation, at **Process step 1 only**. Everything below is a *finding* or a *proposal
> pending Process steps 2–5*. The `Where … tightens our design` section in particular is a list of
> candidate implications for the project owner to adjudicate — **no decision, gate, pre-registration,
> axis definition, or roadmap item has been changed on the basis of this review.**

---

## Background

RLVR is gated by task supply. As policies improve, fixed distributions saturate; hand curation cannot keep pace; naive synthetic generation yields tasks that are trivial, impossible, or ill-posed. The natural fix is to train a *generator* with RL, rewarding well-formed tasks of appropriate difficulty. The obstacle is measurement cost: scoring a candidate task requires running the solver, and on SWE-bench-style tasks a single rollout takes tens of minutes. A reliable difficulty signal needs many rollouts per candidate. Solver-in-the-loop generator RL is therefore intractable at the scale that matters.

PROPEL — *Probe Rewards for Optimizing Problems at the Edge of Learning* — amortises the solver. A small activation probe is trained **once** on an offline corpus of (task, solver-outcome) pairs, reading hidden states from a **frozen reference generator**. During RL the probe replaces live rollouts, collapsing per-step reward cost from *k* solver trials to one forward pass. It builds on RLFR (Prasad et al., 2026), extending it to multi-turn agent trajectories and adding explicit treatment of fixed-probe mode collapse.

For `assay` this matters for a reason that has nothing to do with task generation: **it is the same epistemic move.** `assay` claims a battery of inference-only probes forecasts what GRPO will do to a policy. PROPEL claims an activation probe forecasts what a solver would do to a task. Same genre, adjacent target, published first, and *it works*.

---

## Key Ideas

### 1. The utility signal is a pass-rate band — and that band is not centred on 0.5
For solver *S* and *K* attempts, with mean solve rate μ_S(x):

**U_S(x) = 𝟙[a ≤ μ_S(x) ≤ b]**, with **K=8, a=1/8, b=3/8**.

A task solved 1–3 times of 8 is positive; 0/8 is too hard; 4–8/8 is saturated. The band is taken from Self-play SWE-RL (Wei et al., 2025), whose injection reward "peaks for solver pass rates near the middle of the 0–1 range." For SWE they use K=3 and the band [1/3, 2/3]. *For us:* **this is A3, owned and operationalised.** Note the disagreement `assay` must resolve — `docs/conceptual.md` defines A3 as *distance from p=0.5*; PROPEL targets **[0.125, 0.375]**, i.e. deliberately skewed hard. One of these is wrong, or they optimise different things (advantage variance vs. headroom-per-step). Reconcile before the pre-registration locks.

### 2. Probe reads a *frozen* reference, closing one Goodhart channel by construction
Activations come from `π_ref`, a frozen copy of the base generator — **not** from the policy `π_θ` being trained. The paper is explicit about why: *"the policy cannot improve its reward by shifting the activations the probe reads."* Reward is a function of the rendered task, not of the policy's internals. *For us:* an elegant, cheap structural defence worth naming in `assay`'s design vocabulary. It closes activation-hacking. **It does not close text-level hacking** — the generator can still find task *text* the probe scores high and the solver does not actually sit at frontier on. That residual is where §Challenge C1 lives.

### 3. RVP — reward variance under policy — a pre-RL screening statistic that beats accuracy
Probe accuracy and calibration turned out insufficient: *"A probe that classifies held-out examples well but assigns almost the same reward to every task sampled from the current policy results in almost no signal during RL."* They define **RVP(f_φ, π_ref) = Var_{x∼π_ref}[R(x, φ)]**, measured before RL on a fixed n=512 base-policy sample, and use it as a selection diagnostic. Empirically: *"Downstream RL gain tracks RVP more closely than validation balanced accuracy"* — AZR (highest RVP, 0.145) shows the largest gain; math (lowest, 0.008) the smallest.

*For us:* **this is `assay`'s thesis, validated in a neighbouring setting, by someone else, before us.** A cheap pre-training measurement that predicts downstream RL outcome better than the obvious quality metric. It is also the *same construct* as Phase 0.1's `p⁸+(1−p)⁸`: a group whose members all pass or all fail carries zero advantage, hence no gradient. RVP measures that degeneracy empirically on the generator side; `p⁸+(1−p)⁸` derives it in closed form on the solver side. **Cite RVP; position ours as the analytic complement.**

### 4. Mode collapse under fixed-probe optimisation — characterised, quantified, partially mitigated
This is the paper's second contribution and its most useful result for us. Optimising a single fixed probe produces **semantic concentration**: on AZR/3B, ~74% of generated tasks collapse onto the single topic `sorting_order`. From Table 4:

| AZR, 3B solver | Utility ↑ | Valid ↑ | Self-BLEU ↓ | Distinct-3 ↑ | Top-topic ↓ |
|---|---|---|---|---|---|
| Base | 0.1009 | 0.6654 | 0.711 | 0.376 | **0.318** |
| PROPEL | **0.1995** | **0.9085** | 0.809 | 0.266 | **0.735** |
| PROPEL + WCO | 0.1725 | 0.8346 | 0.714 | 0.360 | 0.685 |

Utility nearly doubles; top-topic concentration also more than doubles. Mitigations: **worst-case optimisation** over a two-probe ensemble (reward = min_j ℓ_φj) and **adversarial probe co-evolution** (probe-high-but-solver-failed generations mined as negatives for the next probe). WCO on 7B moves top-topic 0.67 → 0.54 while retaining gains; on 3B only 0.74 → 0.69. *For us:* a live, measured instance of proxy-optimisation degradation, with the mitigation ladder already tried and reported as **partial**.

### 5. KL is a collapse *driver*, not only a brake
The β ablation shows a clean utility/diversity trade-off: low β gives high utility and high topic collapse. *"At KL 0.02 the training run on math had fully collapsed."* Their related-work explicitly notes KL regularisation "itself capable of driving collapse rather than preventing it (GX-Chen et al., 2025)." *For us:* the same contested KL role `waterline` resolved first-hand at #6 (model-dependent; KL is *a* driver, not the sole one). The two projects now share a literature thread.

### 6. The proxy beats the true signal — and costs less
Solver-in-the-loop (SIL) RL on AZR: 53.7k online solver trials, utility 14.04 (+3.95 lift). PROPEL: 22.6k offline trials, zero online, utility **19.95 ± 0.62 (+9.86 lift)**. The amortised proxy *outperforms* direct optimisation against ground truth at less than half the trial budget. Plausibly a variance-reduction effect — the probe is a smoothed, dense signal where μ_S at K=8 is noisy and sparse. *For us:* directly relevant to `assay`'s framing. Cheap ≠ worse; a well-chosen proxy can dominate the expensive signal it replaces. But see C2 — nobody checked *which* direction the smoothing biases.

### 7. Cold cross-family probe transfer
Probe + reference model held fixed at Qwen3.5-4B; only the trainable policy swaps. Mistral-7B-Instruct-v0.3: validity 39.6% → 67.4%, utility 5.2% → 9.6% on AZR, and 7.4% → 26.4% on math, with **no per-family tuning**. Phi-3.5-mini transfers more modestly. *For us:* evidence the encoded difficulty signal is a property of the *task*, not the generator — which is the optimistic reading for whether `assay`'s battery transfers across environments.

---

## Results

### For a smart high-schooler
To train an AI to get better at problems, you need problems that are the right level — not so easy it always wins, not so hard it never does. But checking whether a problem is the right level means making the AI try it eight times, and for real software-bug problems each try takes tens of minutes. Too slow to use while teaching a second AI to *write* problems. So: run the slow check once on a big pile of problems, and train a tiny "difficulty guesser" that reads the problem-writer's internal brain-state and predicts the answer. Then use the guesser instead. It works — roughly twice as many good-difficulty problems — and it's cheaper *and better* than doing the slow thing. The catch: the problem-writer figures out one kind of problem the guesser loves and writes it over and over. Three-quarters of its output became sorting problems. They tried using two guessers and taking the harsher score, which helped some, not enough.

### For an undergraduate
PROPEL trains a task generator with GRPO where the reward is an activation probe predicting whether a target solver's pass-rate lands in [1/8, 3/8] at K=8. The probe is a small MLP over pooled layer-11 hidden states of a *frozen* reference generator, trained offline on a few thousand labelled (task, solve-count) pairs. Reward is hard-gated: invalid generations get a fixed penalty, valid ones get the raw probe logit. Frontier-band rate roughly doubles on code induction (10.1→20.0% at 3B solver; 5.3→12.6% at 7B), lifts 1.7× on math, and 2.0× on held-out SWE repositories with a 27B generator/solver. It beats solver-in-the-loop RL while using fewer than half the solver trials. Probe *selection* is the subtle part: balanced accuracy is a poor predictor of RL gain, and reward variance under the base policy (RVP) tracks it better. The dominant failure mode is semantic mode collapse — top-topic share 0.32 → 0.74 — mitigated partially by ensembling (worst-case optimisation) and by adversarial probe co-evolution.

### For an early graduate student
Three things matter for `assay`.

**(a) The genre is identical and they got there first.** "Train a cheap predictor of an expensive RL-relevant quantity, then verify it against real training outcomes" is `assay`'s entire thesis. PROPEL executes it for *difficulty* on the generation side. `assay` executes it for *grader integrity and transfer* on the environment side. The related-work section must now open with PROPEL, and `docs/related-work.md`'s provisional novelty sentence needs rewriting: `assay` is no longer "nobody predicts RL outcomes cheaply" — that is now false — it is "nobody predicts *whether the gain is real* cheaply."

**(b) A3 is taken, which is good news for H3.** PROPEL owns the pass-rate-band axis: it operationalises it, cites the source for the optimal band, and doubles frontier-rate against it. `assay` should not re-derive A3 and must not present it as a contribution. But this *sharpens* H3 — the claim that grader degeneracy (A2) dominates the transfer gap while pass-rate band (A3) predicts learning *speed* rather than the *gap* was, before this paper, a hypothesis about our own battery's internals. It is now a **direct empirical claim about the axis a well-resourced lab has built its generator around**. That raises the stakes and makes H3 the paper's most interesting result if it holds — and a clean, publishable negative if it doesn't.

**(c) Their strongest domain has no held-out grader, and their own sister paper says that matters.** PROPEL's SWE utility label is: apply generator's bug patch, run *k*=3 solver trials, a trial is solved iff *"the solver's final edit passes the verifier suite."* The verifier suite is the repository's own tests — the same artefact the bug was validated against. Nothing checks whether the solver's fix addresses the root cause versus special-casing the failing test. A task whose "fix" is exploitable sits *perfectly* in the frontier band (solver passes 1–3 of 8 — exactly what a partially-discovered shortcut looks like), gets labelled positive, and teaches the probe the activation signature of exploitable-at-the-right-rate. **That is Goodhart amplification routed through the generator, and it is distinct from the semantic mode collapse they characterise** — mode collapse is a diversity failure that is visible; this is a validity failure that looks like success. Their own unix-ctf paper (#14) names precisely this gap in SWE-smith-style filters: *"Both check solvability without checking non-triviality."* PROPEL builds on SWE-smith. **The two papers are in tension, and that tension is `assay`'s opening.**

---

## Key Quotes

> "The limiting resource for training agents via reinforcement learning (RL) is increasingly frontier task supply: valid, solvable tasks just difficult enough to train the current model."

The motivation `assay` shares. Cite alongside Epoch's economics (#6) as the demand-side statement of why environment quality is the bottleneck.

> "A probe that classifies held-out examples well but assigns almost the same reward to every task sampled from the current policy results in almost no signal during RL."

The RVP justification — and the single most `assay`-shaped sentence in the paper. Discriminating power under the *deployment* distribution beats held-out accuracy. Our `p⁸+(1−p)⁸` is the closed-form version of the same insight, one level down.

> "Downstream RL gain (Table 4) tracks RVP more closely than validation balanced accuracy."

A zero-training-step statistic predicts the training outcome better than the obvious quality metric. **This is `assay`'s thesis, demonstrated by a third party, in an adjacent problem.** Strongest single piece of external support the project has; put it in the intro.

> "Because 𝑅 depends only on the rendered task and the frozen 𝜋ref, the policy cannot improve its reward by shifting the activations the probe reads."

The structural anti-Goodhart move. Adopt the vocabulary; note what it does *not* close.

> "PROPEL optimizing a single probe achieves the most significant gains in terms of utility, but results in stronger semantic concentration… it concentrates approximately 74% of generated tasks on the single semantic topic sorting_order."

The measured collapse. Note it is reported honestly and prominently — this is a well-run paper, and the failure mode is *theirs to fix*, not a gotcha.

> "Our mitigations (probe ensembling, adversarial co-evolution) are partial and trade efficacy for diversity."

Limitations, verbatim. The collapse is **named as unsolved**. This is the opening, and it should be engaged as a contribution to their program, not a criticism of it.

> "The probe is trained once on the reference generator's activations and held fixed during RL; under sustained policy drift this surrogate can degrade, and the mode-collapse behavior characterized in Section 6 is one consequence."

Proxy degradation under distribution shift, stated as a known limitation. Directly *Labels Not Loss* territory: when does continued optimisation against a fixed proxy start to hurt the true objective, and is there a measurable trigger?

> "In practice, the solver is itself being trained, and the utility of a task depends on what the solver has already mastered… Doing so induces an inner-outer loop that is expensive to optimize jointly; how to stabilize it without compounding instabilities across loops is an open question."

Future work — and the bridge to PopuLoRA (#15), which is their co-evolutionary answer. The three Vmax papers are one arc.

---

## Study Questions

**Warm-up:** (1) Write down U_S(x) and say why 0/8 and 8/8 are both worthless. (2) Why is the probe read from a *frozen* reference rather than the training policy? (3) What is RVP and why did they need it on top of balanced accuracy?

**Intermediate:** (4) PROPEL targets a solve-rate band of [1/8, 3/8]; `assay`'s A3 is "distance from p=0.5." Derive the group-degeneracy rate `p^K + (1−p)^K` at K=8 for p=0.25 and p=0.5 — which band actually maximises expected gradient signal, and what does the discrepancy tell you about what Wei et al.'s band is optimising? (5) PROPEL beats solver-in-the-loop RL *at lower cost*. Give two mechanisms by which a noisy-but-dense proxy can outperform a sparse ground-truth signal, and say what evidence would distinguish them. (6) Top-topic concentration rises 0.318 → 0.735 while utility doubles. Is that a failure, or the expected consequence of conditioning on a narrow band? Design the control that separates the two.

**Advanced:** (7) PROPEL's SWE solve label is "the solver's final edit passes the verifier suite," and the bug was validated against the same suite. Construct a generated task that lands squarely in the frontier band *because* it is exploitable, and describe how the probe would come to prefer that family. What is the cheapest experiment that detects this? (8) Their adversarial co-evolution mines "probe-high but solver-failed" outputs as negatives. That catches false positives on *difficulty*. Sketch the analogous mining loop for false positives on *integrity* — what plays the role of the solver-failed label when there is no held-out grader? (9) The probe transfers cold across generator families (Qwen → Mistral, Phi). Does that imply the *hackability* of a task is likewise family-invariant, or is there a reason difficulty transfers while exploitability doesn't? Tie your answer to `assay`'s H2.

---

## Challenge Corner

**C1 — PROPEL closes activation-hacking by construction, but not text-level Goodhart, and the mode collapse they report may be the benign half of the problem.** Freezing `π_ref` guarantees the policy cannot game the probe by shifting its own internals. It does not stop the policy from finding a *region of task-space* the probe systematically over-scores. They observed one such region and named it topic collapse (`sorting_order`) — visible, because it shows up in Distinct-3 and top-topic. The invisible version is a family the probe scores high and the solver *does* land in-band on, but for the wrong reason (exploitable). Nothing in their metric stack would surface it: utility recomputes from solver rollouts against the same grader. **If that family exists, PROPEL's headline number is partly measuring it.** This is the sharpest available `assay` contribution to their program, and it is testable cheaply with a held-out grader on a sample of generations.

**C2 — "The proxy beats ground truth" is under-explained and could be a smoothing artefact with a direction.** SIL uses K=8 noisy Bernoulli draws per task; PROPEL uses a dense calibrated logit. Variance reduction is the charitable reading. But a smoothed proxy also *systematically* re-weights the task distribution toward whatever the probe represents well — and they do not report which task families SIL finds that PROPEL misses. A per-family breakdown of the SIL-vs-PROPEL delta would settle whether PROPEL is strictly better or trading coverage for band-precision. Their own diversity metrics suggest the latter is at least partly true.

**C3 — Balanced accuracies are 0.59–0.66. The probe is barely better than a coin.** Table 2: math/3B 0.611, AZR/3B 0.649, SWE 0.594 with ECE 0.294 pre-calibration. That such a weak classifier drives a ~2× utility gain is genuinely surprising and deserves more scrutiny than it gets. Two readings: (i) RL needs only a *ranking* signal with variance, not accuracy — consistent with the RVP finding; or (ii) the gain is concentrated in a small, easily-identified slice of task-space, which is also what mode collapse looks like. These have different implications for whether the recipe generalises, and the paper does not separate them. *For `assay`:* a direct warning about our own `assay_score` — fitted weighting on a constructed family can produce a weak classifier that still moves the outcome, and we would be tempted to report the outcome.

**C4 — Three Vmax papers, three disjoint diversity/quality metric stacks, no common baseline.** PROPEL: Self-BLEU-3, Distinct-3, top-topic. unix-ctf (#14): a five-dimension LLM-judge rubric plus a mechanical hardcoded-assertion count. PopuLoRA (#15): AST depth, cyclomatic complexity, LoC, variable count, policy entropy. All three measure "did the generated distribution degenerate," none share a metric, and none is validated against the others. Their own job description calls for *"normative baselines for measuring the quality of RL environments"* — the gap is internal and acknowledged. **`canary` has a cross-validated battery for exactly this**, including the negative result that effective dimension (participation ratio) is non-monotonic and therefore a poor primary, and that collapse *severity* is seed-noisy enough to need replicates. PROPEL's WCO effect on 3B is 0.74 → 0.69 at a single seed per condition for SWE; that is inside `canary`'s measured seed-noise band for a comparable statistic.

**C5 — Does PROPEL scoop `assay`? No — but the provisional novelty claim in `docs/related-work.md` is now false as written.** PROPEL predicts *difficulty* from internals to *generate* tasks. `assay` predicts *exploitability and transfer* from black-box probes to *audit* environments. Different quantity, different access model (activations vs. inference-only), different consumer (generator training vs. procurement/pre-flight). But the sentence "nobody can tell you an environment is good before spending the compute" is no longer defensible unqualified — PROPEL tells you one thing about a task before spending the compute, and validates it. Rewrite to the quantity, not the genre.

---

## Connection to Project

### Verdict

**Novelty survives; framing must change.** `assay` is not the first cheap-forecaster-of-RL-outcomes — PROPEL is, in an adjacent problem, with a stronger access model (activations) and a validated cross-family transfer result. `assay`'s claim narrows to the quantity being forecast: **not "is this task the right difficulty" but "does passing this grader mean the skill was learned."** That is a genuinely different axis, it is unoccupied, and unix-ctf (#14) confirms Vmax considers it important while PROPEL shows their flagship pipeline does not measure it.

### Differentiation table

| Dimension | PROPEL (Vmax, 2026) | `assay` |
|---|---|---|
| **Quantity forecast** | Solver pass-rate band (difficulty) | Hacking gap + transfer efficiency η (integrity) |
| **Access model** | White-box: activations of a frozen reference generator | **Black-box: inference-only probes**, no weights needed |
| **Consumer** | Generator RL reward, inside the loop | Pre-flight audit / procurement, outside the loop |
| **Validation target** | Realized utility recomputed from solver rollouts, same grader | Post-GRPO gap slope + η against an **independently authored** grader |
| **Grader integrity** | Not measured; SWE label is the repo's own suite | **The object of study** (A1, A2, A5) |
| **Scale** | 4B–27B generators, 3B–27B solvers, real budget | 0.6B–1.7B, ~$100, free tiers |
| **Degeneracy statistic** | RVP, empirical, generator-side | `p^K+(1−p)^K`, closed-form, solver-side |

### What to adopt

- **RVP** as a named prior-art statistic; position Phase 0.1's `p⁸+(1−p)⁸` as its analytic solver-side complement, not as an independent invention. This is a *strengthening* citation.
- **The frozen-reference trick** as design vocabulary for closing self-referential Goodhart channels.
- **The [1/8, 3/8] band + Wei et al. 2025 as the source** for A3 — and resolve the disagreement with our "distance from 0.5" formulation before pre-registration.
- **Their honest reporting norm**: utility always recomputed from solver rollouts, probe score never used at evaluation. Mirror this — `assay_score` must never appear on the y-axis.
- **Adversarial co-evolution** as the template for an integrity-mining loop (Study Q8).

### What to differentiate (for reviewers)

We do not forecast difficulty — PROPEL does, better, with activations we do not require. We forecast **whether the reward is honest**, black-box, and validate against transfer to an independently authored environment. Cite PROPEL as the closest methodological neighbour and as **evidence the genre works**; cite unix-ctf (#14) for the axis they leave open. Never frame either as a competitor: the strongest version of `assay` is a component their pipeline is missing, and their own two papers say so.

### Where PROPEL tightens our design

- **Rewrite `docs/related-work.md`'s novelty sentence** to the forecast *quantity*. Current wording is falsified. **Blocks the Phase 0.5 gate.**
- **Demote A3 to cited-not-contributed**, and promote H3 to the paper's load-bearing claim.
- **Reconcile the band** ([1/8,3/8] vs distance-from-0.5) — a real technical discrepancy with a derivable answer.
- **Add a mode-collapse guard to our own `assay_score` fitting** (C3): a weak fitted classifier that still moves an outcome is exactly what we would be tempted to over-report.
- **The SWE-grader-integrity gap (C1) is now the single best candidate for `assay`'s headline external validation** — a cheap experiment on someone else's published pipeline, in their strongest domain.

---

## Synthesis Pointers
*(Feeds `synthesis-engagement.md` — tagged `Source: #13`.)*
1. **The genre is validated externally** — a cheap pre-training statistic (RVP) predicts RL gain better than the obvious quality metric. Strongest third-party support `assay` has. `Source: #13`
2. **A3 is taken** — cite PROPEL + Wei et al. 2025; do not re-derive. H3 becomes the load-bearing claim. `Source: #13`
3. **Band discrepancy** — [1/8, 3/8] vs distance-from-0.5. Resolve analytically before pre-reg. `Source: #13`
4. **`p^K+(1−p)^K` ↔ RVP** — same construct, two sides; frame ours as the closed-form complement. `Source: #13`
5. **PROPEL's SWE label has no held-out grader**, and their sister paper (#14) names that exact gap. The tension is `assay`'s opening. `Source: #13`
6. **Mode collapse is measured, mitigated only partially, and named as a limitation** — and their three papers use three disjoint diversity metric stacks. `canary`'s battery is directly transferable. `Source: #13`
7. **Novelty sentence in `docs/related-work.md` is falsified as written.** Blocks Phase 0.5. `Source: #13`

---

## Discussion Notes
*To be filled during interactive discussion (Process step 3).* Open threads:
- Does the band discrepancy (§Key Idea 1 / Study Q4) resolve analytically, or is it an empirical claim of Wei et al. we must take on trust?
- Is the SWE grader-integrity probe (C1) cheap enough to run as an `assay` external-validation arm — auditing published generations rather than our own `bisect` variants? That would be a much stronger paper than a self-contained study.
- Should `assay` reposition explicitly as *the integrity axis of the Vmax program* rather than a standalone pre-flight tool? That is a strategic framing question with consequences for venue.
