# Review: unix-ctf — Procedural Environments for Unix-Competence Reinforcement Learning

**Paper:** Geoffrey Bradway, Roger Creus Castanyer, Lorenz Wolf, Maxwill Lin, Matthew James Sargent, Augustine N. Mavor-Parker. **Vmax.** arXiv:2605.29115v1 [cs.CR], 2026-05-27 (dated May 29, 2026). *(Reading-list #14 — clusters K2 "closest prior art / scoop" + K5 "borrowed substrates".)*
**Reviewed:** 2026-08-06
**Reviewer:** Claude (structured review for interactive discussion)
**PDF:** `https://arxiv.org/pdf/2605.29115` · **ID ⬤-verified 2026-08-06** (title/authors/affiliation/date read from the PDF itself).
**Added under the recursive novelty-perimeter rule** — surfaced 2026-08-06 while researching Vmax. Appended, not renumbered.

> **This paper states `assay`'s thesis in one sentence, and it is not our sentence:** *"Both check solvability without checking non-triviality."* It also **directly reproduces Endless Terminals (our #9) and measures it failing** — 17.4% portable vs their 87.5% — and mechanically counts degenerate graders in the survivors: **99% of Endless Terminals tasks carry ≥1 hardcoded exact-string assertion, vs 0% of theirs.** This is simultaneously the strongest external validation `assay`'s premise has received and the most serious novelty threat on the list. **Novelty survives — narrowly, and only on generality.** Full analysis in **Connection to Project**.

> **Scope of this review (read first).** Drafted 2026-08-06 by an out-of-project session doing
> interview preparation, at **Process step 1 only**. Everything below is a *finding* or a *proposal
> pending Process steps 2–5*. The `Where … tightens our design` section in particular is a list of
> candidate implications for the project owner to adjudicate — **no decision, gate, pre-registration,
> axis definition, or roadmap item has been changed on the basis of this review.**

---

## Background

Terminal agents are evaluated on InterCode, Terminal-Bench, AgentBench-OS, NYU CTF, Cybench, and trained by pipelines like Endless Terminals and Nemotron-Terminal. The paper's opening claim is a **construct-validity** one: this landscape collapses two different skills into one. *Unix competence* — using shell and OS primitives as first-class tools — is distinct from *coding-in-a-terminal*, where the deliverable is a non-trivial program that happens to be written through a shell. A solver fluent in Python but weak in Unix passes a substantial fraction of Terminal-Bench 2.0; the reverse profile is rarely exercised.

They make the distinction operational and build a training surface for the neglected half. `unix-ctf` is a procedural generator: each task hides a `flag{...}` token in a fresh Linux container using **a single Unix feature**, and the agent must recover it. CTF format is chosen because *"the reward is mechanical and unambiguous, and because a planted flag can be tied to a single Unix feature"* — i.e. deliberately for grader-integrity reasons.

For `assay` this is a K2 paper wearing K5 clothes. On the surface it is a substrate contribution. Underneath it is an argument about **environment quality measurement**, it lands a quantitative comparison against a generator already on our list, and it uses a mechanical degeneracy probe that is close to a working implementation of our A2.

---

## Key Ideas

### 1. The bidirectional contract — a two-sided grader-integrity filter
The LLM writes only a parameterised `plant.sh` / `recovery.sh` pair. Two filters are then applied:

- **Non-triviality side:** a recursive filesystem search must **not** find the flag in plaintext. (Kills the `grep -r flag /` degenerate solve.)
- **Genuineness side:** re-running `recovery.sh` **on a fresh directory with a fresh flag** must recover it and exit zero. (Kills inlining the original flag or path — i.e. a recovery script that "works" by memorising the answer.)

*For us:* **this is a hand-built A1+A2 for one domain.** The first filter is a hackability probe (is there a trivial path to reward that bypasses the skill?). The second is a contamination/memorisation probe (does the reference solution encode the answer rather than the method?). It is bespoke, cheap, mechanical, and — per their numbers — the single largest driver of quality.

### 2. The thesis sentence, and the yield gap it explains
> *"Upstream pipelines use only one half: Endless Terminals filters on whether at least one of sixteen OpenAI o3 samples solves; SWE-smith and R2E-Gym filter on whether a generated change invalidates a unit test. **Both check solvability without checking non-triviality.**"*

Combined with pipeline factoring (a pre-debugged `ctf-base` image handles the system layer; the LLM never re-derives infrastructure), this drives **87.5% end-to-end portability (656/750)** versus **17.4% in their own n=120 Endless Terminals reproduction**. Failure analysis of the reproduction: 71.7% of failures at Docker-build or test-validation, only 10.8% at LLM task generation. *For us:* the model is good at inventing the *idea*; the yield problem is infrastructure and filtering. That is an argument for `bisect`'s "no Docker, subprocess + timeout" scope discipline, from independent evidence.

### 3. Mechanical grader-degeneracy counting — an A2 implementation in one line
Table 1's caption carries the most `assay`-relevant number in the paper:

> *"Grading-robustness dimension omitted; mechanical count: **99% of Endless Terminals tasks have ≥1 hardcoded exact-string assertion vs. 0% unix-ctf.**"*

They dropped grading robustness from the *judged* rubric and counted it *mechanically* instead. *For us:* hardcoded-exact-string-assertion rate is a cheap, deterministic, language-agnostic degeneracy statistic — **a working A2 probe requiring zero API calls.** Adopt it directly. It is also a reproach to `assay`'s current A2 definition ("cluster reward-maximizing trajectories; count behaviour clusters that are not the target skill"), which is API-expensive and judge-dependent where a `grep` would do much of the work.

### 4. The five-dimension quality audit, and its judge methodology
441 `unix-ctf` variants vs 120 Endless Terminals task dirs, scored by **three LLM judges from three vendors** (gpt-5.4-nano, grok-4-20-reasoning, Kimi-K2.5), **deliberately excluding Claude because Claude appears in the generation loop**.

| Dimension | unix-ctf (n=439) | Endless Terminals (n=120) | Δ |
|---|---|---|---|
| Input richness | 4.74 ± 0.45 | 1.81 ± 0.32 | +2.93 |
| **Solution non-triviality** | **3.45 ± 0.34** | **2.54 ± 0.46** | **+0.90** |
| Format independence | 4.42 ± 1.00 | 1.07 ± 0.04 | +3.35 |
| Tool surface diversity | 3.44 ± 0.83 | 2.85 ± 0.78 | +0.59 |
| Discoverability | 3.05 ± 0.10 | 1.12 ± 0.08 | +1.93 |
| **Aggregate** | **3.82 ± 0.43** | **1.88 ± 0.30** | **+1.94** |

Agreement is reported as pairwise Pearson *r* on per-task aggregates (gpt×grok 0.82, gpt×kimi 0.83, grok×kimi 0.93, n=559). *For us:* **excluding the generation-loop model from the judge panel is exactly `synthoracle`'s contamination discipline**, arrived at independently — adopt and cite. But the reliability treatment is thin: Pearson *r* on aggregates, no position-bias correction, no chance-corrected agreement. See C3.

### 5. The training result — the surface is trainable, not just measurable
Qwen3-8B + LoRA + GRPO on the surface: solve rate **11.6% → 43.6%** on a 15-skill multi-family holdout (n=225); **+33 pp in Forensics**; 32/100 on InterCode-CTF; and it *redistributes* which InterCode-CTF tasks the model solves rather than uniformly lifting. *For us:* a clean demonstration that environment quality translates to policy capability at exactly `waterline`'s model scale (Qwen3-8B, LoRA, GRPO) — and a live reminder that `assay`'s claims must eventually connect to a capability delta, not just a gap statistic.

### 6. Skill separability as a construct-validity argument
The taxonomy work (155 canonical technique IDs across 16 families) exists to defend a claim that *Unix competence is a separate skill*, testable by showing a Python-fluent solver passes terminal benchmarks without it. *For us:* this is the same move `assay`'s η makes — "the benchmark you are passing is not the skill you think you are buying" — applied to skill decomposition rather than environment authorship. Structurally an ally of E2 (grader idiom > environment idiom).

---

## Results

### For a smart high-schooler
To teach an AI to use a Linux terminal properly, you need lots of practice puzzles. So they built a puzzle factory: hide a secret code somewhere in a computer using one specific Unix trick, and the AI has to dig it out. The clever part is the two checks every puzzle must pass. First, you must not be able to just search the whole disk for the code — otherwise the AI learns "search everything" instead of the trick. Second, the official solution has to still work when they change the secret code and the folder — otherwise the "solution" was just memorising the answer. A rival system that skips these checks produced usable puzzles 17% of the time; theirs works 87% of the time. And when they inspected the rival's puzzles, 99% had the answer hardcoded into the grader. Training an 8-billion-parameter model on their puzzles took it from solving 12% to 44%.

### For an undergraduate
`unix-ctf` procedurally generates CTF tasks where a flag is hidden via a single Unix feature. An LLM-assisted pipeline harvests candidate techniques and rewrites them as parameterised `plant.sh`/`recovery.sh` pairs; the container, layout, and grading harness are fixed and pre-debugged. Two filters constitute a bidirectional contract: no plaintext trace of the flag on disk, and successful re-recovery on a fresh directory with a fresh flag. Yield is 87.5% (656/750) against 17.4% for a controlled Endless Terminals reproduction, with the failure mass in infrastructure rather than idea generation. 656 variants dedupe to 441 and canonicalise to 155 technique IDs across 16 families. A three-vendor LLM-judge audit shows a +1.94 aggregate quality gap; separately, 99% of Endless Terminals tasks contain a hardcoded exact-string assertion versus 0%. GRPO+LoRA on Qwen3-8B lifts holdout solve rate 11.6% → 43.6% and shifts the InterCode-CTF solve profile, with +33 pp in Forensics.

### For an early graduate student
Three things matter for `assay`.

**(a) They named our gap first, and in better words than `docs/conceptual.md` currently uses.** "Check solvability without checking non-triviality" is a cleaner statement of `assay`'s premise than anything in our own docs. The novelty question is therefore not *whether* the axis is ours — it isn't — but whether our contribution survives. It does, on **generality and validation target**. Their answer is a domain-specific pair of shell predicates that work because CTF has a mechanical, unambiguous reward and a single hidden token. Neither filter ports: "no plaintext trace of the flag" has no analogue in math, SWE, or agentic web tasks, because those have no flag. `assay` proposes a **portable battery**, and validates it not against a judge rubric but against **what GRPO actually does to a policy**. That distinction must be stated in one sentence in the related-work section, or a reviewer will ask.

**(b) Our #9 changes status, and so does the gap's evidentiary basis.** `Endless Terminals` was listed as Block C supporting material — "env supply at scale; a consumer of `assay`, not a competitor." It is now a **documented low-quality generator with two independent failure measurements**: 17.4% portability under a controlled reproduction, and 99% hardcoded-assertion rate in its outputs. That is the strongest concrete evidence in the entire reading list that environment quality is a real, measurable, currently-unaddressed problem — and it was produced by someone else, which makes it citable rather than self-serving. Upgrade #9 from Block C to Block A and re-scope its key question from "does it compete?" to "is its measured failure profile the motivating exhibit for `assay`?"

**(c) The tension with PROPEL (#13) is the opening, and it is internal to Vmax.** This paper explicitly criticises SWE-smith for filtering only on "whether a generated change invalidates a unit test." PROPEL's SWE domain **is built on SWE-smith**, and its utility label is solver-passes-the-repo's-own-suite. So Vmax's environment paper names a failure mode that Vmax's generator paper inherits and does not measure. This is not a gotcha — different papers, different scopes, and the unix-ctf authors clearly know it. But it is the precise shape of a contribution: **the bidirectional contract generalised into a portable diagnostic, applied to the generator pipeline that currently lacks one.** That is `assay`, stated in their vocabulary, aimed at their stack.

---

## Key Quotes

> "Upstream pipelines use only one half: Endless Terminals filters on whether at least one of sixteen OpenAI o3 samples solves; SWE-smith and R2E-Gym filter on whether a generated change invalidates a unit test. Both check solvability without checking non-triviality."

**The thesis sentence.** `assay`'s premise, stated by a third party, about named systems, with a measured consequence. Quote it in the intro. It converts our motivation from an assertion into a citation.

> "Grading-robustness dimension omitted; mechanical count: 99% of Endless Terminals tasks have ≥ 1 hardcoded exact-string assertion vs. 0% unix-ctf."

A one-line, zero-cost, deterministic grader-degeneracy statistic with a dramatic separation. **Adopt as an A2 sub-probe.** Also note what it implies: they judged grading robustness too unreliable for an LLM rubric and measured it mechanically instead — a methodological verdict we should heed.

> "the hide script must leave no plaintext trace of the flag on disk, and the find script must recover the flag in a fresh directory."

The bidirectional contract in full. Domain-specific by construction; the general form is `assay`'s problem.

> "The model is competent at inventing hiding techniques but rarely succeeds at the surrounding infrastructure re-derived from scratch each time."

Independent support for `bisect`'s scope discipline (no Docker, fixed harness, vary only the grader configuration). Cite when defending the variant-grid design.

> "We deliberately exclude Claude because it appears in the unix-ctf generation loop."

Contamination discipline in judge selection, arrived at independently of `synthoracle`. Adopt, cite, and extend with the reliability machinery they omit.

> "the reward is mechanical and unambiguous, and because a planted flag can be tied to a single Unix feature."

Their stated reason for choosing CTF is *grader integrity*. Environment-quality reasoning is driving substrate choice at Vmax — the audience is pre-sold on `assay`'s premise.

---

## Study Questions

**Warm-up:** (1) State the bidirectional contract's two halves and what degenerate solve each one kills. (2) Why is 17.4% vs 87.5% not simply "their LLM is better"? (3) What does a hardcoded exact-string assertion let a policy do?

**Intermediate:** (4) Both halves of the contract exploit CTF-specific structure (a single hidden token). Write the closest analogue of each for (a) a SWE bug-fix task and (b) a competition math task — and say which one has no analogue at all. (5) Their reproduction shows 71.7% of Endless Terminals failures at Docker-build/test-validation and 10.8% at generation. If you could only fix one, which buys more, and how does that inform `bisect`'s "no Docker" pin? (6) They report judge agreement as pairwise Pearson *r* on aggregate scores. Name two failure modes that statistic cannot detect, and what `crit-thinking`'s machinery would add.

**Advanced:** (7) Generalise the bidirectional contract into a domain-independent pair of predicates. What is the *type signature* of each half, and what must an environment supply for them to be computable? Is `assay`'s A1/A5 pair that generalisation, or is something still missing? (8) 99% hardcoded-assertion rate is a *grader* statistic; the LLM rubric's non-triviality score is a *task* statistic; they differ by only +0.90 while other dimensions differ by +2 to +3.35. Why might the mechanical measure separate so much more cleanly than the judged one, and what does that predict about `assay_score`'s fitted weighting? (9) unix-ctf trains Qwen3-8B with LoRA+GRPO to 43.6% on a holdout. Design the minimal experiment that shows whether that gain is *Unix competence* or *unix-ctf idiom* — i.e. compute η for their own surface. What would you need from them, and what could you do with only public artefacts?

---

## Challenge Corner

**C1 — Their contract is a *generator-side* filter, not a *policy-side* diagnostic, and the distinction is load-bearing for `assay`'s novelty.** Both halves are applied to the authored `plant.sh`/`recovery.sh` pair before the task ships. Neither asks what a *trained policy* converges to. A task can pass both filters — flag not in plaintext, reference solution generalises — and still be solvable by a shortcut the reference author never considered. Their non-triviality score is judged by an LLM reading the task, not measured by adversarially probing it. **`assay`'s A1 (frontier best-of-N with an adversarial system prompt) is the missing half**, and it is exactly the half that requires a policy in the loop. State this explicitly; it is the cleanest differentiation available.

**C2 — The 87.5% vs 17.4% comparison is real but partly confounded by construction.** unix-ctf fixes the container, layout, and grading harness; Endless Terminals generates them per task. Of course yield differs — they removed the failing component from the per-attempt budget. The authors are transparent about this (it *is* their contribution, "pipeline factoring"), but the headline framing invites reading it as "our filters are better" when 71.7% of the delta is attributable to not re-deriving Docker each time. The **filters' independent contribution is not isolated**. For `assay`: a warning about attributing a composite improvement to the axis you care about, and a reason to keep the variant grid factorial.

**C3 — Judge methodology is the weakest part of an otherwise careful paper.** Three vendors and Claude-exclusion are good. But: agreement reported as pairwise Pearson *r* on aggregates (which is inflated by between-task variance and says nothing about per-dimension reliability), no chance-corrected statistic, no position-bias control, no report of whether judges saw conditions in a fixed order, and n=120 vs n=439 unbalanced arms scored by the same panel that could plausibly infer condition from surface features. `crit-thinking`'s position-bias correction and Krippendorff's α are a direct, citable upgrade — and this is a place where `assay`'s A4 (judge instability) has an immediate external application.

**C4 — "Solution non-triviality" separates least (+0.90) among the five judged dimensions.** Input richness (+2.93), format independence (+3.35), and discoverability (+1.93) all separate far more strongly than the dimension the paper's thesis is about. Two readings: (i) LLM judges are poor at assessing non-triviality — supported by their own decision to measure grading robustness mechanically instead; or (ii) Endless Terminals' tasks are not actually much more trivial, and the quality gap is mostly about presentation and format. The paper does not distinguish these, and the mechanical 99%/0% count arguably contradicts (ii). **For `assay`: this is direct evidence that judged non-triviality is an unreliable measurement and that mechanical/adversarial probes should carry A2, not an LLM rubric.**

**C5 — Does it scoop `assay`? Not quite, and the margin is generality plus validation target.** They: one domain, generator-side, hand-designed predicates, validated against an LLM rubric and a downstream training run. We: multi-domain battery, policy-side adversarial probing, validated against *the post-GRPO gap slope and transfer efficiency against an independently authored grader*. The margin is real but it is narrower than `docs/related-work.md` currently assumes, and it now rests on **delivering the η leg**. If η is cut for budget (the §8 cut-order permits it), `assay` reduces to "a portable version of unix-ctf's contract" — still useful, considerably less novel. **This raises the cost of cutting η and should be recorded as such.**

---

## Connection to Project

### Verdict

**Novelty survives on generality and validation target, not on the idea.** The axis — solvability is not integrity — is now published, named, quantified, and owned by the lab `assay` would most want to impress. `assay`'s remaining contribution is (i) a **portable** battery where theirs is bespoke, (ii) **policy-side adversarial** probing where theirs is author-side filtering, and (iii) validation against **training outcomes and independent-author transfer** where theirs is an LLM rubric plus a single training run. All three must be stated explicitly.

### Differentiation table

| Dimension | unix-ctf (Vmax, 2026) | `assay` |
|---|---|---|
| **Integrity mechanism** | Bidirectional contract: no-plaintext + fresh-dir replay | Portable battery A1–A6 across domains |
| **Where applied** | Generator side, pre-ship, on authored scripts | **Policy side**, adversarial best-of-N against the live grader |
| **Domain** | CTF only (needs a single hidden token) | Domain-independent by design; `bisect` as the instance |
| **Non-triviality measured by** | LLM-judge rubric (+0.90 separation) + mechanical assertion count (99%/0%) | Adversarial exploit discovery + held-out grader disagreement |
| **Validated against** | Judge rubric; one GRPO run (11.6→43.6%) | **Post-GRPO gap slope + η vs an independently authored grader** |
| **Scale** | Qwen3-8B, LoRA, GRPO, real budget | 0.6–1.7B, ~$100 |

### What to adopt

- **The thesis quote**, verbatim, in `assay`'s intro. Converts our motivation from assertion to citation.
- **The hardcoded-exact-string-assertion count as an A2 sub-probe** — deterministic, free, and it separated 99%/0% where the LLM rubric managed +0.90.
- **Claude-exclusion from the judge panel** when the generation loop uses Claude (it does — `bisect` seeds and A1 probes).
- **Pipeline factoring** as independent support for `bisect`'s fixed-harness / vary-only-the-grader design.
- **Their Endless Terminals failure taxonomy** (71.7% infra / 10.8% generation) as the argument for "no Docker" in W1.

### What to differentiate (for reviewers)

Theirs is a **filter**; ours is a **diagnostic**. Theirs is authored per domain; ours is measured per environment. Theirs validates against a rubric; ours validates against what training does. Say it in three sentences and move on — and cite them warmly, because the strongest version of `assay` completes their argument rather than competing with it.

### Where unix-ctf tightens our design

- **Promote #9 Endless Terminals from Block C to Block A** and re-scope its question: it is now the motivating exhibit, with two independent failure measurements.
- **Rewrite A2** to lead with a mechanical degeneracy count and use trajectory clustering as the expensive second tier — cheaper, more reliable, and now externally validated.
- **Record that cutting η materially damages the novelty claim** (C5). Amend the §8 cut-order note in `docs/stages.md`.
- **Add the judge-reliability upgrade (C3) as an explicit A4 deliverable** with an external application, not just an internal control.

---

## Synthesis Pointers
*(Feeds `synthesis-engagement.md` — tagged `Source: #14`.)*
1. **"Both check solvability without checking non-triviality"** — `assay`'s premise, published, by Vmax, about named systems. The single best motivating citation available. `Source: #14`
2. **Endless Terminals (#9) has a measured failure profile**: 17.4% portability, 99% hardcoded assertions. Promote to Block A as the motivating exhibit. `Source: #14`
3. **Mechanical assertion-count = a free A2 sub-probe** that outperformed a judged rubric on the same construct. `Source: #14`
4. **Novelty margin is generality + validation target, and it is narrower than assumed.** Cutting η collapses it. `Source: #14`
5. **Judged non-triviality separates weakly (+0.90)** while the mechanical measure separates absolutely — evidence against LLM-rubric A2. `Source: #14`
6. **Internal Vmax tension:** this paper criticises SWE-smith-style filtering; PROPEL (#13) builds on SWE-smith. That gap is `assay`'s opening. `Source: #14`
7. **Claude-exclusion from judge panels** — contamination discipline, independently arrived at; adopt and cite. `Source: #14`

---

## Discussion Notes
*To be filled during interactive discussion (Process step 3).* Open threads:
- Is the general form of the bidirectional contract actually A1+A5, or is there a third predicate we are missing (Study Q7)? This is the most important conceptual question the paper raises.
- Should `assay` run its battery over `unix-ctf`'s 155 canonical techniques as a public field-report arm? The artefacts appear to be released, the environments are cheap (no Docker per task at inference), and it would be an external validation on a *good* environment — a useful contrast to the Hub sweep, which will mostly find bad ones.
- Does η on unix-ctf's own surface (Study Q9) constitute a better G-stage headline than η on Hub pairs? Same skill, published training run, and an independent evaluation surface already exists (InterCode-CTF).
