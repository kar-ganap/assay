# Review: Features as Rewards — Scalable Supervision for Open-Ended Tasks via Interpretability (RLFR)

**Paper:** Aaditya Vikram Prasad\*, Connor Watts\*, Jack Merullo, Dhruvil Gala, Owen Lewis, Thomas McGrath, Ekdeep Singh Lubana. *(\*equal contribution.)* **Goodfire AI** (all authors). arXiv:2602.10067v3 [cs.LG], 2026-02-18. 89 pp. *(Reading-list #16 — clusters K2 "closest prior art" + K4 "reachability".)*
**Reviewed:** 2026-08-06
**Reviewer:** Claude (structured review for interactive discussion)
**PDF:** `https://arxiv.org/pdf/2602.10067` · **ID ⬤-verified 2026-08-06** — resolved from #13's own bibliography (PROPEL p. 726) and confirmed against the PDF.
**⚠️ ID correction:** #13 cites this as *"RL from Feature Rewards (RLFR)"*, which is the **framework** name. The **paper title** is *"Features as Rewards: Scalable Supervision for Open-Ended Tasks via Interpretability."* Searching the list's provisional title would not have found it. Exactly the failure mode the ⬤ rule exists for.
**Added under the recursive novelty-perimeter rule** — the upstream dependency #13 builds on and does not fully explain.

> **Scope of this review (read first).** Drafted 2026-08-06 by an out-of-project session doing
> interview preparation, at **Process step 1 only**. Everything below is a *finding* or a *proposal
> pending Process steps 2–5*. The `Where … tightens our design` section in particular is a list of
> candidate implications for the project owner to adjudicate — **no decision, gate, pre-registration,
> axis definition, or roadmap item has been changed on the basis of this review.**

---

> **This is the load-bearing dependency under the whole Vmax stack, and it names `assay`'s question as its own open problem — in the limitations section, in one sentence.** RLFR invents the frozen-probe-as-reward recipe that PROPEL (#13) inherits wholesale. It is also, formally, a **reward-model identifiability paper**: it frames its own risk as *non-identifiability under limited data coverage → reward hacking*, and its defence is to constrain probe expressivity. Then it says the defence held *in this case* and that **whether it survives further optimisation is unknown.** That is a dose-response question about proxy-optimisation failure, asked by the people who built the proxy, sitting directly upstream of a production pipeline. Full analysis in **Connection to Project**.

---

## Background

Verifiable domains (math, code) admit cheap deterministic checks that can serve as sparse success-based rewards. Most *desirable* behaviours are not like that: verifying factuality, helpfulness, or non-sycophancy requires an LLM judge with search and tool access, and cost grows rapidly with the number of claims per rollout. So open-ended behaviours are effectively unreachable by RLVR.

The paper's move: interpretability has established that models internally represent abstract concepts as linearly-decodable *features*, and — critically — **features are often present even when the model cannot reliably verbalise or act on the concept.** If a probe's readout is well-calibrated against ground truth, that readout can serve as a dense, near-free reward. They instantiate this as **RLFR** on hallucination reduction: a four-stage pipeline (localise and classify candidate hallucination spans → intervene → reward sampled interventions → RL) on Gemma-3-12B-IT.

For `assay`, this is the paper that establishes the *recipe* #13 applies to task generation. It matters here for two reasons that have nothing to do with hallucinations: it states the reward-hacking risk of its own method in **identifiability** terms, and its limitations name the exact experiment `assay`'s premise implies.

---

## Key Ideas

### 1. Features as reward: calibrated internal belief as dense supervision
The central claim is conditional and precisely stated: *"if a model's features are well-calibrated, i.e., if the confidence of the readout of a concept from model features correlates with ground-truth validity of the concept, can one use these features as sources of supervision?"* They read the probe output as a **posterior belief** — the model's uncertainty about whether a concept holds. Prior work used such readouts for *monitoring* and *steering*; the novel affordance is using them as a **training signal**. *For us:* the load-bearing assumption is **calibration**, and calibration is a property that can degrade under distribution shift — which is precisely what optimising against the probe induces.

### 2. The reward-hacking risk, framed as non-identifiability — and the expressivity defence
This is the paragraph that matters most for `assay`:

> *"The central difficulty with such approaches is **non-identifiability under limited data coverage**: a highly expressive reward model can learn many mappings consistent with limited data, yielding brittle generalization and reward hacking… Since we fit probes based upon features to labeled data, our pipeline also falls under the purview of **inverse-RL**; however, our use of **extremely low-expressivity probing architectures** constrains the expressible solutions in our pipeline, and the use of already pretrained features as inputs further induces sample efficiency."*

Their defence against Goodhart is **hypothesis-class restriction**: a linear or tiny-MLP probe cannot express the pathological mappings a large learned reward model can. *For us:* this is the same formal problem as multi-stakeholder preference identifiability — many reward vectors consistent with the observed labels, and the question of which one you converge to determines whether more optimisation helps or harms. The expressivity argument is a *sufficient-condition* style defence and is not accompanied by an identifiability analysis or a measurable condition for when it fails.

### 3. The frozen-parameter defence — the design PROPEL inherits
Probes are run on a **frozen set of parameters**, plus constraints on generations to keep text natural. Rationale, stated in limitations: *"A common claim about training against such monitors is that the student model will learn to evade the monitor (Bailey et al., 2024)."* Freezing means the student cannot lower its loss by reshaping the activations being read. *For us:* #13's *"the policy cannot improve its reward by shifting the activations the probe reads"* is a direct inheritance, citing this paper. **One architectural defence carries two papers and a production pipeline.**

### 4. Reward-pipeline transfer: probe ordering is invariant to activation source
They checked whether the reward pipeline degrades when run on the *policy's* activations rather than the base model's. Fixed and Retracted rates were essentially unchanged (best-of-32, single seed), *"suggesting RLFR preserved the reward probe's ordering."* Practical payoff: no need to host both models at test time. *For us:* this is the **empirical evidence base for the frozen-reference choice** — and it is one seed, one setting, one behaviour, measured at the end of a training run that the paper itself says may not have applied enough optimisation pressure to break the defence. Load-bearing, thinly evidenced.

### 5. Headline results and cost
Gemma-3-12B-IT: **58% less likely to hallucinate** than the base model when run with the probing harness, with standard benchmarks preserved (HellaSwag 83.8→83.6, MMLU 69.2→67.2, GSM8K 72.3→73.0, GPQA 26.3→27.3 — Table 1). Cost: **~90× cheaper per rewarded intervention** than the ground-truth supervision source, ~2 orders of magnitude cheaper than Gemini 2.5 Pro as judge. Targeted-update evidence: average KL between base and policy is 10–20% larger on Not-Supported than Supported tokens, *"though it is important to note the large variation between seeds."*

### 6. Off-target checks — the norm to copy
A full battery: standard benchmarks, per-token KL split by supported/unsupported spans, entity-count comparison, blind Gemini preference rating on paired completions, qualitative review of randomly-selected (not only cherry-picked) interventions, plus red-teaming and manual auditing of both detection outputs and reward labels. *For us:* this is the off-target discipline `assay`'s Run stage should mirror. Note especially that they report **randomly-selected** examples in an appendix alongside the cherry-picked figure ones.

### 7. Future work: cascaded verifiers gated on probe confidence
> *"given we found our probes to be decently well-calibrated, it is worth considering the use of a more expensive verifier when the confidence of the probe is low."*

A cheap probe triages; an expensive verifier adjudicates the uncertain tail. *For us:* structurally identical to `assay`'s two-tier measurement design (cheap gap on every variant, full η on the confirmatory arms). Independent convergence on the same cost architecture.

---

## Results

### For a smart high-schooler
Teaching an AI not to make things up is hard because checking whether a statement is true is slow and expensive — you need another powerful AI to go look it up. But it turns out models internally "know" when they're unsure, even when they confidently say the wrong thing out loud. So you can train a tiny detector that reads that internal uncertainty, and use *it* as the grading signal instead of the expensive fact-checker. Doing that made the model 58% less likely to hallucinate, about ninety times cheaper, without hurting it on normal tests. The obvious worry is that the model learns to fool the detector rather than actually stop making things up. Their fix: read the detector off a *frozen* copy, so the model can't just change its own internals to look confident. It worked here — and they say plainly that they don't know whether it keeps working if you train harder.

### For an undergraduate
RLFR converts interpretability probes into RL reward functions for open-ended behaviours. Pipeline: localise candidate hallucination spans; have the policy intervene (retract or correct); score interventions using probes that read the frozen base model's activations as a calibrated posterior over factual validity; optimise with RL. On Gemma-3-12B-IT this yields a 58% hallucination reduction with benchmark performance preserved, at ~90× lower cost per rewarded intervention than ground-truth supervision. The Goodhart risk is framed as non-identifiability of a reward model under limited data coverage, and mitigated two ways: extremely low-expressivity probe architectures (restricting the hypothesis class) and running probes on frozen parameters (removing the activation-shaping channel). Reward-pipeline transfer experiments find probe ordering invariant to whether activations come from base or trained policy. Limitations name the evasion question as unresolved under further optimisation, and note that the evaluation pipeline reuses the data-collection prompting stack, so most labels are "internally consistent by construction."

### For an early graduate student
Three things matter for `assay`.

**(a) The stack has a single point of failure and it is one architectural trick, evidenced once.** The frozen-parameter defence is what makes probe-as-reward safe from activation-hacking. RLFR introduces it, tests it in one setting on one behaviour at one seed, and PROPEL inherits it verbatim to reward a *task generator* — a different policy, a different objective, a different optimisation budget, and a different failure surface (PROPEL's generator emits task *text*, where the exploitable degrees of freedom are enormous compared to a retraction span). Neither paper measures how the defence behaves as optimisation pressure increases. **RLFR says so explicitly.**

**(b) The open question is a dose-response question, and it is the shape of a study already built.** In limitations: *"in this case it is easier for the student to learn the behavior we are trying to teach than to evade the probe. It is plausible that with further optimization this changes, but further work is needed to ascertain the (in)efficacy of this mitigation."* Read alongside #13's *"under sustained policy drift this surrogate can degrade, and the mode-collapse behavior characterized in Section 6 is one consequence"*, the two papers jointly assert: **the defence holds at the pressures tested, degrades at some higher pressure, and nobody has located the threshold.** That is structurally identical to `waterline`'s λ\* — sweep an optimisation-pressure knob, find where grader-gaming emerges, distinguish visible from covert failure — with the knob being optimisation steps or KL against a frozen probe rather than CoT-grading coverage. The rig for that class of experiment exists and is validated (`waterline` `loop.py` + Modal, emergence-curve callback, early-stop on hack rate).

**(c) It is a reward-identifiability paper in interpretability clothing, which puts it squarely on `Labels Not Loss`'s formal territory.** They name the risk as non-identifiability under limited coverage, place themselves explicitly within inverse-RL, and defend by restricting the hypothesis class. What they do *not* provide is any measurable condition for when the fitted probe's implied objective diverges from the true one — the question `Labels Not Loss` answers for multi-stakeholder reward vectors via the cosine between trained weights and the target. Whether that transfers from a linear reward vector over stakeholder utilities to a linear probe over residual-stream features is a real open question, not a slogan — the geometry is analogous (a direction fitted to limited labels, then optimised against) but the target's identifiability structure differs. **Worth stating as a question, never as a claim.**

---

## Key Quotes

> "The central difficulty with such approaches is non-identifiability under limited data coverage: a highly expressive reward model can learn many mappings consistent with limited data, yielding brittle generalization and reward hacking… our use of extremely low-expressivity probing architectures constrains the expressible solutions in our pipeline."

The Goodhart risk and the defence, in the authors' own framing. Identifiability language, inverse-RL framing, hypothesis-class restriction as mitigation. The most directly `Labels Not Loss`-adjacent paragraph in any of the four Vmax-stack papers.

> "A common claim about training against such monitors is that the student model will learn to evade the monitor (Bailey et al., 2024). We mitigate this issue by running the probe on a frozen set of parameters… **in this case it is easier for the student to learn the behavior we are trying to teach than to evade the probe. It is plausible that with further optimization this changes, but further work is needed to ascertain the (in)efficacy of this mitigation.**"

**The single most important sentence across all four papers for `assay`.** The defence that carries the entire stack is reported as holding *at the pressures tested*, with its behaviour under more pressure named as unknown, by the authors who invented it. This is a named open problem, upstream of a production pipeline.

> "if a model's features are well-calibrated, i.e., if the confidence of the readout of a concept from model features correlates with ground-truth validity of the concept, can one use these features as sources of supervision?"

The conditional the whole method rests on. Calibration is assumed, measured once, and then optimised against — and optimising against a calibrated signal is a standard way to decalibrate it.

> "As our evaluation pipeline reuses the same prompting and tooling stack as our data-collection pipeline, most labels are internally consistent by construction."

A circularity admission in limitations. Honest, and exactly the contamination `synthoracle` was built to avoid. Their mitigation is red-teaming and manual audit rather than an independent oracle.

> "given we found our probes to be decently well-calibrated, it is worth considering the use of a more expensive verifier when the confidence of the probe is low."

Cascaded verification gated on cheap-probe confidence — independent convergence on `assay`'s two-tier cost architecture.

> "One rare but salient behavior we observed was the degeneration of inline completions as repeated or severe interventions pushed the model solidly out of distribution."

Degeneration under repeated intervention. A third instance, across the four papers, of optimisation pressure producing distributional pathology.

---

## Study Questions

**Warm-up:** (1) Why can a probe detect a concept the model cannot reliably verbalise, and why does that make features attractive as reward? (2) State the two anti-Goodhart mechanisms and which failure channel each closes. (3) What does "non-identifiability under limited data coverage" mean for a reward model?

**Intermediate:** (4) The frozen-parameter trick removes the activation-shaping channel. Enumerate the channels it leaves open, and rank them by how much freedom the policy has in each — for RLFR's retraction task versus PROPEL's task-generation task. (5) Calibration is the load-bearing assumption. Sketch the mechanism by which optimising against a calibrated probe decalibrates it, and name the measurement that would detect this mid-training. (6) Their reward-pipeline transfer check (ordering invariant to activation source) is single-seed, end-of-training. Design the version of that check that would actually license the frozen-probe defence at PROPEL's optimisation budget.

**Advanced:** (7) Formalise the dose-response experiment their limitation names: what is the pressure knob, what is the outcome variable, what distinguishes "the student learned the behaviour" from "the student learned to satisfy the probe," and what is the cheapest ground truth that separates them? Compare against `waterline`'s λ-sweep design and say which components port. (8) `Labels Not Loss` characterises when optimising a fitted reward vector helps versus harms a hidden target, via the cosine between trained weights and the true objective. State precisely what would have to hold for that condition to transfer to a linear probe over residual-stream features — and what breaks if it doesn't. (9) They restrict probe expressivity to constrain the solution set. Is low expressivity a *reliable* Goodhart defence, or does it merely change which pathological mappings are reachable? Construct a case where a linear probe is *more* exploitable than a larger one.

---

## Challenge Corner

**C1 — The defence that carries the entire stack is validated once, in the easiest setting, and the authors say so.** Frozen parameters + low expressivity are tested on hallucination-retraction with Gemma-3-12B-IT, where the policy's action space is narrow (retract or correct a span) and the natural-text constraint bites hard. PROPEL applies the same defence to a *generator* emitting arbitrary task text, at 4B–27B, over 30 RL steps, with a vastly larger space of probe-satisfying outputs. RLFR's own limitation — *"plausible that with further optimization this changes"* — is inherited without re-testing. **Whether the defence transfers across action-space size is unmeasured and load-bearing.**

**C2 — "Ordering invariant to activation source" is a weaker result than it is used for.** Single seed, best-of-32, measured at the end of training. It shows the probe's *ranking* survived this run's optimisation; it does not show ranking survives more pressure, nor that the ranking was ever correct in the tail that matters. The tail is precisely where a cascaded verifier would be invoked (their own future work), i.e. where they already suspect the probe is weakest.

**C3 — Circular evaluation, acknowledged but not resolved.** Eval and data-collection pipelines share a prompting/tooling stack; labels are *"internally consistent by construction."* Red-teaming and manual audit are the mitigation. This is the failure `synthoracle` was designed against — the fix is ground truth that is structurally independent of the labeller, not more auditing of a shared stack. For `assay`: a live example of how a careful, well-resourced team still ends up with a partly self-referential oracle, and an argument for the contamination-free-by-construction stance.

**C4 — Low expressivity constrains the solution set; it does not obviously shrink the *pathological* part of it.** A linear probe over a rich pretrained feature basis can still admit directions that correlate with the target on the training distribution and diverge off it — arguably more so, since it cannot represent the corrections that would rule them out. The paper asserts the restriction helps and cites the overoptimisation literature, but offers no identifiability argument or empirical sweep over probe capacity. **This is the cleanest technical gap in the paper**, and it is directly adjacent to published work on reward-vector identifiability.

**C5 — Does it threaten `assay`? No — and it is the strongest *support* on the list.** RLFR forecasts a *property of a completion* (is this claim hallucinated) from internals, to reward an intervention. `assay` forecasts *what training will do to a policy* from black-box probes, to audit an environment. Different quantity, different access model, different consumer. What it supplies is (i) the strongest available statement that cheap proxies for expensive supervision *work*, (ii) the identifiability framing that connects the whole Vmax stack to `Labels Not Loss`, and (iii) a named open problem — the pressure-threshold question — that `assay` and `waterline` are jointly shaped to answer.

---

## Connection to Project

### Verdict

**No novelty threat; the highest-value paper on the list for positioning.** It is the upstream dependency of #13, it frames the Vmax stack's central risk in *identifiability* terms that connect directly to published prior work of ours, and its limitations name a dose-response experiment that `waterline`'s rig was built to run. Read together, #16 and #13 assert a threshold exists and that nobody has located it.

### Differentiation table

| Dimension | RLFR (Goodfire, 2026) | `assay` |
|---|---|---|
| **Forecast quantity** | Is this claim hallucinated (property of a completion) | What will RL do to the policy (property of an environment) |
| **Access model** | White-box: probes over residual-stream features | **Black-box: inference-only** |
| **Consumer** | Reward signal inside the RL loop | Pre-flight audit outside the loop |
| **Anti-Goodhart** | Frozen params + low-expressivity hypothesis class | **Measure the gap** against a held-out grader |
| **Ground truth** | LLM-judge labels from a shared tooling stack (acknowledged circularity) | Independently authored grader / held-out suite |
| **Pressure-threshold** | Named as open in limitations | The kind of question `assay`/`waterline` are built for |

### What to adopt

- **The identifiability framing** — "non-identifiability under limited data coverage → reward hacking" is a cleaner statement of `assay`'s motivating mechanism than our current docs use, and it is the field's own vocabulary.
- **The off-target battery** (benchmarks + split KL + blind preference rating + randomly-selected qualitative examples alongside cherry-picked ones) as the Run-stage template.
- **Cascaded verification gated on cheap-signal confidence** — independent convergence on the two-tier design; cite it as such.
- **The corrected citation** — *"Features as Rewards…"*, arXiv:2602.10067, Goodfire AI. #13 refers to it only by framework name.

### What to differentiate (for reviewers)

RLFR uses internals to *supply* a reward; `assay` uses black-box probes to *audit* one. Their Goodhart defence is architectural and validated once; ours is a measurement, and measurement is what tells you whether the architecture held. The two are complements, and their limitations section says as much without naming us.

### Where RLFR tightens our design *(proposals — owner adjudicates)*

- The **pressure-threshold question** (C1, Study Q7) may be a better external-validation arm than the Hub field report: it is named as open by the authors, sits upstream of a production pipeline, and reuses a rig that already exists.
- **Probe-capacity sweep** (C4) is a cheap, self-contained study with an obvious null and a published prior to argue against.
- Our contamination stance gains a **concrete cautionary exhibit** (C3) from a careful team — useful in the writeup, better than asserting the principle.
- Whether the `Labels Not Loss` condition transfers to feature-space probes (Study Q8) is a **genuinely open question and must be posed as one.** The geometry is analogous; the identifiability structure of the target is not obviously the same. Overclaiming here would be the single easiest way to lose credibility with this audience.

---

## Synthesis Pointers
*(Feeds `synthesis-engagement.md` — tagged `Source: #16`.)*
1. **The frozen-probe defence is one trick carrying the whole stack**, validated in one narrow setting, inherited by #13 without re-testing at a larger action space. `Source: #16`
2. **The authors name the pressure-threshold question as open** — *"plausible that with further optimization this changes… further work is needed."* Read with #13's policy-drift limitation, a threshold is asserted to exist and left unlocated. `Source: #16`
3. **The stack's risk is framed as reward-model non-identifiability under limited coverage**, explicitly within inverse-RL — the formal bridge to `Labels Not Loss`. Transfer to feature-space probes is **open, not established**. `Source: #16`
4. **Circular evaluation acknowledged** (eval reuses the data-collection stack) — a cautionary exhibit for the contamination-free stance. `Source: #16`
5. **Cascaded verification gated on probe confidence** — independent convergence on two-tier measurement. `Source: #16`
6. **ID lesson:** #13 cites this by framework name (RLFR); the real title is different. The ⬤ rule earned its keep. `Source: #16`
7. **Off-target battery** — adopt as the Run-stage template, including reporting randomly-selected examples beside cherry-picked ones. `Source: #16`

---

## Discussion Notes
*To be filled during interactive discussion (Process step 3).* Open threads:
- Does the `Labels Not Loss` cosine condition actually transfer to a linear probe over residual-stream features, or only analogise? This needs a careful answer before it is said out loud to anyone at Vmax or Goodfire — it is the highest-upside and highest-risk connection in the whole reading.
- Is the pressure-threshold experiment (C1/Q7) an `assay` arm, a `waterline` extension, or its own thing? It reuses `waterline`'s rig almost exactly but answers an `assay`-shaped question about a third party's method.
- Goodfire is a distinct organisation from Vmax (Connor Watts spans both, as equal-contribution first author here and second author on #13). Worth knowing which conversation is which.
