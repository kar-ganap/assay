# What we've built and what we've learned — in plain English

> Written 2026-08-06, covering Phases 0.1 through 0.4 complete and 0.5 in progress.
> No jargon. Every number traceable to a committed script. For the technical versions see
> `docs/phases/*-retro.md`; for the numbers, `experiments/*/results/`.

---

## The problem this project is about

When you train an AI system by rewarding good behaviour, something has to decide what counts as
good. That something is a piece of software — a **grader**. It might run the code the AI wrote and
check whether the tests pass, or check whether a maths answer is correct.

Graders are software, so graders have bugs. And a system being trained by reward is, by
construction, searching hard for anything that scores well. If there is a cheap way to make the
grader say "good" without actually doing the task, the system will usually find it. Practitioners
building these training environments report this as their single biggest problem.

The expensive part is that **nobody can tell you whether a training environment is any good until
after they've paid to train on it.** Industry estimates put that at roughly $2,400 per task. You
build the environment, you spend the compute, and only then do you discover whether you taught the
skill or taught the loophole.

**This project asks whether you can find out beforehand, for almost nothing** — by probing the
environment with a frontier model at inference time, before any training, and predicting what
training would have done.

The catch, and the reason this is a research project rather than a product: **that prediction has to
be checked.** Which means actually training models on deliberately flawed environments and seeing
whether the cheap advance warning matched the expensive outcome.

### Why this project exists at all

Every previous project in this portfolio measured things about AI systems without ever training one.
That is a real gap: for the labs and startups building these environments, training is the whole
question. So there is a hard constraint on this project — **it must produce a training curve that is
genuinely ours**, built and understood from the ground up rather than borrowed.

That constraint drove the entire first stage.

---

## Stage 0 — "Crawl": prove the machinery works before trusting it

The work is organised into four stages, each independently publishable if we stop there. Stage 0 has
one job: establish that we can train models, that our measurements mean what we think, and that the
central assumption the whole project rests on is actually true.

Total budget for the stage: about $17. Spent so far: about **$15.50**, and the two most important
results in it cost **nothing at all**.

---

## Phase 0.1 — Building the training algorithm by hand

### What we did and why

We wrote the training algorithm from scratch rather than using an existing library.

That looks like wasted effort — working implementations are freely available. The reason is that the
project's later claims depend on knowing exactly what the training loop does. If we later say "this
environment taught the loophole," we need to be certain that isn't an artifact of a setting we
didn't understand in a library we didn't read. Writing it by hand is how you earn that certainty.

We also deliberately **broke** the algorithm in four specific ways, each designed to make one hidden
mechanism visible.

### What happened

**The main result: it works.** On three-digit addition, the model went from getting 43% right to
getting **92% right** (±1.8% across repeated runs). Gradients flow, we own the curve, the constraint
is satisfied.

**The four deliberate breakages taught more than the success did:**

**Breakage B is the one that matters most for the whole project.** We built a grader that pays out
for something other than being correct — a deliberately bad grader. The model comprehensively
exploited it: it scored **99.3% by the grader that paid** while achieving only **47.4% by the grader
that actually checked correctness.** That gap — 52 percentage points between "what we rewarded" and
"what we wanted" — is the phenomenon this entire project exists to predict.

We also tested a safety mechanism meant to stop models drifting too far from their original
behaviour. **It didn't help.** Removing it actually made the gap *smaller*, consistently across
every repeat. The mechanism was consuming more than half the training signal and restraining
nothing.

**Breakage D was confirmed exactly, and it is the most expensive lesson.** We built a situation
where the training signal is mathematically guaranteed to be zero — and the training ran anyway, for
200 steps, three times, burning about 40 minutes of GPU time each, producing precisely nothing. The
software gave no indication anything was wrong. This directly informed later phases: a training run
can look completely healthy while doing no learning at all.

**Breakage A was a falsified prediction, and we published it as one.** We predicted a specific
signature and got the opposite. Rather than quietly reinterpreting, we recorded it as failed, then
worked out that the comparison was *structurally incapable* of answering the question for three
independent reasons — and rebuilt it. The rebuilt version found that a textbook-predicted benefit
simply doesn't appear in our setting.

### What it implies

- The phenomenon we want to predict is real, large, and reproducible on demand. We can manufacture a
  52-point gap between the reward and the goal whenever we need one.
- A standard mitigation that the field reaches for did not work here — and would have been reported
  as working if we hadn't tested both directions.
- **Three separate claims that looked solid from one run reversed once we ran three.** One flipped
  sign entirely. This became a standing rule and it has since saved us twice.

**Cost: ~$25, against a plan that said $5.** Most of it went on crashes during setup that produced no
usable output. That overrun is why every later phase measures cost before committing to it.

---

## Phase 0.2 — Translating to the tools everyone else uses

### What we did and why

We rebuilt the same task using the standard community toolchain and published it, then had an
independent trainer we didn't control train on it.

The rationale: the hand-built version proves we understand the algorithm, but every later stage runs
on the shared toolchain, and the project's central artifact has to be expressed in a form other
people can use. Doing the translation on the one task we understand completely means any
disagreement is a fact about the translation rather than about the task.

### What happened

**Everything passed, and it cost nothing** — we found a free compute tier and used it.

The headline number is that the independent trainer reached **99.8% accuracy** on held-out problems.
But the more meaningful number is the very first measurement, before any training: **58.8%**, against
our hand-built version's **57.1% ± 1.9%**. An independent system, on an independently published
environment, starting inside the band we'd measured ourselves. That agreement is the actual evidence
the translation is faithful.

**Three things we learned by hitting them:**

- The published environment **cannot import our research code** — it runs on someone else's
  infrastructure. This forced a cleaner separation than we'd have chosen voluntarily, and the design
  is better for it.
- **The standard toolchain's headline progress number is computed over a filtered subset** of the
  data, not everything. If you read it naively during training you get a systematically optimistic
  picture. This mattered enormously in Phase 0.4.
- The free tier has **three separate undocumented restrictions**, discovered one at a time.

### What it implies

We can now put an environment in front of the wider ecosystem and have it work. That is a
prerequisite for the field-report work planned in the final stage, and it's what made the next two
phases free instead of costly.

---

## Phase 0.3 — The reproduction that turned out to be impossible

### What we did and why

The plan was to reproduce a well-known public result — a small model learning a genuine reasoning
task — to prove our training loop works on something harder than arithmetic.

Arithmetic is pattern-matching; a model can do it without anything resembling reasoning. Reproducing
a reasoning result would retire the objection that our loop only works on easy problems.

### What happened

**We never ran it, and that was the right call.** Three cheap measurements — $2.21 against a $10
budget — showed the reproduction was ill-posed before we spent the money.

**First: the task is unlearnable at any scale we can afford.** We measured how often the models
solve the problem by chance before training. If that number is too low, there's nothing for training
to amplify and a failure would tell us nothing about our loop. At both model sizes we could afford,
the task was far below the threshold we'd set in advance. A failed run would have been attributable
to the task, not our code — which means it couldn't have retired the assumption it existed to retire.

**Second, and more consequential: the target had no number to reproduce.** Our own rules require
recording the original published value and computing the difference from ours. The target publishes
a cost estimate and a qualitative claim — no accuracy figure, no configuration, no metrics. **There
was nothing to compare against, and no amount of compute would have created one.** That was knowable
from a five-minute read on day one.

We also found that a claim we'd inherited from an earlier automated research pass — that a certain
model size learns this task — was **simply wrong**, and our own measurement contradicted it.

**Third: a fast-sampling shortcut is not free.** Everyone speeds up training by generating text with
one piece of software and scoring it with another. We measured whether the two agree. Over a
realistic response length, the correction factor between them ranges from **0.22 to 1.55** where we
needed it inside 0.9 to 1.1. It stays correct on average and becomes useless for any individual
response. We'd have adopted this optimisation without checking if we hadn't measured it.

### What it implies

- **A new standing rule:** read a reproduction target's paper for a specific publishable number
  *before* putting it on the schedule. This one line would have saved the whole phase.
- A safeguard we'd previously disabled on the grounds it wasn't needed turns out to be needed as
  soon as we adopt the standard fast-sampling setup. That's now flagged for the next stage.
- **Cheap measurements that kill an expensive plan are the best value in the project.** $2.21 saved
  $10 and, more importantly, saved us from a result that couldn't have meant anything.

---

## Phase 0.4 — The one that had to work: is the loophole even findable?

### What we did and why

This phase tested the assumption everything else depends on.

The concern was concrete. A published industry result found that models only started exploiting
reward loopholes after about 1,500 training steps — and we can afford roughly 200. If loopholes are
only findable far beyond our budget, then every environment we test will look clean for the wrong
reason, and the entire project is unbuildable.

So we built a deliberately broken grader — it pays full reward whenever a specific word appears,
regardless of whether the task was done — and measured how many training steps a small model needs
to discover that.

We used three words chosen to differ in how often the model says them unprompted, spanning a
hundred-fold range. The reasoning: if what the model *already does occasionally* is what training
amplifies, then the common word should be exploited fastest and the rare one slowest. That gives a
prediction rather than just a demonstration.

**One accident made the phase much more interesting.** When we measured how often our model says
each word, the ordering came out *different* from the published one — reliably, on a re-measurement
at four times the sample size. That set up two predictions that contradict each other: theirs said
one word would be exploited first, ours said the other. Only one could be right, which is a far
better experiment than "do we get the same numbers."

### What happened

**The gate passed decisively, and this is the phase's most important result.** Every single one of
15 training runs learned to exploit the grader completely, in **8 to 40 steps** — not 1,500. The
project's biggest risk does not apply at this scale. And it cost **nothing**.

**A detail worth knowing:** five runs were reported as *failed* by the platform. They hadn't failed.
They'd learned the exploit so thoroughly that every attempt scored perfectly, which left the training
software with nothing to learn from, so it gave up. They failed by succeeding. A pipeline trusting
the platform's status would have thrown away five of the cleanest results in the phase.

**The interesting prediction came back inconclusive.** The two words whose ordering was in dispute
turned out to be statistically indistinguishable. The first nine runs leaned our way; six more leaned
the other way; together they cancel.

This is a real answer rather than a shrug, and the reason is worth stating: the design was capable of
detecting a difference as large as the published one, and it found nothing that size. Our best
estimate of the gap between them is about 5 steps, and the range consistent with our data runs from
about −8 to +10 steps. **The published claim is a 26-step difference. That is ruled out.** Anything
smaller than about 10 steps, we simply can't see.

**The prediction does survive in weaker form.** The word the model says 15–35 times more often than
the others *is* exploited much faster — 9 steps against 29. So how often a model already does
something predicts how fast training amplifies it **when the difference is large**, and predicts
nothing when it's small. Worth flagging: that conclusion rests on a single comparison, since we only
tested three words.

**And we found something we weren't looking for, which may be the phase's most useful output.**

The project has a pre-registered screening rule for deciding which environments are worth training
on: measure how often the model exploits the grader by chance in 64 attempts, and reject anything
below 1-in-64 as unreachable. **Two of our three words fell below that line and were exploited
anyway**, comfortably, well inside the budget.

The diagnosis is sharper than "the threshold is wrong." One-in-64 is simply *the smallest thing you
can see with 64 samples*. It was never a statement about reachability — it was our sampling budget
wearing a scientific label. We could only see our own rates at all because we'd measured them with
16,000 samples. **A literal application of the screen would have wrongly rejected a usable
environment 42–68% of the time.**

### What it implies

- **The core assumption holds.** Loopholes are cheap and fast to find at small scale. The project is
  buildable.
- **The screening rule needs rebuilding before the expensive stage**, and the fix is cheap: take more
  samples so the screen can see what it's meant to see, then set the threshold from measured
  evidence rather than from sample size.
- **"We measured our own screening tool's error rate" is a better contribution than a screen that
  passes quietly.** This is now the phase's most novel finding.
- **A significant limitation, stated plainly:** this grader was trivial — say one word, get paid,
  with no genuine task to trade off against. The model wasn't choosing between an honest route and a
  shortcut; there was only the shortcut. So what we've retired is *"small models find a free token
  in our budget,"* not *"small models will exploit a realistic environment."* That distinction now
  sits at the top of the limitations rather than the bottom.

---

## The part that generalises: our own quality controls failed, twice

Two failures in this phase are worth more than the results, because they'll recur.

### We wrote the scoring code before we had the data, and it still gave a wrong answer

To prevent unconsciously picking the analysis that flatters the result, we wrote and committed the
scoring code while all the training runs were still going. The timestamps prove it. That's textbook
practice and we followed it.

**It still reported the prediction confirmed when the data said nothing.** The code compared two
averages and reported which was smaller, without ever considering how spread out the underlying runs
were. On numbers whose spread was three times the difference being measured, that's meaningless.

The root cause is subtle and worth carrying: **the plan we wrote in advance listed only two possible
outcomes — prediction confirmed, or prediction refuted.** It never anticipated "no difference." Given
no box for the actual answer, the code returned the nearest box it had. Writing the analysis in
advance protects against choosing it afterwards; it does nothing about an analysis with nowhere to
put the real result — and that failure is harder to spot, because every visible part of the process
looks correct.

### Then we made the same mistake in the opposite direction

Having caught that, we wrote up the result as **refuted** — in four documents, including one intended
for external readers. That was also wrong. The scoring code said *unresolved* throughout, and the
prediction's literal wording was actually *satisfied* by every average we measured.

Failing to find an effect is not the same as showing there isn't one. We'd committed the identical
error one step over — and it was harder to notice, because "we falsified our own hypothesis" is a
flattering story to tell about yourself. A bias toward appearing rigorous is still a bias.

Both were caught by a structured review at the stage boundary: three independent reviewers with no
prior context, given the documents and told to attack them. Two of the three converged on the same
problems without conferring. **That review is now the most valuable process step in the project**,
and everything above reflects its corrections.

---

## Where things stand

| Phase | Status | Cost |
|---|---|---|
| 0.1 — training loop built by hand | ✅ complete | ~$25 |
| 0.2 — ported to standard toolchain, published | ✅ complete | $0 |
| 0.3 — reproduction disqualified, three screens run | ✅ complete | $2.21 |
| 0.4 — loophole reachability | ✅ complete | **$0** |
| 0.5 — literature review | 🔄 in progress | $0 |

**Total spent: about $15.50 of ~$17 authorised for this stage.** The two highest-value phases were
free.

### What Phase 0.5 has already changed

We're reading the key related work first-hand rather than relying on summaries. The first paper read
in full **contradicted our stated novelty claim** — we had written that nobody had asked whether an
environment's outcome is predictable before training, and that paper explicitly recommends exactly
that. The claim narrowed to something narrower and more defensible: others have *proposed*
pre-training checks; nobody has *validated one against the training outcome it claims to predict*.

A broader review then found more. Two papers from June 2026 do work close to ours, one auditing real
training environments and correlating the result to a downstream outcome. And the theoretical basis
for our own core assumption — that training amplifies what a model already does rather than
discovering something new — turns out to be an active, contested literature we hadn't engaged with
at all. That's a gap worth closing, and closing it makes our position stronger rather than weaker.

### The two open questions before the next stage

1. **Rebuild the screening rule** — more samples, threshold set from evidence, error rate reported.
2. **Resolve a contradiction we've created.** We've now written that the screening measure must not
   be used to rank environments. But the next stage's main method *is* a ranking, and it uses that
   measure as one input. Both can't stand. This needs deciding before any money is spent.

---

## The honest summary

The machinery works: we can train models, we can manufacture the failure we want to predict, we can
publish environments other people's systems can use, and we've shown the failure is cheap to
reproduce at small scale.

The elegant secondary hypothesis came back inconclusive, and the write-up initially claimed more than
the data supported — twice, in opposite directions. Both were caught before anything left the
building, by a review process built for exactly that.

And the most useful result was an accident: the tool we'd planned to use for screening environments
has a measured error rate of 42–68% on precisely the cases it exists to catch. Finding that now, for
nothing, is worth considerably more than the hypothesis we set out to test.
