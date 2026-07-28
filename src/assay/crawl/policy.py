"""The model seam.

``train()`` never touches ``transformers`` or an optimizer directly — it talks to this protocol.
That is what makes the loop testable on a machine with no GPU, which matters more than it sounds:
**a wrongly-masked loss still trains, it just trains the wrong thing.** Masking and shape bugs are
silent, so they have to be caught by fast local tests rather than discovered on a GPU run.

The division of labour is deliberate (``CLAUDE.md`` §7):

- **In ``train()``** — sampling, grading, grouping, advantages, and *building the loss*. The loss is
  the heart of policy gradient and the phase's learning target; hiding it behind an adapter would
  defeat the point.
- **Behind ``Policy``** — generation, gathering per-token log-probs, gradient buffers, the optimizer
  step. Torch bookkeeping.

``ToyPolicy`` uses **real autograd** on a tiny parameter, with *scripted* generation so tests control
the reward distribution exactly. Real gradients, deterministic rewards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from assay.crawl.tasks import Prompt


@dataclass(frozen=True)
class Rollout:
    """One sampled completion, with everything the loss needs.

    ``token_ids`` are the **completion** tokens only — prompt tokens must never enter the loss, or
    the policy is rewarded for text it did not choose.
    """

    prompt: Prompt
    text: str
    token_ids: list[int]

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


class Policy(Protocol):
    """What the loop needs from a model. Everything torch-shaped lives behind here."""

    def generate(self, prompts: Sequence[Prompt], *, k: int) -> list[list[Rollout]]:
        """Sample ``k`` completions per prompt, returned grouped ``[prompt][rollout]``.

        The grouping is load-bearing: a group is one prompt's rollouts, which is what makes the
        baseline an estimate of ``E[R|x]``. Flattening here would make it impossible to recover.
        """
        ...

    def logprobs(self, rollouts: Sequence[Rollout]) -> Any:
        """Summed log-prob of each rollout's completion tokens. Differentiable, shape ``[n]``."""
        ...

    def entropy(self, rollouts: Sequence[Rollout]) -> float:
        """Mean per-token policy entropy over the batch. Ablation B's slower-moving signal."""
        ...

    def kl_to_reference(self, rollouts: Sequence[Rollout]) -> Any:
        """Per-rollout KL to the frozen reference policy. Differentiable, shape ``[n]``."""
        ...

    def optimize(self, loss_first_half: Any, loss_second_half: Any) -> tuple[float, float]:
        """Backward both halves, step the optimizer, return ``(grad_norm, half_batch_cosine)``.

        Taking the halves separately is what makes ablation A's primary metric available: the two
        halves are independent samples of the same expected gradient, so their cosine measures
        estimator variance directly, with no trend to correct for.

        **This costs no extra backward passes** — every sample is still backwarded exactly once,
        accumulated into two buffers rather than one, and summed for the actual update.

        **Contract: the caller scales each half by the FULL rollout count**, so ``loss_a + loss_b``
        is already exactly the full-batch mean. Implementations must therefore *not* divide the
        accumulated gradient — halving would be correct only for equally-sized halves, which fails
        whenever the group count is odd.

        **The caller must split by GROUP, never by rollout.** Advantages inside a group sum to
        exactly zero, so any within-group split hands the two halves complementary pieces of one
        contrast rather than independent replicates — their common-mode components end up
        anti-correlated by construction, biasing the cosine downward on *both* arms and compressing
        the very difference ablation A is trying to detect. Splitting by group gives each half
        disjoint prompts, hence genuinely independent draws.

        When either half's gradient is ~0 the cosine is mathematically undefined; implementations
        return ``0.0``. Read it together with ``grad_norm``: a near-zero norm means the cosine
        carries no information (this is the normal state under ablation D).
        """
        ...


class ToyPolicy:
    """Test double with **real autograd** and *scripted* generation.

    Real gradients, because the bugs worth catching locally are in masking, shapes and the
    two-buffer gradient trick — faking those would test nothing. Scripted text, because tests need
    to control the reward distribution exactly rather than hope a toy model produces one.

    The parameter is a single logit vector over a tiny vocabulary, so log-probs are position- and
    prompt-independent. That is enough to exercise every tensor path the loop takes.
    """

    def __init__(
        self,
        *,
        p_correct: float = 0.5,
        p_parse_fail: float = 0.0,
        vocab: int = 16,
        seed: int = 0,
        lr: float = 0.1,
        min_tokens: int = 3,
        max_tokens: int = 9,
        reward_from_tokens: bool = False,
        signal_token: int = 0,
    ) -> None:
        import torch

        torch.manual_seed(seed)
        self.logits = torch.nn.Parameter(torch.randn(vocab))
        self.reference = self.logits.detach().clone()
        self.opt = torch.optim.SGD([self.logits], lr=lr)
        self._vocab = vocab
        self._p_correct = p_correct
        self._p_parse_fail = p_parse_fail
        self._seed = seed
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._call = 0

        # When True, correctness depends on the *sampled tokens* (does ``signal_token`` appear?)
        # instead of a coin flip. Without this the reward is independent of the tokens, so
        # E[A * grad log pi] = E[A] * E[grad log pi] = 0 — the true gradient is exactly zero and
        # there is no signal-to-noise ratio to measure. Required for any local check of ablation A.
        self._reward_from_tokens = reward_from_tokens
        self._signal_token = signal_token

    def generate(self, prompts: Sequence[Prompt], *, k: int) -> list[list[Rollout]]:
        import random

        self._call += 1
        out: list[list[Rollout]] = []
        for prompt in prompts:
            rng = random.Random(f"{self._seed}:{self._call}:{prompt.prompt_id}")
            group = []
            for _ in range(k):
                # Varying lengths on purpose: a mask bug that ignores length is invisible when
                # every sequence is the same size.
                n = rng.randint(self._min_tokens, self._max_tokens)
                ids = self._sample(n, rng)

                if self._reward_from_tokens:
                    correct, parse_fail = self._signal_token in ids, False
                else:
                    draw = rng.random()
                    correct = draw < self._p_correct
                    parse_fail = self._p_correct <= draw < self._p_correct + self._p_parse_fail

                if correct:
                    text = f"<answer>{prompt.answer}</answer>"
                elif parse_fail:
                    text = "not sure"
                else:
                    text = f"<answer>{int(prompt.answer) + rng.choice([-2, -1, 1, 2])}</answer>"
                group.append(Rollout(prompt=prompt, text=text, token_ids=ids))
            out.append(group)
        return out

    def _sample(self, n: int, rng: Any) -> list[int]:
        """Draw ``n`` token ids **from the policy's own distribution**.

        Not a detail. Drawing uniformly instead breaks ``E[grad log pi] = 0``, the identity that
        makes a baseline unbiased and makes the "push everything up" common component cancel.
        Without it the no-baseline arm keeps a large shared component and its two half-gradients
        agree spuriously — ablation A's cosine comes out *backwards*, which is exactly what an
        earlier version of this double produced.
        """
        import torch

        generator = torch.Generator().manual_seed(rng.randrange(2**31))
        probs = torch.softmax(self.logits.detach(), dim=-1)
        return [int(t) for t in torch.multinomial(probs, n, replacement=True, generator=generator)]

    def logprobs(self, rollouts: Sequence[Rollout]) -> Any:
        import torch

        log_probs = torch.log_softmax(self.logits, dim=-1)
        return torch.stack([log_probs[r.token_ids].sum() for r in rollouts])

    def entropy(self, rollouts: Sequence[Rollout]) -> float:
        import torch

        del rollouts
        log_probs = torch.log_softmax(self.logits, dim=-1)
        return float(-(log_probs.exp() * log_probs).sum().item())

    def kl_to_reference(self, rollouts: Sequence[Rollout]) -> Any:
        import torch

        log_p = torch.log_softmax(self.logits, dim=-1)
        log_q = torch.log_softmax(self.reference, dim=-1)
        per_token = (log_p.exp() * (log_p - log_q)).sum()
        return torch.stack([per_token * len(r.token_ids) for r in rollouts])

    def optimize(self, loss_first_half: Any, loss_second_half: Any) -> tuple[float, float]:
        import torch

        self.opt.zero_grad()
        loss_first_half.backward(retain_graph=True)
        grad_a = self._flat_grad().clone()

        loss_second_half.backward()
        grad_total = self._flat_grad().clone()
        grad_b = grad_total - grad_a

        norm_a, norm_b = grad_a.norm(), grad_b.norm()
        if float(norm_a) < 1e-12 or float(norm_b) < 1e-12:
            cosine = 0.0  # undefined; read alongside grad_norm
        else:
            cosine = float(torch.dot(grad_a, grad_b) / (norm_a * norm_b))

        # No division: the caller scales each half by the FULL rollout count, so the accumulated
        # g_A + g_B is already exactly the full-batch gradient. Halving here would be correct only
        # when the two halves contain equally many rollouts.
        grad_norm = float(self._flat_grad().norm())
        self.opt.step()
        return grad_norm, cosine

    def _flat_grad(self) -> Any:
        import torch

        grad = self.logits.grad
        return torch.zeros_like(self.logits) if grad is None else grad.detach().flatten()
