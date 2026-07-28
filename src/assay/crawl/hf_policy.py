"""``Policy`` backed by ``transformers`` + LoRA. Runs on Modal; cannot be imported locally.

Deliberately thin. Everything error-prone — the causal shift, completion masking, the k3 KL
estimator — lives in ``assay.crawl.logprob``, which needs no ``transformers``, no GPU and no model,
and is tested exhaustively on a laptop. What is left here is glue that a first ``modal run`` will
exercise.

Two implementation notes worth knowing before reading:

**No second model for the reference.** With LoRA the base weights are frozen, so disabling the
adapter *is* the reference policy. That saves a full model copy of GPU memory and guarantees the
reference cannot drift.

**Generation is left-padded.** Decoder-only generation requires it, and it has a second benefit:
every sequence's completion then begins at the same index, which is what makes
``build_completion_mask`` take a single scalar ``prompt_len``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from assay.crawl.logprob import (
    build_completion_mask,
    completion_logprobs,
    entropy_over_completions,
    sequence_kl,
)
from assay.crawl.policy import Rollout
from assay.crawl.tasks import Prompt


class HFPolicy:
    """Llama-class causal LM with a LoRA adapter, behind ``assay.crawl.policy.Policy``."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None,
        learning_rate: float,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        lora_rank: int = 16,
        generate_batch: int = 4,
    ) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        self.model = get_peft_model(
            base,
            LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        # Recompute activations during backward instead of storing them. A 1B model over
        # [128 rollouts, ~50 positions] retains roughly 7-8 GB of activations across 16 layers
        # without this; the trade is ~30% more compute for an order of magnitude less memory.
        # With a frozen base and only LoRA trainable, the inputs to each checkpointed segment do
        # not require grad, and checkpointing silently saves nothing. This makes them require it.
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False  # incompatible with checkpointing, and unused here

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=learning_rate)

        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        # Prompts per generate() call. With k=8 this is 4*8 = 32 sequences in flight rather than
        # 128 — same total work, a quarter of the generation-time peak. Purely a memory knob; it
        # cannot affect results because each prompt's group is generated independently.
        self._generate_batch = generate_batch
        self._prompt_len = 0  # set by generate(), consumed by logprobs()

    # -- generation ---------------------------------------------------------------------

    def _render(self, prompts: Sequence[Prompt]) -> list[str]:
        """Same chat template as the calibration sweep — a base rate measured under a different
        prompt is not the base rate that predicts this run."""
        return [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p.question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    def _completion_length(self, ids: Any) -> int:
        """Tokens up to and including the first EOS.

        Counting to EOS rather than to the first pad matters because ``pad_token`` is set to
        ``eos_token``: after an early stop the two are indistinguishable by value. Choosing to stop
        is an action the policy took, so the EOS itself is included.
        """
        eos = self.tokenizer.eos_token_id
        for position, token in enumerate(ids.tolist()):
            if token == eos:
                return position + 1
        return int(ids.shape[0])

    def generate(self, prompts: Sequence[Prompt], *, k: int) -> list[list[Rollout]]:
        import torch

        out: list[list[Rollout]] = []
        for start in range(0, len(prompts), self._generate_batch):
            batch = list(prompts[start : start + self._generate_batch])
            encoded = self.tokenizer(
                self._render(batch), return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            self._prompt_len = int(encoded["input_ids"].shape[1])

            with torch.no_grad():
                sequences = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    max_new_tokens=self._max_new_tokens,
                    num_return_sequences=k,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            completions = sequences[:, self._prompt_len :]
            for i, prompt in enumerate(batch):
                group = []
                for j in range(k):
                    ids = completions[i * k + j]
                    n = self._completion_length(ids)
                    kept = ids[:n]
                    group.append(
                        Rollout(
                            prompt=prompt,
                            text=self.tokenizer.decode(kept, skip_special_tokens=True),
                            token_ids=[int(t) for t in kept.tolist()],
                        )
                    )
                out.append(group)
        return out

    # -- scoring ------------------------------------------------------------------------

    def _batch(self, rollouts: Sequence[Rollout]) -> tuple[Any, Any, int]:
        """Re-encode prompt+completion into one right-padded tensor, plus the completion mask."""
        import torch

        prompt_ids = self.tokenizer(
            self._render([r.prompt for r in rollouts]),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = int(prompt_ids.shape[1])

        lengths = [r.n_tokens for r in rollouts]
        total = prompt_len + max(lengths)
        ids = torch.full(
            (len(rollouts), total), self.tokenizer.pad_token_id, dtype=torch.long
        )
        ids[:, :prompt_len] = prompt_ids
        for row, rollout in enumerate(rollouts):
            if rollout.n_tokens:
                ids[row, prompt_len : prompt_len + rollout.n_tokens] = torch.tensor(
                    rollout.token_ids, dtype=torch.long
                )

        mask = build_completion_mask(prompt_len, lengths, total)
        return ids.to(self.model.device), mask.to(self.model.device), prompt_len

    def _trunk_and_head(self) -> tuple[Any, Any]:
        """The transformer body and the LM head, separately.

        LoRA lives inside the body's attention modules, so calling the body directly still applies
        the adapter — and disabling the adapter still yields the reference policy.
        """
        base = self.model.get_base_model()
        return base.model, base.lm_head

    def _logits(self, ids: Any, span: tuple[int, int], *, adapter: bool) -> Any:
        """Logits over ``[lo, hi)`` only — **never over the whole sequence**.

        This is the difference between a run that fits and one that does not. The head projects to
        a 128k vocabulary, so full-sequence logits are ``[batch, seq, 128256]``. One completion
        hitting ``max_new_tokens`` stretches the padded sequence to ~336 positions, and at 128
        rollouts that single tensor is **~11 GB in bf16** — while only ~10 of those positions are
        ever scored.

        Hidden states are cheap by comparison (``2048`` wide, not ``128256``), so the body runs over
        the full sequence — it must, for causal attention — and only the completion span is
        projected through the head. Roughly a 30x reduction on the tensor that actually hurts.
        """
        import torch

        trunk, head = self._trunk_and_head()
        lo, hi = span
        if adapter:
            hidden = trunk(input_ids=ids).last_hidden_state
            return head(hidden[:, lo:hi, :])
        with torch.no_grad(), self.model.disable_adapter():
            hidden = trunk(input_ids=ids).last_hidden_state
            return head(hidden[:, lo:hi, :])

    def logprobs(self, rollouts: Sequence[Rollout]) -> Any:
        """The step's **only** grad-carrying forward. Its logits are cached for the other readouts.

        Recomputing them in ``kl_to_reference`` would retain a *second* full activation graph over
        the same batch — which is what put a 1B model over 39 GB on an A100-40GB. Entropy is taken
        from the same pass too, which also fixes a timing bug: ``entropy()`` is called after
        ``optimize()``, so an independent forward there would describe the *post-update* policy
        while every other metric describes the batch that was collected.
        """
        import torch

        from assay.crawl.logprob import completion_span

        ids, mask, _ = self._batch(rollouts)
        span = completion_span(mask)
        if span is None:
            raise RuntimeError("no completion tokens in the batch — nothing to score")

        # One position earlier than the completion starts: scoring position p needs the logits at
        # p-1. Everything downstream then works on consistently sliced views, so the logprob
        # helpers need no notion of an offset — they re-derive the span from the sliced mask.
        lo, hi = max(0, span[0] - 1), span[1]
        window = (slice(None), slice(lo, hi))
        ids_w, mask_w = ids[window], mask[window]

        logits = self._logits(ids, (lo, hi), adapter=True)
        self._cache = (id(rollouts), ids, ids_w, mask_w, (lo, hi), logits)
        with torch.no_grad():
            self._entropy = entropy_over_completions(logits, mask_w)
        return completion_logprobs(logits, ids_w, mask_w)

    def _cached_for(self, rollouts: Sequence[Rollout]) -> tuple[Any, Any, Any, Any, Any]:
        cache = getattr(self, "_cache", None)
        if cache is None or cache[0] != id(rollouts):
            raise RuntimeError(
                "logprobs() must be called before entropy()/kl_to_reference() on the same "
                "rollouts — the forward pass is shared, not recomputed"
            )
        return cache[1], cache[2], cache[3], cache[4], cache[5]

    def entropy(self, rollouts: Sequence[Rollout]) -> float:
        self._cached_for(rollouts)  # fail loudly rather than silently returning a stale value
        return float(self._entropy)

    def kl_to_reference(self, rollouts: Sequence[Rollout]) -> Any:
        ids_full, ids_w, mask_w, span, policy_logits = self._cached_for(rollouts)
        # Only the reference needs a fresh pass, and it runs under no_grad — the adapter disabled
        # IS the frozen base policy, so no second model is loaded.
        reference = self._logits(ids_full, span, adapter=False)
        return sequence_kl(policy_logits, reference, ids_w, mask_w)

    # -- the update ---------------------------------------------------------------------

    def _flat_grad(self) -> Any:
        import torch

        return torch.cat(
            [
                (torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten()
                for p in self.params
            ]
        )

    def optimize(self, loss_first_half: Any, loss_second_half: Any) -> tuple[float, float]:
        """Two backwards into one accumulating buffer — see ``Policy.optimize``.

        Gradients accumulate in PyTorch, so the second backward leaves ``g_A + g_B`` and ``g_B``
        recovers by subtraction. Every rollout is still backwarded exactly once; the only extra
        cost is holding one flattened copy of ``g_A``.
        """
        import torch

        self.opt.zero_grad(set_to_none=True)
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

        # No division: the caller scales each half by the FULL rollout count, so g_A + g_B is
        # already exactly the full-batch gradient. Halving here would be correct only when the two
        # halves hold equally many rollouts, which fails for an odd number of groups.
        #
        # clip_grad_norm_ returns the norm *before* clipping, which is what we want logged.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0))
        self.opt.step()
        return grad_norm, cosine
