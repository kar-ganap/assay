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
        generate_batch: int = 16,
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
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=learning_rate)

        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
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

    def _logits(self, ids: Any, *, adapter: bool) -> Any:
        """Forward pass. ``adapter=False`` disables LoRA, which *is* the reference policy."""
        import torch

        if adapter:
            return self.model(input_ids=ids).logits
        with torch.no_grad(), self.model.disable_adapter():
            return self.model(input_ids=ids).logits

    def logprobs(self, rollouts: Sequence[Rollout]) -> Any:
        ids, mask, _ = self._batch(rollouts)
        self._cached = (ids, mask)
        return completion_logprobs(self._logits(ids, adapter=True), ids, mask)

    def entropy(self, rollouts: Sequence[Rollout]) -> float:
        import torch

        ids, mask, _ = self._batch(rollouts)
        with torch.no_grad():
            return entropy_over_completions(self._logits(ids, adapter=True), mask)

    def kl_to_reference(self, rollouts: Sequence[Rollout]) -> Any:
        ids, mask, _ = self._batch(rollouts)
        return sequence_kl(
            self._logits(ids, adapter=True), self._logits(ids, adapter=False), ids, mask
        )

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
