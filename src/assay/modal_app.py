"""Modal wrapper — remote execution shim.

Backend policy (``CLAUDE.md`` §6): the exploratory grid targets the Prime Sprints free queue
(Llama-3.2-1B) where available; **confirmatory arms run here** (H100 $3.95/h, A100-80GB ~$3.20/h).
Tinker is the fallback that removes infra work.

**Why the calibration sweep runs here too, from the start.** The dev machine is a 2019 Intel
MacBook Pro — no MPS, no CUDA — so there is no local GPU path and ``vllm`` will not install on
macOS x86_64. That turns out to be the right sequencing anyway: the sweep is a cheap ~20-minute job,
so it proves the Modal wiring *before* a training run is bet on it.

Generation goes through HF ``generate`` rather than vLLM so the screen and the hand-rolled GRPO loop
share one code path. A base rate measured on a different sampler is not the base rate that predicts
training. vLLM arrives at Phase 0.2 with ``trl``/``verifiers``.

House pattern (from ``../waterline``): ship the ``assay`` source into the image via
``add_local_python_source``; only third-party deps go through ``pip_install``. This module stays a
thin shim — everything worth testing lives in ``assay.crawl``, which is typechecked and runs on
``FakeSampler`` with zero GPU.

Secrets come from ``.env`` parsed directly by ``_dotenv`` and shipped via ``Secret.from_dict``, so
they reach the *remote* container without ever entering the local process environment (§9).
``Secret.from_dotenv()`` was rejected: it needs a ``python-dotenv`` dependency for six lines of
parsing, and routing through ``load_dotenv`` semantics is exactly what §9 forbids.

Excluded from mypy (see ``pyproject.toml``) because the heavy stack is not installed locally.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
from pathlib import Path

try:
    import modal
except ImportError:  # pragma: no cover - optional dependency
    modal = None  # type: ignore[assignment]


APP_NAME = "assay"

#: Phase 2.2's confirmatory arms. The calibration sweep does not need this tier.
GPU = "H100"

#: Phase 0.1's calibration sweep: ~9k short completions from a 1B model.
SWEEP_GPU = "L4"

#: Phase 0.1's ladder runs. Inference-only fits on an L4, but training adds optimizer state,
#: gradients and a frozen reference model for the KL term. A10G (24 GB) is the smallest tier with
#: headroom. A10G (24 GB) OOMed on the first real run: Llama-3.2 has a 128k vocabulary, so the
#: logits tensor alone is ~1.6 GB and log_softmax over it doubled that. Slicing to the completion
#: span cut ~5x of it, but A100-40GB is bought for margin rather than run against the ceiling.
TRAIN_GPU = "A100-40GB"
TRAIN_TIMEOUT_S = 60 * 120

TIMEOUT_S = 60 * 90

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

#: PINNED 2026-07-27 (desideratum 12 — never resolve to ``main``, which moves).
#: Regenerate with an authenticated GET of ``https://huggingface.co/api/models/{MODEL_ID}`` and read
#: ``.sha``; the token comes from ``.env`` read directly, never via ``load_dotenv`` (§9).
MODEL_REVISION: str | None = "9213176726f574b556790deb65791e0c5aa438b6"

RESULTS_DIR = Path("experiments/phase-0.1-grpo-by-hand/results")

#: Raw generations. Gitignored (``experiments/*/raw/``), never modified, always written — an
#: unexpected summary is undebuggable without them.
RAW_DIR = Path("experiments/phase-0.1-grpo-by-hand/raw")

#: Cap on raw completions returned per (family, setting), to keep the payload sane on the fine pass.
RAW_PER_SETTING = 400

#: Durable artifact store, so a run does not depend on this laptop staying awake.
#:
#: Without it the *local* process is load-bearing: it receives the result and writes the files, so
#: a closed lid or a dropped connection means paying for the compute and keeping nothing. The remote
#: function now persists everything itself and commits before returning; the local write is a
#: convenience. Pair with ``modal run --detach`` so the app also survives the client going away, and
#: recover afterwards with the ``fetch`` entrypoint.
VOLUME_NAME = "assay-phase01"
VOLUME_PATH = "/artifacts"


def _image():  # type: ignore[no-untyped-def]
    """One image for both the sweep and the ladder.

    ``add_local_python_source`` must be the **last** step — Modal rejects any build step after a
    local-file step, because local files are attached at container startup rather than baked in.
    So every ``pip_install`` belongs here, not chained on at the call site.
    """
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "peft")
        .add_local_python_source("assay")
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def _git_dirty() -> bool:
    """True if the working tree differs from HEAD.

    A dirty tree means ``git_sha`` does **not** identify the code that produced the run, so the
    result is not reproducible from the manifest. Recorded rather than hidden — a manifest that
    quietly names the wrong commit is worse than one that admits it.
    """
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return True


def _dotenv(path: Path = Path(".env")) -> dict[str, str]:
    """Parse ``.env`` directly into a dict (§9).

    Deliberately **not** ``Secret.from_dotenv()`` / ``load_dotenv()``: nothing here touches
    ``os.environ``, so a shell-exported token can never silently win over the project-scoped key in
    ``.env``. Also avoids a ``python-dotenv`` dependency for six lines of parsing.

    Returns empty inside the Modal container, where ``.env`` is absent by design — the secret is
    resolved locally at deploy time and injected server-side.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


if modal is not None:
    app = modal.App(APP_NAME)
    artifacts = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    class HFSampler:
        """``assay.crawl.sampling.Sampler`` backed by ``transformers.generate``.

        Left-padded batched sampling with ``num_return_sequences=k``, so one call yields a whole
        group per prompt — the same shape GRPO consumes.
        """

        def __init__(self, model, tokenizer):  # type: ignore[no-untyped-def]
            self.model = model
            self.tokenizer = tokenizer

        def sample(self, prompts, *, k, cfg):  # type: ignore[no-untyped-def]
            import torch

            from assay.crawl.sampling import Completion

            torch.manual_seed(cfg.seed)
            out: list[list[Completion]] = []

            for start in range(0, len(prompts), 16):
                batch = prompts[start : start + 16]
                texts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p.question}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in batch
                ]
                enc = self.tokenizer(
                    texts, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(self.model.device)

                with torch.no_grad():
                    generated = self.model.generate(
                        **enc,
                        do_sample=True,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_new_tokens=cfg.max_new_tokens,
                        num_return_sequences=k,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                prompt_len = enc["input_ids"].shape[1]
                completions = generated[:, prompt_len:]
                for i in range(len(batch)):
                    row = []
                    for j in range(k):
                        ids = completions[i * k + j]
                        n_tokens = int((ids != self.tokenizer.pad_token_id).sum())
                        row.append(
                            Completion(
                                text=self.tokenizer.decode(ids, skip_special_tokens=True),
                                n_tokens=max(1, n_tokens),
                            )
                        )
                    out.append(row)
            return out

    @app.function(
        gpu=SWEEP_GPU,
        timeout=TIMEOUT_S,
        image=_image(),
        secrets=[modal.Secret.from_dict({"HF_TOKEN": _dotenv().get("HF_TOKEN", "")})],
    )
    def sweep_remote(n_prompts: int, k: int, seed: int, max_new_tokens: int) -> dict:
        """Run the calibration sweep and return summaries + the provenance the manifest needs."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from assay.crawl import calibrate, tasks
        from assay.crawl.sampling import SamplerConfig

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # decoder-only generation requires left padding

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()

        cfg = SamplerConfig(temperature=1.0, top_p=1.0, max_new_tokens=max_new_tokens, seed=seed)

        # Raw rollouts are persisted, never modified (experiments/README.md). Without them an
        # unexpected summary is undebuggable — you only ever see the aggregate.
        raw: list[dict] = []

        def collect(family_name, setting, prompts, completions):  # type: ignore[no-untyped-def]
            budget = RAW_PER_SETTING
            for prompt, row in zip(prompts, completions, strict=True):
                for c in row:
                    if budget <= 0:
                        return
                    budget -= 1
                    raw.append(
                        {
                            "family": family_name,
                            "setting": setting,
                            "prompt_id": prompt.prompt_id,
                            "question": prompt.question,
                            "answer": prompt.answer,
                            "completion": c.text,
                            "n_tokens": c.n_tokens,
                        }
                    )

        summaries = calibrate.run_sweep(
            tasks.all_families(),
            sampler=HFSampler(model, tokenizer),
            n_prompts=n_prompts,
            k=k,
            cfg=cfg,
            seed=seed,
            observer=collect,
        )
        selection = calibrate.select(summaries)

        from assay.crawl.rewards import grader_fingerprint

        chat_template = tokenizer.chat_template or ""
        return {
            "summaries": [dataclasses.asdict(s) for s in summaries],
            "selection": dataclasses.asdict(selection),
            "raw": raw,
            "provenance": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "sampler": dataclasses.asdict(cfg),
                "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest()[:16],
                "prompt_template_sha256": tasks.template_fingerprint(),
                "grader": grader_fingerprint(),
                "fewshot_examples": 0,
                "gpu": SWEEP_GPU,
            },
        }

    @app.function(
        gpu=TRAIN_GPU,
        timeout=TRAIN_TIMEOUT_S,
        image=_image(),
        secrets=[modal.Secret.from_dict({"HF_TOKEN": _dotenv().get("HF_TOKEN", "")})],
        volumes={VOLUME_PATH: artifacts},
    )
    def train_remote(config: dict, provenance: dict) -> dict:
        """Run one ladder configuration and return its per-step logs plus derived metrics.

        The loop itself is ``assay.crawl.loop.train`` — **user-written** (``CLAUDE.md`` §7). This
        function is only the remote wrapper: it rebuilds the config, writes the manifest before the
        first step, hands off, and brings the logs home.

        Raises ``NotImplementedError`` until the loop exists. That is the intended state; the wiring
        is proven by the calibration sweep, which shares this image and secret path.
        """
        from pathlib import Path

        from assay.crawl import runlog
        from assay.crawl.config import LadderConfig
        from assay.crawl.hf_policy import HFPolicy
        from assay.crawl.loop import train

        cfg = LadderConfig(**config)
        run_dir = Path("/tmp/run") / cfg.run_id
        runlog.write_manifest({**provenance, "config": config}, run_dir)

        policy = HFPolicy(
            MODEL_ID,
            revision=MODEL_REVISION,
            learning_rate=cfg.learning_rate,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
        )
        import torch

        raw: list[dict] = []

        def capture(step, rollouts, grades):  # type: ignore[no-untyped-def]
            # A sample from the first few steps and then periodically. Enough to see what the
            # policy is actually emitting without shipping every rollout home.
            if step > 2 and step % 10:
                return
            for rollout, (proxy, true) in list(zip(rollouts, grades))[:8]:
                raw.append({
                    "step": step,
                    "question": rollout.prompt.question,
                    "answer": rollout.prompt.answer,
                    "completion": rollout.text,
                    "n_tokens": rollout.n_tokens,
                    "proxy_reward": proxy.reward,
                    "true_outcome": true.outcome.value,
                })

        logs = train(cfg, run_dir, policy=policy, observer=capture)
        peak_gb = float(torch.cuda.max_memory_allocated()) / 1e9
        print(f"peak CUDA memory: {peak_gb:.2f} GB")
        payload = {
            "peak_memory_gb": peak_gb,
            "summary": runlog.summarize_run(cfg, logs),
            "steps": [dataclasses.asdict(log) for log in logs],
            "provenance": {**provenance, "config": config},
            "raw": raw,
        }

        # Persist here, before returning. If the caller has gone away — closed laptop, dropped
        # connection — the artifacts still exist and `fetch` recovers them.
        artifact_dir = Path(VOLUME_PATH) / cfg.run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        artifacts.commit()
        print(f"committed artifacts to volume {VOLUME_NAME}:/{cfg.run_id}")
        return payload

    @app.local_entrypoint()
    def main(n_prompts: int = 64, k: int = 8, seed: int = 0, max_new_tokens: int = 256) -> None:
        """Coarse pass defaults to 64 prompts; rerun with ``--n-prompts 200`` for the fine pass.

        ``k`` defaults to 8 = the GRPO group size, which is what makes the unanimity count a direct
        estimate of the step-0 dead-group rate rather than an inference from a mean.
        """
        result = sweep_remote.remote(n_prompts, k, seed, max_new_tokens)
        result["provenance"]["git_sha"] = _git_sha()
        result["provenance"]["git_dirty"] = _git_dirty()
        result["provenance"]["n_prompts"] = n_prompts
        result["provenance"]["k"] = k
        if result["provenance"]["git_dirty"]:
            print(
                "WARNING: working tree is dirty — git_sha does not identify the code that "
                "produced this run, so it is not reproducible from the manifest."
            )

        run_id = f"n{n_prompts}-k{k}-seed{seed}"
        raw_rows = result.pop("raw", [])

        RAW_DIR.joinpath(run_id).mkdir(parents=True, exist_ok=True)
        raw_path = RAW_DIR / run_id / "completions.jsonl"
        with raw_path.open("w") as fh:
            for row in raw_rows:
                fh.write(json.dumps(row) + "\n")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"calibration-{run_id}.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

        for s in result["summaries"]:
            print(
                f"{s['family']:>11s} {s['setting']:<14s} "
                f"dead={s['dead_group_fraction']:.3f} pass@1={s['pass_at_1']:.3f} "
                f"headroom={s['headroom']:.3f} parse_fail={s['parse_fail_rate']:.3f} "
                f"tokens={s['median_completion_tokens']:.0f}"
            )
        chosen = result["selection"]["chosen"]
        print(f"\nchosen: {chosen['family']}/{chosen['setting']}" if chosen else "\nchosen: NONE")
        for e in result["selection"]["excluded"]:
            print(f"excluded {e['family']}/{e['setting']}: {e['reason']}")
        print(f"\nwrote {out}")

    @app.local_entrypoint()
    def fetch(into: str = "experiments/phase-0.1-grpo-by-hand") -> None:
        """Recover every run the volume holds into the local experiment layout.

            modal run src/assay/modal_app.py::fetch

        The point of the volume: a run that finished while this laptop was asleep is not lost, and
        neither is one whose client disconnected. Idempotent — re-fetching simply overwrites.
        """
        phase_dir = Path(into)
        recovered = 0
        for entry in artifacts.listdir("/", recursive=True):
            if not entry.path.endswith("result.json"):
                continue
            payload = json.loads(b"".join(artifacts.read_file(entry.path)).decode())
            _fetch_into(payload, phase_dir)
            summary = payload["summary"]
            print(
                f"  {summary['run_id']:<28} steps={summary['n_steps']:<4} "
                f"live={summary['live_fraction_in_slope_window']} "
                f"peak={payload.get('peak_memory_gb', float('nan')):.1f} GB"
            )
            recovered += 1
        print(f"\nrecovered {recovered} run(s) into {phase_dir}")

    @app.local_entrypoint()
    def overfit(steps: int = 50, lr: float = 3e-4) -> None:
        """Can the loop learn *anything*? Precondition for the LR probe.

            modal run --detach src/assay/modal_app.py::overfit

        One prompt, high learning rate. A correct implementation drives reward on that single
        example to ~1.0. If it cannot overfit one example, the gradient path is broken and no
        learning rate will help — which is what the first probe spent a whole run discovering,
        because a missing attention mask made every gradient meaningless.

        Exercises generation, grading, advantages, the loss, masking, backward and the optimizer
        step end to end. Pass: proxy_reward >= 0.95 within `steps`.
        """
        from assay.crawl.config import LadderConfig

        # prompts_per_step=2 is the minimum the half-batch split allows; both groups draw from the
        # same seed, so this is very nearly a single example repeated.
        cfg = LadderConfig(
            run_id="overfit-check", steps=steps, prompts_per_step=2, group_size=8,
            baseline="group_loo", normalize_by_std=True, kl_coef=0.0, learning_rate=lr,
        )
        result = train_remote.remote(dataclasses.asdict(cfg), _provenance())
        rows = result["steps"]

        print(f"\n{'step':>5} {'reward':>8} {'dead':>7} {'|g|':>9} {'entropy':>9}")
        for row in rows[:: max(1, len(rows) // 12)]:
            print(f"{row['step']:>5} {row['proxy_reward']:8.3f} "
                  f"{row['frac_degenerate_groups']:7.3f} {row['grad_norm']:9.4f} "
                  f"{row['policy_entropy']:9.4f}")

        best = max(r["proxy_reward"] for r in rows)
        print(f"\nbest reward reached: {best:.3f}")
        print("PASS — the loop can learn" if best >= 0.95 else
              "FAIL — cannot overfit one example; the gradient path is broken, not the LR")

    @app.local_entrypoint()
    def probe_lr(
        rates: str = "1e-5,3e-5,1e-4,3e-4", steps: int = 30, setting: str = "add-3digit"
    ) -> None:
        """Learning-rate probe. Rule pre-committed in the phase plan, before this ran.

            modal run src/assay/modal_app.py::probe_lr

        Reject a rate for: non-finite loss/grad, entropy halving inside `steps` (collapse before
        training began), or kl_to_ref staying ~0 (the policy never moved). Among survivors take the
        LARGEST — faster movement leaves more of the 200-step budget past the transient.
        """
        from assay.crawl import runlog
        from assay.crawl.config import LadderConfig

        provenance = _provenance()
        print(f"{'lr':>8} {'reward 0->end':>16} {'kl_end':>9} {'entropy 0->end':>18} "
              f"{'|g| mean':>9} {'dead_end':>9} {'live':>6}")
        print("-" * 78)
        for rate in [float(r) for r in rates.split(",")]:
            cfg = LadderConfig(
                run_id=f"probe-{setting}-lr{rate:g}", setting=setting, steps=steps,
                baseline="group_loo",
                normalize_by_std=True, kl_coef=0.04, learning_rate=rate,
            )
            result = train_remote.remote(dataclasses.asdict(cfg), provenance)
            rows = result["steps"]
            first, last = rows[0], rows[-1]
            finite = all(
                math.isfinite(r["grad_norm"]) and math.isfinite(r["proxy_reward"]) for r in rows
            )
            print(
                f"{rate:>8.0e} {first['proxy_reward']:7.3f}->{last['proxy_reward']:<7.3f} "
                f"{last['kl_to_ref']:9.4f} {first['policy_entropy']:8.3f}->"
                f"{last['policy_entropy']:<8.3f} "
                f"{sum(r['grad_norm'] for r in rows)/len(rows):9.4f} "
                f"{last['frac_degenerate_groups']:9.3f}"
                + f" {sum(1 for r in rows if r['frac_degenerate_groups'] < 1.0):>5}"
                + ("" if finite else "   NON-FINITE -> reject")
            )
            runlog.write_results(result["summary"], RESULTS_DIR)

    @app.local_entrypoint()
    def ladder(runs: str = "", setting: str = "", seeds: str = "0") -> None:
        """Launch ladder runs: ``modal run ... ::ladder --runs run1,run7 --seeds 0,1``.

        The ladder table lives in ``assay.crawl.ladder`` and is **the user's** (§7). This entry
        point only dispatches it and lands the artefacts:

            raw/<run_id>/steps.jsonl   per-step logs, gitignored, never modified
            results/<run_id>.json      derived metrics; every figure regenerates from these

        ``--seeds`` exists because gate 1 needs >= 2 seeds on run 7, and that seed band is what
        every ablation threshold should ultimately be stated against.

        ``--setting`` defaults to **empty**, meaning *use whatever the table pins*. It used to
        default to a task name, which silently overrode the table: after the primary arm was
        swapped to ``add-3digit`` this entry point kept dispatching ``add-2digit``, and the test
        asserting the pinned arm still passed because it checked the *table* rather than what
        actually ran.
        """
        from assay.crawl import runlog
        from assay.crawl.ladder import LADDER

        wanted = [r.strip() for r in runs.split(",") if r.strip()] or sorted(LADDER)
        seed_list = [int(s) for s in seeds.split(",") if s.strip()]
        provenance = _provenance()
        if provenance["git_dirty"]:
            print("WARNING: dirty tree — git_sha does not identify the code producing these runs.")

        for run_id in wanted:
            for seed in seed_list:
                # Empty `setting` means the table decides. Only an explicit value overrides it.
                overrides: dict = {"seed": seed}
                if setting:
                    overrides["setting"] = setting
                resolved = dataclasses.replace(LADDER[run_id], **overrides)
                tag = f"{run_id}-{resolved.setting}-seed{seed}"
                cfg = dataclasses.replace(resolved, run_id=tag)
                print(f"\n=== {tag} ===")
                result = train_remote.remote(dataclasses.asdict(cfg), provenance)

                run_dir = RAW_DIR / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / "steps.jsonl").open("w") as fh:
                    for step in result["steps"]:
                        fh.write(json.dumps(step) + "\n")
                runlog.write_manifest(
                    {**provenance, "config": dataclasses.asdict(cfg)}, run_dir
                )
                runlog.write_results(result["summary"], RESULTS_DIR)

                s = result["summary"]
                print(
                    f"  gap_slope(50-200)={s['gap_slope_50_200']}  "
                    f"proxy {s['proxy_reward_first']:.3f}->{s['proxy_reward_last']:.3f}  "
                    f"dead {s['frac_degenerate_first']:.3f}->{s['frac_degenerate_last']:.3f}  "
                    f"live {s['live_fraction_in_slope_window']}"
                )


def _fetch_into(payload: dict, phase_dir: Path) -> None:  # type: ignore[type-arg]
    """Land one remote payload in the local experiment layout."""
    from assay.crawl import runlog

    run_id = payload["summary"]["run_id"]
    run_dir = phase_dir / "raw" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "steps.jsonl").open("w") as handle:
        for step in payload["steps"]:
            handle.write(json.dumps(step) + "\n")
    runlog.write_manifest(payload["provenance"], run_dir)
    runlog.write_results(payload["summary"], phase_dir / "results")


def _provenance() -> dict:  # type: ignore[type-arg]
    """Everything a run's manifest needs to be reproducible from itself."""
    from assay.crawl.rewards import grader_fingerprint
    from assay.crawl.tasks import template_fingerprint

    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_template_sha256": template_fingerprint(),
        "grader": grader_fingerprint(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "backend": f"modal-{TRAIN_GPU}",
    }


def tasks_fingerprint() -> str:
    from assay.crawl.tasks import template_fingerprint

    return template_fingerprint()


def build_app():  # type: ignore[no-untyped-def]
    """Return the Modal app. Kept for callers that construct it programmatically."""
    if modal is None:  # pragma: no cover
        raise RuntimeError("modal is not installed — `uv sync --extra modal`")
    return app
