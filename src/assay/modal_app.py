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
#: **Back to L4 after measurement.** This was raised to A10G then A100-40GB while chasing an OOM
#: whose real cause turned out to be gradient checkpointing never engaging (the model was left in
#: eval mode). Once fixed, six 200-step runs peaked at **13.5-14.5 GB** — comfortably inside an
#: L4's 24 GB. The larger tier was never needed and cost roughly 3-4x per hour for ~1.5 h of runs.
#:
#: The lesson is the session's recurring one: a bigger machine masked a bug instead of fixing it,
#: and the cost of that stayed switched on long after the bug was found.
TRAIN_GPU = "L4"
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


def _vllm_image():  # type: ignore[no-untyped-def]
    """M2 only. Separate from ``_image()`` on purpose.

    vLLM pins its own ``torch``, and folding it into the shared image would silently change the
    torch version underneath **every** Phase 0.1 artifact's code path — the scorer M2 exists to
    measure. Two images cost a build; one image would cost the comparison.

    **Why a ``-devel`` CUDA base rather than ``debian_slim``.** vLLM's V1 engine runs its model
    through ``torch.compile`` at startup, and inductor shells out to ``nvcc`` for CUDA codegen. The
    pip wheels ship the CUDA *runtime*, not the *toolkit*, so on ``debian_slim`` engine init dies
    with ``Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`` — after the
    GPU is already allocated. ``enforce_eager=True`` does not avoid it: that suppresses CUDA-graph
    capture, which is a later stage than compilation.
    """
    return (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .pip_install("vllm==0.26.0", "transformers", "accelerate", "huggingface_hub")
        .add_local_python_source("assay")
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return "unknown"


def _require_clean_tree(provenance: dict, *, allow_dirty: bool) -> None:  # type: ignore[type-arg]
    """Refuse to launch from a dirty tree. **A check that reports instead of blocking is not a check.**

    This was a ``print("WARNING: ...")`` that fell straight through into the launch loop, and on
    2026-08-01 it let ``run3`` and ablations B/C/D run at a ``git_sha`` that does not identify their
    code — four 200-step runs, ~$2.50, unusable for strict reproduction under
    ``experiments/README.md``. Three compounding failures, all now fixed by raising:

    - a ``print`` does not stop anything;
    - it fired **once**, before a loop that then launched N runs;
    - it was paired with ``--detach``, whose whole purpose is that nobody is watching the terminal.

    Third instance of this pattern in Phase 0.1, after ``make check | tail && git commit`` swallowing
    the exit code and the ``--setting`` default silently overriding the pinned ladder table.

    ``--allow-dirty`` exists for deliberate throwaway smoke tests, and stamps the artifacts so they
    can never be mistaken for the real thing.
    """
    if not provenance["git_dirty"]:
        return
    if allow_dirty:
        print("WARNING: --allow-dirty — git_sha does NOT identify this code. Throwaway runs only.")
        return
    raise SystemExit(
        "refusing to launch from a dirty working tree: git_sha would not identify the code "
        "producing these runs, so their artifacts cannot enter analysis.\n"
        "  commit first, or pass --allow-dirty for a throwaway smoke test."
    )


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

        def __init__(self, model, tokenizer, use_chat_template: bool = True):  # type: ignore[no-untyped-def]
            self.model = model
            self.tokenizer = tokenizer
            # **False for base models.** Qwen ships a chat template even on its base checkpoints, so
            # `apply_chat_template` succeeds and silently wraps the prompt in an instruct format the
            # model was never tuned on. TinyZero is R1-Zero style — base models, raw prompts — and
            # measuring a base rate through the wrong prompt format would answer a different
            # question than the screen is asking.
            self.use_chat_template = use_chat_template

        def sample(self, prompts, *, k, cfg):  # type: ignore[no-untyped-def]
            import torch

            from assay.crawl.sampling import Completion

            torch.manual_seed(cfg.seed)
            out: list[list[Completion]] = []

            for start in range(0, len(prompts), 16):
                batch = prompts[start : start + 16]
                texts = [
                    p.question
                    if not self.use_chat_template
                    else self.tokenizer.apply_chat_template(
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

    @app.function(
        gpu=TRAIN_GPU,
        timeout=TRAIN_TIMEOUT_S,
        image=_image(),
        secrets=[modal.Secret.from_dict({"HF_TOKEN": _dotenv().get("HF_TOKEN", "")})],
        volumes={VOLUME_PATH: artifacts},
    )
    def probe_remote(config: dict, provenance: dict) -> dict:
        """Ablation A's paired fixed-policy probe. See ``assay.crawl.probe`` for why it exists.

        Same image, secret and volume as ``train_remote`` — the probe reuses the ladder's sampler
        and grader, and its whole point is that it draws from the same distribution the ladder
        trains on.
        """
        from pathlib import Path

        import torch

        from assay.crawl import runlog
        from assay.crawl.hf_policy import HFPolicy
        from assay.crawl.probe import ProbeConfig, probe

        cfg = ProbeConfig(**config)
        ladder = cfg.as_ladder_config()
        run_dir = Path("/tmp/run") / cfg.run_id
        runlog.write_manifest({**provenance, "config": config}, run_dir)

        policy = HFPolicy(
            MODEL_ID,
            revision=MODEL_REVISION,
            learning_rate=ladder.learning_rate,
            max_new_tokens=ladder.max_new_tokens,
            temperature=ladder.temperature,
            top_p=ladder.top_p,
            seed=cfg.seed,
        )
        result = probe(cfg, run_dir, policy=policy)
        result["peak_memory_gb"] = float(torch.cuda.max_memory_allocated()) / 1e9
        result["provenance"] = {**provenance, "config": config}
        print(f"peak CUDA memory: {result['peak_memory_gb']:.2f} GB")
        print(f"verdict: {result['verdict']['verdict']}")

        artifact_dir = Path(VOLUME_PATH) / cfg.run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "probe.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        artifacts.commit()
        print(f"committed artifacts to volume {VOLUME_NAME}:/{cfg.run_id}")
        return result

    # ── Phase 0.3 / M1: the Countdown base-rate screen ──────────────────────────────────
    #
    #: L4, on measurement rather than caution. Qwen2.5-3B is 6 GB of bf16 weights and the sampler's
    #: 128-sequence batch needs 2.4 GB of KV cache at 512 tokens — comfortably inside 24 GB, and this
    #: is generation-only (no gradients, no optimizer, no reference model). Phase 0.1's lesson was
    #: that a bigger machine masks bugs and the cost outlives them; escalate only on a measured OOM.
    SCREEN_GPU = "L4"

    #: Base models, pinned. TinyZero is R1-Zero style: no SFT, no chat template.
    SCREEN_MODELS = {
        "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
        "qwen2.5-3b": ("Qwen/Qwen2.5-3B", "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"),
    }

    @app.function(
        gpu=SCREEN_GPU,
        timeout=TIMEOUT_S,
        image=_image(),
        secrets=[modal.Secret.from_dict({"HF_TOKEN": _dotenv().get("HF_TOKEN", "")})],
        volumes={VOLUME_PATH: artifacts},
    )
    def screen_remote(
        model_id: str,
        model_revision: str,
        n_prompts: int,
        k: int,
        seed: int,
        max_new_tokens: int,
        provenance: dict,
        settings: list | None = None,
    ) -> dict:
        """Countdown base-rate screen for one model. Generation only.

        Answers Phase 0.3's first question: is Countdown a task GRPO can get traction on, at any
        scale we can afford? The statistic is ``dead_group_fraction`` at ``k = G = 8``, so unanimity
        is a direct estimate of the step-0 dead-group rate — the same instrument Phase 0.1 used to
        pick its task after finding the original criterion was stated on the wrong statistic.
        """
        import json as _json
        from pathlib import Path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from assay.crawl import calibrate, tasks
        from assay.crawl.rewards import grader_fingerprint
        from assay.crawl.sampling import SamplerConfig

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()

        cfg = SamplerConfig(temperature=1.0, top_p=1.0, max_new_tokens=max_new_tokens, seed=seed)
        raw: list[dict] = []

        def collect(family_name, setting, prompts, completions):  # type: ignore[no-untyped-def]
            budget = RAW_PER_SETTING
            for prompt, row in zip(prompts, completions, strict=True):
                for c in row:
                    if budget <= 0:
                        return
                    budget -= 1
                    raw.append({
                        "family": family_name, "setting": setting,
                        "prompt_id": prompt.prompt_id, "question": prompt.question,
                        "answer": prompt.answer, "completion": c.text, "n_tokens": c.n_tokens,
                    })

        countdown = next(f for f in tasks.all_families() if f.name == "countdown")
        summaries = calibrate.run_sweep(
            [countdown],
            # use_chat_template=False: these are BASE checkpoints (see HFSampler).
            sampler=HFSampler(model, tokenizer, use_chat_template=False),
            n_prompts=n_prompts, k=k, cfg=cfg, seed=seed, observer=collect,
            # M3 screens a subset. Re-running the four settings M1 already settled would spend most
            # of the budget re-measuring a known answer. An unknown name raises in run_sweep.
            settings=settings or None,
        )

        payload = {
            "summaries": [dataclasses.asdict(s) for s in summaries],
            "raw": raw,
            "peak_memory_gb": float(torch.cuda.max_memory_allocated()) / 1e9,
            "provenance": {
                # Caller provenance FIRST, then the screen's own values. Spread last, `_provenance()`
                # clobbers `model_id` with the module-level MODEL_ID — which on 2026-08-03 made both
                # screen artifacts claim they ran Llama-3.2-1B-Instruct when they ran Qwen. The
                # numbers were right and the label was wrong, which is the worse failure.
                **provenance,
                "model_id": model_id,
                "model_revision": model_revision,
                "sampler": dataclasses.asdict(cfg),
                "prompt_template_sha256": tasks.template_fingerprint(),
                "grader": grader_fingerprint(),
                "use_chat_template": False,
                "gpu": SCREEN_GPU,
            },
        }
        tag = f"screen-countdown-{model_id.split('/')[-1]}-seed{seed}"
        out = Path(VOLUME_PATH) / tag
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(_json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifacts.commit()
        print(f"peak CUDA memory: {payload['peak_memory_gb']:.2f} GB")
        print(f"committed artifacts to volume {VOLUME_NAME}:/{tag}")
        return payload

    @app.local_entrypoint()
    def screen_countdown(
        models: str = "qwen2.5-1.5b,qwen2.5-3b",
        n_prompts: int = 200,
        k: int = 8,
        seed: int = 0,
        max_new_tokens: int = 512,
        allow_dirty: bool = False,
    ) -> None:
        """``modal run --detach src/assay/modal_app.py::screen_countdown``.

        Pre-registered decision band on ``dead_group_fraction`` lives in
        ``docs/phases/phase-0.3-r0-plan.md``: <=0.50 workable, 0.50-0.75 marginal, >0.75 starved.
        """
        provenance = _provenance()
        _require_clean_tree(provenance, allow_dirty=allow_dirty)

        for name in (m.strip() for m in models.split(",") if m.strip()):
            model_id, revision = SCREEN_MODELS[name]
            print(f"\n=== screening {model_id} ===")
            result = screen_remote.remote(
                model_id, revision, n_prompts, k, seed, max_new_tokens, provenance
            )

            tag = f"screen-countdown-{model_id.split('/')[-1]}-seed{seed}"
            # RESULTS_DIR is Phase 0.1's; a 0.3 artifact belongs under 0.3.
            screen_dir = Path("experiments/phase-0.3-r0/results")
            screen_dir.mkdir(parents=True, exist_ok=True)
            (screen_dir / f"{tag}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            for s in result["summaries"]:
                print(f"  {s['setting']:<8} pass@1 {s['pass_at_1']:.3f}  "
                      f"dead {s['dead_group_fraction']:.3f}  "
                      f"parse_fail {s['parse_fail_rate']:.3f}")

    @app.function(
        gpu=SCREEN_GPU,
        timeout=TIMEOUT_S,
        image=_vllm_image(),
        secrets=[modal.Secret.from_dict({"HF_TOKEN": _dotenv().get("HF_TOKEN", "")})],
        volumes={VOLUME_PATH: artifacts},
    )
    def mismatch_remote(
        model_id: str,
        model_revision: str,
        n_prompts: int,
        seed: int,
        max_new_tokens: int,
        provenance: dict,
    ) -> dict:
        """M2 — sample with vLLM, score the *same token ids* with our HF scorer.

        The comparison is only interpretable at ``temperature=1.0, top_p=1.0``. vLLM returns the
        log-probs of the **processed** distribution — after temperature scaling and any top-p mask —
        so at any other setting the difference would be dominated by our own sampler config rather
        than by the two implementations, and M2 would answer a question nobody asked. Both are
        asserted below rather than trusted, since the failure is silent and the number still looks
        plausible.

        Two rig checks ride along, because a mismatch number with no control is unfalsifiable:

        1. **HF vs HF** on the same ids must be *identically* zero. If it is not, the harness is
           wrong and no vLLM figure from this code path means anything.
        2. **pass@1 against M1**, which measured this exact model, task, setting and seed at 0.024.
           A vLLM sampler producing a wildly different pass rate is broken in a way the log-prob
           statistics might not reveal.
        """
        import json as _json
        from pathlib import Path

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vllm import LLM, SamplingParams

        from assay.crawl import tasks
        from assay.crawl.logprob import (
            _token_logprobs,
            build_attention_mask,
            build_completion_mask,
        )
        from assay.crawl.mismatch import mismatch_statistics, mismatch_verdict
        from assay.crawl.rewards import Outcome, grade_countdown, grader_fingerprint

        temperature, top_p = 1.0, 1.0
        assert temperature == 1.0 and top_p == 1.0, (
            "vLLM reports log-probs of the post-processing distribution; away from T=1/top_p=1 "
            "this measures our sampler config, not the two implementations"
        )

        countdown = next(f for f in tasks.all_families() if f.name == "countdown")
        prompts = countdown.generate("cd-3", n_prompts, seed=seed)

        # --- sample with vLLM -------------------------------------------------------------
        # gpu_memory_utilization is held low on purpose: the HF scorer has to load into the *same*
        # container afterwards, and vLLM's default (~0.9) would leave it nothing.
        llm = LLM(
            model=model_id, revision=model_revision, dtype="bfloat16",
            gpu_memory_utilization=0.40, max_model_len=1024, seed=seed, enforce_eager=True,
        )
        params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, logprobs=0, seed=seed,
        )
        outputs = llm.generate([p.question for p in prompts], params)

        sampled = []
        for prompt, out in zip(prompts, outputs, strict=True):
            gen = out.outputs[0]
            if not gen.token_ids:
                continue
            # logprobs[i] maps token_id -> Logprob for position i; the sampled token is always
            # present at logprobs=0, so this reads the chosen token's own log-prob.
            per_token = [float(lp[tid].logprob) for tid, lp in zip(gen.token_ids, gen.logprobs,
                                                                   strict=True)]
            sampled.append({
                "prompt_id": prompt.prompt_id, "question": prompt.question, "answer": prompt.answer,
                "prompt_token_ids": list(out.prompt_token_ids),
                "completion_token_ids": list(gen.token_ids),
                "text": gen.text, "vllm_logprobs": per_token,
            })

        del llm
        torch.cuda.empty_cache()

        # --- score the same ids with the HF path the loss actually differentiates ----------
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=model_revision, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()

        def hf_score(rows: list[dict]) -> list[list[float]]:
            """Per-token HF log-probs for exactly the ids vLLM emitted. Never re-tokenizes.

            Round-tripping vLLM's *text* through the tokenizer can change the ids, which would make
            this a measurement of re-tokenization rather than of the two samplers.
            """
            out: list[list[float]] = []
            for chunk in [rows[i : i + 16] for i in range(0, len(rows), 16)]:
                prompt_len = max(len(r["prompt_token_ids"]) for r in chunk)
                comp_lens = [len(r["completion_token_ids"]) for r in chunk]
                total = prompt_len + max(comp_lens)
                ids = torch.full((len(chunk), total), tokenizer.pad_token_id, dtype=torch.long)
                prompt_attn = torch.zeros((len(chunk), prompt_len), dtype=torch.long)
                for row, r in enumerate(chunk):
                    p, c = r["prompt_token_ids"], r["completion_token_ids"]
                    ids[row, prompt_len - len(p) : prompt_len] = torch.tensor(p)  # left-pad
                    prompt_attn[row, prompt_len - len(p) :] = 1
                    ids[row, prompt_len : prompt_len + len(c)] = torch.tensor(c)
                mask = build_completion_mask(prompt_len, comp_lens, total)
                attention = build_attention_mask(prompt_attn, comp_lens, total)
                with torch.no_grad():
                    logits = model(
                        input_ids=ids.cuda(), attention_mask=attention.cuda()
                    ).logits.float()
                    tok, m = _token_logprobs(logits, ids.cuda(), mask.cuda())
                tok, m = tok.cpu(), m.cpu()
                for row, n in enumerate(comp_lens):
                    keep = m[row].bool()
                    out.append([float(x) for x in tok[row][keep][:n]])
            return out

        hf_logprobs = hf_score(sampled)
        # The vLLM side is truncated to match: the shift in `_token_logprobs` drops a completion's
        # first token when it begins at position 0, and alignment is asserted, never assumed.
        vllm_logprobs = [
            r["vllm_logprobs"][len(r["vllm_logprobs"]) - len(h) :]
            for r, h in zip(sampled, hf_logprobs, strict=True)
        ]

        control = hf_score(sampled)  # rig check 1: HF vs HF must be identically zero
        control_stats = mismatch_statistics(hf_logprobs, control)

        stats = mismatch_statistics(hf_logprobs, vllm_logprobs)
        verdict = mismatch_verdict(stats, operating_length=max_new_tokens)

        # `grade_countdown` returns a Grade, not a float — read `.outcome`, exactly as
        # `calibrate` does, so this pass@1 is the same quantity M1 reported.
        grades = [grade_countdown(r["text"], r["question"]) for r in sampled]
        pass_at_1 = sum(g.outcome is Outcome.CORRECT for g in grades) / max(1, len(grades))
        parse_fail = sum(g.outcome is Outcome.PARSE_FAIL for g in grades) / max(1, len(grades))

        payload = {
            "verdict": verdict,
            "control": {
                "max_abs": control_stats["max_abs"],
                "mean_abs": control_stats["mean_abs"],
                "is_exactly_zero": control_stats["max_abs"] == 0.0,
            },
            "sampler_cross_check": {
                "vllm_pass_at_1": pass_at_1,
                "vllm_parse_fail_rate": parse_fail,
                "m1_hf_pass_at_1": 0.024375,
                "m1_hf_parse_fail_rate": 0.271875,
                "note": "M1 measured this model/task/setting/seed with the HF sampler.",
            },
            "raw": [{k: v for k, v in r.items() if k != "prompt_token_ids"} for r in sampled[:32]],
            "peak_memory_gb": float(torch.cuda.max_memory_allocated()) / 1e9,
            "provenance": {
                **provenance,
                "model_id": model_id,
                "model_revision": model_revision,
                "sampler": {
                    "temperature": temperature, "top_p": top_p,
                    "max_new_tokens": max_new_tokens, "seed": seed,
                },
                "task": {"family": "countdown", "setting": "cd-3", "n_prompts": n_prompts},
                "prompt_template_sha256": tasks.template_fingerprint(),
                "grader": grader_fingerprint(),
                "use_chat_template": False,
                "gpu": SCREEN_GPU,
            },
        }
        tag = f"mismatch-vllm-{model_id.split('/')[-1]}-seed{seed}"
        out_dir = Path(VOLUME_PATH) / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(_json.dumps(payload, indent=2, sort_keys=True) + "\n")
        artifacts.commit()
        print(f"committed artifacts to volume {VOLUME_NAME}:/{tag}")
        return payload

    @app.local_entrypoint()
    def screen_difficulty(
        model: str = "qwen2.5-3b",
        settings: str = "cd-3-easy,cd-3-mid",
        n_prompts: int = 200,
        k: int = 8,
        seed: int = 0,
        max_new_tokens: int = 512,
        allow_dirty: bool = False,
    ) -> None:
        """M3 — ``modal run --detach src/assay/modal_app.py::screen_difficulty``.

        The four admission criteria and the tie-break were pre-registered in
        ``docs/phases/phase-0.3-r0-plan.md`` §M3, in the commit *before* these settings existed.
        They are applied here by ``assay.crawl.admission``, not by eye.
        """
        from assay.crawl.admission import ADMISSION, admission_report
        from assay.crawl.calibrate import SettingSummary

        provenance = _provenance()
        _require_clean_tree(provenance, allow_dirty=allow_dirty)

        wanted = [x.strip() for x in settings.split(",") if x.strip()]
        model_id, revision = SCREEN_MODELS[model]
        print(f"=== M3: difficulty screen, {model_id}, settings {wanted} ===")
        result = screen_remote.remote(
            model_id, revision, n_prompts, k, seed, max_new_tokens, provenance, wanted
        )

        out_dir = Path("experiments/phase-0.3-r0/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"screen-difficulty-{model_id.split('/')[-1]}-seed{seed}"
        (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

        summaries = [SettingSummary(**row) for row in result["summaries"]]
        print(f"\n  criteria (pre-registered): {ADMISSION}\n")
        for row in admission_report(summaries):
            c = row["criteria"]
            print(f"  {row['setting']:<12} dead {c['dead_group_fraction']['value']:.3f}  "
                  f"explore {c['exploration_ratio']['value']:.2f}x  "
                  f"tok {c['median_completion_tokens']['value']:.0f}  "
                  f"parse_fail {c['parse_fail_rate']['value']:.3f}  "
                  f"-> {'ADMITTED' if row['admitted'] else 'rejected: ' + ','.join(row['failed'])}")

    @app.local_entrypoint()
    def measure_mismatch(
        model: str = "qwen2.5-1.5b",
        n_prompts: int = 128,
        seed: int = 0,
        max_new_tokens: int = 512,
        allow_dirty: bool = False,
    ) -> None:
        """``modal run --detach src/assay/modal_app.py::measure_mismatch``.

        Pre-registered band on the implied sequence ratio at the operating length lives in
        ``docs/phases/phase-0.3-r0-plan.md`` §M2: inside [0.9, 1.1] negligible, outside not free.
        """
        provenance = _provenance()
        _require_clean_tree(provenance, allow_dirty=allow_dirty)

        model_id, revision = SCREEN_MODELS[model]
        print(f"=== M2: vLLM vs HF log-probs, {model_id} ===")
        result = mismatch_remote.remote(
            model_id, revision, n_prompts, seed, max_new_tokens, provenance
        )

        out_dir = Path("experiments/phase-0.3-r0/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"mismatch-vllm-{model_id.split('/')[-1]}-seed{seed}"
        (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

        v, s = result["verdict"], result["verdict"]["stats"]
        print(f"\n  control (HF vs HF)   max|delta| = {result['control']['max_abs']:.2e}  "
              f"exactly zero: {result['control']['is_exactly_zero']}")
        print(f"  pass@1  vLLM {result['sampler_cross_check']['vllm_pass_at_1']:.4f}  "
              f"vs M1 HF {result['sampler_cross_check']['m1_hf_pass_at_1']:.4f}")
        print(f"  per-token  mean {s['mean']:+.5f}  std {s['std']:.5f}  "
              f"max|d| {s['max_abs']:.4f}  max_off_policy {s['max_off_policy']:.4f}")
        print(f"  independence_ratio {s['independence_ratio']:.2f}  (1 = iid; >>1 = correlated)")
        for length, r in sorted(v["by_length"].items()):
            print(f"  ratio @ {length:<5} median {r['median']:.4f}  "
                  f"[{r['lo']:.4f}, {r['hi']:.4f}]")
        print(f"\n  VERDICT: {v['verdict']}  (band {v['band']} at L={v['operating_length']})")

    @app.local_entrypoint()
    def probe_a(
        seeds: str = "0,1,2", warmups: str = "0,50", batches: int = 40,
        length_normalize: bool = True, allow_dirty: bool = False,
    ) -> None:
        """``modal run --detach src/assay/modal_app.py::probe_a``.

        The pre-registered grid is 3 seeds x {base policy, 50-step warmup}; the gate lives in
        ``assay.crawl.probe.probe_verdict`` and was written before this ever ran.
        """
        import dataclasses as dc

        from assay.crawl.probe import ProbeConfig

        provenance = _provenance()
        _require_clean_tree(provenance, allow_dirty=allow_dirty)

        grid = [
            ProbeConfig(
                # Batch count is in the id: N is the thing being varied to tighten the interval,
                # so an N=160 run must not overwrite the N=40 artifact it is meant to be compared
                # against (desideratum: raw artefacts are never modified).
                run_id=f"probeA-w{warmup}-seed{seed}"
                + ("" if length_normalize else "-nolennorm")
                + ("" if batches == 40 else f"-n{batches}"),
                seed=seed,
                warmup_steps=warmup,
                batches=batches,
                length_normalize=length_normalize,
            )
            for warmup in (int(w) for w in warmups.split(",") if w.strip())
            for seed in (int(s) for s in seeds.split(",") if s.strip())
        ]
        for cfg in grid:
            print(f"\n=== {cfg.run_id} ===")
            result = probe_remote.remote(dc.asdict(cfg), provenance)

            run_dir = RAW_DIR / cfg.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "probe.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            (RESULTS_DIR / f"{cfg.run_id}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            verdict = result["verdict"]
            print(f"  verdict {verdict['verdict']}  NSR {verdict.get('nsr')}")

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
            summary = payload.get("summary")
            if summary is None:
                # Not a ladder run — the volume also holds screens and probes, whose payloads have a
                # different shape. Recovery must not abort on the first one it does not recognise:
                # before this guard, one screen artifact raised KeyError and stopped the whole fetch,
                # which is the opposite of what a recovery path is for.
                name = entry.path.rsplit("/", 1)[0].strip("/") or "root"
                print(f"  {name:<40} (not a ladder run — skipped)")
                continue
            _fetch_into(payload, phase_dir)
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
    def ladder(
        runs: str = "", setting: str = "", seeds: str = "0", suffix: str = "",
        allow_dirty: bool = False,
    ) -> None:
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

        ``--suffix`` appends to the run tag, so re-measuring an arm cannot overwrite the artefacts
        of an earlier one (``docs/desiderata.md`` — raw rollouts are never modified). Required when
        a table entry changes what it *records* while leaving what it *does* untouched, as the
        centred-cosine diagnostic did on rungs 1-3: same update, same seed, same curve, but the old
        run has no ``half_batch_grad_cosine_centered`` and both readings must survive.
        """
        from assay.crawl import runlog
        from assay.crawl.ladder import LADDER

        wanted = [r.strip() for r in runs.split(",") if r.strip()] or sorted(LADDER)
        seed_list = [int(s) for s in seeds.split(",") if s.strip()]
        provenance = _provenance()
        _require_clean_tree(provenance, allow_dirty=allow_dirty)

        for run_id in wanted:
            for seed in seed_list:
                # Empty `setting` means the table decides. Only an explicit value overrides it.
                overrides: dict = {"seed": seed}
                if setting:
                    overrides["setting"] = setting
                resolved = dataclasses.replace(LADDER[run_id], **overrides)
                tag = f"{run_id}-{resolved.setting}-seed{seed}" + (f"-{suffix}" if suffix else "")
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
