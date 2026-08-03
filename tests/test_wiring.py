"""Wiring smoke tests — no GPU, no API, no network.

These do not test behaviour (there is none yet). They test that the scaffold's *contracts* are
importable, coherent, and that the pre-registered constants match the documents. When a phase
implements a stub, its real tests land beside it; these stay as the structural floor.
"""

from __future__ import annotations

import inspect

import pytest

from assay import (
    battery,
    cli,
    config,
    env,
    grader,
    loop,
    measure,
    report,
    screen,
)


def test_package_imports() -> None:
    import assay

    assert assay.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "module",
    [config, env, grader, screen, battery, loop, measure, report, cli],
)
def test_module_has_docstring(module: object) -> None:
    """Every module states what it is for and which phase implements it."""
    doc = inspect.getdoc(module)
    assert doc, f"{module} has no docstring"


# --------------------------------------------------------------------------------------
# Pre-registered constants must match docs/pre-registration.md §2 and §4.
# If one of these fails, either the code drifted or the pre-registration changed without a
# change-log entry. Both are bugs.
# --------------------------------------------------------------------------------------


def test_screen_admission_band_matches_prereg() -> None:
    cfg = config.ScreenConfig()
    assert cfg.k_rollouts == 64
    assert cfg.temperature == 1.0
    assert cfg.admit_min == pytest.approx(1 / 64)
    assert cfg.admit_max == pytest.approx(0.30)


def test_train_defaults_match_prereg_pins() -> None:
    cfg = config.TrainConfig(model_id="placeholder")
    assert cfg.steps == 200, "P-steps"
    assert cfg.max_completion_tokens == 512, "P-algo / L6 short horizon"
    assert cfg.group_size == 8
    assert cfg.use_lora is True


def test_grader_variant_carries_ground_truth_pathology() -> None:
    """H1 scores predictions against a pathology label assigned by construction."""
    fields = {f for f in config.GraderVariant.__dataclass_fields__}
    assert "expected_pathology" in fields
    assert "is_positive_control" in fields


def test_step_log_exposes_both_reward_legs() -> None:
    """The gap is proxy minus true. Both legs must be logged every step."""
    fields = set(loop.StepLog.__dataclass_fields__)
    assert {"proxy_reward", "true_reward", "step"} <= fields
    log = loop.StepLog(
        step=1,
        proxy_reward=0.8,
        true_reward=0.3,
        policy_entropy=1.0,
        distinct_completions=64,
        kl_to_ref=0.0,
        kl_loss_fraction=0.0,
        grad_norm=1.0,
        half_batch_grad_cosine=0.9,
        max_abs_advantage=2.0,
        group_pass_rate=0.5,
        frac_degenerate_groups=0.0,
        tokens=100,
        wall_clock_s=1.0,
    )
    assert log.gap == pytest.approx(0.5)


def test_step_log_carries_every_ablation_signature() -> None:
    """Each ablation's pre-registered signature must be computable from what is logged.

    Writing the signatures is what surfaced ``distinct_completions`` (B) and ``max_abs_advantage``
    (C) as gaps — discovering them after the runs would mean re-running.
    """
    fields = set(loop.StepLog.__dataclass_fields__)
    assert {"grad_norm", "half_batch_grad_cosine"} <= fields, "ablation A"
    assert {"policy_entropy", "distinct_completions"} <= fields, "ablation B"
    assert {"tokens", "max_abs_advantage"} <= fields, "ablation C"
    assert {"kl_to_ref", "kl_loss_fraction"} <= fields, "ablation B rig-check"
    assert {"frac_degenerate_groups", "grad_norm"} <= fields, "ablation D"


def test_transfer_efficiency_records_pair_provenance() -> None:
    """Hub-sourced and model-authored A/A' pairs are never pooled (desideratum 11)."""
    fields = set(measure.TransferEfficiency.__dataclass_fields__)
    assert {"pair_source", "idiom_correlation", "eta"} <= fields


def test_assay_report_records_weighting_provenance() -> None:
    """assay_score is fitted; every report says so (desideratum 10)."""
    assert "weighting_source" in battery.AssayReport.__dataclass_fields__


def test_assay_score_has_no_default_weighting() -> None:
    """Passing a weighting must be a deliberate act, not a default that hides provenance."""
    sig = inspect.signature(battery.assay_score)
    assert sig.parameters["weights"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------------------
# Stubs must fail loudly, not silently return None.
# --------------------------------------------------------------------------------------


def test_unimplemented_stubs_raise() -> None:
    with pytest.raises(NotImplementedError):
        config.core_grid()
    with pytest.raises(NotImplementedError):
        config.positive_control()


def test_cli_parser_builds() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["screen", "bisect", "-k", "8"])
    assert args.command == "screen"
    assert args.k == 8


def test_cli_run_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError):
        cli.main(["run", "bisect"])
