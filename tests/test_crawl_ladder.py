"""The ladder table — comparability is a property of the table, so it is tested, not assumed.

An ablation that changes two things at once produces a result attributable to neither. These tests
make that structural rather than a matter of care while editing.
"""

from __future__ import annotations

from dataclasses import fields

from assay.crawl.config import LadderConfig
from assay.crawl.ladder import ABLATION_B_CONTROL, ABLATION_C, BETA, LADDER, RUN7


def _differences(a: LadderConfig, b: LadderConfig) -> set[str]:
    """Fields that differ, ignoring identity fields overwritten at dispatch.

    ``diagnostic_centered_cosine`` is ignored because it is *provably* not a switch: it adds two
    backward passes on an already-built graph and restores the gradient buffer, and
    ``test_the_centred_cosine_does_not_change_the_update`` asserts bit-identical parameters with it
    on and off. Counting it as a difference would make rungs 1-3 look non-comparable to run 7 over a
    field that cannot affect any of them.
    """
    ignore = {"run_id", "setting", "seed", "diagnostic_centered_cosine"}
    return {
        f.name
        for f in fields(a)
        if f.name not in ignore and getattr(a, f.name) != getattr(b, f.name)
    }


def test_the_cut_is_reflected_in_the_table() -> None:
    """Runs 1, 2, 3, 7 plus four ablations and two length arms. Runs 4-6 are cut."""
    assert set(LADDER) == {
        "run1", "run2", "run3", "run7",
        "ablation_b", "ablation_b_control", "ablation_c", "ablation_d",
        "run7_nolennorm", "ablation_c_nolennorm",
    }
    assert not {"run4", "run5", "run6"} & set(LADDER)


def test_the_rungs_differ_only_in_their_baseline() -> None:
    """Ablation A compares run1 against run2. If anything else differed, the comparison is void."""
    for key in ("run1", "run2"):
        assert _differences(LADDER[key], LADDER["run3"]) == {"baseline"}


def test_run7_adds_exactly_normalisation_and_the_leash_to_rung_3() -> None:
    assert _differences(RUN7, LADDER["run3"]) == {"normalize_by_std", "kl_coef"}


def test_each_ablation_differs_from_its_parent_by_one_switch() -> None:
    """The property that makes each result attributable. Enforced, not trusted."""
    assert _differences(LADDER["ablation_c"], RUN7) == {"reward"}
    assert _differences(LADDER["ablation_d"], RUN7) == {"force_unanimous_groups"}
    assert _differences(LADDER["run7_nolennorm"], RUN7) == {"length_normalize"}
    assert _differences(LADDER["ablation_c_nolennorm"], ABLATION_C) == {"length_normalize"}
    # B is the instructive exception: two changes from run 7, so it routes through a control.
    assert _differences(ABLATION_B_CONTROL, RUN7) == {"reward"}
    assert _differences(LADDER["ablation_b"], ABLATION_B_CONTROL) == {"kl_coef"}


def test_ablation_b_alone_would_have_been_confounded() -> None:
    """Without the control, B changes both the leash and the reward — attributable to neither."""
    assert _differences(LADDER["ablation_b"], RUN7) == {"kl_coef", "reward"}


def test_the_leash_is_on_where_it_should_be_and_off_where_it_should_not() -> None:
    assert LADDER["ablation_b"].kl_coef == 0.0, "B is defined by removing the leash"
    assert LADDER["ablation_b_control"].kl_coef == BETA, "the control keeps it"
    assert LADDER["run7"].kl_coef == BETA
    for rung in ("run1", "run2", "run3"):
        assert LADDER[rung].kl_coef == 0.0, "the leash arrives at rung 5, which is cut"


def test_the_centred_cosine_is_on_exactly_where_ablation_a_needs_it() -> None:
    """Rungs 1-3 only. Ignored by ``_differences``, so nothing else would catch it drifting."""
    on = {key for key, cfg in LADDER.items() if cfg.diagnostic_centered_cosine}
    assert on == {"run1", "run2", "run3"}


def test_every_entry_carries_the_pinned_task_and_rate() -> None:
    for key, cfg in LADDER.items():
        assert cfg.setting == "add-3digit", f"{key} is not on the primary arm"
        assert cfg.learning_rate == 1e-5, f"{key} does not carry the probed rate"
        assert cfg.group_size == 8 and cfg.epochs_per_batch == 1


def test_clipping_is_inert_everywhere() -> None:
    """Single epoch pins the importance ratio at 1, so no entry may claim to clip."""
    assert not any(cfg.clipping_is_active for cfg in LADDER.values())
