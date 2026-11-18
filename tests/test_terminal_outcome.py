"""Historical attempt diagnostics. Public scoring is tested by the evaluation harness."""
import pytest

def terminal(stage="STEP_BUDGET_EXHAUSTED", reason="BUDGET_EXHAUSTED", **extra):
    return dict(success=False, completed=True, outcome_class="real_fail",
                failure_stage=stage, failure_reason=reason, steps=80, **extra)

@pytest.mark.parametrize("result,want", [
    (terminal(), "normal_step_limit"),
    (terminal(reason="UGV_NAV_STALLED"), "normal_step_limit"),
    (terminal(reason="UGV_PATROL_LOOP_ABORTED"), "normal_step_limit"),
    (terminal("FALSE_CONFIRM", "UGV_CONFIRMED_WRONG_OBJECT"), "normal_wrong_confirm"),
    (dict(success=True, completed=True, outcome_class="success", failure_stage=None,
          failure_reason="SUCCESS"), "normal_success"),
    (terminal("EPISODE_WALL_TIMEOUT"), "abnormal_timeout"),
    (terminal("VLM_BUDGET_EXHAUSTED"), "abnormal_guard_stop"),
    (terminal("SIM_TIME_EXHAUSTED"), "abnormal_guard_stop"),
    (dict(success=False, completed=False, outcome_class="infra_error",
          failure_stage="SUBPROCESS_CRASH", failure_reason="RUNTIME_EXCEPTION"), "abnormal_infrastructure"),
    (terminal("UNKNOWN", "UNKNOWN_FAILURE"), "unknown"),
    (terminal("FALSE_CONFIRM", "UNKNOWN_FAILURE"), "unknown"),
])
def test_terminal_classifier_cases(result, want):
    from benchmark.parallel.outcome import classify_terminal, is_normal_completion
    assert classify_terminal(result) == want
    assert is_normal_completion(result) is want.startswith("normal_")


@pytest.mark.parametrize("key,value", [("success", 0), ("completed", 1),
    ("completed", "true"), ("steps", True), ("steps", 79),
    ("outcome_class", "infra_error"), ("failure_reason", "SUCCESS")])
def test_contradictions_and_non_boolean_flags_are_unknown(key, value):
    from benchmark.parallel.outcome import classify_terminal
    result = terminal()
    result[key] = value
    assert classify_terminal(result) == "unknown"


def test_frozen_step_limit_and_wall_signal_during_normal_step_finish():
    from benchmark.parallel.outcome import classify_terminal
    result = terminal(supervisor_wall_timeout=True)
    assert classify_terminal(result) == "normal_step_limit"
    assert classify_terminal(result, step_limit=81) == "unknown"
