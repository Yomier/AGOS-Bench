"""Scientific terminal outcomes; no simulator, model, or scoring dependencies.

Callers supply the frozen decision limit. Result quality, cleanup receipts and
supervisor timeout signals do not change an otherwise consistent terminal.
Classification does not grant permission to retry an abnormal attempt.
"""
from collections.abc import Mapping


INFRASTRUCTURE_STAGES = frozenset({
    "CARLA_CONNECT_TIMEOUT", "AIRSIM_CONNECT_TIMEOUT", "EMPTY_CAMERA_FRAME",
    "RPC_TIMEOUT", "SERVER_PROCESS_EXITED", "RESULT_FILE_CORRUPT",
    "ROAD_GRAPH_LOAD_ERROR", "TARGET_SPAWN_FAILED", "UGV_SPAWN_FAILED",
    "VLM_INFRA_FAILURE", "METHOD_RUNTIME_EXCEPTION", "SUBPROCESS_CRASH",
    "RUNTIME_EXCEPTION", "INFRA_ERROR", "VLM_CONSECUTIVE_FAILURE",
    "ORACLE_UNREACHABLE", "ORACLE_NOT_REACHABLE",
})
NORMAL_TERMINALS = frozenset({"normal_success", "normal_wrong_confirm", "normal_step_limit"})
STEP_REASONS = frozenset({"BUDGET_EXHAUSTED", "UGV_NAV_STALLED", "UGV_PATROL_LOOP_ABORTED"})
INFRASTRUCTURE_REASONS = frozenset({
    "RUNTIME_EXCEPTION", "INFRA_ERROR", "VLM_CONSECUTIVE_FAILURE",
    "NO_VALID_CONFIRM_NODE", "ORACLE_UNREACHABLE", "ORACLE_NOT_REACHABLE",
})


def classify_terminal(result, step_limit=80):
    """Return a terminal category, or unknown for incomplete/conflicting data."""
    if not isinstance(result, Mapping) or type(step_limit) is not int or step_limit <= 0:
        return "unknown"
    success, completed = result.get("success"), result.get("completed")
    if type(success) is not bool or type(completed) is not bool:
        return "unknown"
    stage, reason, outcome = (result.get(k) for k in
                              ("failure_stage", "failure_reason", "outcome_class"))
    if (stage is not None and not isinstance(stage, str)) or not isinstance(reason, str) or not isinstance(outcome, str):
        return "unknown"
    if success:
        return ("normal_success" if completed and outcome == "success" and
                reason == "SUCCESS" and stage in (None, "") else "unknown")
    if reason == "SUCCESS" or outcome == "success":
        return "unknown"
    if completed and outcome == "real_fail":
        if stage == "FALSE_CONFIRM" and reason == "UGV_CONFIRMED_WRONG_OBJECT":
            return "normal_wrong_confirm"
        if stage == "STEP_BUDGET_EXHAUSTED" and reason in STEP_REASONS:
            steps = result.get("steps")
            return ("normal_step_limit" if type(steps) is int and steps >= step_limit else "unknown")
    # Explicitly accept old wall/guard serialization as abnormal: historical
    # completed=True/real_fail/BUDGET_EXHAUSTED never meant normal completion.
    legacy = completed and outcome == "real_fail"
    abnormal = not completed and outcome in ("infra_error", "guard_stop")
    if legacy or abnormal:
        if stage == "EPISODE_WALL_TIMEOUT" and reason in ("BUDGET_EXHAUSTED", "EPISODE_WALL_TIMEOUT"):
            return "abnormal_timeout"
        if stage in ("VLM_BUDGET_EXHAUSTED", "SIM_TIME_EXHAUSTED") and reason in (
                "BUDGET_EXHAUSTED", "GUARD_STOP", stage):
            return "abnormal_guard_stop"
    if (not completed and outcome == "infra_error" and
            stage in INFRASTRUCTURE_STAGES and
            (reason == stage or reason in INFRASTRUCTURE_REASONS)):
        return "abnormal_infrastructure"
    return "unknown"


def is_normal_completion(result, step_limit=80):
    return classify_terminal(result, step_limit) in NORMAL_TERMINALS
