"""Public causal request/arrival and event-call scheduling specifications."""
from agos.lite_core import LiteObservation


def make_controller(arm="agos_b"):
    from agos.methods.cooperative_search.controller import CooperativeController
    return CooperativeController(arm, {"routine_progress": 0.5, "routine_per_role": 1})


def obs(role="UGV", time=0):
    result = LiteObservation("ep", 1, 80, role)
    result.coop_event_time = time
    return result


def request(key="R1", sender="UAV", available_at=0):
    return {"request_id": key, "message_id": "msg-" + key,
            "sender": sender, "receiver": "UGV" if sender == "UAV" else "UAV",
            "type": "assistance_request", "candidate_id": "C1", "available_at": available_at}


def test_request_waits_for_full_action_boundary():
    c = make_controller()
    c.transition("UGV", {"ready_at": 100, "effect_at": 0})
    c.ingest_message(request())
    assert c.boundary("UGV", obs(time=0), {})["call_reason"] is None
    assert c.boundary("UGV", obs(time=100), {})["call_reason"] == "request_receipt"


def test_receipt_and_arrival_are_separate_calls():
    c = make_controller()
    c.ingest_message(request())
    receipt = c.boundary("UGV", obs(), {})
    assert receipt["phase"] == "receive"
    assert c.apply_decision("UGV", {"request_reply": "accept", "request_id": "R1"}) is True
    approach = c.boundary("UGV", obs(time=1), {"arrived": False})
    assert approach["phase"] == "approach"
    assert approach["call_reason"] is None
    arrival = c.boundary("UGV", obs(time=2), {"arrived": True})
    assert arrival["call_reason"] == "arrival"
    c.apply_decision("UGV", {"action": "STOP_OBSERVE"})
    assert c.boundary("UGV", obs(time=3), {})["call_reason"] == "verify"


def test_defer_is_retained_without_duplicate_wake():
    c = make_controller()
    c.ingest_message(request())
    c.boundary("UGV", obs(), {})
    c.apply_decision("UGV", {"request_reply": "defer", "request_id": "R1"})
    c.ingest_message(request())
    boundary = c.boundary("UGV", obs(time=1), {})
    assert boundary["call_reason"] is None
    assert "R1" in boundary["pending_request_ids"]


def test_unbindable_model_reply_is_dropped_not_fatal():
    c = make_controller()
    c.apply_decision("UGV", {"request_reply": "accept", "request_id": None})
    assert c.roles["UGV"]["phase"] == "coverage"
    assert c.roles["UGV"]["current_receipt"] is None
    c.apply_decision("UGV", {"action": "STOP_OBSERVE", "owner_choice": "algorithm"})
    assert c.roles["UGV"]["owner"] == "algorithm"


def test_reverse_assistance_follows_same_sequence():
    c = make_controller()
    c.ingest_message(request(sender="UGV"))
    assert c.boundary("UAV", obs("UAV"), {})["call_reason"] == "request_receipt"
    c.apply_decision("UAV", {"request_reply": "accept", "request_id": "R1"})
    assert c.boundary("UAV", obs("UAV", 1), {"arrived": True})["call_reason"] == "arrival"


def test_candidate_handoff_does_not_freeze_uav():
    c = make_controller()
    c.apply_decision("UAV", {"sender_policy": "resume", "candidate_id": "C1"})
    state = c.boundary("UAV", obs("UAV", 1), {})
    assert state["owner"] == "algorithm"
    assert state["phase"] == "coverage"
    assert state.get("frozen", False) is False


def test_verify_exit_requests_one_reallocation():
    c = make_controller()
    c.apply_decision("UGV", {"phase": "verify"})
    c.apply_decision("UGV", {"action": "REJECT"})
    first = c.boundary("UGV", obs(time=1), {})
    second = c.boundary("UGV", obs(time=2), {})
    assert first["reallocate_reason"] == "verification_end"
    assert second["reallocate_reason"] is None
    assert second["phase"] == "coverage"


def test_multiple_pending_requests_survive_recent_window():
    c = make_controller()
    for n in range(8):
        c.ingest_message(request(f"R{n}"))
    state = c.boundary("UGV", obs(time=1), {})
    assert len(state["pending_request_ids"]) == 8
    assert state["request"]["request_id"] == "R0"


def test_future_time_message_does_not_block_earlier_visible_request():
    c = make_controller()
    c.ingest_message(request("future", available_at=100))
    c.ingest_message(request("now", available_at=1))
    assert c.boundary("UGV", obs(time=2), {})["request"]["request_id"] == "now"


def test_routine_checkpoint_occurs_once_but_events_are_unlimited():
    c = make_controller()
    assert c.boundary("UAV", obs("UAV", 1), {"progress": 0.5})["call_reason"] == "routine_check"
    assert c.boundary("UAV", obs("UAV", 2), {"progress": 0.8})["call_reason"] is None
    assert c.boundary("UAV", obs("UAV", 3), {"evidence_id": "frame1"})["call_reason"] == "new_evidence"
    assert c.boundary("UAV", obs("UAV", 4), {"evidence_id": "frame1"})["call_reason"] is None
    assert c.boundary("UAV", obs("UAV", 5), {"evidence_id": "frame2"})["call_reason"] == "new_evidence"


def test_routine_per_role_zero_never_wakes():
    """E4 (w/o routine-check) is pure config: a zero routine budget must never
    fire a routine wake even at full progress, while real events still wake."""
    from agos.methods.cooperative_search.controller import CooperativeController
    c = CooperativeController("agos_b", {"routine_progress": 0.5, "routine_per_role": 0})
    assert c.boundary("UAV", obs("UAV", 1), {"progress": 0.5})["call_reason"] is None
    assert c.boundary("UAV", obs("UAV", 2), {"progress": 1.0})["call_reason"] is None
    assert c.boundary("UAV", obs("UAV", 3), {"evidence_id": "frame1"})["call_reason"] == "new_evidence"


def test_a_calls_at_coverage_decisions_b_skips_without_event():
    assert make_controller("agos_a").boundary("UAV", obs("UAV"), {})["call_reason"] == "observation"
    assert make_controller("baseline_a").boundary("UAV", obs("UAV"), {})["call_reason"] == "observation"
    assert make_controller().boundary("UAV", obs("UAV"), {})["call_reason"] is None


def test_wake_factor_is_orthogonal_to_coverage():
    """``agos_c`` is fixed coverage with event wake: it must skip the routine
    observation call exactly like ``agos_b`` and still answer real events."""
    idle = make_controller("agos_c").boundary("UAV", obs("UAV"), {})
    assert idle["call_reason"] is None and idle["owner"] == "algorithm"
    busy = make_controller("agos_c").boundary("UAV", obs("UAV", 1), {"evidence_id": "frame9"})
    assert busy["call_reason"] == "new_evidence"

def test_resume_search_clears_the_request_without_a_terminal_claim():
    c = make_controller()
    c.ingest_message(request())
    c.boundary("UGV", obs(), {})
    c.apply_decision("UGV", {"request_reply": "accept", "request_id": "R1"})
    c.boundary("UGV", obs(time=1), {"arrived": True})
    assert c.apply_decision("UGV", {"resume_search": True}) is False
    state = c.roles["UGV"]
    assert state["phase"] == "coverage" and state["active_request"] is None
    assert c.boundary("UGV", obs(time=2), {})["call_reason"] is None


def test_receipt_binds_when_the_model_names_another_agents_request():
    """A stale ID in the reply still binds to the pending request."""
    c = make_controller()
    c.ingest_message(request("UGV:2"))
    receipt = c.boundary("UGV", obs(), {})
    assert receipt["call_reason"] == "request_receipt"
    applied = c.apply_decision("UGV", {"request_reply": "accept", "request_id": "UAV:1"})
    assert applied is True
    assert c.roles["UGV"]["phase"] == "approach"
    assert c.roles["UGV"]["active_request"]["request_id"] == "UGV:2"
    assert c.boundary("UGV", obs(time=1), {"arrived": False})["call_reason"] is None


def test_reply_without_any_pending_receipt_is_still_dropped():
    c = make_controller()
    assert c.apply_decision("UGV", {"request_reply": "accept", "request_id": "UAV:1"}) is False
    assert c.roles["UGV"]["phase"] == "coverage"


def test_accept_wins_over_resume_search_in_the_same_object():
    """An accepted handoff starts the approach even when the reply also asks to resume."""
    c = make_controller()
    c.ingest_message(request("UGV:2"))
    assert c.boundary("UGV", obs(), {})["call_reason"] == "request_receipt"
    applied = c.apply_decision("UGV", {"request_reply": "accept", "request_id": "UGV:2",
                                       "resume_search": True})
    assert applied is True
    assert c.roles["UGV"]["phase"] == "approach"
    assert c.boundary("UGV", obs(time=1), {"arrived": False})["call_reason"] is None


def test_resume_search_retires_a_pending_receipt_instead_of_looping():
    c = make_controller()
    c.ingest_message(request("UGV:3"))
    assert c.boundary("UGV", obs(), {})["call_reason"] == "request_receipt"
    assert c.apply_decision("UGV", {"resume_search": True}) is False
    assert c.roles["UGV"]["phase"] == "coverage"
    assert c.boundary("UGV", obs(time=1), {})["call_reason"] is None
