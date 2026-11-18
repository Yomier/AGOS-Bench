"""Model-event cost and no-CoT output contracts, without network requests."""
import pytest

from agos.lite_core import LiteObservation


class Transport:
    def __init__(self, *args, **kwargs):
        self.responses = []

    def decide(self, system, prompt, images):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, "ok", "raw-json"


def policy(monkeypatch):
    from agos.methods.cooperative_search import policy as module
    monkeypatch.setattr(module, "VLMClient", Transport)
    return module.DecisionPolicy({"vlm": {"temperature": 0}})


def observation(role="UGV"):
    return LiteObservation("ep", 1, 80, role,
                           available_actions=["STOP_OBSERVE", "CONFIRM", "REJECT"]
                           if role == "UGV" else ["MOVE_TO_UAV_WAYPOINT", "WAIT"],
                           target_description="blue van")


@pytest.mark.parametrize(('role_models', 'expected'), [
    ({}, ['shared', 'shared']),
    ({'uav_model': 'aerial', 'ugv_model': 'ground'}, ['aerial', 'ground']),
])
def test_chat_payload_selects_model_for_each_role(monkeypatch, role_models, expected):
    import json
    from types import SimpleNamespace

    from agos.agents.http_accounting import capture_http_attempts
    from agos.methods.cooperative_search.policy import DecisionPolicy

    endpoint = 'https://example.invalid/v1/chat/completions'
    requests_seen = []

    def respond(url, **kwargs):
        payload = kwargs['json']
        prompt = json.loads(payload['messages'][1]['content'][-1]['text'])
        role = prompt['role']
        requests_seen.append((url, role, payload['model']))
        answer = json.dumps({'decision': {'owner_choice': 'mllm'},
                             'action': {'role': role, 'action': 'WAIT'}})
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {'choices': [{'message': {'content': answer}}]})

    monkeypatch.setattr('agos.agents.agents.requests.post', respond)
    policy = DecisionPolicy({'vlm': {'model': 'shared', 'api_url': endpoint,
                                     'api_key_env': 'AGOS_TEST_UNUSED_KEY', **role_models}})
    with capture_http_attempts() as attempts:
        for role in ('UAV', 'UGV'):
            result = policy.decide(observation(role), {'phase': 'coverage'}, [], 'observation', 1)
            assert result['action']['role'] == role
    assert attempts == [2]
    assert requests_seen == [(endpoint, 'UAV', expected[0]), (endpoint, 'UGV', expected[1])]


def test_algorithm_boundary_without_event_has_zero_model_calls(monkeypatch):
    p = policy(monkeypatch)
    assert p.decide(observation(), {"phase": "coverage"}, [], None, 100) == {}
    assert p.drain_calls() == 0


def test_primary_decision_can_directly_confirm_without_aerial_gate(monkeypatch):
    p = policy(monkeypatch)
    p.vlm.responses = [{"action": {"role": "UGV", "action": "CONFIRM",
                                     "evidence_view": "V1", "target_verdict": "MATCH"}}]
    result = p.decide(observation(), {"phase": "verify"}, [], "verify", 1)
    assert result["action"]["action"] == "CONFIRM"
    assert p.drain_calls() == 1
    assert p.drain_calls() == 0


def test_no_budget_for_second_crop_call(monkeypatch):
    p = policy(monkeypatch)
    p.vlm.responses = [{"action": {"role": "UAV", "action": "SUSPECT_CANDIDATE"}}]
    obs = observation("UAV")
    p.decide(obs, {"phase": "coverage"}, [], "new_evidence", 1)
    verification = p.verify_proposals(obs, [{"proposal_id": "P1"}])
    assert verification["status"] == "budget_exhausted"
    assert p.drain_calls() == 1


def test_capture_root_writes_prompt_images_with_hashes(monkeypatch, tmp_path):
    import hashlib
    import pathlib
    from agos.methods.cooperative_search import policy as module
    monkeypatch.setattr(module, "VLMClient", Transport)
    p = module.DecisionPolicy({"vlm": {"temperature": 0}, "capture_root": str(tmp_path)})
    p.vlm.responses = [{"action": {"role": "UAV", "action": "WAIT"}}]
    obs = observation("UAV")
    obs.target_reference_image = b"reference-bytes"
    obs.agent_image = b"scene-bytes"
    p.decide(obs, {"phase": "coverage"}, [], "new_evidence", 5)
    events = p.drain_events()
    assert len(events) == 1
    event = events[0]
    assert event["target"]["action"]["action"] == "WAIT"
    assert [pathlib.Path(path).read_bytes() for path in event["image_paths"]] == [
        b"reference-bytes", b"scene-bytes"]
    assert [hashlib.sha256(block).hexdigest() for block in (
        b"reference-bytes", b"scene-bytes")] == event["image_sha256"]
    assert p.drain_events() == []


def test_crop_request_and_primary_are_counted_separately(monkeypatch):
    p = policy(monkeypatch)
    p.vlm.responses = [{"action": {"role": "UAV", "action": "SUSPECT_CANDIDATE"}},
                       {"selected_proposal": "P1", "target_verdict": "MATCH"}]
    obs = observation("UAV")
    p.decide(obs, {"phase": "coverage"}, [], "new_evidence", 2)
    result = p.verify_proposals(obs, [{"proposal_id": "P1"}])
    assert result["selected_proposal"] == "P1"
    assert p.drain_calls() == 2
    assert [event["call_kind"] for event in p.events] == ["primary_decision", "gsam_crop"]


def test_exception_still_drains_real_started_request(monkeypatch):
    p = policy(monkeypatch)
    p.vlm.responses = [RuntimeError("transport-fault")]
    with pytest.raises(RuntimeError, match="transport-fault"):
        p.decide(observation(), {"phase": "verify"}, [], "verify", 1)
    assert p.drain_calls() == 1
    assert p.events[0]["status"] == "error"


def test_fixed_phase_and_reason_are_input_not_predicted_cot(monkeypatch):
    p = policy(monkeypatch)
    p.vlm.responses = [{"decision": {"owner_choice": "algorithm"},
                       "action": {"role": "UGV", "action": "STOP_OBSERVE"},
                       "reasoning": "unwanted prose", "task_phase": "future_success"}]
    result = p.decide(observation(), {"phase": "coverage"}, [], "routine_check", 1)
    assert result["owner_choice"] == "algorithm"
    assert "reasoning" not in result and "task_phase" not in result
    event = p.events[0]
    assert event["public_input"]["task_phase"] == "coverage"
    assert event["public_input"]["call_reason"] == "routine_check"
    assert event["target"] == result

def test_capture_directory_separates_towns_with_the_same_episode_id(monkeypatch, tmp_path):
    from agos.lite_core import LiteObservation
    from agos.methods.cooperative_search import policy as module

    class Transport:
        def __init__(self, *args, **kwargs):
            pass

        def decide(self, *args):
            return {"action": "WAIT"}, "ok", "json"

    monkeypatch.setattr(module, "VLMClient", Transport)
    p = module.DecisionPolicy({"vlm": {"temperature": 0}}, capture_root=str(tmp_path))
    first = LiteObservation("lite_easy_00002", 1, 80, "UAV")
    second = LiteObservation("lite_easy_00002", 1, 80, "UAV")
    first.target_reference_image = b"scene-one"
    second.target_reference_image = b"scene-two"
    assert p._capture_dir(first) != p._capture_dir(second)
    assert p._capture_dir(first).parent == tmp_path


def test_verify_call_uses_the_flat_answer_system_primary_does_not(monkeypatch):
    """Only crop verification uses the flat-answer prompt when enabled."""
    from agos.methods.cooperative_search import policy as module
    monkeypatch.setattr(module, "VLMClient", Transport)
    p = module.DecisionPolicy({"vlm": {"temperature": 0},
                               "cooperative_search": {"verify_system_v2": True}})
    p.vlm.responses = [{"action": {"role": "UAV", "action": "SUSPECT_CANDIDATE"}},
                       {"selected_proposal": "P1", "target_verdict": "MATCH"}]
    obs = observation("UAV")
    p.decide(obs, {"phase": "coverage"}, [], "new_evidence", 2)
    p.verify_proposals(obs, [{"proposal_id": "P1"}])
    kinds = [(event["call_kind"], event["system_prompt"]) for event in p.events]
    assert kinds[0][0] == "primary_decision"
    assert kinds[0][1] is module.SYSTEM
    assert kinds[1][0] == "gsam_crop"
    assert "no wrappers" in kinds[1][1]
    # The default configuration shares the primary prompt.
    p_default = policy(monkeypatch)
    p_default.vlm.responses = [{"action": {"role": "UAV", "action": "SUSPECT_CANDIDATE"}},
                               {"selected_proposal": "P1", "target_verdict": "MATCH"}]
    p_default.decide(obs, {"phase": "coverage"}, [], "new_evidence", 2)
    p_default.verify_proposals(obs, [{"proposal_id": "P1"}])
    assert p_default.events[-1]["system_prompt"] is module.SYSTEM


def test_expert_verify_site_carries_the_same_system_as_deployment_verify(monkeypatch):
    """Byte parity: the frozen expert_verify row trains on the exact system
    prompt the deployment verify call uses, with and without the v2 flag."""
    from agos.methods.cooperative_search import policy as module
    for flag in (False, True):
        monkeypatch.setattr(module, "VLMClient", Transport)
        p = module.DecisionPolicy({"vlm": {"temperature": 0},
                                   "cooperative_search": {"verify_system_v2": flag}})
        obs = observation("UAV")
        p.record_expert_verify(obs, [{"proposal_id": "P1"}],
                               {"selected_proposal": "P1", "target_verdict": "MATCH"},
                               "new_evidence")
        p._remaining = 2
        p.vlm.responses = [{"action": {"role": "UAV", "action": "SUSPECT_CANDIDATE"}},
                           {"selected_proposal": "P1", "target_verdict": "MATCH"}]
        p.decide(obs, {"phase": "coverage"}, [{"proposal_id": "P1"}], "new_evidence", 2)
        p.verify_proposals(obs, [{"proposal_id": "P1"}])
        expert = next(e for e in p.events if e["call_kind"] == "expert_verify")
        deployed = next(e for e in p.events if e["call_kind"] == "gsam_crop")
        assert expert["system_prompt"] == deployed["system_prompt"]


def test_verify_reads_answers_nested_the_way_the_system_prompt_asks(monkeypatch):
    # The shared system prompt places these fields under action.
    p = policy(monkeypatch)
    p.vlm.responses = [{"action": {"role": "UAV", "action": "SUSPECT_CANDIDATE"}},
                       {"decision": {"owner_choice": "mllm"},
                        "action": {"selected_proposal": "P1", "target_verdict": "MATCH"}}]
    obs = observation("UAV")
    p.decide(obs, {"phase": "coverage"}, [], "new_evidence", 2)
    result = p.verify_proposals(obs, [{"proposal_id": "P1"}])
    assert result["selected_proposal"] == "P1"
    assert result["target_verdict"] == "MATCH"
    assert result["status"] == "ok"
    assert p.drain_calls() == 2


def test_verify_reads_the_system_declared_proposal_id_alias(monkeypatch):
    # The primary system prompt calls the selected field proposal_id.
    p = policy(monkeypatch)
    p.vlm.responses = [{"action": {"role": "UAV", "action": "SUSPECT_CANDIDATE"}},
                       {"decision": {"resume_search": True},
                        "action": {"role": "UAV", "action": "SUSPECT_CANDIDATE",
                                   "waypoint_id": "U_0024", "candidate_id": "P1",
                                   "target_verdict": "MATCH", "evidence_view": "P1",
                                   "proposal_id": "P1",
                                   "fallback_action": "MOVE_TO_UAV_WAYPOINT",
                                   "fallback_waypoint_id": "U_0024"}}]
    obs = observation("UAV")
    p.decide(obs, {"phase": "coverage"}, [], "new_evidence", 2)
    result = p.verify_proposals(obs, [{"proposal_id": "P1"}])
    assert result["selected_proposal"] == "P1"
    assert result["target_verdict"] == "MATCH"
    assert p.events[-1]["verify_raw"]["selected_proposal"] == "P1"


def test_verify_prefers_top_level_then_action_then_decision_and_keeps_raw(monkeypatch):
    p = policy(monkeypatch)
    p.vlm.responses = [{"action": {"role": "UAV", "action": "SUSPECT_CANDIDATE"}},
                       {"selected_proposal": "P0", "target_verdict": "UNKNOWN",
                        "decision": {"selected_proposal": "P2", "target_verdict": "UNKNOWN"},
                        "action": {"selected_proposal": "P9", "target_verdict": "MATCH"}}]
    obs = observation("UAV")
    p.decide(obs, {"phase": "coverage"}, [], "new_evidence", 2)
    result = p.verify_proposals(obs, [{"proposal_id": "P0"}])
    assert result["selected_proposal"] == "P0"  # top level wins over wrappers
    assert p.events[-1]["verify_raw"]["selected_proposal"] == "P0"
    # nested-only answers: action is the SYSTEM-declared field home, then decision
    p.vlm.responses = [{"decision": {"selected_proposal": "P2", "target_verdict": "UNKNOWN"},
                        "action": {"selected_proposal": "P9", "target_verdict": "MATCH"}}]
    p._remaining = 2
    result = p.verify_proposals(obs, [{"proposal_id": "P9"}])
    assert result["selected_proposal"] == "P9"
    assert result["target_verdict"] == "MATCH"
    p.vlm.responses = [{"decision": {"selected_proposal": "P2", "target_verdict": "UNKNOWN"}}]
    p._remaining = 2
    result = p.verify_proposals(obs, [{"proposal_id": "P2"}])
    assert result["selected_proposal"] == "P2"
    assert result["target_verdict"] == "UNKNOWN"


def test_verify_invalid_choice_still_sanitises_but_the_raw_answer_survives(monkeypatch):
    p = policy(monkeypatch)
    p.vlm.responses = [{"action": {"role": "UAV", "action": "SUSPECT_CANDIDATE"}},
                       {"action": {"selected_proposal": "GHOST", "target_verdict": "MATCH"}}]
    obs = observation("UAV")
    p.decide(obs, {"phase": "coverage"}, [], "new_evidence", 2)
    result = p.verify_proposals(obs, [{"proposal_id": "P1"}])
    assert result["selected_proposal"] == "NONE"
    assert result["target_verdict"] == "UNKNOWN"
    event = p.events[-1]
    assert event["verify_raw"] == {"selected_proposal": "GHOST", "target_verdict": "MATCH"}
