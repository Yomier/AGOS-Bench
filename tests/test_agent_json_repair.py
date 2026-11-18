"""The shared VLM JSON repair must accept the cooperative policy's nested object."""
import json

from agos.agents.agents import model_field_error, repair_json


NESTED = {
    "decision": {"owner_choice": "algorithm", "request_reply": "accept",
                 "request_id": None, "request_assistance": False, "sender_policy": "resume"},
    "action": {"role": "UAV", "action": "MOVE_TO_UAV_WAYPOINT", "waypoint_id": "U_0071",
               "target_verdict": "UNKNOWN", "fallback_action": "WAIT"},
}


def test_nested_three_level_object_parses():
    raw = json.dumps(NESTED, indent=2)
    assert repair_json(raw) == NESTED
    assert repair_json("```json\n" + raw + "\n```") == NESTED
    assert model_field_error(NESTED) == ""


def test_prose_wrapped_object_still_parses():
    raw = "Here is the decision:\n" + json.dumps(NESTED) + "\nDone."
    assert repair_json(raw) == NESTED
