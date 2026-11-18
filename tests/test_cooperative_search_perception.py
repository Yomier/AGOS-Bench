"""Public frame screening and meaningful evidence deduplication."""
import io
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from agos.lite_core import LiteObservation, UAVFrame
from agos.methods.common.grounded_localizer import GroundResult, GroundedProposal


def png():
    result = io.BytesIO()
    Image.new("RGB", (100, 60), "blue").save(result, format="PNG")
    return result.getvalue()


def screen(monkeypatch):
    from agos.methods.cooperative_search import perception as module

    class Localizer:
        def __init__(self, *args, **kwargs):
            self.result = GroundResult("ok", [GroundedProposal(
                "P1", [20, 20, 40, 40], detector_score=0.9, identity_score=0.8)])

        def ground_frame(self, frame, description, reference, segment=False):
            return self.result

    monkeypatch.setattr(module, "GroundedLocalizer", Localizer)
    return module.ProposalScreen({"wake_identity_min": 0.72, "evidence_viewpoint_m": 20})


def obs(x=0, frame=1):
    result = LiteObservation("ep", 1, 80, "UAV", agent_position=(x, 0, 80),
                             agent_image=png(), target_reference_image=png())
    result.uav_frame = UAVFrame(png(), frame, 0, x, 0, 80, 0, -90, 90, 100, 60)
    return result


def test_new_frame_at_same_view_does_not_repeatedly_wake(monkeypatch):
    p = screen(monkeypatch)
    first, second = p.screen(obs(frame=1)), p.screen(obs(frame=2))
    assert first["evidence_id"]
    assert second["evidence_id"] is None
    assert second["proposals"][0]["proposal_id"] == "P1"


def test_fresh_viewpoint_can_wake_after_prior_evidence(monkeypatch):
    p = screen(monkeypatch)
    first, second = p.screen(obs()), p.screen(obs(x=40, frame=2))
    assert first["evidence_id"] != second["evidence_id"]
    assert second["evidence_id"]


def test_low_retrieval_score_is_not_a_wake_or_a_negative_label(monkeypatch):
    p = screen(monkeypatch)
    p.air.result.proposals[0].identity_score = 0.4
    result = p.screen(obs())
    assert result["evidence_id"] is None
    assert len(result["proposals"]) == 1
    assert "target_verdict" not in result["proposals"][0]


def test_tool_failure_is_public_fault_not_empty_scene(monkeypatch):
    p = screen(monkeypatch)
    p.air.result = GroundResult("error", error="service_timeout")
    result = p.screen(obs())
    assert result["status"] == "error"
    assert result["fault_id"]
    assert result["evidence_id"] is None


def test_invalid_frame_binding_is_hard_contract_error(monkeypatch):
    p = screen(monkeypatch)
    p.air.result = GroundResult("invalid_source", error="wrong_frame")
    with pytest.raises(ValueError, match="invalid_source"):
        p.screen(obs())
