"""Public role control, request lifecycles, and event-based decision sites."""
from copy import deepcopy

from .contracts import ARMS, EVENT_WAKE_ARMS


class CooperativeController:
    def __init__(self, arm: str, config: dict):
        if arm not in ARMS:
            raise ValueError("cooperative_search: unknown arm")
        self.arm, self.config = arm, config
        self.messages = set()
        self.requests = {}
        self.roles = {
            role: {"owner": "algorithm", "phase": "coverage", "ready_at": 0.0,
                   "active_request": None, "routine_calls": 0, "evidence_seen": set(),
                   "faults_seen": set(), "reallocate_reason": None,
                   "sender_policy": "resume", "current_receipt": None}
            for role in ("UAV", "UGV")
        }

    def ingest_message(self, message: dict) -> None:
        identity = message.get("message_id") or message.get("request_id")
        if not identity or identity in self.messages:
            return
        if message.get("type") not in ("assistance_request", "candidate_request"):
            return
        if message.get("receiver") not in self.roles or message.get("sender") not in self.roles:
            raise ValueError("cooperative_search: explicit public sender and receiver required")
        if message["sender"] == message["receiver"] or not message.get("request_id"):
            raise ValueError("cooperative_search: invalid request identity")
        self.messages.add(identity)
        request = deepcopy(message)
        request["state"] = "pending"
        # A message update may make a deferred request actionable again, but only
        # with a fresh public message identity, never by repeat delivery.
        self.requests[request["request_id"]] = request

    def boundary(self, role: str, observation, route_status: dict) -> dict:
        state = self.roles[role]
        now = float(getattr(observation, "coop_event_time", 0.0))
        visible = [request for request in self.requests.values()
                   if request["receiver"] == role
                   and float(request.get("available_at", 0)) <= now
                   and request["state"] in ("pending", "deferred")]
        visible.sort(key=lambda request: (float(request.get("available_at", 0)), request["request_id"]))
        out = {"owner": state["owner"], "phase": state["phase"], "call_reason": None,
               "request": deepcopy(state["active_request"]),
               "pending_request_ids": [request["request_id"] for request in visible],
               "reallocate_reason": None, "sender_policy": state["sender_policy"],
               "action_ready": now >= state["ready_at"]}
        if not out["action_ready"]:
            return out
        out["reallocate_reason"] = state["reallocate_reason"]
        state["reallocate_reason"] = None
        pending = next((request for request in visible if request["state"] == "pending"), None)
        if pending is not None:
            state["current_receipt"] = pending["request_id"]
            out.update(owner="mllm", phase="receive", call_reason="request_receipt",
                       request=deepcopy(pending))
            return out
        if state["phase"] == "approach" and route_status.get("arrived"):
            state["phase"], state["owner"] = "arrival", "mllm"
        if state["phase"] in ("arrival", "verify"):
            out.update(owner="mllm", phase=state["phase"],
                       call_reason="arrival" if state["phase"] == "arrival" else "verify")
            return out
        fault_id = route_status.get("fault_id")
        if fault_id and fault_id not in state["faults_seen"]:
            state["faults_seen"].add(fault_id)
            out.update(owner="mllm", phase="recover", call_reason="fault")
            return out
        evidence_id = route_status.get("evidence_id")
        if evidence_id and evidence_id not in state["evidence_seen"]:
            state["evidence_seen"].add(evidence_id)
            out.update(owner="mllm", call_reason="new_evidence")
            return out
        # Accepted tasks use automatic legal approach until a new public event.
        if state["phase"] == "approach":
            return out
        if self.arm not in EVENT_WAKE_ARMS:
            out.update(owner="mllm", call_reason="observation")
        elif (state["routine_calls"] < int(self.config.get("routine_per_role", 1))
              and float(route_status.get("progress", 0)) >= float(self.config.get("routine_progress", 0.5))):
            state["routine_calls"] += 1
            out.update(owner="mllm", call_reason="routine_check")
        return out

    def apply_decision(self, role: str, decision: dict) -> bool:
        """Apply a model control decision; returns True only for an applied reply."""
        state = self.roles[role]
        # 2026-10-31 v5 (the actual root cause of the whole P0 detour): the 4B emits
        # ``request_reply=accept`` AND ``resume_search=true`` in the same object
        # (80/80 UGV steps in v5smoke). The old order tested resume_search first and
        # returned, so the accept was swallowed on every step: the receipt was never
        # consumed, the phase stayed ``receive`` (79 of 80 steps), no route ever
        # activated, and the UGV never drove to the candidate -- which is why v2's
        # confirms were all made from 50-90 m away. A reply therefore wins; deferring
        # the search only applies when no reply was applied.
        reply = decision.get("request_reply")
        if reply in ("accept", "defer"):
            key = decision.get("request_id")
            request = self.requests.get(key) if key else None
            if request is None or request["receiver"] != role or key != state["current_receipt"]:
                # The model may name another agent's request id; the framework owns
                # the pending identity, so the reply binds to that receipt. With no
                # pending receipt the reply is still dropped, never guessed.
                key = state["current_receipt"]
                request = self.requests.get(key) if key else None
                if request is None or request["receiver"] != role:
                    reply = None
            if reply in ("accept", "defer"):
                request["state"] = "accepted" if reply == "accept" else "deferred"
                state["current_receipt"] = None
                if reply == "accept":
                    state["active_request"] = deepcopy(request)
                    state.update(owner="algorithm", phase="approach", reallocate_reason="request_accepted")
                return True
        if decision.get("resume_search"):
            # Leaving an unverifiable arrival is a distinct non-terminal control:
            # no CONFIRM/REJECT claim is made and the coverage task resumes. A pending
            # receipt is retired rather than left visible, or the boundary re-fires it
            # every step (that is the 79/80 request_receipt loop).
            key = state["current_receipt"]
            pending = self.requests.get(key) if key else None
            if pending is not None and pending["receiver"] == role and pending["state"] == "pending":
                pending["state"] = "declined"
                state["reallocate_reason"] = "receipt_declined"
            state.update(owner="algorithm", phase="coverage", active_request=None,
                         current_receipt=None, reallocate_reason=state.get("reallocate_reason") or "verify_resume")
            return False
        if state["phase"] == "arrival":
            state.update(owner="mllm", phase="verify")
        # The session may explicitly enter independent ground verification after
        # a public visual discovery; this field is not a predicted phase label.
        if decision.get("phase") == "verify":
            state.update(owner="mllm", phase="verify")
        action = decision.get("action")
        if isinstance(action, dict):
            action = action.get("action")
        if action in ("REJECT", "CONFIRM"):
            state.update(owner="algorithm", phase="coverage", active_request=None,
                         reallocate_reason="verification_end")
        owner = decision.get("owner_choice")
        if owner in ("algorithm", "mllm") and state["phase"] == "coverage":
            state["owner"] = owner
        sender_policy = decision.get("sender_policy")
        if sender_policy in ("hold", "reposition", "resume"):
            state["sender_policy"] = sender_policy
            if sender_policy == "resume" and state["phase"] == "coverage":
                state["owner"] = "algorithm"
        return False

    def transition(self, role: str, receipt: dict) -> None:
        state = self.roles[role]
        ready = float(receipt.get("ready_at", state["ready_at"]))
        if ready < float(receipt.get("effect_at", ready)):
            raise ValueError("cooperative_search: ready_at precedes immediate action effect")
        state["ready_at"] = max(state["ready_at"], ready)
