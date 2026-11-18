"""Direct per-role VLM decisions for Minimal-Agent."""
import base64

from agos.agents.agents import VLMClient
from agos.benchmark_protocol import format_confirm_distance
from agos.lite_core import LiteAction
from agos.methods.common.legacy_sixview_collage import render_legacy_sixview_collage
from agos.methods.common.uav_annotation import bind_nadir_annotation, prepare_clean_uav
from .prompts import UAV_SYSTEM, UGV_SYSTEM, build_uav_prompt, build_ugv_prompt


class MinimalAgent:
    def __init__(self, role, config):
        self.role = role
        self.config = config
        vlm_config = config['vlm']
        self.vlm = VLMClient(vlm_config, max_retries=vlm_config.get('max_retries_per_call', 1))

    def decide(self, obs):
        if self.role == 'UAV':
            prepare_clean_uav(obs)
            system = UAV_SYSTEM.format(target_description=obs.target_description,
                                       max_annotations=self.config['uav']['max_candidate_annotations'])
            user = build_uav_prompt(obs)
            images = (obs.agent_image, obs.target_reference_image, obs.teammate_reference_image)
        else:
            if not obs.agent_image and obs.ugv_views:
                obs.agent_image = render_legacy_sixview_collage(obs.ugv_views)
            system = UGV_SYSTEM.format(target_description=obs.target_description,
                                       confirm_budget=self.config['ugv']['confirm_budget'],
                                       confirm_distance_m=format_confirm_distance(self.config))
            user = build_ugv_prompt(obs)
            images = (obs.agent_image, obs.target_reference_image)

        parsed, status, raw = self.vlm.decide(system, user, [base64.b64encode(i).decode() for i in images if i])
        self.last_raw_response = raw
        self.last_parse_status = status
        if not parsed:
            if status in ('vlm_auth_error', 'vlm_config_error'):
                raise RuntimeError(f'VLM fatal config error ({status}): {raw}')
            if status in ('vlm_timeout', 'vlm_http_error', 'vlm_empty_response'):
                return LiteAction(self.role, 'INFRA_ERROR', reason=f'vlm_infra_failed: {status}')
            return LiteAction(self.role, 'WAIT', reason=f'vlm_invalid_output: {status}')

        action = LiteAction(
            role=parsed.get('role', self.role), action=parsed.get('action', 'WAIT'),
            waypoint_id=parsed.get('waypoint_id'), grid_cell=parsed.get('grid_cell'),
            candidate_id=parsed.get('candidate_id'), confidence=parsed.get('confidence', 0.5),
            visual_description=parsed.get('visual_description', ''), evidence=parsed.get('evidence', ''),
            reason=parsed.get('reason', ''), message=parsed.get('message', ''),
        )
        return bind_nadir_annotation(obs, action) if self.role == 'UAV' else action


class MinimalAgentMethod:
    method_id = 'minimal_agent'

    def __init__(self, config, services=None, profile=None):
        self.uav = MinimalAgent('UAV', config)
        self.ugv = MinimalAgent('UGV', config)
        self.last_raw_response = ''
        self.last_parse_status = ''

    def get_mode_name(self):
        return self.method_id

    def role_capabilities(self, role):
        return {'uses_vlm': True}

    def decide(self, obs):
        agent = self.ugv if obs.role == 'UGV' else self.uav
        action = agent.decide(obs)
        self.last_raw_response = agent.last_raw_response
        self.last_parse_status = agent.last_parse_status
        return action

    def start_episode(self, episode_context):
        pass

    def on_transition(self, observation, proposed_action=None, executed_action=None, execution_result=None):
        pass

    def build_trace(self, obs):
        return {}

    def close_episode(self):
        pass
