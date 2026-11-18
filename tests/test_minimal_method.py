"""Offline Minimal-Agent decision parity at the sensor, HTTP and action boundaries."""
import base64
from dataclasses import replace
import hashlib
import io
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from agos.agents.agents import validate_action
from agos.lite_core import LiteObservation, UAVFrame, UGV_VIEW_REGISTRY
from agos.methods import registry
from agos.runtime import runner
from agos.runtime.observation_maps import ONLINE_MAP_CONTRACT, ObservationMapProtocol
from agos.runtime.paper_audit import decide_and_account, http_accounting_result


CONFIG = {
    'vlm': {'model': 'fixture', 'api_url': 'http://example.invalid/v1/chat/completions',
            'temperature': 0.0, 'max_output_tokens': 1024, 'top_p': 1.0, 'seed': 20260725,
            'max_retries_per_call': 1, 'retry_backoff_max': 0,
            'chat_template_kwargs': {'enable_thinking': False}},
    'uav': {'max_candidate_annotations': 999}, 'ugv': {'confirm_budget': 1},
    'benchmark': {'observation_contract': ONLINE_MAP_CONTRACT},
    'observation_map': {'online_map_enabled': True, 'variant_id': 'R32_ONLINE_MAP'},
}
PLAN = {'teammate_reference': {'protocol': 'AGOS_TEAMMATE_TOPDOWN_V1'}}


def png(color):
    buffer = io.BytesIO()
    Image.new('RGB', (16, 12), color).save(buffer, 'PNG')
    return buffer.getvalue()


def observation(role, status=None):
    peer = 'UAV' if role == 'UGV' else 'UGV'
    bus = runner.MessageBus()
    for i in range(8):
        bus.send(peer, f'note {i}', i, receiver=role, step=i)
    bus.send(peer, 'note 7', 8, receiver=role, step=8)
    bus.send(peer, 'future', 10, receiver=role, step=9)
    bus.send(role, 'my message', 1, receiver=peer, step=1)
    messages = bus.get_recent_for(role, 8)
    assert [m.content for m in messages] == [f'note {i}' for i in range(3, 8)]
    candidate = SimpleNamespace(candidate_id='C007', status=status, region_center_xy=(12., 8.), confidence=.8)
    obs = LiteObservation(
        'synthetic', 7, 80, role, agent_position=(3., 4., 80.), region_boundary=(0, 40, 0, 40),
        target_description='a blue compact car', target_reference_image=png('blue'),
        candidates=[candidate] if status else [], recent_messages=messages,
        active_candidate=candidate if status in ('unverified', 'assigned_to_ugv') else None,
        topdown_map=b'forbidden city image', confirm_budget=1,
    )
    if role == 'UAV':
        obs.uav_memory_text = ['visited U_01']
        obs.agent_image = png('yellow')
        obs.teammate_reference_image = png('white')
        obs.available_waypoints = [dict(waypoint_id='U_02', world_xy=(9., 11.), distance_m=9)]
        obs.available_actions = ['WAIT', 'MOVE_TO_UAV_WAYPOINT', 'NO_RELIABLE_CANDIDATE', 'ANNOTATE_CANDIDATE']
    else:
        obs.ugv_memory_text = ['previous move FORWARD_2']
        obs.available_waypoints = [dict(action='FORWARD_1', description='short option', distance_m=7., target_xy=(9., 8.))]
        obs.available_actions = ['FORWARD_1', 'STOP_OBSERVE', 'CONFIRM', 'REJECT']
        colors = ('red', 'green', 'blue', 'yellow', 'purple', 'orange')
        obs.ugv_views = tuple(replace(v, image_bytes=png(c)) for v, c in zip(UGV_VIEW_REGISTRY, colors))
    return obs


def bind_inputs(obs, tmp_path):
    protocol = ObservationMapProtocol(CONFIG, PLAN, 'minimal_agent', str(tmp_path))
    assert protocol.excludes_fullcity and protocol.online_map is None
    if obs.role == 'UAV':
        frame = UAVFrame(png('red'), 123, 1., 10., 15., 80., 0., -90., 90., 16, 12)
        camera = SimpleNamespace(capture_uav_frame=lambda: frame)
        obs.raw_agent_image = runner._capture_observation_uav(camera, protocol, 1.)
        assert runner._bind_uav_frame('minimal_agent', False)
        runner._attach_uav_observation_frame(obs, protocol.last_frame, obs.raw_agent_image)
    protocol.attach(obs, 1., 0., (0, 0), [])
    assert obs.topdown_map == b'' and obs.map_metadata['source'] == 'none'
    return obs


# SHA-256 of rendered system and user messages from the frozen method on the
# synthetic inputs above. Each candidate state changes the ground prompt.
SYSTEM_HASH = {
    'UAV': '3c32ada2914ae18904f1809588d44d7d0597fdbbf20677aaddf654e40ceec18f',
    'UGV': '8771c69456992b689e4087b8c9fd83dc3be0e3116b2e690f0f56641f14551af7',
}
USER_HASH = {
    ('UAV', None): 'cfc70e792e4db430c62e65c195fcd7eed7bd4c1b6fa86fa8cbade9aa4d6c4127',
    ('UAV', 'unverified'): '5b8eafeff07a2e3450bfa1a90979419bf44963394725644562bb0dfc1ce5c553',
    ('UGV', None): '042604c4684d5af440a05f4cca4028a375f95e28ce09008904c7261d413d4c94',
    ('UGV', 'unverified'): '47251db4c2d9c36d6fe9eff7d65a5b56b278bf5ae571a1cf193f614ddd27ce28',
    ('UGV', 'assigned_to_ugv'): '941907583ac1a4483eb6c3fe9c92d8d9895f192683c319ead0b4bbf0625e8ee3',
    ('UGV', 'rejected'): '89f4836a83e9edb6411bf580ea21c4d0b16784f2b1b506ef35ef91aa1e7573e5',
}


@pytest.mark.parametrize('role,status', USER_HASH)
def test_frozen_prompt_images_action_and_http_accounting(role, status, tmp_path, monkeypatch):
    assert registry.resolve_method_id('minimal_agent') == ('minimal_agent', None)
    method = registry.create('minimal_agent', CONFIG)
    obs = bind_inputs(observation(role, status), tmp_path)
    action_name = ('ANNOTATE_CANDIDATE' if not status else 'WAIT') if role == 'UAV' else ('CONFIRM' if status else 'FORWARD_1')
    reply = dict(role=role, action=action_name, message='copy', confidence=.9, evidence='visible')
    if action_name == 'ANNOTATE_CANDIDATE':
        reply.update(position_xy=[999, 999], grid_cell='D2')
    if role == 'UGV' and obs.active_candidate:
        reply['candidate_id'] = 'C007'
    sent = []

    def post(url, *, json, **kwargs):
        sent.append(json)
        if len(sent) == 1:
            return SimpleNamespace(status_code=503)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {'choices': [{'message': {'content': raw}}]})

    raw = json.dumps(reply)
    monkeypatch.setattr('agos.agents.agents.requests.post', post)
    state = SimpleNamespace()
    assert runner._role_uses_vlm('minimal_agent', method, role)
    action = decide_and_account(lambda: runner._method_decide('minimal_agent', method, obs), state, method, role)
    assert http_accounting_result(state)['vlm_http_attempts'] == 2
    assert getattr(state, role.lower() + '_vlm_http_attempts') == 2
    assert len(sent) == 2 and sent[0] == sent[1]
    assert method.last_parse_status == 'ok' and method.last_raw_response == raw
    assert action.role == role and action.action == action_name and action.message == 'copy'
    assert validate_action(vars(action), obs, {'U_02', 'FORWARD_1'}) == (True, 'valid')
    payload = sent[-1]
    assert {k: v for k, v in payload.items() if k != 'messages'} == {
        'model': 'fixture', 'temperature': 0.0, 'max_tokens': 1024, 'top_p': 1.,
        'seed': 20260725, 'chat_template_kwargs': {'enable_thinking': False},
    }
    system, user = payload['messages']
    assert system['role'] == 'system' and user['role'] == 'user'
    components = user['content']
    user_text = components[-1]['text']
    assert hashlib.sha256(system['content'].encode()).hexdigest() == SYSTEM_HASH[role]
    assert hashlib.sha256(user_text.encode()).hexdigest() == USER_HASH[role, status]
    assert 'no topdown_map image is provided' in system['content'] and 'without map polygons' in user_text
    images = [base64.b64decode(part['image_url']['url'].split(',', 1)[1]) for part in components[:-1]]
    if role == 'UAV':
        assert images == [obs.raw_agent_image, obs.target_reference_image, obs.teammate_reference_image]
        assert 'not a starting-location photo or current heading/position' in user_text
        if action_name == 'ANNOTATE_CANDIDATE':
            assert action.position_xy == (10., 15.) and action.grid_cell is None
            assert action.source_frame_id == 123 and action.localization_source == 'uav_nadir_point'
            assert action.source_image_sha256 == hashlib.sha256(obs.raw_agent_image).hexdigest()
            lifecycle = SimpleNamespace(step=obs.step)
            runner._note_nadir_lifecycle(lifecycle, action)
            assert lifecycle.uav_frozen and lifecycle.uav_frozen_since_step == obs.step
    else:
        assert len(images) == 2 and images[1] == obs.target_reference_image
        collage = Image.open(io.BytesIO(images[0]))
        assert collage.size == (1440, 540)
        for i, view in enumerate(obs.ugv_views):
            assert collage.getpixel((480*(i%3)+240, 270*(i//3)+135)) == Image.open(io.BytesIO(view.image_bytes)).getpixel((8, 6))


@pytest.mark.parametrize('role', ['UAV', 'UGV'])
@pytest.mark.parametrize('status,expected', [('json_parse_failed', 'WAIT'), ('vlm_timeout', 'INFRA_ERROR'),
                                            ('vlm_auth_error', 'fatal'), ('vlm_config_error', 'fatal')])
def test_invalid_model_output_and_transport_failures_stay_distinct(role, status, expected, monkeypatch, tmp_path):
    method = registry.create('minimal_agent', CONFIG)
    obs = bind_inputs(observation(role), tmp_path)
    client = method.uav.vlm if role == 'UAV' else method.ugv.vlm
    monkeypatch.setattr(client, 'decide', lambda *args: (None, status, 'example response'))
    if expected == 'fatal':
        with pytest.raises(RuntimeError, match=status):
            runner._method_decide('minimal_agent', method, obs)
        return
    action = runner._method_decide('minimal_agent', method, obs)
    assert action.action == expected and status in action.reason and method.last_parse_status == status


def test_runner_records_public_identity_and_provenance():
    session = runner.JointLiteRunner(CONFIG, agent_mode='minimal_agent')
    assert runner._agent_name_for_mode(session.agent_mode) == 'MinimalAgent'
    assert not runner._collect_method_trace(session.agent_mode)
    meta = runner._build_run_meta(CONFIG, session.agent_mode)
    assert meta['agent_mode'] == 'minimal_agent'
    assert meta['code_files_md5']['agos/methods/minimal_agent/method.py'] != 'MISSING'


def test_joint_runner_publishes_sensor_point_and_scores_ground_decision(tmp_path, monkeypatch):
    """Run two scheduled decisions with synthetic sensors, road assets and HTTP replies."""
    config = dict(CONFIG, eval={'max_steps': 2, 'max_vlm_calls': 2},
                  uav=dict(CONFIG['uav'], waypoint_spacing_m=40),
                  ugv=dict(CONFIG['ugv'], action_space='sparse_nav_v1', intersection_max_distance_m=50,
                           stop_observe_yaws=[-45, 0, 45, -135, 180, 135]),
                  candidate={'duplicate_suppression_radius_m': 30, 'max_candidates_total': None},
                  success={'require_stop_observe': False, 'require_gt_visible_in_stop_6view': True,
                           'require_los': True, 'max_confirm_distance_m': 30.})
    ugv_start = dict(x=0., y=0., z=.5, yaw=0.)
    plan = dict(PLAN, episode_id='synthetic', map_name='Synthetic',
                region=dict(x_min=-50., x_max=50., y_min=-50., y_max=50.),
                start={'uav': dict(x=8., y=1., z=80.), 'ugv': ugv_start},
                uav_grid_waypoints=[], ugv_road_waypoints=[],
                target_oracle={'transform': {'x': 17., 'y': 2., 'z': .5}},
                target_task={'target_description': 'blue compact car', 'target_reference_image': 'target.png'},
                static_solvability_precheck={'certified_route_length_m': 17.})
    sensor = UAVFrame(png('red'), 432, 0., 10., 1., 80., 0., -90., 90., 16, 12)
    views = observation('UGV').ugv_views
    target_xyz = SimpleNamespace(**plan['target_oracle']['transform'])
    target = SimpleNamespace(is_alive=True, get_location=lambda: target_xyz, destroy=lambda: None)
    ground_xyz = SimpleNamespace(**ugv_start)
    iface = SimpleNamespace(
        set_uav_pose=lambda *args: None, set_ugv_transform=lambda *args, **kwargs: SimpleNamespace(
            location=ground_xyz, rotation=SimpleNamespace(yaw=0.)),
        _ugv_actor=SimpleNamespace(is_alive=True, get_location=lambda: ground_xyz), _cam_actor=None,
        spawn_target_from_plan=lambda _: target, spawn_distractors_from_plan=lambda _: [],
        capture_uav_frame=lambda: sensor, capture_ugv_views=lambda **kwargs: views, get_world=lambda: None,
    )
    target_image, identity = png('blue'), png('white')
    monkeypatch.setattr('agos.episode_io.load_reference_assets', lambda *a, **kw: (target_image, {}))
    monkeypatch.setattr('agos.episode_io.load_teammate_reference', lambda *a, **kw: identity)
    store = SimpleNamespace(nodes={'p': ground_xyz}, edges={}, manifest=SimpleNamespace(graph_id='g', asset_sha256='g'))
    locator = SimpleNamespace(locate=lambda *a, **kw: SimpleNamespace(fine_node_id='p', is_valid=True))
    sparse = SimpleNamespace(release_id='synthetic', get_maneuver_options=lambda *a, **kw: {})
    monkeypatch.setattr(runner, '_load_roadnet_for_episode', lambda *a: (store, locator, None, None, None, None))
    monkeypatch.setattr(runner, '_load_sparse_nav_adapter', lambda *a: (sparse, 'synthetic'))
    monkeypatch.setattr(runner, '_validate_sparse_start_contract', lambda *a, **kw: 'p')
    # The synthetic one-port world has no deployment/SPL certificate.
    monkeypatch.setattr('agos.eval.navigation_oracle.oracle_required', lambda *a: False)
    monkeypatch.setattr(runner, 'probe_uav_observation', lambda *a: {'valid': True, 'visible': True})
    monkeypatch.setattr('agos.runtime.evaluation_observations.check_visibility_6view',
                        lambda *a, **kw: (True, ['V1'], {}, True))
    payloads = []

    def post(url, *, json, **kw):
        payloads.append(json['messages'])
        action = ('{"role":"UAV","action":"ANNOTATE_CANDIDATE","message":"check the road"}'
                  if len(payloads) == 1 else '{"role":"UGV","action":"CONFIRM"}')
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {'choices': [{'message': {'content': action}}]})

    monkeypatch.setattr('agos.agents.agents.requests.post', post)
    result = runner.JointLiteRunner(config, 'minimal_agent').run_episode(iface, plan, str(tmp_path / 'episode'))
    assert result['completed'] and result['success'] and result['mode'] == 'joint_minimal_agent'
    assert result['steps'] == result['vlm_calls'] == result['vlm_http_attempts'] == 2
    for role in ('uav', 'ugv'):
        assert result[role + '_vlm_calls'] == result[role + '_vlm_http_attempts'] == 1
    candidate, = result['candidates']
    assert candidate['region_center_xy'] == [10., 1.] and candidate['source_frame_id'] == sensor.frame_id
    assert candidate['source_image_sha256'] == hashlib.sha256(sensor.image_bytes).hexdigest()
    assert result['candidate_created_count'] == 1 and result['invalid_action_count'] == 0
    assert result['gva_tp'] == 1 and result['confirm_count'] == 1
    assert [len(messages[1]['content']) - 1 for messages in payloads] == [3, 2]
    assert 'check the road' in payloads[1][1]['content'][-1]['text']
    assert f"candidate {candidate['candidate_id']}" in payloads[1][1]['content'][-1]['text']
