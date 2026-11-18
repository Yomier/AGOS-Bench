"""Portable corpus contracts with a generated image and two synthetic expert calls."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from training.corpus import build_corpus
from training.freeze import freeze


def capture(tmp_path):
    episode = tmp_path / 'run/episodes/train1100/Town01/example'
    attempt = episode / 'attempt-0001'
    attempt.mkdir(parents=True)
    frame = tmp_path / 'frame.png'
    Image.new('RGB', (8, 8), (20, 40, 60)).save(frame)
    digest = hashlib.sha256(frame.read_bytes()).hexdigest()
    result = attempt / 'example_result.json'
    result.write_text(json.dumps({'episode_id': 'example', 'success': True, 'failure_reason': 'SUCCESS'}))
    terminal = {'outcome': 'success', 'attempt': attempt.name, 'result': result.name,
                'result_sha256': hashlib.sha256(result.read_bytes()).hexdigest(), 'vision_errors': 0}
    (episode / 'terminal.json').write_text(json.dumps(terminal))
    from agos.lite_core import LiteObservation
    from agos.methods.cooperative_search.policy import DecisionPolicy
    from training.expert import EventExpert

    policy = DecisionPolicy({'vlm': {}, 'cooperative_search': {'verify_system_v2': True}},
                            capture_root=str(tmp_path / 'captures'))
    expert = EventExpert({'confirm_capture': True})
    uav = LiteObservation('example', 1, 80, 'UAV')
    uav.target_reference_image = uav.agent_image = frame.read_bytes()
    policy.record_expert_verify(uav, [], {'selected_proposal': 'NONE', 'target_verdict': 'UNKNOWN'}, 'verify')
    ugv = LiteObservation('example', 2, 80, 'UGV')
    ugv.target_reference_image = ugv.agent_image = frame.read_bytes()
    verdict = expert.confirm_verdict({'active_candidate': {'region_center_xy': [10, 0]},
                                      'ugv': {'position': [0, 0], 'views': [{'yaw_offset_deg': 0}]}},
                                     {'target_xy': [10, 0]})
    policy.record_expert_confirm(ugv, verdict, 'verify')
    with (attempt / 'ag_os_trace.jsonl').open('w', encoding='utf-8') as stream:
        for call in policy.drain_events():
            assert call['image_sha256'][0] == digest
            stream.write(json.dumps({'step': call['step'], 'role': call['role'], 'model_calls': [call]}) + '\n')
    return tmp_path / 'run'


def test_freeze_and_partition_preserve_input_order_and_images(tmp_path):
    run = capture(tmp_path)
    frozen, corpus = tmp_path / 'frozen', tmp_path / 'corpus'
    assert freeze(run, frozen)['rows'] == 2
    report = build_corpus(frozen, corpus)
    assert report['rows'] == {'mixed': 2, 'uav': 1, 'ugv': 1}
    assert report['unique_images'] == 1
    rows = (frozen / 'rows.jsonl').read_bytes().splitlines(keepends=True)
    assert (corpus / 'mixed.jsonl').read_bytes() == b''.join(rows)
    assert (corpus / 'uav.jsonl').read_bytes() == rows[0]
    assert (corpus / 'ugv.jsonl').read_bytes() == rows[1]
    ref = json.loads(rows[1])['images'][0]
    assert hashlib.sha256((corpus / ref).read_bytes()).hexdigest() == Path(ref).stem


def test_corrupt_image_does_not_publish_partial_corpus(tmp_path):
    frozen, corpus = tmp_path / 'frozen', tmp_path / 'corpus'
    freeze(capture(tmp_path), frozen)
    next((frozen / 'images').iterdir()).write_bytes(b'corrupt bytes')
    with pytest.raises(ValueError, match='SHA256'):
        build_corpus(frozen, corpus)
    assert not corpus.exists()


def test_changed_terminal_result_is_refused(tmp_path):
    run = capture(tmp_path)
    next(run.rglob('*_result.json')).write_text('{}')
    with pytest.raises(ValueError, match='locked result changed'):
        freeze(run, tmp_path / 'frozen')
    assert not (tmp_path / 'frozen').exists()
