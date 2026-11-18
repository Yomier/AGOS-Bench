"""GSAM returns source-bound RGB crops, including detections at image borders."""
import base64
from contextlib import nullcontext
import hashlib
import io
import threading
from types import SimpleNamespace

from PIL import Image
import pytest

from agos.methods.common.grounded_localizer import GroundedLocalizer
from agos.methods.common.source_guard import verify_proposal_source
from carlaair.perception.gsam import GsamEngine


class Feature:
    def __mul__(self, other):
        return self

    def sum(self):
        return SimpleNamespace(item=lambda: 0.6)


class DetectorProcessor:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, **kwargs):
        return {}

    def post_process_grounded_object_detection(self, *args, **kwargs):
        return [{'boxes': [SimpleNamespace(tolist=lambda: self.bbox)],
                 'scores': [SimpleNamespace(item=lambda: 0.4)], 'labels': ['vehicle']}]


def engine(bbox):
    server = GsamEngine.__new__(GsamEngine)
    server._Image = Image
    server._torch = SimpleNamespace(no_grad=nullcontext)
    server._lock = threading.Lock()
    server.device = 'cpu'
    server.det_proc = DetectorProcessor(bbox)
    server.det = lambda **kwargs: None
    server._ref_feat = lambda raw: Feature()
    server._feat = lambda image: Feature()
    server.sam_model_name = server.sam_loaded_path = 'test-sam'
    server.detector_model = server.encoder_model = 'fixture'
    return server


def ground(bbox):
    image = Image.new('RGB', (480, 270), (80, 100, 150))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    raw = buffer.getvalue()
    response = engine(bbox).ground(image_bytes=raw, ref_bytes=raw, description='vehicle',
                                   box_threshold=0.2, text_threshold=0.18, nms_iou=0.5,
                                   topk=3, segment=False)
    return raw, response


@pytest.mark.parametrize('bbox', [
    [178.53366088867188, 220.44712829589844, 481.34295654296875, 268.9267578125],
    [-1.4, 20.0, 50.0, 90.0], [10.0, -2.0, 80.0, 271.5],
])
def test_real_border_box_metadata_matches_clipped_rgb_and_client(bbox):
    raw, response = ground(bbox)
    assert response['status'] == 'ok'
    frame = SimpleNamespace(image_bytes=raw, width=480, height=270, valid=True, frame_id=1)
    client = GroundedLocalizer('fixture', post=lambda *args: response)
    result = client.ground_frame(frame, 'vehicle', raw, segment=False)
    assert result.status == 'ok', result.error
    proposal = result.proposals[0]
    x1, y1, x2, y2 = proposal.bbox_xyxy
    assert 0 <= x1 < x2 <= 480
    assert 0 <= y1 < y2 <= 270
    assert verify_proposal_source(raw, proposal)[0]


def test_detector_box_wholly_outside_image_is_empty():
    _, result = ground([-80.0, 20.0, -10.0, 90.0])
    assert result['status'] == 'empty' and result['proposals'] == []


@pytest.mark.parametrize('kind', ['invalid', 'bad_b64', 'bad_image'])
def test_sixview_error_panel_hash_is_actual_bytes_or_empty(kind):
    raw = b'not-an-image' if kind == 'bad_image' else b'valid-bytes'
    encoded = '%%%' if kind == 'bad_b64' else base64.b64encode(raw).decode()
    result = engine([1, 2, 3, 4]).ground_ugv_six(
        bundle_id='B1', views=[dict(view_id='V0', image_b64=encoded, valid=kind != 'invalid')],
        ref_bytes=b'ref', detection_query='vehicle', box_threshold=0.2, text_threshold=0.18,
        nms_iou=0.5, panel_limit=8, display_limit=3)
    panel = result['panels'][0]
    assert panel['source_sha256'] == ('' if kind == 'bad_b64' else hashlib.sha256(raw).hexdigest())
    assert panel['status'] == 'invalid_source'
