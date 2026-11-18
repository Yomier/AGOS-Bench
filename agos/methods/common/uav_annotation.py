"""Current no-tool annotation: report the observing UAV's GPS nadir point.

No target coordinates or model-supplied localization are consumed here.
"""
from dataclasses import replace
import hashlib
import math
from numbers import Real

UAV_ANNOTATION_REVISION = 'AGOS_UAV_POINT_V2_20261013'
RETIRED_UAV_GRID_OPTIONS = (
    'uav_annotation_repair_enabled', 'uav_roi_verifier_enabled',
    'r18_q_consistency_repair_enabled', 'r18_uav_grounder_enabled',
    'r18_uav_auto_annotate_enabled', 'r18_uav_precise_geometry_enabled',
    'r19_uav_geometry_refiner_enabled', 'candidate_ranker_enabled',
    'candidate_ranker_dispatch_enabled', 'r20_candidate_revision_enabled',
)


def finite_xy(value):
    return (isinstance(value, (tuple, list)) and len(value) == 2
            and all(isinstance(v, Real) and not isinstance(v, bool)
                    and math.isfinite(float(v)) for v in value))


def prepare_clean_uav(obs):
    if getattr(obs, 'role', '') == 'UAV':
        raw = getattr(obs, 'raw_agent_image', b'')
        frame = getattr(obs, 'uav_frame', None)
        if raw:
            obs.agent_image = raw
        elif frame is not None:
            obs.agent_image = frame.image_bytes
    return obs


def bind_nadir_annotation(obs, action):
    """Only a UAV annotation decision is localized; other/invalid roles pass through."""
    if (getattr(obs, 'role', '') != 'UAV' or action.role != 'UAV'
            or action.action != 'ANNOTATE_CANDIDATE'):
        return action
    pose = getattr(obs, 'agent_position', ())
    xy = list(pose[:2]) if isinstance(pose, (tuple, list)) else []
    if not finite_xy(xy):
        return replace(action, action='WAIT', grid_cell=None, geometry_type=None,
                       position_xy=None, reason='invalid UAV GPS; annotation withheld')
    frame = getattr(obs, 'uav_frame', None)
    image = (getattr(frame, 'image_bytes', b'') or
             getattr(obs, 'raw_agent_image', b'') or getattr(obs, 'agent_image', b''))
    return replace(action, waypoint_id=None, grid_cell=None, geometry_type='point',
                   position_xy=(float(xy[0]), float(xy[1])),
                   source_frame_id=getattr(frame, 'frame_id', None),
                   source_image_sha256=hashlib.sha256(image).hexdigest() if image else '',
                   localization_source='uav_nadir_point')
