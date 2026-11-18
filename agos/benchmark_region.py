"""Public task rectangle: hard UAV fence, soft UGV search boundary."""
import math
from collections.abc import Mapping

REGION_POLICY_REVISION = 'AGOS_REGION_UAV_HARD_UGV_SOFT_V1_20261013'

def region_bounds(region):
    if isinstance(region, Mapping):
        region = tuple(region[k] for k in ('x_min', 'x_max', 'y_min', 'y_max'))
    try:
        values = tuple(float(v) for v in region)
    except (TypeError, ValueError):
        raise ValueError('invalid task region') from None
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError('invalid task region')
    if not (values[0] < values[1] and values[2] < values[3]):
        raise ValueError('invalid task region extent')
    return values

def inside_task_region(xy, region):
    b = region_bounds(region)
    try:
        x, y = (float(v) for v in xy[:2])
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and math.isfinite(y) and b[0] <= x <= b[1] and b[2] <= y <= b[3]

def observation_region(obs):
    region = getattr(obs, 'region_boundary', None)
    # Dataclass defaults on legacy diagnostic observations mean "unspecified".
    # Runtime episode rectangles are always validated strictly by the catalog.
    if region is None or tuple(region) == (0, 0, 0, 0):
        return None
    return region_bounds(region)

def region_prompt_lines(obs):
    b = observation_region(obs)
    if b is None:
        return []
    inside = inside_task_region(obs.agent_position, b)
    lines = [f'- Task search region (world GPS metres): X=[{b[0]:.2f}, {b[1]:.2f}], Y=[{b[2]:.2f}, {b[3]:.2f}]; +X north, +Y east.',
             '- Your region status: ' + ('INSIDE' if inside else 'OUTSIDE') + '.',
             '- These bounds describe the search area, not the target location.']
    if obs.role == 'UAV':
        lines.append('- UAV rule: select only supplied legal waypoints inside or on this rectangle; never fly outside it.')
    else:
        lines.append('- UGV rule: use the full legal road network. A temporary detour outside this rectangle is allowed to bypass disconnected in-region roads; return to the search region along legal roads when feasible.')
        if not inside:
            lines.append('WARNING: OUTSIDE TASK REGION. Plan to return inside along legal roads. You may continue a necessary temporary detour; do not start unrelated outside-area exploration. This warning does not mask road actions or override your action.')
    return lines

def enforce_uav_move(action, catalog):
    """Executor guard also covers Oracle/debug modes that bypass VLM validation."""
    if action.action == 'MOVE_TO_UAV_WAYPOINT' and not catalog.is_legal_waypoint(action.waypoint_id):
        from agos.lite_core import LiteAction
        return LiteAction(role='UAV', action='WAIT', reason='invalid: UAV waypoint outside legal task region/menu')
    return action
