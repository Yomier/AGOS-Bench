
import math





MIN_SEPARATION_M = 100.0

REGION_MARGIN_M = 25.0

MAX_MASK_LOSS = 0.005  # rasterization tolerance, not permission for partial occlusion

MIN_MASK_PIXELS = 200

class ReferencePhotoError(RuntimeError):
    pass

def staging_location_allowed(location, deployed_location, region=None):
    if math.dist(location[:2], deployed_location[:2]) < MIN_SEPARATION_M:
        return False
    if region and (region['x_min'] - REGION_MARGIN_M <= location[0] <= region['x_max'] + REGION_MARGIN_M
                   and region['y_min'] - REGION_MARGIN_M <= location[1] <= region['y_max'] + REGION_MARGIN_M):
        return False
    return True

def check_ground_gaps(gaps):
    from .vehicle_grounding import check_ground_gaps as check
    return check(gaps)

TEAMMATE_PROTOCOL = 'AGOS_TEAMMATE_TOPDOWN_V1'

def actor_identity(actor):
    """Blueprint and visual attributes, excluding actor role/physics controls."""
    keys = ('color', 'driver_id', 'object_type')
    return dict(blueprint=actor.type_id,
                attributes={key: str(actor.attributes[key]) for key in keys if key in actor.attributes})

def assert_actor_identity(actor, identity):
    actual = actor_identity(actor)
    if not identity or actual != identity:
        raise ReferencePhotoError(f'UGV identity mismatch: expected={identity}, actual={actual}')
