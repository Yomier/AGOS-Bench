"""Physical deployment and five-point road contact; never exposed to an agent."""
import math

DEPLOYMENT_PROTOCOL = 'AGOS_GROUNDED_VEHICLES_V3'
LEGACY_PROTOCOL = 'AGOS_FROZEN_VEHICLES_V1'


class VehicleGroundingError(ValueError):
    """A geometry/settling rejection, distinct from transport/runtime failure."""


def check_ground_gaps(gaps):
    valid = len(gaps) == 5 and all(math.isfinite(g) for g in gaps)
    passed = valid and min(gaps) >= -0.05 and max(gaps) <= 0.10 and max(gaps) - min(gaps) <= 0.08
    return dict(passed=bool(passed), bottom_to_road_gaps_m=gaps,
                allowed_gap_m=[-0.05, 0.10], max_surface_variation_m=0.08)


def settle_vehicle(world, actor, initial):
    import carla
    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    heights = []
    # spawn_target supports heights through road_z+2m. The measured firetruck
    # trajectory needs106 ticks;100 cuts it off during suspension convergence.
    # Allow up to5 simulated seconds at the production60Hz, exiting as soon as
    # the original quality gates pass. A moving/oscillating car still rejects.
    for tick in range(300):
        world.tick()
        tf = actor.get_transform()
        velocity = actor.get_velocity()
        heights.append(tf.location.z)
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        if tick >= 59 and speed < 0.03 and max(heights[-10:]) - min(heights[-10:]) < 0.003:
            if (math.hypot(initial.x - tf.location.x, initial.y - tf.location.y) > 1.0
                    or abs(tf.rotation.pitch) > 3 or abs(tf.rotation.roll) > 3):
                raise VehicleGroundingError('vehicle drifted or tilted while settling')
            actor.set_simulate_physics(False)
            return dict(ticks=tick + 1, speed_m_s=speed,
                        height_variation_m=max(heights[-10:]) - min(heights[-10:]),
                        pitch_deg=tf.rotation.pitch, roll_deg=tf.rotation.roll)
    raise VehicleGroundingError(
        f'vehicle did not settle within 300 ticks: speed={speed!r}m/s, '
        f'height_variation={max(heights[-10:]) - min(heights[-10:])!r}m, '
        f'pitch={tf.rotation.pitch!r}, roll={tf.rotation.roll!r}')


def ground_at_pose(world, actor, pose):
    """Probe physical support while the caller keeps the actor lifted/frozen.

    cast_ray includes query-only RoutePlanner trigger volumes. project_point
    returns the nearest blocking surface, so triggers cannot hide the road and
    a solid non-road obstacle still rejects the deployment.
    """
    import carla
    bbox = actor.bounding_box
    gaps = []
    for dx, dy in [(0, 0), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]:
        point = pose.transform(carla.Location(x=bbox.location.x + dx * bbox.extent.x,
            y=bbox.location.y + dy * bbox.extent.y, z=bbox.location.z - bbox.extent.z))
        first = world.project_point(
            carla.Location(x=point.x, y=point.y, z=point.z + 0.6),
            carla.Vector3D(x=0, y=0, z=-1), 1.2)
        if first is None or not point.z - 0.6 <= first.location.z <= point.z + 0.6:
            return check_ground_gaps([])
        if str(first.label) not in ('Roads', 'RoadLines'):
            return check_ground_gaps([])
        gaps.append(point.z - first.location.z)
    return check_ground_gaps(gaps)


def deployment_ground_contact(world, actor):
    import carla
    pose = actor.get_transform()
    lifted = carla.Location(x=pose.location.x, y=pose.location.y,
                            z=pose.location.z + 300)
    try:
        # set_transform(get_transform()) needlessly converts Euler -> quaternion
        # -> Euler. It can change the strict Oracle snapshot even while frozen.
        actor.set_location(lifted)
        world.tick()
        return ground_at_pose(world, actor, pose)
    finally:
        actor.set_location(pose.location)
        world.tick()


def settle_deployed_vehicle(world, actor, initial):
    actor.set_simulate_physics(True)
    settling = settle_vehicle(world, actor, initial)
    pose = actor.get_transform()
    command = dict(x=pose.location.x, y=pose.location.y, z=pose.location.z,
                   yaw=pose.rotation.yaw, pitch=pose.rotation.pitch, roll=pose.rotation.roll)
    # Save the command before applying it once. QC/Oracle use its readback, while
    # replay applies this exact command, never the post-conversion Euler angles.
    actor.set_transform(pose)
    world.tick()
    ground = deployment_ground_contact(world, actor)
    if not ground['passed']:
        raise VehicleGroundingError('deployed vehicle fails actual road contact')
    return dict(protocol=DEPLOYMENT_PROTOCOL, settling=settling, ground=ground,
                deployment_command_pose=command)


def saved_deployment_pose(iface, actor):
    receipt = getattr(iface, '_vehicle_deployment_receipts', {}).get(actor.id, {})
    command = receipt.get('deployment_command_pose')
    if receipt.get('protocol') != DEPLOYMENT_PROTOCOL or not command:
        raise VehicleGroundingError('missing V3 deployment command pose')
    return dict(command)


def restore_certified_pose(world, actor, expected):
    """Apply the saved immutable command after independent settling agrees with it.

    Physical micrometre jitter must not invalidate the exact Oracle binding.
    Saved poses that lack road support remain inadmissible.
    """
    import carla
    actual = actor.get_transform()
    values = [float(expected[k]) for k in ('x', 'y', 'z', 'yaw')]
    values.extend(float(expected.get(k, 0)) for k in ('pitch', 'roll'))
    if not all(math.isfinite(v) for v in values):
        raise VehicleGroundingError('nonfinite certified deployment pose')
    xy = math.hypot(actual.location.x - values[0], actual.location.y - values[1])
    dz = abs(actual.location.z - values[2])
    if xy > 0.25 or dz > 0.25:
        raise VehicleGroundingError('independent settled pose disagrees with certified deployment')
    actor.set_transform(carla.Transform(carla.Location(x=values[0], y=values[1], z=values[2]),
        carla.Rotation(yaw=values[3], pitch=values[4], roll=values[5])))
    world.tick()
    contact = deployment_ground_contact(world, actor)
    if not contact['passed']:
        raise VehicleGroundingError('certified deployment lacks actual road contact')
    return dict(protocol=DEPLOYMENT_PROTOCOL, xy_difference_m=xy, z_difference_m=dz, ground=contact)
