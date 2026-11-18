# -*- coding: utf-8 -*-
"""episode_io.py — 统一的 episode plan / asset 加载与验证（P0-08 / P0-02）

所有入口（generator 写后自检、Oracle Gate、run_single_oracle、run_lite_joint、
parallel episode_job、报告重放工具）必须共用这一套验证，杜绝"并行入口拒绝、
单集入口放行"的几何/数据混用。

Fail-closed 原则（P0-02）：任一正式校验失败即抛 EpisodeValidationError，
不允许静默退回 fallback。
"""
import os
import json
import hashlib
import logging
import math

logger = logging.getLogger("LiteV1.EpisodeIO")


class EpisodeValidationError(RuntimeError):
    """正式 plan/asset 验证失败（fail-closed）。"""


# Camera spec 必须与生成器冻结的 spec 一致。
EXPECTED_CAMERA_SPEC = {
    "height_m": 80.0, "yaw_deg": 0.0, "pitch_deg": -90.0,
    "fov_deg": 90.0, "img_w": 1024, "img_h": 576, "rows": 5, "cols": 5,
}

MIN_SCHEMA_VERSION = 2


def _sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def expected_camera_spec_hash() -> str:
    return _sha256_json(EXPECTED_CAMERA_SPEC)


def current_calibration():
    """加载当前环境的 UAV 标定（返回 dict 或 None）。"""
    try:
        from agos.perception import uav_grid_geometry as _ug
        calib = getattr(_ug, "_calibration", None) or _ug.load_calibration()
        return calib
    except Exception as e:
        logger.warning("uav_grid_geometry unavailable or calibration load failed: %s", e)
        return None


def current_calibration_hash():
    calib = current_calibration()
    return _sha256_json(calib) if calib else None


def validate_geometry_contract(plan, plan_path="<plan>", formal=True):
    """P0-5/P0-02：几何契约 fail-closed 验证。

    正式模式要求：
      1. schema_version >= MIN_SCHEMA_VERSION
      2. geometry_backend == calibration
      3. plan 的 calibration_sha256 非空
      4. 当前机器标定存在
      5. 当前标定 hash 与 plan 完全一致
      6. camera spec hash 完全一致
    任一失败即抛错（fail-closed）。
    """
    schema = plan.get("schema_version")
    contract = plan.get("geometry_contract") or {}
    backend = contract.get("geometry_backend", "")

    # 1+2. schema + backend
    if schema is None or int(schema) < MIN_SCHEMA_VERSION \
            or backend != "uav_grid_geometry.calibration":
        raise EpisodeValidationError(
            f"geometry contract violation for {plan_path}: "
            f"schema_version={schema!r}, geometry_backend={backend!r}. "
            f"Legacy (pre-P0-1-fix) or fallback-generated plan; regenerate with "
            f"the calibrated generator.")

    # 6. camera spec hash
    exp_cam = expected_camera_spec_hash()
    if contract.get("camera_spec_sha256") != exp_cam:
        raise EpisodeValidationError(
            f"camera spec mismatch for {plan_path}: "
            f"plan={contract.get('camera_spec_sha256')}, expected={exp_cam}. "
            f"Camera geometry changed; regenerate.")

    # 3. plan must carry a non-empty calibration hash
    plan_cal_hash = contract.get("calibration_sha256")
    if not plan_cal_hash:
        raise EpisodeValidationError(
            f"missing calibration_sha256 in {plan_path}: formal plans must pin "
            f"the calibration hash.")

    # 4+5. current machine must have calibration and it must match (fail-closed)
    cur_hash = current_calibration_hash()
    if cur_hash is None:
        raise EpisodeValidationError(
            f"no calibration available on this machine for {plan_path}: plan "
            f"requires calibration hash {plan_cal_hash[:12]}..., but current "
            f"environment has none. Refusing to fall back to legacy geometry.")
    if plan_cal_hash != cur_hash:
        raise EpisodeValidationError(
            f"calibration mismatch for {plan_path}: plan={plan_cal_hash[:12]}..., "
            f"current={cur_hash[:12]}.... Calibration changed; regenerate.")


def validate_reference_certificate(plan, plan_dir, formal=True):
    """Fail-closed reference validation: legacy three-view v2 or explicit random-single v3.

    r9.5：证书 v2 要求 target_reference.views[]（三张独立 PNG 的
    kind/relpath/sha256）——独立参考图的来源，缺失即拒绝。
    """
    tr = plan.get("target_reference") or {}
    if tr.get('certificate_version') not in (None, 1, 2, 3):
        raise EpisodeValidationError('unsupported reference certificate version')
    single = tr.get('certificate_version') == 3
    if single and tr.get('mode') != 'random_single':
        raise EpisodeValidationError('reference v3 requires random_single mode')
    if tr.get('mode') == 'random_single' and not single:
        raise EpisodeValidationError('single reference requires certificate_version=3')
    expected_sha = tr.get("png_sha256")
    rel = tr.get("png_relpath") or plan.get("target_task", {}).get("target_reference_image", "")
    if not expected_sha or not rel:
        raise EpisodeValidationError(
            "target_reference certificate incomplete: missing png_sha256 or path.")
    png_path = os.path.join(plan_dir, rel)
    if not os.path.exists(png_path):
        raise EpisodeValidationError(f"reference PNG not found: {png_path}")
    with open(png_path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != expected_sha:
        raise EpisodeValidationError(
            f"reference PNG hash mismatch for {png_path}: plan={expected_sha[:12]}..., "
            f"actual={actual[:12]}.... PNG may be stale/corrupted.")
    qg = tr.get("quality_gate") or {}
    panels = tr.get("panels") or []
    if single and (qg.get('required_panels') != 1 or len(panels) != 1):
        raise EpisodeValidationError('random_single reference requires exactly one panel')
    if qg.get("required_panels", 3) != len(panels):
        raise EpisodeValidationError(
            f"reference panel count mismatch: required={qg.get('required_panels')}, "
            f"got={len(panels)}.")
    for p in panels:
        if "transform" not in p or "bbox_2d" not in p:
            raise EpisodeValidationError(
                f"reference panel '{p.get('kind')}' missing transform/bbox certificate.")
    # r9.5：独立参考 views 证书（独立参考图，fail-closed）
    views = tr.get("views") or []
    if formal:
        got_kinds = {v.get("kind") for v in views}
        expected_kinds = {'random'} if single else {'top', 'front', 'side'}
        if got_kinds != expected_kinds or len(views) != len(expected_kinds):
            raise EpisodeValidationError(
                f"reference views certificate missing/incomplete: got kinds "
                f"{sorted(str(k) for k in got_kinds)}, expected "
                f"{'one random view' if single else 'top/front/side'}. "
                f"Check the dataset release and reference-image certificates.")
        for v in views:
            if not v.get("sha256") or not v.get("relpath"):
                raise EpisodeValidationError(
                    f"reference view '{v.get('kind')}' certificate incomplete.")
            vp = os.path.join(plan_dir, v["relpath"])
            if not os.path.exists(vp):
                raise EpisodeValidationError(f"reference view PNG not found: {vp}")
            with open(vp, "rb") as f:
                actual_v = hashlib.sha256(f.read()).hexdigest()
            if actual_v != v["sha256"]:
                raise EpisodeValidationError(
                    f"reference view '{v.get('kind')}' hash mismatch for {vp}: "
                    f"plan={v['sha256'][:12]}..., actual={actual_v[:12]}....")
        if single:
            view, panel = views[0], panels[0]
            if (view['relpath'] != rel or view['sha256'] != expected_sha or panel.get('kind') != 'random'
                    or panel.get('los_clear') is not True or panel.get('los_waived_by_strong_bbox') is not False):
                raise EpisodeValidationError('single reference asset/visibility certificate mismatch')
            sampling = tr.get('sampling') or {}
            if sampling.get('distribution') != 'uniform_surface_band':
                raise EpisodeValidationError('single reference sampling certificate missing')
            try:
                distance = float(sampling['distance_m'])
                low, high = float(sampling['elevation_min_deg']), float(sampling['elevation_max_deg'])
                elevation = float(panel['elevation_deg'])
                azimuth = float(panel['azimuth_deg'])
                actual_distance = float(panel['distance_m'])
                center = [float(x) for x in panel['target_center']]
                location = [float(x) for x in panel['transform']['location']]
                values = [distance, low, high, elevation, azimuth, actual_distance, *center, *location]
                if (not all(math.isfinite(x) for x in values) or distance <= 0
                        or not 0 <= low < high <= 90 or not low <= elevation <= high
                        or not 0 <= azimuth < 360 or len(center) != 3 or len(location) != 3
                        or not math.isclose(actual_distance, distance, abs_tol=1e-6)
                        or not math.isclose(math.dist(center, location), distance, abs_tol=0.005)):
                    raise ValueError('sampling geometry mismatch')
                if sampling.get('azimuth_range_deg') != [0.0, 360.0]:
                    raise ValueError('azimuth support mismatch')
                delta = [location[i] - center[i] for i in range(3)]
                actual_elevation = math.degrees(math.atan2(delta[2], math.hypot(*delta[:2])))
                actual_azimuth = math.degrees(math.atan2(delta[1], delta[0])) % 360
                angle_error = lambda a, b: abs((a - b + 180) % 360 - 180)
                tf = panel['transform']
                yaw, pitch, roll, fov = [float(tf[k]) for k in ('yaw', 'pitch', 'roll', 'fov')]
                if (not all(math.isfinite(x) for x in (yaw, pitch, roll, fov))
                        or abs(actual_elevation - elevation) > 0.005
                        or angle_error(actual_azimuth, azimuth) > 0.005
                        or angle_error(yaw, actual_azimuth + 180) > 0.005
                        or abs(pitch + actual_elevation) > 0.005 or abs(roll) > 0.005
                        or fov != qg['fov'] or fov != 90.0):
                    raise ValueError('camera position/aim contradicts sampling')
                from PIL import Image
                with Image.open(png_path) as image:
                    image_size = list(image.size)
                    image.verify()
                output = panel['output_size']
                crop = panel['crop_offset']
                target_px = panel['target_center_px']
                source_size = qg['img_size']
                if (source_size != [480, 270] or output != image_size or len(crop) != 2 or len(target_px) != 2
                        or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in crop)
                        or any(crop[i] + output[i] / 2 != source_size[i] / 2 for i in range(2))
                        or target_px != [output[0] / 2, output[1] / 2]):
                    raise ValueError('reference crop does not center the target')
                bounds = panel['bbox_2d']
                if (len(bounds) != 4 or not all(math.isfinite(float(x)) for x in bounds)
                        or not crop[0] <= bounds[0] < bounds[2] <= crop[0] + output[0]
                        or not crop[1] <= bounds[1] < bounds[3] <= crop[1] + output[1]):
                    raise ValueError('reference bbox clipped by crop')
                from agos.episode.reference_photo import (staging_location_allowed, check_ground_gaps,
                    MIN_SEPARATION_M, REGION_MARGIN_M, MAX_MASK_LOSS, MIN_MASK_PIXELS)
                staging = panel['staging']
                photo, deployed = staging['photo_location'], staging['deployment_location']
                if (len(photo) != 3 or len(deployed) != 3 or not all(math.isfinite(float(x)) for x in photo + deployed)
                        or staging['strategy'] != 'separate_appearance_matched_clone'
                        or staging['min_separation_m'] != MIN_SEPARATION_M
                        or staging['region_margin_m'] != REGION_MARGIN_M
                        or staging['background_policy'] != 'neutral_rgb_220'
                        or not staging_location_allowed(photo, deployed, staging['deployment_region'])
                        or not math.isclose(staging['separation_m'], math.dist(photo[:2], deployed[:2]), abs_tol=0.005)):
                    raise ValueError('photo location is not isolated from deployment')
                photo_center = [float(x) for x in staging['photo_target_center']]
                if (len(photo_center) != 3 or not all(math.isfinite(x) for x in photo_center)
                        or math.dist(photo_center, center) > 0.005 or math.dist(photo_center, photo) > 6.0):
                    raise ValueError('camera target contradicts photo location')
                if plan.get('target_oracle'):
                    deployed_tf = plan['target_oracle']['transform']
                    if (math.dist(deployed[:2], [deployed_tf['x'], deployed_tf['y']]) > 0.01
                            or staging['deployment_region'] != plan.get('region')):
                        raise ValueError('photo provenance contradicts deployed task')
                ground = staging['ground']
                if ground.get('passed') is not True or not check_ground_gaps(ground['bottom_to_road_gaps_m'])['passed']:
                    raise ValueError('photo vehicle is not grounded')
                settled = staging['settling']
                if (not 60 <= settled['ticks'] <= 100 or not 0 <= settled['speed_m_s'] < 0.03
                        or not 0 <= settled['height_variation_m'] < 0.003
                        or abs(settled['pitch_deg']) > 3 or abs(settled['roll_deg']) > 3):
                    raise ValueError('photo vehicle did not settle')
                quality = panel['render_quality']
                n, overlap, extra = (quality[k] for k in ('ideal_pixels', 'overlap_pixels', 'extra_pixels'))
                missing, outside = (quality[k] for k in ('interior_missing_pixels', 'interior_extra_pixels'))
                if (quality.get('passed') is not True or quality['method'] != 'paired_instance_silhouette'
                        or quality['same_frame_rgb_and_mask'] is not True
                        or quality['ideal_translation_m'] != 300.0
                        or quality['boundary_tolerance_px'] != 1
                        or not all(isinstance(x, int) and not isinstance(x, bool) for x in (n, overlap, extra, missing, outside))
                        or n < MIN_MASK_PIXELS or not 0 <= overlap <= n or extra < 0
                        or not 0 <= missing <= n - overlap or not 0 <= outside <= extra
                        or quality['visible_pixels'] != overlap + extra
                        or (n - missing) / n < 1 - MAX_MASK_LOSS or outside / n > MAX_MASK_LOSS
                        or not 1 - MAX_MASK_LOSS <= quality['visible_pixels'] / n <= 1 + MAX_MASK_LOSS
                        or quality['min_visible_fraction'] != 1 - MAX_MASK_LOSS
                        or not math.isclose(quality['visible_fraction'], (n - missing) / n, abs_tol=1e-9)
                        or not math.isclose(quality['raw_visible_fraction'], quality['visible_pixels'] / n, abs_tol=1e-9)):
                    raise ValueError('reference render is occluded or lacks a quality certificate')
            except (KeyError, TypeError, ValueError, OverflowError) as e:
                raise EpisodeValidationError(f'single reference sampling certificate invalid: {e}') from e


def load_reference_assets(episode_plan, extra_search_dirs=()):
    """加载参考资产（r9.5 统一入口，runner 各处共用）。

    Returns:
        (collage_bytes, views_bytes)：collage 供 legacy VLM 方法；
        views_bytes = {top/front/side: png_bytes} 供方法使用（当前格式中
        唯一参考来源）。views 缺文件时该 kind 跳过并保持 dict 完整性
        （证书校验已在 load_and_validate_episode_plan fail-closed）。
    """
    plan_dir = episode_plan.get("_plan_dir", "")
    tr = episode_plan.get("target_reference") or {}
    rel = tr.get("png_relpath") or \
        episode_plan.get("target_task", {}).get("target_reference_image", "")

    def _read(relpath):
        search = []
        if plan_dir:
            search.append(os.path.join(plan_dir, relpath))
        for d in extra_search_dirs:
            search.append(os.path.join(d, relpath))
        search.append(relpath)
        for p in search:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return f.read()
        return None

    collage = _read(rel) if rel else None
    views = {}
    for v in tr.get("views") or []:
        kind, vrel = v.get("kind"), v.get("relpath")
        if not kind or not vrel:
            continue
        data = _read(vrel)
        if data is not None:
            views[kind] = data
    return collage or b"", views


def load_teammate_reference(episode_plan, actor=None, require_new=False):
    """Validate new identity assets; absent legacy assets return None explicitly."""
    from agos.episode.reference_photo import TEAMMATE_PROTOCOL, assert_actor_identity, ReferencePhotoError
    certificate = episode_plan.get('teammate_reference')
    if certificate is None:
        if require_new:
            raise EpisodeValidationError('new benchmark requires teammate reference identity asset')
        return None
    try:
        if certificate.get('protocol') != TEAMMATE_PROTOCOL:
            raise ValueError('unsupported teammate protocol')
        from agos.eval.navigation_oracle import validate_certificate, NavigationOracleError
        navigation = episode_plan.get('navigation_oracle')
        try:
            validate_certificate(navigation, navigation.get('binding', {}) if isinstance(navigation, dict) else {})
        except NavigationOracleError as error:
            raise ValueError(f'new teammate benchmark requires valid navigation certificate: {error}') from error
        identity = episode_plan.get('ugv_identity')
        if not identity or identity != certificate.get('identity') or not identity.get('blueprint'):
            raise ValueError('teammate identity contract mismatch')
        relative = certificate['png_relpath']
        # A certificate asset must remain within its own dataset.
        if os.path.isabs(relative) or '..' in relative.replace('\\', '/').split('/'):
            raise ValueError('teammate asset path escapes dataset')
        path = os.path.join(episode_plan.get('_plan_dir', ''), relative)
        with open(path, 'rb') as stream:
            data = stream.read()
        if hashlib.sha256(data).hexdigest() != certificate.get('png_sha256'):
            raise ValueError('teammate reference PNG hash mismatch')
        quality = certificate['quality']; framing = certificate['framing']
        ground = certificate['staging']['ground']; camera = certificate['camera']
        from agos.episode.reference_photo import check_ground_gaps
        if not all(math.isfinite(float(value)) for value in (
                quality.get('raw_visible_fraction', 0), quality.get('visible_fraction', 0),
                framing.get('margin_px', 0), framing.get('longest_fraction', 0),
                camera.get('height_above_roof_m', 0), camera.get('pitch', 0))):
            raise ValueError('teammate reference quality values must be finite')
        if not (quality.get('passed') is True and quality.get('same_frame_rgb_and_mask') is True
                and quality.get('frame') is not None
                and .995 <= float(quality.get('raw_visible_fraction', 0)) <= 1.005
                and float(quality.get('visible_fraction', 0)) >= .995
                and ground.get('passed') is True
                and check_ground_gaps(ground.get('bottom_to_road_gaps_m', []))['passed']
                and framing.get('passed') is True and framing.get('head_up') is True
                and framing.get('background_rgb') == [220,220,220]
                and framing.get('output_size_px') == 512 and framing.get('margin_px', 0) >= 16
                and .5 <= float(framing.get('longest_fraction', 0)) <= .75
                and camera.get('height_above_roof_m') == 5.0 and camera.get('pitch') == -90.0):
            raise ValueError('teammate reference lacks valid quality/camera certificate')
        import io
        import numpy as np
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'PNG' or image.size != (512,512):
                raise ValueError('teammate reference must be 512px PNG')
            pixels = np.asarray(image.convert('RGB'))
        if not np.all(pixels[:16] == 220) or not np.all(pixels[-16:] == 220) \
                or not np.all(pixels[:,:16] == 220) or not np.all(pixels[:,-16:] == 220) \
                or np.count_nonzero(np.any(pixels != 220, axis=2)) < 200:
            raise ValueError('teammate reference empty or non-neutral/clipped border')
        if actor is not None:
            assert_actor_identity(actor, identity)
        return data
    except (KeyError, ValueError, TypeError, OSError, ReferencePhotoError) as error:
        raise EpisodeValidationError(f'teammate reference validation failed: {error}') from error


def load_and_validate_episode_plan(plan_path, formal=True, check_reference=True):
    """统一入口：加载 plan 并做 fail-closed 验证，注入 _plan_dir。

    Returns:
        plan dict（已注入 _plan_dir）
    Raises:
        EpisodeValidationError: 任一验证失败
    """
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    plan["_plan_dir"] = os.path.dirname(os.path.abspath(plan_path))
    validate_geometry_contract(plan, plan_path, formal=formal)
    if check_reference:
        validate_reference_certificate(plan, plan["_plan_dir"], formal=formal)
        load_teammate_reference(plan)
    return plan
