"""carla_interface.py — CARLA-AIR 连接 + UAV/UGV 控制 + 图像捕获"""
import time, io, math, logging, threading
import numpy as np
from dataclasses import replace as dataclasses_replace
from PIL import Image

from agos.lite_core import DirectionalView, UGV_VIEW_REGISTRY, UAVFrame

logger = logging.getLogger("LiteV1.CarlaInterface")

class CarlaInterface:
    def __init__(self, host="127.0.0.1", rpc_port=2000, airsim_port=41451):
        global carla, airsim
        import carla, airsim
        from benchmark.native_runtime import require_client_guard
        require_client_guard()
        self.rpc_port = rpc_port
        self.airsim_port = airsim_port
        self.client = carla.Client(host, rpc_port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        self.airsim = airsim.MultirotorClient(port=airsim_port)
        self.airsim.confirmConnection()
        self._cleanup_world_actors()
        self._setup_sensors()
        logger.info("CARLA + AirSim connected")

    def _cleanup_world_actors(self):
        """清理 world 中残留的车辆和相机，避免影响 UGV/target spawn。

        Carla-AIR 单进程多 episode 运行时，上一 episode 的 actor 可能未被销毁，
        导致下一 episode 的 UGV 在起点因碰撞而生成失败。只清理 vehicle 和 sensor.camera.*，
        保留交通标志、路灯、行人等环境 actor。
        """
        actors = self.world.get_actors()
        removed = 0
        for actor in actors:
            type_id = actor.type_id
            if 'vehicle.' in type_id or type_id.startswith('sensor.camera.'):
                try:
                    actor.destroy()
                    removed += 1
                except Exception as e:
                    logger.warning(f"_cleanup_world_actors: failed to destroy {actor.id}: {e}")
        if removed:
            logger.info(f"Cleaned up {removed} stale actors from previous episodes")


    def _setup_sensors(self):
        """初始化相机占位（实际参数在每次 capture 时由 _create_camera 设置）。

        2026-07-22: UGV 观测统一为"几何中心锚点 + 车顶0.3m"，
        行进=forward view(90°, 六视图之一)，停下=六视图(90°×6)。"""
        self._cam_transform = carla.Transform(carla.Location(x=0, y=0, z=1.8))
        self._cam_actor = None
        self._cam_image = None
        self._cam_event = None
        self._ugv_actor = None

    def _create_camera(self, transform, width=960, height=540, fov=120):
        if self._cam_actor is not None:
            try:
                self._cam_actor.destroy()
            except:
                pass
        # 清空旧帧 + 新建 event，保证后续 capture 拿到的是本相机的新帧，
        # 避免上一台相机的残留图像被误用（旧帧复用 bug）
        self._cam_image = None
        self._cam_frame = None      # r9 D1：sensor frame 号（_on_camera 接力）
        self._cam_timestamp = None  # r9 D1：sensor timestamp（同上）
        self._cam_event = threading.Event()
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        self._cam_actor = self.world.spawn_actor(bp, transform)
        self._cam_actor.listen(lambda img: self._on_camera(img))
        return self._cam_actor

    def _on_camera(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        self._cam_image = arr[:, :, [2, 1, 0]].copy()
        # r9 D1：sensor 帧号/timestamp 随帧接力（numpy 数组上无这些属性，
        # 旧 getattr 回退导致 DirectionalView.frame_id 恒 None）
        self._cam_frame = getattr(image, "frame", None)
        self._cam_timestamp = getattr(image, "timestamp", None)
        if self._cam_event:
            self._cam_event.set()

    def get_map(self):
        return self.world.get_map()

    def get_world(self):
        return self.world

    def get_all_actors(self):
        return self.world.get_actors()

    # === UAV Control (AirSim for position + CARLA camera for capture) ===
    def set_uav_pose(self, x, y, z=80, yaw_deg=0, pitch_deg=-90):
        """设置 UAV 位姿（AirSim simSetVehiclePose for position only）

        2026-07-24 修正：yaw_deg 默认改为 0（朝北），使 UAV 俯视图方向为
        上北下南左西右东（图像向右=世界东，图像向下=世界南）。
        2026-08-02 修正（bug review）：删除本 docstring 中"图像向下=北、
        需要 FLIP_TOP_BOTTOM"的过时错误注释——那是 07-20 旧管线时代的说法，
        与同函数上一行的正确描述自相矛盾，且正是旧标定矩阵被做成南北镜像
        的误导根源。正确结论（已由生产数据验证）：yaw=0, pitch=-90 的原始
        图像就是上=北，不需要任何翻转，与 MapFrame 的 py=(x_max-x)/x_range
        （图像向上=北）天然一致。
        """
        pose = airsim.Pose()
        pose.position = airsim.Vector3r(x, y, -z)  # AirSim NED: z negative = up
        pose.orientation = airsim.to_quaternion(0, 0, math.radians(yaw_deg))  # pitch=0, AirSim ignores pitch for multirotor
        self.airsim.simSetVehiclePose(pose, True)
        self._uav_pos = {"x": x, "y": y, "z": z, "yaw": yaw_deg}
        time.sleep(0.3)

    def capture_uav_image(self, width=1024, height=576, fov=90):
        """Historical bytes API, with the same RGB quality checks and retries."""
        return self.capture_uav_frame(width, height, fov).image_bytes

    def capture_uav_frame(self, width=1024, height=576, fov=90):
        """捕获 UAV 俯视图 — 使用 CARLA 相机直接拍（AirSim pitch 不生效的修复方案）"""
        if not hasattr(self, "_uav_pos"):
            # B20（2026-09-17 审计，铁律 15）：禁止世界原点兜底——UAV 位姿未设置即
            # 拍照属上游协议违规，静默兜底会产出几何错误的图像（生产路径
            # set_uav_pose 必先于 capture，本分支不可达；诊断脚本误用时立刻暴露）。
            raise RuntimeError("UAV pose 未设置即拍照（禁止世界原点兜底，铁律 15）")

        import carla as _carla
        uav_x = self._uav_pos["x"]
        uav_y = self._uav_pos["y"]
        uav_z = self._uav_pos["z"]
        uav_yaw = self._uav_pos.get("yaw", 0)

        # Spawn CARLA camera at UAV position, looking straight down (pitch=-90)
        cam_transform = _carla.Transform(
            _carla.Location(x=uav_x, y=uav_y, z=uav_z),
            _carla.Rotation(yaw=uav_yaw, pitch=-90)  # pitch=-90 = looking straight down
        )

        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))

        # P1-01: 多槽位时增加图像质量校验，避免空图/黑图/旧帧进入 VLM
        for attempt in range(3):
            # B16（2026-09-17 审计）：相机 spawn 改 try_spawn_actor + 小重试（首试
            # + 2 次重试），仍失败抛带上下文的 RuntimeError（上层已归 infra 处理），
            # 不再用会抛裸异常的 spawn_actor。
            cam_actor = None
            for _spawn_try in range(3):
                cam_actor = self.world.try_spawn_actor(bp, cam_transform)
                if cam_actor is not None:
                    break
                time.sleep(0.2)
            if cam_actor is None:
                raise RuntimeError(
                    "UAV capture camera spawn failed at (%.1f, %.1f, %.1f) yaw=%.0f "
                    "after 3 try_spawn_actor attempts (quality attempt %d/3)"
                    % (uav_x, uav_y, uav_z, uav_yaw, attempt + 1))
            image_data = [None]
            def _on_img(image):
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))
                transform = getattr(image, "transform", None)
                pose_source = "sensor_transform" if transform is not None else "capture_transform"
                transform = transform if transform is not None else cam_transform
                pose = (float(transform.location.x), float(transform.location.y),
                        float(transform.location.z), float(transform.rotation.yaw),
                        float(transform.rotation.pitch))
                # Assign one immutable tuple: a newer callback can never pair
                # its RGB with an earlier callback's frame number or transform.
                image_data[0] = (arr[:, :, [2, 1, 0]].copy(),
                                 getattr(image, "frame", None), getattr(image, "timestamp", None),
                                 pose, pose_source, int(image.width), int(image.height))

            cam_actor.listen(_on_img)
            try:
                self.world.tick()
            except:
                pass
            time.sleep(0.5)
            cam_actor.stop()
            cam_actor.destroy()

            if image_data[0] is not None:
                img, frame_id, timestamp_s, pose, pose_source, image_w, image_h = image_data[0]
                # 质量校验：非全黑、非全白、尺寸正确
                mean_val = img.mean()
                std_val = img.std()
                if mean_val > 5 and std_val > 2:  # 非全黑/全白/纯色
                    from PIL import Image as PILImage
                    pil_img = PILImage.fromarray(img)
                    # 2026-07-24 修正：CARLA 相机 yaw=0, pitch=-90 时图像向上=北（不需要 FLIP）
                    # 用户反馈 RGB 底图上下放反了，说明原始图像就是上=北，FLIP 反而变成上=南
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    return UAVFrame(buf.getvalue(), frame_id, timestamp_s, *pose,
                                    float(fov), image_w, image_h, pose_source=pose_source)
                else:
                    logger.warning("UAV image quality check failed (mean=%.1f std=%.1f), retry %d/3",
                                   mean_val, std_val, attempt + 1)
            else:
                logger.warning("UAV image capture empty, retry %d/3", attempt + 1)

        # 3 次都失败，返回空
        logger.error("UAV image capture failed after 3 attempts")
        return UAVFrame(b"", None, None, uav_x, uav_y, uav_z, uav_yaw, -90,
                        float(fov), int(width), int(height), valid=False)

    def capture_ugv_topdown_reference(self, x, y, width=512, height=512, fov=90):
        """Legacy local-background reference only; new plans load certified identity PNG.

        Retained unchanged to reproduce historical plan inputs. Never use this
        camera as a fallback for AGOS_TEAMMATE_TOPDOWN_V1.
        """
        import carla as _carla
        ref_z = 5.0  # 从 10m 降到 5m，UGV 更大更清晰
        cam_transform = _carla.Transform(
            _carla.Location(x=x, y=y, z=ref_z),
            _carla.Rotation(yaw=0, pitch=-90)  # 2026-07-24: 统一 yaw=0 (north-up)
        )
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        # P0-fix (2026-07-22): spawn_actor 可能失败（位置被占用），需要检查并重试
        cam_actor = self.world.try_spawn_actor(bp, cam_transform)
        if cam_actor is None:
            logger.warning("Failed to spawn camera for UGV topdown reference at (%.1f, %.1f), trying offset", x, y)
            # 尝试在 x,y 附近偏移 spawn
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]:
                cam_transform = _carla.Transform(
                    _carla.Location(x=x + dx, y=y + dy, z=ref_z),
                    _carla.Rotation(yaw=0, pitch=-90)
                )
                cam_actor = self.world.try_spawn_actor(bp, cam_transform)
                if cam_actor is not None:
                    logger.info("Spawned camera at offset (%.1f, %.1f)", dx, dy)
                    break
        if cam_actor is None:
            logger.error("Failed to spawn camera for UGV topdown reference at (%.1f, %.1f) after trying offsets", x, y)
            return b""
        image_data = [None]
        def _on_img(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            image_data[0] = arr[:, :, [2, 1, 0]].copy()
        cam_actor.listen(_on_img)
        try:
            self.world.tick()
        except:
            pass
        time.sleep(0.5)
        cam_actor.stop()
        cam_actor.destroy()
        if image_data[0] is not None:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image_data[0])
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()
        return b""

    def capture_global_topdown(self, region, width=2048, height=2048, fov=90):
        """Spec v2 §4.2/§6.3: capture a full-map RGB top-down image covering the episode region.

        The camera is placed high above the region center, looking straight down.
        Altitude is computed from the region diagonal and fov so the whole region
        fits inside the frame (with ~10% margin).

        Args:
            region: dict with x_min/x_max/y_min/y_max (world meters)
            width/height: image resolution (square recommended)
            fov: camera horizontal fov in degrees

        Returns:
            (png_bytes, capture_meta_dict) where capture_meta has the camera pose
            and the world extent the image covers, for map_metadata alignment.
        """
        import carla as _carla
        cx = (region["x_min"] + region["x_max"]) / 2.0
        cy = (region["y_min"] + region["y_max"]) / 2.0
        span_x = region["x_max"] - region["x_min"]
        span_y = region["y_max"] - region["y_min"]
        span = max(span_x, span_y) * 1.1  # 10% margin
        # altitude so that the ground footprint (square) >= span
        half_fov_rad = math.radians(fov / 2.0)
        alt = (span / 2.0) / math.tan(half_fov_rad)
        alt = max(alt, 100.0)  # never lower than 100m
        cam_transform = _carla.Transform(
            _carla.Location(x=cx, y=cy, z=alt),
            _carla.Rotation(yaw=0, pitch=-90)  # 2026-07-24: yaw=0 (north), image right=world east, image down=world south, 与 MapFrame 一致
        )
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        cam_actor = self.world.spawn_actor(bp, cam_transform)
        image_data = [None]
        def _on_img(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            image_data[0] = arr[:, :, [2, 1, 0]].copy()
        cam_actor.listen(_on_img)
        # Tick multiple times to ensure textures load at high altitude
        for _ in range(3):
            try:
                self.world.tick()
            except Exception:
                pass
        time.sleep(2.0)  # high-altitude full-map capture needs longer for texture streaming
        cam_actor.stop()
        cam_actor.destroy()
        # world footprint actually covered (square, side = 2*alt*tan(fov/2))
        side = 2.0 * alt * math.tan(half_fov_rad)
        capture_meta = {
            "cam_world_x": round(cx, 2), "cam_world_y": round(cy, 2), "cam_altitude_m": round(alt, 2),
            "fov_deg": fov, "image_width_px": width, "image_height_px": height,
            "covered_world_x_min": round(cx - side / 2.0, 2), "covered_world_x_max": round(cx + side / 2.0, 2),
            "covered_world_y_min": round(cy - side / 2.0, 2), "covered_world_y_max": round(cy + side / 2.0, 2),
            "meters_per_pixel": round(side / width, 4),
        }
        if image_data[0] is not None:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image_data[0])
            # 2026-07-24 修正：CARLA 相机 yaw=0, pitch=-90 时图像向上=北（不需要 FLIP）
            # 用户反馈 RGB 底图上下放反了，说明原始图像就是上=北，FLIP 反而变成上=南
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue(), capture_meta
        return b"", capture_meta

    # === UGV Control (CARLA) ===
    def configure_ugv_identity(self, identity=None):
        """Freeze appearance before spawn; None explicitly selects legacy behavior."""
        if identity is not None:
            if not isinstance(identity, dict) or not identity.get('blueprint') \
                    or not isinstance(identity.get('attributes'), dict):
                raise ValueError('invalid UGV identity contract')
            from agos.episode.reference_photo import assert_actor_identity
            actor = getattr(self, '_ugv_actor', None)
            if actor is not None and getattr(actor, 'is_alive', False):
                assert_actor_identity(actor, identity)
        import copy
        self._ugv_identity = copy.deepcopy(identity)

    def set_ugv_transform(self, x, y, z=0.5, yaw_deg=0.0, fast_teleport=False, strict_kinematic=False):
        """设置 UGV 位姿（CARLA set_transform），返回实际生效的 Transform。

        Args:
            fast_teleport: 为 True 时用于 2m 中间点瞬移，减少 sleep 以加速 wall-clock。
            strict_kinematic: sparse benchmark/SFT 模式；实际 x/y 必须保持在请求航点 0.25m 内，否则失败。

        Returns:
            carla.Transform 或 None（生成失败时）。
        """
        from agos.episode.vehicle_grounding import DEPLOYMENT_PROTOCOL
        canonical_kinematic = (strict_kinematic and
            getattr(self, 'vehicle_deployment_protocol', None) == DEPLOYMENT_PROTOCOL)
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(yaw=yaw_deg)
        )
        # P0-fix: 检查 actor 是否真的活着（is_alive），不只是属性存在和不为 None。
        # 并行评测时前一个 episode 的 cleanup() 可能销毁了 actor 但属性还在，
        # 导致 set_transform 到死 actor 上，UGV 实际不存在。
        _actor_alive = (hasattr(self, "_ugv_actor") and self._ugv_actor is not None
                        and getattr(self._ugv_actor, "is_alive", False))
        if not _actor_alive:
            # Spawn UGV with a fixed blueprint list, deterministic order
            # P0-fix (2026-07-21): UGV 限制为轿车类型（sedan），与目标车区分且避免大型车辆相机高度问题。
            # 目标车: audi.tt(car), tesla.model3(sedan), jeep.wrangler(SUV), firetruck(truck), vw.t2(van)
            # UGV 选轿车，特征与目标车不同（颜色区分），且相机高度正常（~1.5m 车顶）。
            bp_lib = self.world.get_blueprint_library()
            # cars34: an owner-frozen teammate blueprint overrides the default order.
            preferred = getattr(self, '_ugv_preferred_blueprints', None) or [
                "vehicle.dodge.charger_2020", "vehicle.chevrolet.impala",
                "vehicle.mercedes.coupe_2020", "vehicle.bmw.grandtourer",
                "vehicle.audi.etron"]
            bps = []
            frozen_identity = getattr(self, '_ugv_identity', None)
            for bp_id in ([frozen_identity['blueprint']] if frozen_identity else preferred):
                bp = bp_lib.find(bp_id)
                if bp:
                    if frozen_identity:
                        for key, value in frozen_identity['attributes'].items():
                            if not bp.has_attribute(key):
                                raise RuntimeError(f'UGV identity blueprint lacks attribute {key}')
                            attribute = bp.get_attribute(key)
                            if attribute.is_modifiable:
                                bp.set_attribute(key, str(value))
                            else:
                                # CARLA ActorAttribute.__str__ prints a diagnostic
                                # wrapper; actor.attributes contains the raw value.
                                # Read the typed value rather than comparing that
                                # wrapper with the frozen identity's actual string.
                                kind = str(attribute.type).rsplit('.', 1)[-1].lower()
                                if kind == 'string':
                                    actual_value = attribute.as_str()
                                elif kind in ('int', 'integer'):
                                    actual_value = str(attribute.as_int())
                                elif kind == 'bool':
                                    actual_value = str(attribute.as_bool()).lower()
                                elif kind == 'float':
                                    actual_value = str(attribute.as_float())
                                elif kind in ('rgbcolor', 'color'):
                                    color = attribute.as_color()
                                    actual_value = f'{color.r},{color.g},{color.b}'
                                else:
                                    raise RuntimeError(f'UGV identity unsupported immutable attribute type: {key}/{kind}')
                                if actual_value != str(value):
                                    raise RuntimeError(f'UGV identity immutable attribute mismatch: {key}')
                    bps.append(bp)
            if not bps and not frozen_identity:
                bps = list(bp_lib.filter("vehicle.*"))[:10]

            actor = None
            for bp in bps:
                actor = self.world.try_spawn_actor(bp, transform)
                if actor is not None:
                    break

            # If failed, try nearby positions with small offsets (z only, keep x/y)
            if actor is None:
                offsets = [(0,0,0.5),(0,0,1.0),(0,0,1.5),(0,0,2.0),
                           (2,0,0),(-2,0,0),(0,2,0),(0,-2,0)]
                for off in offsets:
                    t2 = carla.Transform(
                        carla.Location(x=x+off[0], y=y+off[1], z=z+off[2]),
                        carla.Rotation(yaw=yaw_deg)
                    )
                    for bp in bps:
                        actor = self.world.try_spawn_actor(bp, t2)
                        if actor is not None:
                            logger.info(f"UGV spawned at offset ({off[0]},{off[1]},{off[2]}) from ({x:.1f},{y:.1f})")
                            break
                    if actor is not None:
                        break

            if actor is None:
                logger.error(f"Failed to spawn UGV at ({x:.1f}, {y:.1f}) after trying offsets")
                return None
            self._ugv_actor = actor
            if frozen_identity:
                from agos.episode.reference_photo import assert_actor_identity
                try:
                    assert_actor_identity(actor, frozen_identity)
                except Exception:
                    actor.destroy()
                    self._ugv_actor = None
                    raise
            # Round-8 audit hardening: benchmark UGV uses kinematic teleport, not CARLA dynamics.
            # Disable physics immediately so gravity/collision cannot perturb the initial pose.
            try:
                self._ugv_actor.set_simulate_physics(False)
            except Exception as e:
                if canonical_kinematic:
                    self._ugv_actor.destroy()
                    self._ugv_actor = None
                    raise
                logger.warning("Failed to disable UGV physics immediately after spawn: %s", e)
            if canonical_kinematic:
                # CARLA spawn may auto-lift and step gravity before freeze. Reapply
                # the exact kinematic command, just as for an already-live UGV.
                self._ugv_actor.set_transform(transform)
            # Return actual transform; if offset used, it differs from requested x/y
            if canonical_kinematic:
                # SetTransform is asynchronous; GetTransform reads the last
                # streamed snapshot. Advance/receive a tick before validating
                # the one submitted command instead of guessing wall delay.
                self.world.tick()
            else:
                time.sleep(0.05 if fast_teleport else 0.5)
            actual_tf = self._ugv_actor.get_transform()
            # P0-fix: 如果 CARLA 自动把 UGV spawn 到过高位置（z > requested + 2m），
            # 说明 x,y 被严重遮挡（如建筑物内部），拒绝 spawn 并返回 None。
            # 否则 RoadnetLocator 会因 z 偏差过大而拒绝定位，导致 UGV 空转。
            if actual_tf.location.z > z + 2.0:
                logger.error(f"UGV spawned at z={actual_tf.location.z:.1f} (requested {z:.1f}), too high, destroying and rejecting")
                try:
                    self._ugv_actor.destroy()
                except Exception:
                    pass
                self._ugv_actor = None
                return None
            # 2026-07-25 新增：检查 x/y 偏移是否超过 10m（防止 spawn 到错误位置）
            xy_offset = math.sqrt((actual_tf.location.x - x)**2 + (actual_tf.location.y - y)**2)
            _xy_tol = 0.25 if strict_kinematic else 10.0
            if xy_offset > _xy_tol:
                logger.error(f"UGV spawned at ({actual_tf.location.x:.1f}, {actual_tf.location.y:.1f}) "
                            f"(requested ({x:.1f}, {y:.1f})), xy offset={xy_offset:.1f}m, destroying and rejecting")
                try:
                    self._ugv_actor.destroy()
                except Exception:
                    pass
                self._ugv_actor = None
                return None
            return actual_tf
        else:
            # Kinematic teleport contract: disable physics BEFORE changing pose.
            try:
                self._ugv_actor.set_simulate_physics(False)
            except Exception:
                if canonical_kinematic:
                    raise
                pass
            self._ugv_actor.set_transform(transform)
            if canonical_kinematic:
                self.world.tick()
            else:
                time.sleep(0.05 if fast_teleport else 0.5)
            # 2026-07-25 新增：set_transform 后也检查 x/y 偏移（防止物理引擎弹到错误位置）
            actual_tf = self._ugv_actor.get_transform()
            xy_offset = math.sqrt((actual_tf.location.x - x)**2 + (actual_tf.location.y - y)**2)
            _xy_tol = 0.25 if strict_kinematic else 10.0
            if xy_offset > _xy_tol and strict_kinematic and not canonical_kinematic:
                # Round-8 并发补丁（2026-09-12）：多引擎并发负载下 RPC 迟滞可能使 get_transform
                # 读到重放前的旧位姿（实测偏差恰为上一 fine 节点距离）。重放一次并加长等待后再判，
                # 不放宽 0.25m 严格容差本身。
                self._ugv_actor.set_transform(transform)
                time.sleep(0.4)
                actual_tf = self._ugv_actor.get_transform()
                xy_offset = math.sqrt((actual_tf.location.x - x)**2 + (actual_tf.location.y - y)**2)
            if xy_offset > _xy_tol:
                logger.error(f"UGV set_transform to ({actual_tf.location.x:.1f}, {actual_tf.location.y:.1f}) "
                            f"(requested ({x:.1f}, {y:.1f})), xy offset={xy_offset:.1f}m, too large, destroying and rejecting")
                try:
                    self._ugv_actor.destroy()
                except Exception:
                    pass
                self._ugv_actor = None
                return None
            return actual_tf

    # ============================================================
    # 2026-08-02 清理（bug review / 死代码删除）：
    # get_road_waypoints_between + _count_turns（基于 CARLA 原生 topology 的
    # Dijkstra 路由）已无任何调用方——正式路径统一走 roadnet.RoadRouter
    # （冻结两层路网图上的 A*）。保留第二套路由实现只会造成两图漂移风险，
    # 故整体删除。历史实现见 git/归档或 2026-07-17 工作日志。
    # ============================================================

    def _get_ugv_camera_anchor(self):
        """返回 UGV 相机锚点（世界坐标 carla.Location）。

        P0-fix (2026-07-22 用户要求)：
        - 水平位置锁定到 UGV **几何中心**（bounding box 中心的世界坐标），
          而非 actor 原点——很多车型的 actor origin 偏车头/车尾，导致相机随移动偏移。
        - 竖直位置 = 车顶 + 0.3m（bounding box 顶 + 0.3），避免拍到车内。

        bbox.location 是 bbox 中心相对 actor 原点的偏移（vehicle 局部坐标系，x 前 y 左 z 上），
        需经 actor transform 旋转变换到世界坐标。
        """
        if not hasattr(self, "_ugv_actor") or self._ugv_actor is None:
            # 兜底：用 actor transform + 默认高度
            t = self._ugv_actor.get_transform() if getattr(self, "_ugv_actor", None) else carla.Transform()
            return carla.Location(t.location.x, t.location.y, t.location.z + 2.2)
        try:
            tf = self._ugv_actor.get_transform()
            bbox = self._ugv_actor.bounding_box
            # bbox 中心的世界坐标（水平 = 几何中心）
            center_world = tf.transform(bbox.location)
            # 车顶高度 = bbox 中心 z + 半高 extent.z；相机 = 车顶 + 0.3m
            cam_z = center_world.z + bbox.extent.z + 0.3
            return carla.Location(center_world.x, center_world.y, cam_z)
        except Exception:
            t = self._ugv_actor.get_transform()
            return carla.Location(t.location.x, t.location.y, t.location.z + 2.2)

    def capture_ugv_front_image(self, width=960, height=540, fov=90):
        """捕获 UGV 前向图（= 六视图的 forward view）。

        2026-07-22 用户要求：行进观测从"独立 120° 前向广角"改为"六视图 forward view"，
        即 Navigation = front only（与六视图同 90° FOV、同相机锚点），
        只有主动停下观察才获得 360° 六视图。
        学术设定：移动阶段是部分可观测 egocentric navigation agent。
        相机锚点：水平=UGV 几何中心，竖直=车顶+0.3m。
        """
        anchor = self._get_ugv_camera_anchor()
        base_yaw = self._ugv_actor.get_transform().rotation.yaw
        cam = self._create_camera(carla.Transform(
            anchor,
            carla.Rotation(yaw=base_yaw, pitch=0)
        ), width=width, height=height, fov=fov)
        # 等待本相机的新帧（event 在 _create_camera 中已重置），超时则兜底 sleep
        if self._cam_event is not None:
            got = self._cam_event.wait(timeout=2.0)
            if not got:
                logger.warning("capture_ugv_front_image: camera frame timeout (2.0s)")
                time.sleep(0.3)
        else:
            time.sleep(0.3)
        result = self._cam_image.copy() if self._cam_image is not None else None
        if self._cam_actor:
            self._cam_actor.destroy()
            self._cam_actor = None
        if result is not None:
            pil_img = Image.fromarray(result)
            if pil_img.size != (width, height):
                pil_img = pil_img.resize((width, height))
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()
        return b""

    def capture_ugv_views(self, yaws=None, panel_w=480, panel_h=270, fov=90):
        """r7 边界重构：原始六视图采集 API（benchmark 只采集，不预处理）。

        时序与旧 capture_ugv_6view 完全一致（六次顺序采样：
        create camera -> wait 2.0s -> copy -> destroy -> sleep 0.1），
        返回带固定相机身份的 DirectionalView 元组；黑帧（超时无图）
        填零图且 valid=False，黑帧计数同样暴露到 self.last_6view_black_count。
        """
        if yaws is None:
            yaws = [-45, 0, 45, -135, 180, 135]
        base_transform = self._ugv_actor.get_transform()
        base_yaw = base_transform.rotation.yaw
        anchor = self._get_ugv_camera_anchor()
        views = []
        black_count = 0
        for i, yaw_offset in enumerate(yaws):
            cam_yaw = base_yaw + yaw_offset
            cam_transform = carla.Transform(
                anchor,
                carla.Rotation(yaw=cam_yaw, pitch=0)
            )
            self._create_camera(cam_transform, width=panel_w, height=panel_h, fov=fov)
            got = self._cam_event.wait(timeout=2.0)
            if not got:
                logger.warning("capture_ugv_views: frame timeout (2.0s) at yaw_offset=%s", yaw_offset)
            frame_id = None
            if self._cam_image is not None:
                pil_img = Image.fromarray(self._cam_image.copy())
                if pil_img.size != (panel_w, panel_h):
                    pil_img = pil_img.resize((panel_w, panel_h))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
                frame_id = self._cam_frame
                valid = True
            else:
                pil_img = Image.fromarray(
                    np.zeros((panel_h, panel_w, 3), dtype=np.uint8))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                image_bytes = buf.getvalue()
                valid = False
                black_count += 1
            if self._cam_actor:
                self._cam_actor.destroy()
                self._cam_actor = None
            time.sleep(0.1)
            template = UGV_VIEW_REGISTRY[i] if i < len(UGV_VIEW_REGISTRY) else None
            if template is not None and template.yaw_offset_deg == float(yaw_offset):
                views.append(dataclasses_replace(
                    template, image_bytes=image_bytes,
                    fov_deg=float(fov), width=panel_w, height=panel_h,
                    frame_id=frame_id,
                    timestamp_s=(self._cam_timestamp
                                 if valid and self._cam_timestamp is not None
                                 else time.time()),
                    valid=valid))
            else:  # 非默认 yaw 配置：退化为按序号命名
                views.append(DirectionalView(
                    "V%d" % i, "VIEW_%d" % i, float(yaw_offset), float(fov),
                    panel_w, panel_h, image_bytes=image_bytes,
                    frame_id=frame_id,
                    timestamp_s=(self._cam_timestamp
                                 if valid and self._cam_timestamp is not None
                                 else time.time()),
                    valid=valid))
        self.last_6view_black_count = black_count
        return tuple(views)

    def capture_ugv_6view(self, yaws=None,
                          panel_w=480, panel_h=270, fov=90):
        """捕获 6 视图并拼接成 2x3 collage。

        .. deprecated:: r7
            采样请改用 capture_ugv_views()（benchmark 原始通道）；
            2x3 collage 属于 需要多视图拼图时使用的方法内
            表示，由 methods/common/legacy_sixview_collage.py 渲染。
            本函数保留用于兼容旧脚本与回归测试。

        R2-P2-02：yaws 改为 None 默认（可变默认参数陷阱）；超时黑图计数暴露到
        self.last_6view_black_count 供 runner 记入 result warnings（黑图静默降级
        会污染免停靠核验的证据链）。
        """
        views = self.capture_ugv_views(yaws=yaws, panel_w=panel_w,
                                       panel_h=panel_h, fov=fov)
        images = []
        for v in views:
            if v.valid:
                images.append(np.array(Image.open(io.BytesIO(v.image_bytes))))
            else:
                images.append(np.zeros((panel_h, panel_w, 3), dtype=np.uint8))

        # r7: collage 渲染收敛到 methods/common/legacy_sixview_collage
        # （旧函数行为由同一实现保证，输出逐像素一致）
        from agos.methods.common.legacy_sixview_collage import (
            render_legacy_sixview_collage)
        return render_legacy_sixview_collage(views), images  # collage + individual images

    # === Actor Management ===
    def configure_vehicle_deployment(self, protocol=None):
        from agos.episode.vehicle_grounding import DEPLOYMENT_PROTOCOL, LEGACY_PROTOCOL
        protocol = protocol or LEGACY_PROTOCOL
        if protocol not in (DEPLOYMENT_PROTOCOL, LEGACY_PROTOCOL):
            raise ValueError('unsupported vehicle deployment protocol: ' + str(protocol))
        if getattr(self, 'vehicle_deployment_protocol', None) != protocol:
            self._vehicle_deployment_receipts = {}
        self.vehicle_deployment_protocol = protocol

    def spawn_target(self, blueprint_filter, color_rgb, transform, max_tries=5, strict_blueprint=True,
                     color_policy=None):
        """生成目标车辆。strict_blueprint=True 时只在指定 filter 内查找，失败不 fallback。
        碰撞时仅在 z 高度上重试，保持 x/y 与计划一致。

        2026-07-25 修复：用 road waypoint 的 z 值 spawn，确保 target 在路面上，
        不被路面遮挡（之前用固定 z=0.5+0.5*i，导致 target 在地下）。

        cars34 颜色合同：color_policy='factory' 时保持原厂涂装（color_rgb 必须
        为 None）；非空 color_rgb 遇不可修改/缺失 color 属性时明确报错，只有
        显式 factory 模式才略过颜色设置。
        """
        bp_lib = self.world.get_blueprint_library()
        bps = bp_lib.filter(blueprint_filter)
        if not bps:
            if strict_blueprint:
                logger.error(f"Blueprint filter '{blueprint_filter}' returned no results")
                return None
            bps = bp_lib.filter("vehicle.*")
            if not bps:
                logger.error("No vehicle blueprints found at all")
                return None
        # Pick first matching blueprint deterministically
        bp = bps[0]
        if color_rgb:
            if not bp.has_attribute("color") or not bp.get_attribute("color").is_modifiable:
                raise ValueError(
                    f"target color contract violation: blueprint '{bp.id}' has no modifiable "
                    "color attribute for a non-null color_rgb; factory policy requires null rgb")
            bp.set_attribute("color", f"{color_rgb[0]},{color_rgb[1]},{color_rgb[2]}")
        elif color_policy == 'factory':
            pass  # explicit factory appearance: keep the blueprint's original livery
        elif color_policy is not None:
            raise ValueError("non-factory color policy requires a non-null color_rgb")
        original_loc = transform.location
        yaw = transform.rotation.yaw

        # 2026-07-25 新增：获取 road waypoint 的 z 值，确保 target 在路面上
        carla_map = self.world.get_map()
        road_z = None
        try:
            loc = carla.Location(x=original_loc.x, y=original_loc.y, z=0.0)
            wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is not None:
                road_z = wp.transform.location.z
        except Exception as e:
            from agos.episode.vehicle_grounding import DEPLOYMENT_PROTOCOL
            if getattr(self, 'vehicle_deployment_protocol', None) == DEPLOYMENT_PROTOCOL:
                raise
            logger.warning(f"Failed to get road waypoint z: {e}")

        if road_z is not None:
            z_base = road_z
        else:
            z_base = 0.5
            logger.warning(f"No road waypoint found at ({original_loc.x:.1f}, {original_loc.y:.1f}), using default z=0.5")

        for i in range(max_tries):
            z = z_base + 0.5 * i
            try_transform = carla.Transform(
                carla.Location(x=original_loc.x, y=original_loc.y, z=z),
                carla.Rotation(yaw=yaw)
            )
            actor = self.world.try_spawn_actor(bp, try_transform)
            if actor is not None:
                from agos.episode.vehicle_grounding import (
                    DEPLOYMENT_PROTOCOL, settle_deployed_vehicle, VehicleGroundingError)
                if getattr(self, 'vehicle_deployment_protocol', None) == DEPLOYMENT_PROTOCOL:
                    try:
                        receipt = settle_deployed_vehicle(self.world, actor, try_transform.location)
                    except VehicleGroundingError as error:
                        logger.warning('Vehicle deployment quality rejection: %s', error)
                        actor.destroy()
                        return None
                    except BaseException:
                        actor.destroy()
                        raise  # Transport/native failure is never a resampling reason.
                    self._vehicle_deployment_receipts[actor.id] = receipt
                    return actor
                try:
                    actor.set_simulate_physics(False)
                except Exception as e:
                    logger.warning(f"Failed to disable target physics: {e}")
                return actor
        logger.error("spawn target failed after all retries")
        return None

    def spawn_distractors_from_plan(self, episode_plan):
        """P0-5: 从 episode plan 重新 spawn 全部干扰车辆，返回 actor 列表"""
        self.configure_vehicle_deployment(episode_plan.get('vehicle_deployment_protocol'))
        distractors = []
        for d in episode_plan.get("distractors_oracle", []):
            color_rgb = d.get("color_rgb")
            color_policy = d.get("color_policy")
            if color_policy not in (None, 'factory') and not color_rgb:
                raise ValueError('distractor replay color contract violation: null rgb without factory policy')
            bp_filter = d.get("blueprint_filter", "vehicle.*")
            transform = carla.Transform(
                carla.Location(x=d["x"], y=d["y"], z=d.get("z", 0.5)),
                carla.Rotation(yaw=d.get("yaw", 0.0))
            )
            actor = self.spawn_target(bp_filter, color_rgb, transform, color_policy=color_policy)
            if actor is not None:
                from agos.episode.vehicle_grounding import DEPLOYMENT_PROTOCOL, restore_certified_pose
                if self.vehicle_deployment_protocol == DEPLOYMENT_PROTOCOL:
                    try:
                        self._vehicle_deployment_receipts[actor.id]['canonical_replay'] = (
                            restore_certified_pose(self.world, actor, d))
                    except BaseException:
                        actor.destroy()
                        raise
                distractors.append(actor)
        logger.info(f"Respawned {len(distractors)}/{len(episode_plan.get('distractors_oracle', []))} distractors")
        return distractors

    def spawn_target_from_plan(self, episode_plan):
        """P0-1: 从 episode plan 重新 spawn target actor"""
        self.configure_vehicle_deployment(episode_plan.get('vehicle_deployment_protocol'))
        oracle = episode_plan["target_oracle"]
        tf = oracle["transform"]
        import carla as _carla
        transform = _carla.Transform(
            _carla.Location(x=tf["x"], y=tf["y"], z=tf.get("z", 0.5)),
            _carla.Rotation(yaw=tf.get("yaw", 0.0))
        )
        color_rgb = oracle.get("color_rgb")
        color_policy = oracle.get("color_policy")
        if color_policy not in (None, 'factory') and not color_rgb:
            raise ValueError('target replay color contract violation: null rgb without factory policy')
        bp_filter = oracle.get("blueprint_filter", "vehicle.*")
        try:
            actor = self.spawn_target(bp_filter, color_rgb, transform, color_policy=color_policy)
            if actor is None:
                logger.error("spawn_target returned None for plan target at (%.1f, %.1f)" % (tf["x"], tf["y"]))
                return None
            from agos.episode.vehicle_grounding import DEPLOYMENT_PROTOCOL, restore_certified_pose
            if self.vehicle_deployment_protocol == DEPLOYMENT_PROTOCOL:
                try:
                    self._vehicle_deployment_receipts[actor.id]['canonical_replay'] = (
                        restore_certified_pose(self.world, actor, tf))
                except BaseException:
                    actor.destroy()
                    raise
            logger.info("Target re-spawned from plan: actor_id=%s at (%.1f, %.1f)" % (actor.id, tf["x"], tf["y"]))
            return actor
        except Exception as e:
            from agos.episode.vehicle_grounding import DEPLOYMENT_PROTOCOL
            if self.vehicle_deployment_protocol == DEPLOYMENT_PROTOCOL:
                raise
            logger.error("Failed to spawn target from plan: %s" % e)
            return None

    def get_actor_bbox(self, actor):
        """获取 actor 的 3D bounding box"""
        bbox = actor.bounding_box
        transform = actor.get_transform()
        corners = []
        for extent in [
            carla.Location(x=bbox.extent.x, y=bbox.extent.y, z=bbox.extent.z),
            carla.Location(x=-bbox.extent.x, y=bbox.extent.y, z=bbox.extent.z),
            carla.Location(x=bbox.extent.x, y=-bbox.extent.y, z=bbox.extent.z),
            carla.Location(x=-bbox.extent.x, y=-bbox.extent.y, z=bbox.extent.z),
            carla.Location(x=bbox.extent.x, y=bbox.extent.y, z=-bbox.extent.z),
            carla.Location(x=-bbox.extent.x, y=bbox.extent.y, z=-bbox.extent.z),
            carla.Location(x=bbox.extent.x, y=-bbox.extent.y, z=-bbox.extent.z),
            carla.Location(x=-bbox.extent.x, y=-bbox.extent.y, z=-bbox.extent.z),
        ]:
            world_loc = transform.transform(extent)
            corners.append((world_loc.x, world_loc.y, world_loc.z))
        return corners

    def raycast(self, start_loc, end_loc):
        """CARLA raycast for LOS check"""
        try:
            return self.world.cast_ray(
                carla.Location(x=start_loc[0], y=start_loc[1], z=start_loc[2]),
                carla.Location(x=end_loc[0], y=end_loc[1], z=end_loc[2])
            )
        except:
            return None

    def check_los_to_target(self, observer_loc, target_actor, self_actor=None):
        """检查从 observer_loc 到 target_actor 是否通视且距离不超过确认半径。

        返回 (visible, distance_m, reason)
        使用多条射线到目标 bbox 采样点，要求至少一条命中目标几何内部且中间
        无其他障碍物（非目标、非自车）才视为 visible。异常时 fail-closed。
        """
        if not target_actor or not target_actor.is_alive:
            return False, 9999.0, "target_not_alive"
        try:
            import carla
            tgt_transform = target_actor.get_transform()
            tgt_loc = tgt_transform.location
            bbox = target_actor.bounding_box
            # Sample target points: center + bbox corners
            local_points = [bbox.location]
            for dx, dy, dz in [
                (bbox.extent.x, bbox.extent.y, bbox.extent.z),
                (-bbox.extent.x, bbox.extent.y, bbox.extent.z),
                (bbox.extent.x, -bbox.extent.y, bbox.extent.z),
                (-bbox.extent.x, -bbox.extent.y, bbox.extent.z),
                (bbox.extent.x, bbox.extent.y, -bbox.extent.z),
                (-bbox.extent.x, bbox.extent.y, -bbox.extent.z),
                (bbox.extent.x, -bbox.extent.y, -bbox.extent.z),
                (-bbox.extent.x, -bbox.extent.y, -bbox.extent.z),
            ]:
                local_points.append(carla.Location(x=dx, y=dy, z=dz))
            world_points = [tgt_transform.transform(p) for p in local_points]

            eye = carla.Location(x=observer_loc[0], y=observer_loc[1], z=observer_loc[2] + 1.0)
            distance = math.hypot(math.hypot(tgt_loc.x - observer_loc[0], tgt_loc.y - observer_loc[1]),
                                  tgt_loc.z - observer_loc[2])

            blocking_labels = {"Building", "Fence", "Pedestrian", "Pole", "TrafficSign",
                               "TrafficLight", "Vegetation", "Wall", "Other"}

            visible = False
            for tgt_pt in world_points:
                hits = self.raycast((eye.x, eye.y, eye.z), (tgt_pt.x, tgt_pt.y, tgt_pt.z))
                if hits is None:
                    # raycast unavailable: fail-closed, do not assume visible
                    continue
                clear = True
                for h in hits:
                    hl = h.location
                    dist_to_hit = math.hypot(math.hypot(hl.x - eye.x, hl.y - eye.y), hl.z - eye.z)
                    dist_to_pt = math.hypot(math.hypot(tgt_pt.x - eye.x, tgt_pt.y - eye.y), tgt_pt.z - eye.z)
                    if dist_to_hit >= dist_to_pt - 0.1:
                        # Hit is at or beyond the target point -> target reached
                        break
                    # Extract actor id from LabelledPoint if available (different API versions)
                    hit_actor = getattr(h, "actor", None)
                    if hit_actor is None:
                        hit_actor = getattr(h, "get_actor", lambda: None)()
                    hit_actor_id = getattr(hit_actor, "id", None) if hit_actor else None
                    # Skip self actor
                    if self_actor is not None and hit_actor_id == self_actor.id:
                        continue
                    # Skip target actor itself
                    if hit_actor_id == target_actor.id:
                        continue
                    label = getattr(h, "label", None)
                    label_name = str(label) if label is not None else ""
                    # If hit is a vehicle but not the target/self, treat as blocking
                    if any(bl in label_name for bl in blocking_labels):
                        clear = False
                        break
                if clear:
                    visible = True
                    break
            if visible:
                return True, distance, ""
            return False, distance, "los_blocked"
        except Exception as e:
            logger.warning(f"check_los_to_target error: {e}")
            return False, 9999.0, str(e)

    def cleanup(self):
        if hasattr(self, "_ugv_actor") and self._ugv_actor:
            try:
                if getattr(self._ugv_actor, "is_alive", False):
                    self._ugv_actor.destroy()
            except: pass
            self._ugv_actor = None
        if self._cam_actor:
            try:
                if getattr(self._cam_actor, "is_alive", False):
                    self._cam_actor.destroy()
            except: pass
            self._cam_actor = None
        if hasattr(self, "_runtime_target") and self._runtime_target:
            try:
                if getattr(self._runtime_target, "is_alive", False):
                    self._runtime_target.destroy()
            except: pass
            self._runtime_target = None
