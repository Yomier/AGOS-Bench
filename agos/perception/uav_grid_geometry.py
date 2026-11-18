"""uav_grid_geometry.py - UAV grid cell <-> world coordinate unified geometry module.

P0-1 fix: All UAV grid mapping functions must go through this module.
No other file is allowed to implement its own project_cell_to_world.

This module supports both:
1. Calibration-matrix mode (preferred): uses a 2x2 linear transform from
   a calibration JSON file produced by calibrate_uav_grid_geometry.py
2. Fallback mode: uses the original yaw-based formula (for backwards compat)
"""
import json, math, os, logging

logger = logging.getLogger("LiteV1.UAVGeometry")

DEFAULT_PARAMS = {
    "height_m": 80.0, "yaw_deg": 0.0, "pitch_deg": -90.0,
    "fov_deg": 90.0, "img_w": 1024, "img_h": 576, "rows": 5, "cols": 5,
}

_calibration = None
# B17（2026-09-17 审计）：标定加载失败哨兵。置位后 _ensure_calibration 不再重复
# 读文件/打日志（此前每次 pixel_to_world/world_to_pixel 都会重读并刷 warning），
# 直接走解析公式回退；显式调用 load_calibration() 仍会重新尝试加载。
_calibration_failed = False
_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "calibration", "uav_topdown_80m_yaw-90_fov90.json")


def _analytic_matrix(fov_deg, img_w, img_h, height_m):
    """North-up 解析真源矩阵：图像 px 向右=世界东(+Y)，py 向下=世界南(-X)。

    2026-08-02 修复依据（bug review 实证）：
    - UAV 相机（yaw=0, pitch=-90）真实朝向为上=北/右=东，已由 210ep 生产数据
      B/E 行标注的物理偏移分布验证（A 行 93% 目标在北侧、E 行 84% 在南侧、
      col1 93% 西、col5 80% 东），且与 _camera_basis/project_point_to_image
      的 CARLA Rotation 物理推导一致。
    - 旧标定矩阵 [[0,90],[160,0]]（image_down=+X）为 2026-07-20 旧相机管线
      （yaw=-90+FLIP 时代）产物，南北镜像，导致 grid_hit 集中于 C 行（0 命中
      于 A/B/E 行）。修正后矩阵 = [[0,-gh],[gw,0]]。
    """
    gw, gh = get_ground_footprint(height_m, fov_deg, img_w, img_h)
    return [[0.0, -gh], [gw, 0.0]]


def _validate_calibration(calib):
    """校验标定矩阵与 north-up 解析公式一致（容差 1%）。不一致则拒绝使用。

    这是一道永久性防线：任何再被引入的镜像/旋转矩阵都会在加载时被拒，
    静默退回正确的解析公式，杜绝 07-20 式错误再次无声上线。
    """
    if not isinstance(calib, dict):
        return False
    M = calib.get("normalized_offset_to_world_xy")
    if not M or len(M) != 2 or any(len(r) != 2 for r in M):
        return False
    fov = float(calib.get("fov_deg", 90.0))
    img_w = int(calib.get("img_w", 1024))
    img_h = int(calib.get("img_h", 576))
    height_m = float(calib.get("height_m", 80.0))
    A = _analytic_matrix(fov, img_w, img_h, height_m)
    tol = 0.01
    for i in range(2):
        for j in range(2):
            scale = max(1.0, abs(A[i][j]))
            if abs(M[i][j] - A[i][j]) > tol * scale:
                logger.error(
                    "UAV calibration matrix rejected: M[%d][%d]=%.3f deviates from "
                    "north-up analytic %.3f (mirrored/rotated calibration?). "
                    "Falling back to the analytic north-up formula.",
                    i, j, M[i][j], A[i][j])
                return False
    return True


def load_calibration(path=None):
    global _calibration, _calibration_failed
    asset_root = os.environ.get('AGOS_ASSET_ROOT')
    p = path or (os.path.join(asset_root, 'calibration', os.path.basename(_CALIBRATION_PATH))
                 if asset_root else _CALIBRATION_PATH)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                calib = json.load(f)
        except (OSError, ValueError) as e:
            # B17：文件不可读/JSON 损坏也算加载失败——置哨兵、只报一次错
            _calibration = None
            _calibration_failed = True
            logger.error("UAV calibration file unreadable at %s: %s; "
                         "falling back to the analytic north-up formula", p, e)
            return None
        if _validate_calibration(calib):
            _calibration = calib
            _calibration_failed = False
            logger.info("Loaded UAV calibration from %s (validated against north-up analytic)", p)
        else:
            _calibration = None
            _calibration_failed = True
    else:
        _calibration = None
        _calibration_failed = True
        logger.warning("UAV calibration file not found at %s; using analytic north-up formula", p)
    return _calibration


def _ensure_calibration():
    # B17：加载已失败（文件缺失/矩阵被拒/文件损坏）时不再重复读文件/打日志
    if _calibration is None and not _calibration_failed:
        load_calibration()


def get_ground_footprint(uav_z, fov_deg, img_w, img_h):
    fov_rad = math.radians(fov_deg)
    gw = 2 * uav_z * math.tan(fov_rad / 2)
    gh = gw * img_h / img_w
    return gw, gh


def cell_label_to_pixel_polygon(cell_label, img_w=1024, img_h=576, rows=5, cols=5):
    if not cell_label or len(cell_label) < 2:
        return []
    row_char = cell_label[0].upper()
    col_num = int(cell_label[1:]) - 1
    row_num = ord(row_char) - ord("A")
    if row_num < 0 or row_num >= rows or col_num < 0 or col_num >= cols:
        return []
    cw = img_w / cols
    ch = img_h / rows
    x0 = col_num * cw
    y0 = row_num * ch
    return [(x0, y0), (x0+cw, y0), (x0+cw, y0+ch), (x0, y0+ch)]


def pixel_to_world(uav_x, uav_y, uav_z, yaw_deg, fov_deg, img_w, img_h, px, py, ground_z=0.0):
    """North-up UAV image geometry.

    Image convention:
      px right = world east  = +Y
      py down  = world south = -X
    World convention:
      +X north, +Y east.
    yaw_deg is ignored in this fixed-north-up mode.

    B9（2026-09-17 审计）：ground_z 为地面高程（默认 0.0=旧行为）。非零时地面
    足迹按有效高度（uav_z - ground_z）计算，使山城/高架场景 cell 尺度正确。
    """
    _ensure_calibration()
    gw, gh = get_ground_footprint(uav_z - float(ground_z), fov_deg, img_w, img_h)
    norm_x = (px / img_w) - 0.5
    norm_y = (py / img_h) - 0.5
    if _calibration and "normalized_offset_to_world_xy" in _calibration:
        M = _calibration["normalized_offset_to_world_xy"]
        # Calibration matrix values are metric offsets at calibration.height_m.
        # B9 follow-up (R27): when ground_z changes, scale the metric transform by
        # effective camera height; otherwise candidate.use_ground_elevation=true
        # is a silent no-op whenever a valid 80 m calibration is loaded.
        calib_h = max(1e-6, float(_calibration.get("height_m", uav_z)))
        effective_h = max(0.0, float(uav_z) - float(ground_z))
        height_scale = effective_h / calib_h
        ox = (M[0][0] * norm_x + M[0][1] * norm_y) * height_scale
        oy = (M[1][0] * norm_x + M[1][1] * norm_y) * height_scale
        return uav_x + ox, uav_y + oy
    else:
        # 2026-07-24 修正：north-up 显式公式，不再依赖 yaw
        # px right = world east (+Y), py down = world south (-X)
        ox = norm_x * gw   # east offset (world +Y)
        oy = norm_y * gh   # south offset (world -X)
        return uav_x - oy, uav_y + ox


def pixel_polygon_to_world_polygon(uav_x, uav_y, uav_z, yaw_deg, fov_deg, img_w, img_h, pixel_polygon,
                                   ground_z=0.0):
    """B9：ground_z 透传给 pixel_to_world（默认 0.0=旧行为）。"""
    wc = [pixel_to_world(uav_x, uav_y, uav_z, yaw_deg, fov_deg, img_w, img_h, px, py,
                         ground_z=ground_z)
          for px, py in pixel_polygon]
    cx = sum(p[0] for p in wc) / len(wc)
    cy = sum(p[1] for p in wc) / len(wc)
    hd = max(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in wc)
    unc = min(max(hd + 8, 20), 45)
    return wc, (cx, cy), unc


def cell_label_to_world_polygon(uav_x, uav_y, uav_z, yaw_deg, fov_deg, img_w, img_h, cell_label,
                                rows=5, cols=5, ground_z=0.0):
    pp = cell_label_to_pixel_polygon(cell_label, img_w, img_h, rows, cols)
    if not pp:
        return [], (0, 0), 20
    return pixel_polygon_to_world_polygon(uav_x, uav_y, uav_z, yaw_deg, fov_deg, img_w, img_h, pp,
                                          ground_z=ground_z)


def world_to_pixel(uav_x, uav_y, uav_z, yaw_deg, fov_deg, img_w, img_h, wx, wy):
    """Inverse of pixel_to_world under the fixed north-up convention.

    Image convention:
      px right = world east  = +Y
      py down  = world south = -X
    World convention:
      +X north, +Y east.
    yaw_deg is ignored in this fixed-north-up mode.
    """
    _ensure_calibration()
    gw, gh = get_ground_footprint(uav_z, fov_deg, img_w, img_h)
    dx = wx - uav_x   # north offset (world +X)
    dy = wy - uav_y   # east offset (world +Y)
    if _calibration and "world_xy_to_normalized_offset" in _calibration:
        M = _calibration["world_xy_to_normalized_offset"]
        # The forward calibrated transform scales offsets by uav_z/calib_h.
        # Undo that scale here; 80 m historical projection remains identical.
        inverse_scale = float(_calibration.get("height_m", uav_z)) / max(1e-6, float(uav_z))
        nx = (M[0][0] * dx + M[0][1] * dy) * inverse_scale
        ny = (M[1][0] * dx + M[1][1] * dy) * inverse_scale
        px = (nx + 0.5) * img_w
        py = (ny + 0.5) * img_h
    else:
        # 2026-07-24 修正：north-up 显式公式，不再依赖 yaw
        # px = east offset (world +Y), py = south offset (world -X)
        ox = dy   # east offset (world +Y)
        oy = -dx  # south offset (world -X)
        px = (ox / gw + 0.5) * img_w
        py = (oy / gh + 0.5) * img_h
    if px < 0 or px >= img_w or py < 0 or py >= img_h:
        return None, None
    return px, py


def world_to_grid_cell(uav_x, uav_y, uav_z, yaw_deg, fov_deg, img_w, img_h, tx, ty, rows=5, cols=5):
    px, py = world_to_pixel(uav_x, uav_y, uav_z, yaw_deg, fov_deg, img_w, img_h, tx, ty)
    if px is None:
        return None, None, None
    col = min(int(px / (img_w / cols)), cols - 1)
    row = min(int(py / (img_h / rows)), rows - 1)
    labels = ["A", "B", "C", "D", "E"]
    return f"{labels[row]}{col + 1}", px, py
