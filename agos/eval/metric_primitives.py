"""metric_primitives.py — ACG/GVA 指标计算公共几何与 confirmable predicate.

纯 Python 实现，不依赖 Shapely。当前 candidate 为简单凸四边形。
"""

import math
from numbers import Real, Integral
from typing import Sequence, Tuple

Point2D = Tuple[float, float]


def point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
    """Ray-casting 判断点是否在多边形内部或边界上。"""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # 边界检查：点到边距离 < 1e-6 视为在边界上
        dx, dy = xj - xi, yj - yi
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            continue
        t = ((x - xi) * dx + (y - yi) * dy) / (dx * dx + dy * dy)
        if 0 <= t <= 1:
            qx, qy = xi + t * dx, yi + t * dy
            if math.hypot(x - qx, y - qy) < 1e-6:
                return True
        # ray-casting
        intersects = ((yi > y) != (yj > y)) and (
            x <= (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_to_segment_distance(
    point: Point2D, start: Point2D, end: Point2D
) -> float:
    """点到线段的最短距离。"""
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / denom
    t = max(0.0, min(1.0, t))
    qx, qy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - qx, py - qy)


def point_to_polygon_distance(
    point: Point2D, polygon: Sequence[Point2D]
) -> float:
    """点到多边形的最短距离（内部为 0）。"""
    if len(polygon) < 3:
        raise ValueError("candidate polygon must contain at least 3 vertices")
    if point_in_polygon(point, polygon):
        return 0.0
    return min(
        point_to_segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )


def candidate_scale(polygon: Sequence[Point2D]) -> float:
    """Candidate 尺度：几何中心到最远顶点的距离。"""
    if len(polygon) < 3:
        raise ValueError("candidate polygon must contain at least 3 vertices")
    cx = sum(x for x, _ in polygon) / len(polygon)
    cy = sum(y for _, y in polygon) / len(polygon)
    return max(math.hypot(x - cx, y - cy) for x, y in polygon)


def compute_acg_score(
    target_xy: Point2D, polygon: Sequence[Point2D]
) -> tuple[float, float, float]:
    """计算单 candidate 的 ACG 距离得分（保留旧接口，供逐 candidate 记录）.

    Returns:
        (score, distance_m, scale_m)
        - score: exp(-distance / max(scale, eps))
        - distance_m: target 到 polygon 的最短距离
        - scale_m: candidate 尺度（几何中心到最远顶点）
    """
    distance_m = point_to_polygon_distance(target_xy, polygon)
    scale_m = candidate_scale(polygon)
    score = math.exp(-distance_m / max(scale_m, 1e-6))
    return score, distance_m, scale_m


# =====================================================================
# =====================================================================
# ACG v3（2026-07-24 外部Agent方案）：引导质量 50/(50+d) × 对数排名折扣 1/log2(j+1)
# =====================================================================
ACG_GUIDANCE_SCALE_M = 50.0  # ACG v3独立平滑尺度；不是当前Confirm30成功阈值。


def candidate_guidance_quality(residual_distance_m: float, scale_m: float = ACG_GUIDANCE_SCALE_M) -> float:
    """单 candidate 的引导质量: scale / (scale + distance)。距离为 0 时质量 1,距离 scale 时质量 0.5。"""
    if (not isinstance(residual_distance_m, Real) or isinstance(residual_distance_m, bool)
            or not math.isfinite(residual_distance_m) or residual_distance_m < 0
            or not math.isfinite(scale_m) or scale_m <= 0):
        raise ValueError('ACG requires finite nonnegative distance and positive scale')
    distance = residual_distance_m
    return scale_m / (scale_m + distance)


def candidate_rank_discount(candidate_rank: int) -> float:
    """对数排名折扣: 1 / log2(rank + 1)。rank=1 -> 1.0, rank=2 -> 0.631, rank=3 -> 0.500。"""
    if candidate_rank < 1:
        raise ValueError("candidate_rank must start from 1")
    return 1.0 / math.log2(candidate_rank + 1.0)


def compute_acg_episode_v3(
    candidate_records: Sequence[dict],
    max_steps: int,
    scale_m: float = ACG_GUIDANCE_SCALE_M,
) -> dict:
    """计算 episode 级 ACG v3。

    ACG = max_j[ 1/log2(j+1) × 50/(50+d_j) ]
    - d_j: 第 j 个 candidate 的 region-to-target residual distance
           （目标中心到候选 polygon 的几何距离，非路网距离）
    - j: candidate 创建序号（1-based）
    - 无 candidate -> ACG = 0
    - 不使用 grid_hit, 不使用 S_find/S_max, 不使用 exp(-dist/D0)

    Args:
        candidate_records: 每条含 distance_m（目标到候选 polygon 的几何距离）和 step_index（创建步）
        max_steps: 该 episode 的步数预算（不再用于 ACG, 保留兼容）
    Returns:
        dict: {score, best_quality, best_rank, best_candidate_id}
    """
    if not candidate_records:
        return {"score": 0.0, "best_quality": 0.0, "best_rank": None,
                "best_candidate_id": None}

    best_utility = 0.0
    best_cid = None
    best_rank = None
    best_quality = 0.0
    for rank, r in enumerate(candidate_records, start=1):
        dist = r.get("distance_m")
        q = candidate_guidance_quality(dist, scale_m)
        discount = candidate_rank_discount(rank)
        utility = q * discount
        if utility > best_utility:
            best_utility = utility
            best_cid = r.get("candidate_id")
            best_rank = rank
            best_quality = q

    return {
        "score": round(best_utility, 6),
        "best_quality": round(best_quality, 6),
        "best_rank": best_rank,
        "best_candidate_id": best_cid,
    }


# =====================================================================
# GVA v3（2026-07-24 外部Agent方案）：episode 级条件成功率
# =====================================================================
def compute_gva_episode_v3(
    task_success: bool,
    oracle_verifiable_reached: bool,
    invalid_confirm: bool,
) -> float | None:
    """计算 episode 级 GVA v3。

    GVA = task_success / (oracle_verifiable_reached OR invalid_confirm)
    - 分母为 0（无核验机会且无错误确认）-> 返回 None（不进入 GVA 分母）
    - 不使用 confidence, 不使用 REJECT 评分, 不使用无核验事件赋 0.5

    Args:
        task_success: episode 最终是否成功
        oracle_verifiable_reached: UGV 是否曾进入合法核验状态
        invalid_confirm: 是否执行了非法 CONFIRM
    Returns:
        float or None: 1.0(成功) / 0.0(失败) / None(不参与)
    """
    if not oracle_verifiable_reached and not invalid_confirm:
        return None  # 不进入 GVA 分母
    return 1.0 if task_success else 0.0


def compute_gva_v3(episodes: Sequence[dict]) -> dict:
    """计算总体 GVA v3。

    Args:
        episodes: 每条含 task_success, oracle_verifiable_reached, invalid_confirm
    Returns:
        dict: {gva, gva_support, num_success, num_oracle_verifiable, num_invalid_confirm}
    """
    numerator = 0
    denominator = 0
    num_oracle = 0
    num_invalid = 0
    num_success = 0

    for ep in episodes:
        success = bool(ep.get("task_success", False))
        oracle = bool(ep.get("oracle_verifiable_reached", False))
        invalid = bool(ep.get("invalid_confirm", False))
        if oracle:
            num_oracle += 1
        if invalid:
            num_invalid += 1
        if success:
            num_success += 1
        if oracle or invalid:
            denominator += 1
            if success:
                numerator += 1

    if denominator == 0:
        return {"gva": None, "gva_support": 0,
                "num_success": num_success, "num_oracle_verifiable": num_oracle,
                "num_invalid_confirm": num_invalid}
    return {
        "gva": round(numerator / denominator, 6),
        "gva_support": denominator,
        "num_success": num_success,
        "num_oracle_verifiable": num_oracle,
        "num_invalid_confirm": num_invalid,
    }


# =====================================================================
# ASE / ACov（2026-07-24 外部Agent方案）：空中搜索效率 + 区域覆盖率
# =====================================================================
def compute_ase_episode(
    first_oracle_visible_uav_step: int | None,
    max_uav_decision_steps: int,
) -> float:
    """计算 episode 级 ASE（Aerial Search Efficiency）。

    ASE = (T - tau + 1) / T, 其中 tau 为目标首次 Oracle 空中可见的 UAV 本地步。
    目标始终不可见时 ASE = 0。

    Args:
        first_oracle_visible_uav_step: 目标首次 Oracle 可见的 UAV 本地步（None=始终不可见）
        max_uav_decision_steps: UAV 最大决策预算（固定值，通常= max_steps）
    Returns:
        float: [0, 1]
    """
    if (not isinstance(max_uav_decision_steps, Integral)
            or isinstance(max_uav_decision_steps, bool) or max_uav_decision_steps <= 0):
        raise ValueError("max_uav_decision_steps must be positive")
    if first_oracle_visible_uav_step is None:
        return 0.0
    tau = first_oracle_visible_uav_step
    if (not isinstance(tau, Integral) or isinstance(tau, bool)
            or tau < 1 or tau > max_uav_decision_steps):
        raise ValueError('ASE first visible step must be an integer within the fixed budget')
    return (max_uav_decision_steps - tau + 1) / max_uav_decision_steps


def compute_ase(episodes: list) -> float | None:
    """计算总体 ASE（所有 episode 平均）。"""
    values = []
    for ep in episodes:
        tau = ep.get("ase_first_oracle_visible_uav_step")
        t_max = ep.get("ase_max_uav_decision_steps", 80)
        values.append(compute_ase_episode(tau, t_max))
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def compute_acov_at_k(
    observed_cells: set,
    total_cells: int,
    k: int,
    observed_events=None,
) -> float:
    """计算 ACov@K（Aerial Coverage Rate at K UAV steps）。

    P1-2 (2026-09-04 external review): 若提供 observed_events
    （(uav_local_step, cell_id) 序列），则严格按"UAV 本地步 <= K"截断计算；
    未提供时退回旧语义（累积集合，忽略 K），兼容旧数据/旧调用。

    Args:
        observed_cells: 已观察网格 ID 集合（最终累积）
        total_cells: 可搜索网格总数
        k: 步骤界限
        observed_events: [(uav_local_step, cell_id), ...] 时间线
    Returns:
        float: [0, 1]
    """
    if total_cells <= 0:
        return 0.0
    if observed_events is not None:
        observed_k = {cell for step, cell in observed_events if step <= k}
        return min(1.0, len(observed_k) / total_cells)
    return min(1.0, len(observed_cells) / total_cells)


def compute_acov(episodes: list, k: int = 80) -> float | None:
    """计算总体 ACov@K（所有 episode 平均）。"""
    values = []
    for ep in episodes:
        total = ep.get("acov_total_cells", 0)
        observed = ep.get("acov_observed_cells", 0)
        if total > 0:
            values.append(min(1.0, observed / total))
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def is_confirmable_observation(
    *,
    has_valid_stop: bool,
    gt_visible: bool,
    distance_to_target_m: float | None,
    max_confirm_distance_m: float,
    require_stop: bool,
    require_los: bool,
    los_clear: bool | None,
) -> bool:
    """与 SuccessChecker 完全一致的 oracle 正类标签函数。

    GVA 和 SuccessChecker 共用同一套逻辑，禁止两套不同的"可确认"判断。
    """
    if require_stop and not has_valid_stop:
        return False
    if distance_to_target_m is None:
        raise RuntimeError("missing oracle distance for verification")
    if not gt_visible:
        return False
    if distance_to_target_m > max_confirm_distance_m:
        return False
    if require_los and los_clear is not True:
        return False
    return True
