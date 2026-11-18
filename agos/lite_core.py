"""lite_core.py — 核心数据结构、配置加载、状态机"""
import yaml, json, math, os, logging, copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

logger = logging.getLogger("LiteV1")

# ============================================================
# Config
# ============================================================
def load_config(path):
    """Read YAML, resolving deployment values in scalar strings after parsing."""
    import re
    from pathlib import Path
    from agos.benchmark_protocol import confirm_distance_m, success_protocol

    defaults = {
        'AGOS_ASSET_ROOT': str(Path(__file__).resolve().parent.parent / 'assets'),
        'AGOS_VLM_URL': 'http://127.0.0.1:8114/v1/chat/completions',
        'AGOS_VLM_MODEL': 'qwen3.5-4b',
        'AGOS_SIDECAR_URL': 'http://127.0.0.1:8201',
    }
    def substitute(match):
        name = match.group(1)
        value = os.environ.get(name, defaults.get(name))
        if not value:
            raise ValueError(f'Missing environment variable {name} in {path}')
        return value.rstrip('/') if name == 'AGOS_SIDECAR_URL' else value

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            return re.sub(r'\$\{([A-Z_][A-Z_0-9]*)\}', substitute, value)
        return value

    with open(path, encoding='utf-8') as stream:
        config = expand(yaml.safe_load(stream))
    config.setdefault('success', {})['max_confirm_distance_m'] = confirm_distance_m(config)
    config.setdefault('benchmark', {})['success_protocol_id'] = success_protocol(config)['id']
    return config



# ============================================================
# Text Memory Management (Simplified: last 5 steps as text)
# ============================================================
class MemoryEntry:
    """单条动作记忆记录（保留结构，但不再做压缩）"""
    def __init__(self, step: int, sim_time: float, role: str, action: str,
                 params: Dict[str, Any] = None, result: str = "", outcome: str = ""):
        self.step = step
        self.sim_time = sim_time
        self.role = role
        self.action = action
        self.params = params or {}
        self.result = result
        self.outcome = outcome
        self.compressed = False

    def to_dict(self):
        return {
            "step": self.step,
            "sim_time": self.sim_time,
            "role": self.role,
            "action": self.action,
            "params": self.params,
            "result": self.result,
            "outcome": self.outcome,
        }


def _format_memory_entry(entry: MemoryEntry) -> str:
    """将单条记忆渲染为紧凑文本行（简单版，无压缩）"""
    step = entry.step
    t = entry.sim_time
    role = entry.role
    action = entry.action
    params = entry.params
    result = entry.result or entry.outcome

    if action == "ANNOTATE_CANDIDATE":
        location = params.get("position_xy")
        label = f"point={location}" if location is not None else f"legacy_grid={params.get('grid_cell', '?')}"
        conf = params.get("confidence", 0.0)
        return f"Step {step} sim_t={t:.0f}s: {role} {action} [{label}, conf={conf:.2f}] -> {result}"
    if action == "GO_TO_CANDIDATE":
        cid = params.get("candidate_id", "?")
        return f"Step {step} sim_t={t:.0f}s: {role} {action} [candidate={cid}] -> {result}"
    if action == "STOP_OBSERVE":
        cid = params.get("candidate_id", "")
        prefix = f"[candidate={cid}] " if cid else ""
        return f"Step {step} sim_t={t:.0f}s: {role} {action} {prefix}-> {result}"
    if action in ("CONFIRM", "REJECT"):
        cid = params.get("candidate_id", "?")
        reason = params.get("reason", "")
        reason_str = f", reason=\"{reason}\"" if reason else ""
        return f"Step {step} sim_t={t:.0f}s: {role} {action} [candidate={cid}{reason_str}] -> {result}"
    if action == "WAIT":
        return f"Step {step} sim_t={t:.0f}s: {role} WAIT -> {result}"
    if action in ("MOVE_TO_UAV_WAYPOINT", "MOVE_TO_ROAD_WAYPOINT"):
        wp = params.get("waypoint_id") or params.get("path", "?")
        return f"Step {step} sim_t={t:.0f}s: {role} {action} [wp={wp}] -> {result}"
    if action in ("FORWARD_1", "FORWARD_2", "TURN_LEFT", "TURN_RIGHT", "TURN_STRAIGHT", "REVERSE"):
        d = params.get("distance_m")
        d_str = f", {d:.0f}m" if isinstance(d, (int, float)) else ""
        return f"Step {step} sim_t={t:.0f}s: {role} {action}[{d_str}] -> {result}"
    if action == "message":
        content = params.get("content", "")
        return f"Step {step} sim_t={t:.0f}s: {role} MESSAGE -> \"{content}\""

    # 默认格式
    param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
    return f"Step {step} sim_t={t:.0f}s: {role} {action} [{param_str}] -> {result}"


def get_memory_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取 memory 配置，并提供默认值"""
    return config.get("memory", {
        "max_entries": 5,
    })

# ============================================================
class ActionType:
    UAV_MOVE = "MOVE_TO_UAV_WAYPOINT"
    UAV_ANNOTATE = "ANNOTATE_CANDIDATE"
    UAV_WAIT = "WAIT"
    UGV_MOVE = "MOVE_TO_ROAD_WAYPOINT"
    UGV_GO_TO_CANDIDATE = "GO_TO_CANDIDATE"
    UGV_STOP_OBSERVE = "STOP_OBSERVE"
    UGV_CONFIRM = "CONFIRM"
    UGV_REJECT = "REJECT"
    UGV_WAIT = "WAIT"

# ============================================================
# Data Types
# ============================================================


def canonical_map_key(map_name_or_path: str) -> str:
    """CARLA map名（可含路径与 _Opt 后缀）→ registry 用的 base map key。

    例："/Game/Carla/Maps/Town03_Opt" -> "Town03"。runner 与 episode_generator
    共用此函数（2026-09-04 收敛，此前两处独立实现易漂移）。
    """
    base = (map_name_or_path or "").replace("\\", "/").rstrip("/").split("/")[-1]
    if base.endswith("_Opt"):
        base = base[:-4]
    return base


@dataclass
class WaypointInfo:
    waypoint_id: str
    world_xy: Tuple[float, float]
    distance_m: float
    visited: bool = False
    description: str = ""

@dataclass
class CandidateInfo:
    candidate_id: str
    status: str  # unverified/assigned_to_ugv/visited/confirmed/rejected/suppressed_duplicate
    grid_cell: Optional[str] = None
    region_center_xy: Tuple[float, float] = (0, 0)
    region_polygon: List[Tuple[float, float]] = field(default_factory=list)
    uncertainty_radius_m: float = 20.0
    confidence: float = 0.5
    source_agent: str = "UAV"
    evidence: str = ""
    created_step: int = 0
    visited_by_ugv: bool = False
    stop_observe_count: int = 0
    confirm_attempts: int = 0
    reject_reason: Optional[str] = None
    # GSAM 两实验（2026-10-10 设计 §5/§9.1）：点类型候选合同。
    # geometry_type='point'：position 即 region_center_xy；region_polygon 为空、
    # grid_cell 可空；服务池用点级可达观察 port。region 为默认（旧行为不变）。
    geometry_type: str = "region"
    source_frame_id: Optional[int] = None
    source_image_sha256: str = ""
    # CandidateServiceContract（外审 P0-C01 修复，Phase-G2 Stage 0 Step G1）：
    # 候选的服务节点池（fine node id 元组），由 runner 在 serviceability 预检通过后
    # 用 compute_service_nodes 固化（半径 = max_confirm_distance_m，与 adapter /
    # approach planner 三处共用同一池）；到达判定按池成员判定。默认 () 兼容既有构造。
    service_node_ids: tuple = ()
    # R27 conservative verification-ready pool: farthest polygon corner <= terminal
    # confirm radius. Broad service pool remains available for diagnostics/backcompat.
    confirmable_service_node_ids: tuple = ()
    prefer_confirmable_service: bool = False
    # STOP_OBSERVE v2（外审 P0-C02 修复）：最近一次 STOP 观察步，
    # 由 CandidateLedger.record_stop_observation 维护，不参与生命周期状态。
    last_observed_step: int = 0

@dataclass
class MessageInfo:
    sender: str
    content: str
    timestamp_step: float = 0.0


@dataclass
class StopObservationContext:
    """记录一次 STOP_OBSERVE 的上下文，用于绑定 CONFIRM/REJECT 到当前证据"""
    observation_id: str
    created_step: int
    sim_time: float
    ugv_pose: Tuple[float, float, float]  # x, y, yaw
    candidate_id: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    is_fresh: bool = True


@dataclass(frozen=True)
class DirectionalView:
    """一个固定相机身份的 UGV 方向视图（r7 边界重构：benchmark 原始传感器通道）。

    benchmark 只负责采集；拼图/检测/编码等视觉预处理是 method 的职责。
    frame_id/timestamp_s 在 V1 采集时序（六次顺序采样）下可为 None。
    """
    view_id: str
    view_name: str
    yaw_offset_deg: float
    fov_deg: float
    width: int
    height: int
    image_bytes: bytes = b""
    frame_id: Optional[int] = None
    timestamp_s: Optional[float] = None
    valid: bool = True


# 与 control/carla_interface.py capture_ugv_6view 的 yaws/labels 严格一致
UGV_VIEW_REGISTRY: Tuple[DirectionalView, ...] = (
    DirectionalView("V0", "FRONT_LEFT", -45.0, 90.0, 480, 270),
    DirectionalView("V1", "FRONT", 0.0, 90.0, 480, 270),
    DirectionalView("V2", "FRONT_RIGHT", 45.0, 90.0, 480, 270),
    DirectionalView("V3", "BACK_LEFT", -135.0, 90.0, 480, 270),
    DirectionalView("V4", "BACK", 180.0, 90.0, 480, 270),
    DirectionalView("V5", "BACK_RIGHT", 135.0, 90.0, 480, 270),
)


@dataclass(frozen=True)
class UAVFrame:
    """One raw UAV RGB capture atomically bound to its own sensor metadata."""
    image_bytes: bytes
    frame_id: Optional[int]
    timestamp_s: Optional[float]
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    fov_deg: float
    width: int
    height: int
    valid: bool = True
    pose_source: str = "sensor_transform"


@dataclass
class LiteObservation:
    episode_id: str
    step: int
    max_steps: int
    role: str
    # 语义（r7 起）：UAV 为 benchmark 原始单图；UGV 为 deprecated/optional
    # 方法可自行生成证据板），benchmark 不再对 UGV 填 2x3 collage。
    agent_image: bytes = b""
    # UGV 六个固定相机身份的独立原始视图（UGV 观测时由 runner 填充）
    ugv_views: Tuple[DirectionalView, ...] = ()
    # 工具方法的独立 TOP/FRONT/SIDE 目标参考图（用于参考图匹配；
    # legacy VLM 方法继续用 target_reference_image 合成图）
    target_reference_views: dict = field(default_factory=dict)
    topdown_map: bytes = b""
    target_reference_image: bytes = b""
    teammate_reference_image: bytes = b""   # UAV 可见的 UGV 10m 纯俯视参考图
    agent_position: Tuple[float, float, float] = (0, 0, 0)
    observation_mode: str = "UAV_AERIAL"
    target_description: str = ""
    candidates: List[CandidateInfo] = field(default_factory=list)
    active_candidate: Optional[CandidateInfo] = None
    available_waypoints: List[WaypointInfo] = field(default_factory=list)
    available_actions: List[str] = field(default_factory=list)
    recent_messages: List[MessageInfo] = field(default_factory=list)
    latest_stop_context: Optional[StopObservationContext] = None
    annotation_count: int = 0
    max_annotations: int = 999  # 2026-07-23: UAV 标注不设上限（原为 5）
    confirm_count: int = 0
    max_confirms: int = 3
    confirm_budget: int = 3
    region_boundary: Tuple[float, float, float, float] = (0, 0, 0, 0)
    uav_frozen: bool = False  # True while UAV is frozen waiting for UGV arrival message

@dataclass
class LiteAction:
    role: str
    action: str
    waypoint_id: Optional[str] = None
    grid_cell: Optional[str] = None
    candidate_id: Optional[str] = None
    confidence: float = 0.5
    visual_description: str = ""
    evidence: str = ""
    reason: str = ""
    selected_objective_id: Optional[str] = None
    message: str = ""   # 用于 UAV 向 UGV 发送引导消息
    # GSAM 两实验（2026-10-10 设计 §9.1）：T1 点动作合同。
    # geometry_type='point' 时：position_xy 必填，grid_cell 可空、polygon 不适用；
    # 核验 MLLM 只选择 proposal ID，不填世界坐标（坐标由工具投影产生）。
    geometry_type: Optional[str] = None
    position_xy: Optional[Tuple[float, float]] = None
    source_frame_id: Optional[int] = None
    source_image_sha256: str = ""
    # GSAM 主调用后备搜索动作（设计 §4.1：工具失败后不被迫 WAIT）
    fallback_waypoint_id: Optional[str] = None
    localization_source: str = ""

# ============================================================
# Episode State
# ============================================================
class EpisodeState:
    def __init__(self, config, episode_plan):
        self.config = config
        self.episode_id = episode_plan.get("episode_id", "lite_easy_000001")
        self.step = 0
        self.max_steps = config["eval"]["max_steps"]
        self.vlm_calls = 0
        self.vlm_http_attempts = 0
        self.uav_vlm_http_attempts = 0
        self.ugv_vlm_http_attempts = 0
        self.max_vlm_calls = config["eval"]["max_vlm_calls"]
        self.uav_vlm_calls = 0
        self.ugv_vlm_calls = 0
        self.terminal = False
        self.success = False
        self.failure_stage = None
        
        # Actor readback and movement are mutable runtime state; the plan is
        # the frozen benchmark/SPL contract. CARLA meter<->centimeter float
        # roundtrips must not rewrite its original target or initial poses.
        self.uav_pos = copy.deepcopy(episode_plan["start"]["uav"])
        self.ugv_pos = copy.deepcopy(episode_plan["start"]["ugv"])
        self.target = copy.deepcopy(episode_plan.get("target_oracle", {}))
        self.target_description = episode_plan["target_task"]["target_description"]
        
        self.candidates: List[CandidateInfo] = []
        self.rejected_regions: List[Tuple[float, float]] = []
        self.annotation_count = 0
        self.confirm_count = 0
        self.confirm_budget = config["ugv"]["confirm_budget"]
        self.false_confirm_count = 0
        self.false_reject_count = 0
        
        self.ugv_observation_mode = "UGV_DRIVING"
        self.ugv_at_approach = False
        self.ugv_stopped = False

        self.grid_hit = False
        self.gt_visible_in_stop6 = False
        self.ne_distance = None  # overwritten with terminal XY error at episode finalization
        self.confirm_distance_m = None  # independent diagnostic at actual CONFIRM
        self.gva_sampling_valid = True
        self.metric_sampling_errors = []
        self.osr = False  # Oracle Success: consumed legal Confirm opportunity (v6)
        # GVA / C1 diagnostic contract: this is a pure post-hoc oracle metric.
        # It must become true whenever ANY UGV decision-state satisfies the same
        # confirmability predicate as SuccessChecker, regardless of the model's
        # chosen action.  It is never exposed to the agent.
        self.oracle_verifiable_reached = False
        self.first_oracle_verifiable_step = None
        # Exact step of a captured predecision six-view opportunity (no +1 inference).
        self.first_predecision_oracle_verifiable_step = None
        self.min_ugv_target_distance_m = None
        self.min_visible_target_distance_m = None

        # UAV freeze/wake coordination: after UAV annotates a candidate it freezes
        # (no VLM calls, waits) until the UGV sends any message (e.g. "arrived").
        self.uav_frozen = False
        # 冻结起始 step（用于框架层冻结超时兜底，杜绝无限冻结）
        self.uav_frozen_since_step = None

        # STOP observation context
        self.latest_stop_context: Optional[StopObservationContext] = None

        # CCG counters
        self.candidate_hit_count = 0

        # P0-6: Split annotation counts
        self.annotation_attempt_count = 0  # Successful creates (suppressed doesn't consume per user decision)
        self.candidate_created_count = 0  # Same as annotation_attempt_count
        self.annotation_action_outputs = 0  # Total ANNOTATE_CANDIDATE outputs from VLM
        self.invalid_action_count = 0

        # Task-level memory (simplified: last N steps as text, no compression)
        self.uav_memory = []  # List of {"step":N, "action":"...", "result":"...", "sim_time":T}
        self.ugv_memory = []  # Same format
        self.uav_visited_waypoints = []  # IDs of visited waypoints
        self.ugv_visited_waypoints = []

        # Memory config (simplified: just max_entries for truncation)
        _mem_cfg = config.get("memory", {})
        self.memory_max_entries = _mem_cfg.get("max_entries", 5)

        # === Discrete event simulation time ===
        self.uav_sim_time = 0.0   # UAV independent clock (seconds)
        self.ugv_sim_time = 0.0   # UGV independent clock (seconds)
        self.max_sim_time = config.get("eval", {}).get("max_sim_time_s", 600.0)  # 10 min default
        self.time_model = get_time_model_config(config)
        self.uav_speed_mps = config.get("uav", {}).get("speed_mps", 5.0)
        self.ugv_speed_mps = config.get("ugv", {}).get("speed_mps", 3.0)

        # Messages & trajectory
        self.messages: List[MessageInfo] = []
        self.uav_trajectory: List[Tuple[float, float]] = []
        self.ugv_trajectory: List[Tuple[float, float]] = []
        self.ugv_route_history: List[Tuple[str, float]] = []
        # 锚点回退（BACK_TO_PREVIOUS_ANCHOR）运行时状态（2026-09-04）
        self.anchor_visit_history: List[str] = []
        self.consecutive_backtracks = 0

        # Spec v2 §4.3 counters (runner increments these)
        self.uav_action_count = 0
        self.ugv_action_count = 0
        self.stop_observe_count = 0
        self.reject_count = 0
        self.spl = 0.0

        # ACG/GVA metrics (2026-07-21, metrics_schema_version=3)
        self.candidate_acg_scores = []  # list of {candidate_id, distance_m, scale_m, acg_score}
        self.gva_events = []  # list of verification events
        self.infra_error_count = 0

        # Frozen road graph metadata
        self.road_graph_id = episode_plan.get("road_graph_id", "")
        self.road_graph_asset_sha256 = episode_plan.get("road_graph_asset_sha256", "")

        region = episode_plan.get("region", {})
        self.region = region

    def add_uav_memory(self, action, params=None, result="", outcome=""):
        """记录 UAV 动作到记忆"""
        self.uav_memory.append({
            "step": self.step,
            "sim_time": round(self.uav_sim_time, 1),
            "role": "UAV",
            "action": action,
            "params": params or {},
            "result": result,
            "outcome": outcome,
        })

    def add_ugv_memory(self, action, params=None, result="", outcome=""):
        """记录 UGV 动作到记忆"""
        self.ugv_memory.append({
            "step": self.step,
            "sim_time": round(self.ugv_sim_time, 1),
            "role": "UGV",
            "action": action,
            "params": params or {},
            "result": result,
            "outcome": outcome,
        })

    def get_uav_memory_lines(self):
        """获取最近 N 条 UAV 记忆文本行（简单截断，无压缩）"""
        entries = [MemoryEntry(**m) for m in self.uav_memory]
        kept = entries[-self.memory_max_entries:]
        return [_format_memory_entry(e) for e in kept]

    def get_ugv_memory_lines(self):
        """获取最近 N 条 UGV 记忆文本行（简单截断，无压缩）"""
        entries = [MemoryEntry(**m) for m in self.ugv_memory]
        kept = entries[-self.memory_max_entries:]
        return [_format_memory_entry(e) for e in kept]

    def add_message(self, sender, content, sim_time=None):
        """添加消息，带仿真时间戳"""
        if sim_time is None:
            sim_time = max(self.uav_sim_time, self.ugv_sim_time)
        self.messages.append(MessageInfo(sender, content, sim_time))
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def get_messages_for(self, role):
        """获取对指定角色可见的消息（sim_time <= 该角色的时钟）"""
        my_time = self.uav_sim_time if role == "UAV" else self.ugv_sim_time
        return [m for m in self.messages if m.timestamp_step <= my_time]

    def has_active_candidate(self):
        return any(c.status in ("unverified", "assigned_to_ugv", "visited") for c in self.candidates)

    def get_active_candidate(self):
        for c in self.candidates:
            if c.status in ("unverified", "assigned_to_ugv", "visited"):
                return c
        return None

# ============================================================
# Failure Stages
# ============================================================
class FailureStage:
    # Infrastructure / budget failures
    STEP_BUDGET_EXHAUSTED = "STEP_BUDGET_EXHAUSTED"
    VLM_BUDGET_EXHAUSTED = "VLM_BUDGET_EXHAUSTED"
    CANDIDATE_BUDGET_EXHAUSTED = "CANDIDATE_BUDGET_EXHAUSTED"
    SIM_TIME_EXHAUSTED = "SIM_TIME_EXHAUSTED"
    RUNTIME_EXCEPTION = "RUNTIME_EXCEPTION"
    # Outcome of final CONFIRM
    FALSE_CONFIRM = "FALSE_CONFIRM"
    # REJECT is no longer a terminal failure stage, only an event metric (false_reject_count)
    # FALSE_REJECT kept here as a string label for backward compatibility in event logs, not terminal.
    FALSE_REJECT = "FALSE_REJECT"



def get_time_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取时间模型配置，提供默认值"""
    return config.get("time_model", {
        "version": "lite_time_v2",
        "stop_observe_s": 10.0,
        "wait_s": 10.0,
        "annotate_candidate_s": 5.0,
        "confirm_s": 5.0,
        "reject_s": 5.0,
        "move_observation_s": 1.0,
        "normal_turn_s": 2.0,
        "u_turn_s": 4.0,
    })


def compute_action_sim_time(action, role, prev_pos, new_pos, speed_mps,
                            road_distance_m=None, normal_turns=0, u_turns=0,
                            observation_time_s=1.0, time_model=None):
    """计算动作的仿真时间消耗（秒）

    简化虚拟移动时间模型：
    - UAV: 移动=水平直线距离/5m/s + 1s 观察时间
    - UGV: 移动=路网距离/5m/s + 2s*普通转弯 + 4s*掉头 + 1s 观察时间
    """
    tm = time_model or {}
    stop_observe_s = tm.get("stop_observe_s", 10.0)
    wait_s = tm.get("wait_s", 10.0)
    annotate_s = tm.get("annotate_candidate_s", 5.0)
    confirm_s = tm.get("confirm_s", 5.0)
    reject_s = tm.get("reject_s", 5.0)
    normal_turn_s = tm.get("normal_turn_s", 2.0)
    u_turn_s = tm.get("u_turn_s", 4.0)

    turn_time = normal_turn_s * normal_turns + u_turn_s * u_turns

    if action in ("MOVE_TO_UAV_WAYPOINT",):
        if prev_pos and new_pos:
            dx = new_pos[0] - prev_pos[0]
            dy = new_pos[1] - prev_pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            return max(1.0, dist / speed_mps + observation_time_s)
        return 5.0  # fallback: min observation time

    if action in ("MOVE_TO_ROAD_WAYPOINT", "GO_TO_CANDIDATE",
                  "FORWARD_1", "FORWARD_2", "TURN_LEFT", "TURN_RIGHT", "TURN_STRAIGHT", "REVERSE"):
        # 2026-09-04 动作空间 v2：操纵式移动动作与路点移动同用路网距离时间模型
        if road_distance_m is not None and road_distance_m > 0:
            return max(1.0, road_distance_m / speed_mps + turn_time + observation_time_s)
        if prev_pos and new_pos:
            dx = new_pos[0] - prev_pos[0]
            dy = new_pos[1] - prev_pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            return max(1.0, dist / speed_mps + turn_time + observation_time_s)
        return 5.0  # fallback

    if action == "STOP_OBSERVE":
        return stop_observe_s

    if action == "ANNOTATE_CANDIDATE":
        return annotate_s

    if action in ("CONFIRM",):
        return confirm_s

    if action in ("REJECT",):
        return reject_s

    if action == "WAIT":
        return wait_s

    return 5.0  # default
