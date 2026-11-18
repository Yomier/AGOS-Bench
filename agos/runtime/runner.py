"""runner.py — Joint/UAV-only/UGV-only Runner + Message Bus + Oracle Agent"""
import sys, os, json, math, time, base64, logging, random, hashlib

try:
    import carla  # oracle 可达性检查用 carla.Transform 瞬移 UGV 实体
except ImportError:  # 本地无 CARLA SDK（静态检查/单测环境）
    carla = None

from agos.lite_core import *
from agos.lite_core import compute_action_sim_time
from agos.agents.agents import validate_action
from agos.agents.agents import VLMCancelledError, request_cancel, clear_cancel  # R2-P1-07
from agos.methods import registry as method_registry  # r7: 三 primary 方法注册表
from agos.methods.base import MethodServices
from agos.methods.common.legacy_sixview_collage import render_legacy_sixview_collage
from agos.runtime.observation_maps import ObservationMapProtocol
from agos.runtime.paper_audit import http_accounting_result, model_call_accounting_result
from benchmark.parallel.outcome import classify_terminal, is_normal_completion
from agos.methods.vln_os_common.contracts import public_transition
from agos.perception.perception import *
from agos.maps.waypoints import *
from agos.roadnet import (
    RoadGraphStore, RoadLocator, RoadRouter, RoadOptionProvider,
    CandidateApproachPlanner, UGVRoadCatalogAdapter, ApproachAdapter, RouteFailure,
    RoadGraphRegistry, AnchorCatalog, compute_service_nodes, compute_confirmable_service_nodes,
    compute_point_service_nodes, point_diagnostic_square,
)
# B2（2026-09-17 审计）：删除死导入 Metrics（runner 全文无使用），仅保留 SuccessChecker
from agos.eval.eval import SuccessChecker
from agos.eval.metric_contract import (metric_protocol, finalize_navigation_error, ACG_POINT_HALF_SIZE_M,
    aggregate_verification_actions)
from agos.benchmark_region import REGION_POLICY_REVISION, enforce_uav_move, inside_task_region
from agos.runtime.evaluation_observations import (actor_xy, probe_ugv_observation,
    record_ugv_observation, probe_uav_observation)
from agos.benchmark_protocol import confirm_distance_m, success_protocol, validate_plan_success_protocol
from agos.eval.contamination import ContaminationMonitor, RuntimeContaminationMonitor
from agos.eval.metric_primitives import (
    compute_acg_score, is_confirmable_observation,
    compute_acg_episode_v3, compute_gva_episode_v3,
    compute_ase_episode, compute_acov_at_k,
)
from agos.perception.perception import compute_grid_cell_from_world

logger = logging.getLogger("LiteV1.Runner")
from agos.methods.common.uav_annotation import UAV_ANNOTATION_REVISION, RETIRED_UAV_GRID_OPTIONS


def _fs_tag(value, fallback="independent"):
    """Filesystem-safe tag for attempt-dir image names.

    Proposal-bound candidate ids look like ``B8c279e50/V3/P3``; embedding one
    raw into a flat filename joins a nested path whose parent directories are
    never created, so the very first ``open(..., "wb")`` crashed the episode
    (internal validation, 2026-10).
    """
    tag = str(value or "").replace(os.sep, "_").replace("/", "_").replace("\\", "_").strip("_")
    return tag or fallback


def _terminal_failure_reason(state, total_sim, guard_abort_info):
    """Normalize the terminal reason before deriving outcome/completed fields."""
    if state.success:
        return "SUCCESS", "任务成功：UGV 在阈值距离内 CONFIRM 真实目标。"
    fs = state.failure_stage or "UNKNOWN"
    if fs == "FALSE_REJECT":
        return "UGV_REJECTED_TRUE_TARGET", "UGV 到达覆盖真实目标的 candidate 附近并执行验证，但最终输出 REJECT（目标实际可见且距离在阈值内）。"
    if fs == "FALSE_CONFIRM":
        return "UGV_CONFIRMED_WRONG_OBJECT", "UGV 对非目标对象执行了 CONFIRM。"
    if fs == "VLM_BUDGET_EXHAUSTED":
        return "GUARD_STOP", "VLM 调用预算耗尽（%d/%d）。" % (state.vlm_calls, state.max_vlm_calls)
    if fs == "SIM_TIME_EXHAUSTED":
        return "GUARD_STOP", "仿真时间预算耗尽（%.0fs > %.0fs）。" % (total_sim, state.max_sim_time)
    if fs == "STEP_BUDGET_EXHAUSTED":
        if guard_abort_info:
            kind, cid = guard_abort_info
            if kind == "UGV_PATROL_LOOP":
                return "UGV_PATROL_LOOP_ABORTED", "UGV 在 candidate %s 周围连续移动超过监护阈值仍未执行验证（CONFIRM/REJECT），监护逻辑强制中止该 candidate；episode 最终因步数预算结束。" % cid
            if kind == "UGV_NAV_STALLED":
                return "UGV_NAV_STALLED", "UGV 前往 candidate %s 的导航长期停滞（assigned_to_ugv 状态超时），监护逻辑强制中止该 candidate；episode 最终因步数预算结束。" % cid
        return "BUDGET_EXHAUSTED", "步数预算耗尽（%d/%d）。" % (state.step, state.max_steps)
    if fs == "EPISODE_WALL_TIMEOUT":
        return "EPISODE_WALL_TIMEOUT", "墙钟保护中断，任务未正常完成。"
    if fs == "TARGET_SPAWN_FAILED":
        return "RUNTIME_EXCEPTION", "目标 actor 生成失败。"
    return "UNKNOWN_FAILURE", "未归一化失败原因：%s" % fs


def _terminal_outcome_fields(state, reason):
    """Derive completion only from the scientific terminal, never finalization."""
    candidate = dict(success=state.success, completed=True,
                     outcome_class="success" if state.success else "real_fail",
                     failure_stage=state.failure_stage, failure_reason=reason,
                     steps=state.step)
    terminal = classify_terminal(candidate, state.max_steps)
    if terminal.startswith("normal_"):
        return candidate["outcome_class"], True
    if terminal == "abnormal_guard_stop":
        return "guard_stop", False
    return "infra_error", False


def _record_terminal_finalization_fault(state, result, component, error, phase):
    """Record post-terminal quality/cleanup separately without unlocking it."""
    if not is_normal_completion(result, state.max_steps):
        return False
    event = {"component": component, "phase": phase,
             "error": f"{type(error).__name__}: {error}"[:500]}
    result.setdefault("post_terminal_faults", []).append(event)
    result["cleanup_state" if phase == "cleanup" else "result_quality"] = "failed"
    if not hasattr(state, "paper_fault_events"):
        state.paper_fault_events = []
    state.paper_fault_events.append({**event, "kind": "execution_lifecycle" if phase == "cleanup" else "implementation",
                                     "status": "post_terminal_failure", "fallback": False})
    logger.warning("Post-terminal %s failed (%s): %s", phase, component, error)
    return True


def _note_nadir_lifecycle(state, action, shadow=False):
    """Preserve no-tool primary suspension; shadows continue aerial search."""
    if getattr(action, 'localization_source', '') == 'uav_nadir_point' and not shadow:
        state.uav_frozen = True
        state.uav_frozen_since_step = state.step


def _load_observation_global_rgb(carla_iface, map_region, map_root, episode_id, protocol):
    """Mandatory exclusion precedes both capture and reading any old cache."""
    if protocol.excludes_fullcity:
        return None
    path = os.path.join(map_root, "global_rgb_topdown.png")
    if not os.path.exists(path):
        try:
            rgb_bytes, cap_meta = carla_iface.capture_global_topdown(map_region)
            if rgb_bytes:
                with open(path, "wb") as out:
                    out.write(rgb_bytes)
                with open(os.path.join(map_root, "global_rgb_capture_meta.json"), "w", encoding="utf-8") as out:
                    json.dump(cap_meta, out, indent=2, ensure_ascii=False)
                logger.info("[%s] Global RGB topdown captured (%d bytes)", episode_id, len(rgb_bytes))
        except Exception as exc:
            logger.warning("Failed to capture global topdown: %s", exc)
    if os.path.exists(path):
        with open(path, "rb") as source:
            return source.read()
    try:
        return carla_iface.capture_global_topdown(map_region, width=1024, height=1024, fov=90)[0]
    except Exception as exc:
        logger.warning("Failed to capture global RGB fallback: %s", exc)
        return None




def _capture_observation_uav(carla_iface, protocol, event_time):
    # Preserve the actual sensor packet for private ASE scoring in every method.
    frame = carla_iface.capture_uav_frame()
    protocol.last_frame = frame
    if protocol.online_map is not None:
        protocol.update_uav(frame, event_time)
    return frame.image_bytes


def _attach_uav_observation_frame(observation, frame, image_bytes):
    """Bind only the packet returned by capture; never reconstruct sensor facts."""
    if frame is None or frame.image_bytes != image_bytes:
        raise RuntimeError('METHOD_CONTRACT_FAILED: UAV_SOURCE_FRAME_MISMATCH')
    observation.uav_frame = frame
    observation.agent_position = (frame.x, frame.y, frame.z)
    observation._uav_source_frame = frame




def _capture_metric_ugv_views(carla_iface, config):
    """Use the same configured cameras for RGB collection and Oracle scoring."""
    cfg = config.get('ugv', {}) or {}
    return carla_iface.capture_ugv_views(
        yaws=cfg.get('stop_observe_yaws', [-45, 0, 45, -135, 180, 135]),
        panel_w=cfg.get('stop_panel_width', 480), panel_h=cfg.get('stop_panel_height', 270),
        fov=cfg.get('stop_camera_fov_deg', 90))


def _is_coop_mode(agent_mode: str) -> bool:
    """AGOS 事件协同搜索（cooperative_search）：四臂共用方法入口。"""
    return str(agent_mode or "").lower().strip() == "cooperative_search"


def _method_start_episode(agent_mode: str, method, context, logger=None):
    """Initialize a method session and preserve cooperative-search failure classification."""
    log = logger if logger is not None else globals()["logger"]
    try:
        method.start_episode(context)
    except Exception as exc:
        if _is_coop_mode(agent_mode):
            raise RuntimeError("METHOD_START_FAILED") from exc
        log.warning("method.start_episode failed: %s", exc)


def _method_decide(agent_mode: str, agent, obs, task_attempt: int = 1,
                   capture_black_count: int = 0):
    """运行期方法异常分级（2026-10-11 batch1 停发复盘修正；
    2026-10-12 batch2 停发复盘再修正）。

    - VLMCancelledError 保持原语义（Runner 墙钟超时合同）；
    - 具名合同违例（sixview_input_contract 等设计内 fail-closed）与
      同任务第 2 次方法异常（系统性信号）→ METHOD_CONTRACT_FAILED
      （implementation，整批停发）；
    - batch2 复盘（02:01 停发）：采集层传感器帧超时产生黑帧
      （valid=False），被 sixview 合同 fail-closed 拒绝——属环境态
      瞬态故障而非实现缺陷。本步采集确有黑帧且首次尝试的具名违例
      → METHOD_RUNTIME_EXCEPTION（infra 重试，复现即升级）；
    - 其余未具名的一次性运行期异常 → METHOD_RUNTIME_EXCEPTION
      （infra 可重试；复现即升级）。旧模式异常行为不变。
    """
    try:
        return agent.decide(obs)
    except VLMCancelledError:
        raise
    except Exception as exc:
        if not _is_coop_mode(agent_mode):
            raise
        named = (isinstance(exc, ValueError) and "sixview_input_contract" in str(exc)) or any(
            tag in str(exc) for tag in ("GS6_IDENTITY_PROTOCOL", "GS6_EVIDENCE_BUDGET", "GS6_PROPOSAL_CONTRACT"))


        if isinstance(exc, RuntimeError) and "METHOD_CONTRACT_FAILED" in str(exc):
            raise
        # batch2 复盘：采集黑帧（传感器帧超时）诱发的具名违例 = 环境态瞬态
        # 故障，首次尝试降级 infra 重试；无黑帧的具名违例（真 misbind）与
        # 第 2 次复现（系统性信号）保持整批停发。
        capture_flake = (named and int(capture_black_count or 0) > 0
                         and int(task_attempt) < 2)
        if (named and not capture_flake) or int(task_attempt) >= 2:
            raise RuntimeError("METHOD_CONTRACT_FAILED") from exc
        raise RuntimeError("METHOD_RUNTIME_EXCEPTION") from exc


def _role_uses_vlm(agent_mode, method, role):
    return agent_mode != 'menu_oracle' and bool(method.role_capabilities(role.upper())['uses_vlm'])


def _bind_uav_frame(agent_mode, gsam_on):
    return agent_mode == 'cooperative_search' or bool(gsam_on)


def _collect_method_trace(agent_mode):
    return agent_mode == 'cooperative_search'


def _agent_name_for_mode(agent_mode):
    return 'CooperativeSearchMethod' if agent_mode == 'cooperative_search' else 'MenuOracleMethod'

def _rebind_navigation_certificate(plan_certificate, live_binding, max_dz_m: float = 0.30):
    """GSAM 两实验（2026-10-10）：SPL 部署证书的物理沉降噪声重绑。

    重放部署与生成部署的 actor settle z 存在毫米级物理噪声（实测 1-18mm，
    xy 全同），全有或全无的 canonical_hash 判定会随机拒绝合法重放。本函数按
    设计 §9.2"重新生成合法证书"路径处置，绝不关闭校验、不编辑 hash 伪装：
      1. 深比较 plan 证书 binding 与 live binding：差异必须仅限
         target_deployment/ugv_deployment/distractor_deployment[*] 的 location[2]
         （z），且 |dz| <= max_dz_m；location 的 x/y 逐位相同；其余一切字段相等。
         z 阈值 0.30m 依据：SPL 度量 distance_metric='unrounded_fine_path_xy_m'
         （纯 xy）且 goal/成功距离判定亦为 xy——z 不参与度量语义；0.30m 覆盖
         物理沉降噪声（18mm 实测上限的 16 倍冗余）且远小于多层路网的层间高差
         （数米），不存在"同 xy 不同层"的混叠窗口；
      2. SPL 度量字段（distance_m/goal/start/actions/steps/fine_path/
         settled_states/goal_pose_kind/version）逐字段原样保留；
      3. 以 build_certificate 同一算法重算 binding_sha256/certificate_sha256；
      4. 返回 (新证书, 最大|dz| 描述)；任何越界差异返回 None（调用方按原错误抛出）。
    本函数位于 runner（不在 SPL 绑定的源码 hash 清单内），不改变绑定文件字节。
    """
    if not isinstance(plan_certificate, dict) or not isinstance(live_binding, dict):
        return None
    old_binding = plan_certificate.get("binding")
    if not isinstance(old_binding, dict):
        return None
    max_dz = [0.0]

    def walk(a, b, in_z):
        """递归校验：仅 deployment 子树内允许物理 spawn 噪声——location[2]（z）
        |dz|<=0.30m 与 rotation 三分量 |dθ|<=0.01rad（≈0.57°，实测 2.8e-5rad 量级）；
        location 的 x/y 逐位相同；其余一切字段（含 deployment 之外）必须相等。
        rotation 不参与 SPL 度量/goal 判定/可见性 anchor，仅是部署快照噪声。"""
        if isinstance(a, dict) and isinstance(b, dict):
            keys = set(a) | set(b)
            if keys - set(a) or keys - set(b):
                return False  # 键集合不同
            for k in keys:
                if k == "location" and isinstance(a.get(k), list) and len(a.get(k)) == 3:
                    la, lb = a[k], b[k]
                    if la[0] != lb[0] or la[1] != lb[1]:
                        return False  # x/y 必须逐位相同
                    dz = abs(float(la[2]) - float(lb[2]))
                    max_dz[0] = max(max_dz[0], dz)
                    if dz > max_dz_m:
                        return False
                    continue
                if k == "rotation" and isinstance(a.get(k), list) and len(a.get(k)) == 3:
                    ra, rb = a[k], b[k]
                    if any(abs(float(x) - float(y)) > 0.01 for x, y in zip(ra, rb)):
                        return False
                    continue
                if k in ("target_deployment", "ugv_deployment", "distractor_deployment"):
                    if not walk(a[k], b[k], True):
                        return False
                    continue
                if not walk(a[k], b[k], False):
                    return False
            return True
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return all(walk(x, y, in_z) for x, y in zip(a, b))
        return a == b

    if not walk(old_binding, live_binding, False):
        return None
    from agos.eval.navigation_oracle import canonical_hash
    new_cert = {k: v for k, v in plan_certificate.items()
                if k not in ("certificate_sha256", "binding", "binding_sha256")}
    new_cert["binding"] = live_binding
    new_cert["binding_sha256"] = canonical_hash(live_binding)
    new_cert["rebind_audit"] = {
        "reason": "physical_settle_z_noise", "max_abs_dz_m": round(max_dz[0], 6),
        "rebound_at": time.time(),
        "spl_metric_fields_untouched": True}
    new_cert["certificate_sha256"] = canonical_hash(new_cert)
    return new_cert, "%.4fm" % max_dz[0]


def _resolve_nav_stall_max_steps(config) -> int:
    """B4（2026-09-17 审计）：NAV stall guard 步数阈值解析。

    优先读 nav 块（审计口径），回退 ugv 块（R14-R20 历史 config 兼容，
    此前实现一直挂在 ugv 下），默认 20=旧行为。
    """
    _cfg = config if isinstance(config, dict) else {}
    _nav_cfg = _cfg.get("nav", {}) or {}
    _ugv_cfg = _cfg.get("ugv", {}) or {}
    return int(_nav_cfg.get("nav_stall_guard_max_steps",
                            _ugv_cfg.get("nav_stall_guard_max_steps", 20)))


_RUN_META_FILES = ('agos/runtime/runner.py', 'agos/methods/cooperative_search/method.py',
                   'agos/methods/cooperative_search/controller.py', 'agos/methods/cooperative_search/policy.py',
                   'agos/eval/navigation_oracle.py', 'agos/methods/menu_oracle/method.py')
_RUN_META_CODE_MD5_CACHE = None
_BEHAVIOR_TREE_MD5_CACHE = None

# G3（发车前外审 20260907）：prompt 结构版本随 Stage0 修复升版，
# i0_gate/机制报告可据此区分 phase-g2（修复前）与 phase-g2-v2（契约修复后）。
_PROMPT_SCHEMA_VERSION = "uav-point-v2-region-v1-20261013"


def _behavior_tree_md5():
    """agos 行为树代码组合哈希（Stage0 G3 新增）。

    遍历 <repo>/agos 下全部 .py（排除 tests、scripts、__pycache__
    与 *.bak* 备份目录——均不参与评测进程 import），按 sorted(relpath:
    file_md5) 拼接后取整体 md5；进程内缓存一次。
    覆盖 _RUN_META_FILES 之外的任何行为树文件改动，供 resume 代码态门
    与外审追溯（c1_run.resume 门与本字段同源比对）。
    """
    global _BEHAVIOR_TREE_MD5_CACHE
    if _BEHAVIOR_TREE_MD5_CACHE is not None:
        return _BEHAVIOR_TREE_MD5_CACHE
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    pkg_root = os.path.join(repo_root, "agos")
    parts = []
    for dirpath, dirnames, filenames in os.walk(pkg_root):
        dirnames[:] = [d for d in sorted(dirnames)
                       if d not in ("__pycache__", "tests", "scripts")
                       and not d.startswith(".")
                       and ".bak" not in d]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, repo_root).replace(os.sep, "/")
            h = hashlib.md5()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            parts.append("%s:%s" % (rel, h.hexdigest()))
    _BEHAVIOR_TREE_MD5_CACHE = hashlib.md5(
        "\n".join(sorted(parts)).encode("utf-8")).hexdigest()
    return _BEHAVIOR_TREE_MD5_CACHE


def task_seed(dataset_id: str, map_key: str, episode_id: str,
                   base_seed: int = 20260725) -> int:
    identity = "%s/%s/%s" % (dataset_id, map_key, episode_id)
    h = hashlib.sha256(json.dumps([int(base_seed), "fixed-ugv-rng", identity],
                                  sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=True).encode()).hexdigest()
    return int(h, 16)


def _build_run_meta(config: dict, agent_mode: str = None) -> dict:
    """构建 result.json 顶层 run_meta。

    code_files_md5（文件读盘哈希，进程内不会变）只在进程首次调用时计算一次
    并缓存；run_id / agent_mode / config_sha256 / timestamp 每次调用现算
    （均为廉价操作，且 run_id 需即时反映环境变量变化）。
    """
    global _RUN_META_CODE_MD5_CACHE
    if _RUN_META_CODE_MD5_CACHE is None:
        # runner.py 位于 <repo_root>/agos/runtime/，向上两级定位仓库根
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        md5s = {}
        for rel in _RUN_META_FILES:
            path = os.path.join(repo_root, *rel.split("/"))
            if os.path.isfile(path):
                h = hashlib.md5()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                md5s[rel] = h.hexdigest()
            else:
                md5s[rel] = "MISSING"
        _RUN_META_CODE_MD5_CACHE = md5s
    try:
        config_sha256 = hashlib.sha256(json.dumps(
            config, sort_keys=True, ensure_ascii=False, default=str
        ).encode("utf-8")).hexdigest()
    except Exception:
        config_sha256 = None
    dataset_id = (config.get("benchmark") or {}).get("dataset_id")
    policy_seed = task_seed(str(dataset_id or ""), os.environ.get("AGOS_EPISODE_MAP_KEY", ""),
        os.environ.get("AGOS_EPISODE_ID", ""),
        int((config.get("episode") or {}).get("seed_base", 20260725) or 20260725))
    meta = {
        "evaluation_binding": {
            "plan_sha256": os.environ.get("AGOS_EPISODE_PLAN_SHA256"),
            "policy_seed": policy_seed,
            "release_sha256": os.environ.get("AGOS_PAPER_RELEASE_SHA256"),
            "dataset_id": dataset_id,
            "group_id": (config.get("paper_experiment") or {}).get("group_id"),
        },
        "run_id": os.environ.get("AGOS_RUN_ID"),
        "agent_mode": agent_mode,
        "ugv_action_space": ("sparse_nav_v1" if _sparse_nav_enabled(config) else "MANEUVER_V2"),
        "config_sha256": config_sha256,
        # Stable source-file provenance. Unlike config_sha256 this is captured
        # before host/rpc/airsim slot overrides in episode_job.
        "source_config_sha256": os.environ.get("AGOS_SOURCE_CONFIG_SHA256"),
        "source_config_basename": os.environ.get("AGOS_SOURCE_CONFIG_BASENAME"),
        "code_files_md5": dict(_RUN_META_CODE_MD5_CACHE),
        "behavior_tree_md5": _behavior_tree_md5(),
        "prompt_schema_version": _PROMPT_SCHEMA_VERSION,
        "metric_protocol": metric_protocol(),
        "region_policy_revision": REGION_POLICY_REVISION,
        "vllm_model": ((config.get("vlm") or {}).get("model")),
        "timestamp": time.time(),
    }
    return meta


# ---------------------------------------------------------------------------
# I1 帧转储钩子（Phase-G Step 6）：默认关（config 无 i1_frame_dump 键零开销）。
# ---------------------------------------------------------------------------
def _save_ugv_step_views(views, img_dir, step) -> int:
    """R28-V0 插桩：单个 UGV 决策步的六个原始视图平铺落盘。

    文件名 ugv_<view_id>_s<step:03d>.png（原字节直写，无重编码）；
    空字节视图跳过。纯落盘，绝不进入决策路径。返回写入数。"""
    n = 0
    for v in (views or ()):
        vid = str(getattr(v, "view_id", "") or "view")
        data = getattr(v, "image_bytes", b"") or b""
        if not data:
            continue
        with open(os.path.join(img_dir, "ugv_%s_s%03d.png" % (vid, int(step))), "wb") as f:
            f.write(data)
        n += 1
    return n


def i1_dump_trigger(step, fvs, oracle_now, assigned_now, assigned_prev, cfg,
                    arrival_step=None):
    """I1 帧转储触发判定（纯函数，可单测）。

    返回触发原因字符串（优先级 oracle_window > candidate_arrival > neg_sample），
    未触发返回 None；cfg 为 None 或未 enabled 时恒 None。
      - oracle_window: oracle_now 为真，或 fvs 非 None 且
        0 <= step - fvs <= cfg["window"]（首次可见后的近场窗口）；
      - candidate_arrival: assigned 上升沿（assigned_now 且非 assigned_prev），
        或距到达沿 arrival_step 在 window 步内（沿所在步由调用方跨步维护）；
      - neg_sample: step % cfg["neg_every"] == 0。
    """
    if not cfg or not cfg.get("enabled"):
        return None
    window = int(cfg.get("window", 0) or 0)
    if oracle_now or (fvs is not None and 0 <= step - fvs <= window):
        return "oracle_window"
    if (assigned_now and not assigned_prev) or (
            arrival_step is not None and 0 <= step - arrival_step <= window):
        return "candidate_arrival"
    neg_every = int(cfg.get("neg_every", 0) or 0)
    if neg_every > 0 and step % neg_every == 0:
        return "neg_sample"
    return None


def _i1_dump_frames(output_dir, state, obs, trigger, target_actor=None):
    """I1 触发时落盘六视图帧 + meta（仅诊断用途，绝不影响主循环）。

    写 output_dir/i1_frames/step_%04d/{trigger.txt, meta.json, view_*.img}；
    视图原字节直写（后缀 .img 避免格式假设）。任何 I/O 失败只告警不抛。
    """
    step = int(getattr(state, "step", 0) or 0)
    try:
        step_dir = os.path.join(output_dir, "i1_frames", "step_%04d" % step)
        os.makedirs(step_dir, exist_ok=True)
        pos = getattr(state, "ugv_pos", {}) or {}
        gt_xy = None
        if target_actor is not None and getattr(target_actor, "is_alive", False):
            loc = target_actor.get_location()
            gt_xy = [round(float(loc.x), 2), round(float(loc.y), 2)]
        cand = getattr(obs, "active_candidate", None)
        board = str(getattr(obs, "dove_board_text", "") or "")
        meta = {
            "artifact_id": "%s:%04d" % (getattr(state, "episode_id", "?"), step),
            "step": step,
            "trigger": trigger,
            "ugv_pos": {"x": round(float(pos.get("x", 0.0) or 0.0), 2),
                        "y": round(float(pos.get("y", 0.0) or 0.0), 2),
                        "yaw": round(float(pos.get("yaw", 0.0) or 0.0), 2)},
            "gt_target_xy": gt_xy,  # target 存活时的真实位置（仅诊断落盘）
            "active_candidate": ({"id": getattr(cand, "candidate_id", ""),
                                  "status": getattr(cand, "status", "")}
                                 if cand is not None else None),
            "dove": board[:200] or None,
        }
        with open(os.path.join(step_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(step_dir, "trigger.txt"), "w", encoding="utf-8") as f:
            f.write(str(trigger))
        for v in (getattr(obs, "ugv_views", None) or []):
            if not getattr(v, "valid", False) or not getattr(v, "image_bytes", b""):
                continue
            vid = getattr(v, "view_id", "unknown")
            with open(os.path.join(step_dir, "view_%s.img" % vid), "wb") as f:
                f.write(v.image_bytes)
        return step_dir
    except Exception as e:
        logger.warning("i1 frame dump failed: %s", e)
        return None




def _public_point_publication_check(ledger, *, q_enabled, q_max_shadow,
                                    suppression_mode, min_shift_m):
    """Expose point admission status without exposing ledger/GT to a method."""
    def check(position_xy, step):
        shadow = bool(q_enabled and (ledger.get_active() is not None or ledger.shadow_count() > 0))
        return ledger.publication_status(
            position_xy, step, shadow=shadow, max_shadow=q_max_shadow,
            suppression_mode=suppression_mode, min_shift_m=min_shift_m)
    return check


def _safe_on_transition(agent, observation, proposed_action, executed_action, execution_result):
    """Deliver the execution receipt to the method."""
    hook = getattr(agent, "on_transition", None)
    if hook is None:
        return
    try:
        result = execution_result or {}
        if getattr(agent, 'public_input_only', False):
            result = public_transition(result)
        hook(observation, proposed_action, executed_action, result)
    except Exception as e:
        if getattr(agent, 'public_input_only', False):
            raise RuntimeError('METHOD_CONTRACT_FAILED: VLN_OS_TRANSITION_FAILED: '+str(e)) from e
        logger.warning("optional agent.on_transition failed: %s", e)


_COOP_RECEIPT_FIELDS = (
    "step", "action", "candidate_id", "grid_cell", "waypoint_id",
    "candidate_center_xy", "geometry_type", "position_xy", "localization_source",
    "source_frame_id", "source_image_sha256",
)


def _coop_public_receipt(action, role, pos, effect_at, ready_at, executed=True,
                         extra=None):
    """Build the public execution receipt handed to a cooperative_search method.

    Only values the benchmark itself publishes.  Ground-truth visibility,
    distance-to-target and hidden telemetry must never enter this dict; the
    caller may pass them in ``extra`` and they are dropped here.
    """
    receipt = {"role": role, "pos": [round(float(pos[0]), 2), round(float(pos[1]), 2)],
               "executed": bool(executed), "status": "ok" if executed else "failed",
               "effect_at": round(float(effect_at), 3), "ready_at": round(float(ready_at), 3)}
    for key in _COOP_RECEIPT_FIELDS:
        value = getattr(action, key, None)
        if key == "action":
            value = getattr(action, "action", None)
        if value not in (None, "", []):
            receipt[key] = value
    if "action" not in receipt:
        receipt["action"] = getattr(action, "action", "") or ""
    if extra:
        for key in ("status", "candidate_center_xy", "distance_m", "duration_s"):
            if extra.get(key) is not None:
                receipt[key] = extra[key]
    return receipt




def _canonical_map_key(name: str) -> str:
    """统一地图名规范化（2026-09-04 收敛到 lite_core.canonical_map_key）"""
    return canonical_map_key(name)


def _load_roadnet_for_episode(config, episode_plan):
    """加载并返回本 episode 的冻结路网服务实例

    支持两种配置模式：
    1. registry 模式（推荐）：config["road_graph"]["registry"] 指向 road_graph_registry.json
    2. dir 模式（兼容）：config["road_graph"]["dir"] 直接指向图目录
    """
    rg_cfg = config.get("road_graph", {})

    # Round-8 audit hardening: fail closed if an episode was generated for
    # sparse_nav_v1 but the runtime silently fell back to MANEUVER_V2 (or vice versa).
    # This is a dataset-integrity contract: a sparse plan must never execute on the
    # legacy fine-graph action surface merely because UGV_ACTION_SPACE was omitted.
    _planned_as = str(episode_plan.get("action_space_version") or "").strip()
    _ugv_cfg = config.get("ugv", {}) or {}
    _configured_as = str(os.environ.get("UGV_ACTION_SPACE") or _ugv_cfg.get("action_space") or "").strip()
    _runtime_as = "sparse_nav_v1" if _configured_as == "sparse_nav_v1" else "MANEUVER_V2"
    if _planned_as and _planned_as != _runtime_as:
        raise RuntimeError(
            "UGV action-space contract mismatch: "
            f"episode_plan={_planned_as!r}, runtime={_runtime_as!r}. "
            "Set UGV_ACTION_SPACE=sparse_nav_v1 (or ugv.action_space) for sparse plans; "
            "refusing silent fallback to protect benchmark/SFT provenance.")

    # Mode 1: registry
    registry_path = rg_cfg.get("registry")
    if registry_path:
        registry = RoadGraphRegistry.load(registry_path)
        map_key = _canonical_map_key(episode_plan.get("map", ""))
        if not map_key:
            map_key = _canonical_map_key(episode_plan.get("map_key", ""))
        if not map_key:
            raise RuntimeError("map key not found in episode_plan; cannot resolve road graph from registry")
        graph_dir = registry.resolve(
            map_key=map_key,
            expected_graph_id=episode_plan.get("road_graph_id") or None,
            expected_asset_sha256=episode_plan.get("road_graph_asset_sha256") or None,
        )
    else:
        # Mode 2: direct dir (legacy)
        graph_dir = rg_cfg.get("dir")
        if not graph_dir:
            graph_dir = episode_plan.get("road_graph_dir")
        if not graph_dir:
            raise RuntimeError("road_graph not configured: set road_graph.registry or road_graph.dir in config")

    expected_map_key = _canonical_map_key(episode_plan.get("map", ""))
    store = RoadGraphStore.load(graph_dir, expected_map_key=expected_map_key or None)
    locator = RoadLocator(store)
    router = RoadRouter(store)
    # 锚点资产：graph_dir 下有 anchors_v3_1.json 则加载（R2-P1-03 定界：仅锚点
    # 经过审计用，不参与 v2 动作菜单生成；回退语义由 REVERSE 承担）
    anchor_catalog = AnchorCatalog.load_for_graph_dir(graph_dir)
    option_provider = RoadOptionProvider(store)
    approach_planner = CandidateApproachPlanner(store)
    return store, locator, router, option_provider, approach_planner, anchor_catalog


# ===== sparse_nav_v1 门控（外审四轮 P0-BENCH；用户 2026-09-11 拍板门控分支）=====
_SPARSE_NAV_MODULE_CACHE = {}


def _sparse_nav_enabled(config) -> bool:
    """UGV_ACTION_SPACE=sparse_nav_v1 时启用锚点网络六动作底座（env 优先，config 回落）。

    未启用 = 现役 MANEUVER_V2（fine 图 RoadOptionProvider），行为零变化。"""
    ugv_cfg = config.get("ugv", {}) or {}
    val = (os.environ.get("UGV_ACTION_SPACE") or ugv_cfg.get("action_space") or "").strip()
    return val == "sparse_nav_v1"


def _sparse_intersection_max_distance(config) -> float:
    """Single source of truth shared with episode_generator's sparse threshold fallback."""
    ugv_cfg = config.get("ugv", {}) or {}
    return float(ugv_cfg.get(
        "sparse_intersection_max_distance_m",
        ugv_cfg.get("intersection_max_distance_m", 50.0)))


def _load_sparse_nav_adapter(config, episode_plan):
    """加载 SemanticNavAdapter（控制图正本）。路径：config ugv.control_graph 优先，
    回落 episode_plan['control_graph']。返回 (adapter, graph_sha256)。"""
    from pathlib import Path
    from agos.scripts.semantic_nav_adapter import SemanticNavAdapter
    from agos.scripts import semantic_nav_adapter
    explicit = (config.get('ugv') or {}).get('control_graph')
    town = _canonical_map_key(episode_plan.get('map') or episode_plan.get('map_key'))
    if not town or not town.isalnum():
        raise ValueError(f'Invalid control graph town: {town!r}')
    assets = Path(os.environ.get('AGOS_ASSET_ROOT', Path(__file__).resolve().parents[2] / 'assets'))
    p = Path(explicit) if explicit else assets / 'navigation' / 'ugv_control_graph' / town / 'ugv_control_graph.json'
    adapter_code_sha = hashlib.sha256(Path(semantic_nav_adapter.__file__).read_bytes()).hexdigest()
    adapter = SemanticNavAdapter.load(p)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    # Fail-closed provenance checks.  Schema-only validation is insufficient:
    # a wrong-town or stale V4 graph is syntactically valid but silently changes
    # the benchmark action surface.
    if adapter.schema_version != 4:
        raise RuntimeError(f"sparse_nav_v1 expects schema 4, got {adapter.schema_version}")
    _plan_as = episode_plan.get("action_space_version")
    if _plan_as not in (None, "", "sparse_nav_v1"):
        raise RuntimeError(f"episode action_space_version={_plan_as!r} is incompatible with sparse_nav_v1")
    _exp_sha = episode_plan.get("control_graph_sha256")
    if _exp_sha and _exp_sha != h:
        raise RuntimeError(f"control graph SHA mismatch: plan={_exp_sha[:12]} actual={h[:12]}")
    _exp_adapter_sha = episode_plan.get("semantic_adapter_sha256")
    if _exp_adapter_sha and _exp_adapter_sha != adapter_code_sha:
        raise RuntimeError(
            f"semantic adapter SHA mismatch: plan={_exp_adapter_sha[:12]} "
            f"actual={adapter_code_sha[:12]}")
    _exp_threshold = episode_plan.get("sparse_intersection_max_distance_m")
    _actual_threshold = _sparse_intersection_max_distance(config)
    if (_exp_threshold is not None and
            abs(float(_exp_threshold) - _actual_threshold) > 1e-9):
        raise RuntimeError(
            f"semantic maneuver threshold mismatch: plan={float(_exp_threshold):g} "
            f"actual={_actual_threshold:g}")
    _exp_rel = episode_plan.get("control_graph_release_id")
    if _exp_rel and _exp_rel != adapter.release_id:
        raise RuntimeError(f"control graph release mismatch: plan={_exp_rel} actual={adapter.release_id}")
    _exp_graph = episode_plan.get("road_graph_id")
    _actual_graph = adapter.graph.get("graph_id")
    if _exp_graph and _actual_graph and _exp_graph != _actual_graph:
        raise RuntimeError(f"control/L0 graph_id mismatch: plan={_exp_graph} control={_actual_graph}")
    _exp_town = _canonical_map_key(episode_plan.get("map", "") or episode_plan.get("map_key", ""))
    _actual_town = _canonical_map_key(adapter.graph.get("town", ""))
    if _exp_town and _actual_town and _exp_town != _actual_town:
        raise RuntimeError(f"control graph town mismatch: episode={_exp_town} control={_actual_town}")
    return adapter, h


def _validate_sparse_start_contract(episode_plan, store, sparse_nav, actual_tf=None,
                                    logical_xy_tol_m=0.05, logical_z_tol_m=0.05,
                                    logical_yaw_tol_deg=1.0, physical_xy_tol_m=0.25):
    """Fail closed unless the episode starts exactly on its declared sparse control port.

    sparse_current_port is authoritative in sparse_nav_v1.  Therefore a stale or
    externally edited plan whose physical start pose disagrees with
    ugv_start_port_fine_node would otherwise create an impossible first transition:
    the logical action begins at port A while CARLA begins at B, and execution skips
    fine_path[0].  That silently corrupts benchmark/SFT trajectories.
    (外审 R9 P0-02，2026-09-13 采纳)
    """
    port = str(episode_plan.get("ugv_start_port_fine_node") or "").strip()
    if not port:
        raise RuntimeError("SPARSE_START_CONTRACT_VIOLATION: missing ugv_start_port_fine_node")
    owner = sparse_nav.port_owner.get(port)
    if owner is None:
        raise RuntimeError(
            f"SPARSE_START_CONTRACT_VIOLATION: start port {port!r} is not a control port")
    if port not in store.nodes:
        raise RuntimeError(
            f"SPARSE_START_CONTRACT_VIOLATION: start port {port!r} is absent from L0 fine graph")
    planned_owner = str(episode_plan.get("ugv_start_control_node_id") or "").strip()
    if planned_owner and planned_owner != owner:
        raise RuntimeError(
            "SPARSE_START_CONTRACT_VIOLATION: control-node owner mismatch: "
            f"plan={planned_owner!r} graph={owner!r}")
    start = (episode_plan.get("start") or {}).get("ugv") or {}
    required = ("x", "y", "z", "yaw")
    if any(k not in start for k in required):
        raise RuntimeError(
            "SPARSE_START_CONTRACT_VIOLATION: start.ugv must contain x/y/z/yaw")
    node = store.nodes[port]
    xy = math.hypot(float(start["x"]) - float(node.x), float(start["y"]) - float(node.y))
    zerr = abs(float(start["z"]) - float(node.z))
    yawerr = abs(((float(start["yaw"]) - float(node.yaw_deg) + 180.0) % 360.0) - 180.0)
    if xy > logical_xy_tol_m or zerr > logical_z_tol_m or yawerr > logical_yaw_tol_deg:
        raise RuntimeError(
            "SPARSE_START_CONTRACT_VIOLATION: plan start pose disagrees with declared port: "
            f"port={port} xy={xy:.4f}m z={zerr:.4f}m yaw={yawerr:.3f}deg")
    if actual_tf is not None:
        pxy = math.hypot(float(actual_tf.location.x) - float(node.x),
                         float(actual_tf.location.y) - float(node.y))
        if pxy > physical_xy_tol_m:
            raise RuntimeError(
                "SPARSE_START_CONTRACT_VIOLATION: CARLA actor is not on declared start port: "
                f"port={port} xy={pxy:.4f}m")
    return port


UGV_MANEUVER_ACTIONS = ("FORWARD_1", "FORWARD_2", "TURN_LEFT", "TURN_RIGHT",
                        "TURN_STRAIGHT", "REVERSE")


def compute_ugv_available_actions(state, active_candidate, maneuver_menu=None):
    """根据当前状态计算 UGV 合法动作集（动作空间 v2）。

    - 移动类：FORWARD_1 / FORWARD_2 / TURN_LEFT / TURN_RIGHT / REVERSE，
      仅当 maneuver_menu（本步选项生成器输出）中存在对应选项时可用；
    - STOP_OBSERVE：语义弱化为"原地等候 10s"（六视图已常态化，不再采集/门控）；
    - CONFIRM / REJECT：任何时候都允许（用户裁定 2026-09-04），
      CONFIRM 仍受 confirm_budget 限制（由 validate_action 校验）。
    """
    actions = {"STOP_OBSERVE"}
    if maneuver_menu:
        for act in UGV_MANEUVER_ACTIONS:
            if act in maneuver_menu:
                actions.add(act)
    actions.update({"CONFIRM", "REJECT"})
    return sorted(actions)


def _check_ugv_osr(state, target_actor, config, carla_iface):
    """模块级 OSR 检查：若目标在 6 视图中几何可见，则设置 state.osr = True。"""
    if state.osr or not target_actor or not target_actor.is_alive:
        return
    _ugv_cfg = (config.get("ugv", {}) or {})
    gt_vis, _, _, _ = check_visibility_6view(
        target_actor,
        (state.ugv_pos["x"], state.ugv_pos["y"]), state.ugv_pos.get("yaw", 0),
        config["ugv"]["stop_observe_yaws"],
        # B6（2026-09-17 审计）：六视图 stop 相机参数 config 化（默认 90/480/270=旧行为）
        _ugv_cfg.get("stop_camera_fov_deg", 90),
        _ugv_cfg.get("stop_panel_width", 480),
        _ugv_cfg.get("stop_panel_height", 270), carla_iface.get_world(),
        observer_actor=getattr(carla_iface, "_ugv_actor", None),
        # B3（2026-09-17 审计）：LOS raycast 异常口径（默认 False=fail-open 旧行为）
        los_fail_closed=bool((config.get("perception", {}) or {}).get("los_fail_closed", False)))
    if gt_vis:
        state.osr = True


def _compute_osr_from_trajectory(ugv_trajectory, target_loc, carla_iface, max_dist_m=50):
    """根据 UGV 实际 2m 轨迹计算 OSR：任一点距目标 <= max_dist_m 且 LOS 通视。"""
    if not ugv_trajectory or target_loc is None:
        return False
    for x, y in ugv_trajectory:
        d = math.hypot(x - target_loc[0], y - target_loc[1])
        if d <= max_dist_m:
            visible, _, _ = carla_iface.check_los_to_target((x, y, 1.8), target_loc[2])
            if visible:
                return True
    return False


def _point_in_polygon(point, polygon):
    """检查点是否在多边形内（射线法）"""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def service_arrival_hit(current_fine_node, service_node_ids, ugv_xy=None,
                        node_xy=None, tol_m=5.0):
    """CandidateServiceContract 到达判定（外审 P0-C01 修复，方法中立，纯函数便于单测）。

    契约：当前 fine node ∈ 候选服务节点池（Candidate.service_node_ids，由创建点
    compute_service_nodes 固化，与 adapter 可服务性检查 / approach planner 三处
    共用同一池、半径同源 = max_confirm_distance_m）即视为到达候选。
    旧判定"UGV 距候选中心 <= approach_distance_min_m(30m)"与三级规划的
    30-50m 外环优先策略数学冲突（外环节点到中心 >= ~39m），导致 UGV 到达自己的
    合法 goal 仍停在 unverified——VERIFY prompt 永不出现。
    几何兜底：current_fine_node 缺失/不在图中时，UGV 坐标到池内任一节点欧氏
    距离 <= tol_m 也算到达（node_xy: node_id -> (x, y) 或 None 的查询函数）。
    """
    pool = set(service_node_ids or ())
    if not pool:
        return False
    # When a logical fine node is available, membership is authoritative.
    # Do not let the geometric tolerance expand the service radius by another
    # tol_m (e.g. a valid sparse port 55 m away matching a pool node at 50 m).
    if current_fine_node is not None:
        return current_fine_node in pool
    if ugv_xy is None or node_xy is None:
        return False
    for nid in pool:
        xy = node_xy(nid)
        if xy is None:
            continue
        if math.hypot(ugv_xy[0] - xy[0], ugv_xy[1] - xy[1]) <= tol_m:
            return True
    return False


def sparse_service_pool_reachable(adapter, current_port, service_node_ids,
                                  intersection_max_distance_m=50.0,
                                  max_route_m=None):
    """Check candidate serviceability on the actual sparse semantic graph.

    Legacy candidate checks use the dense frozen L0 graph.  In sparse_nav_v1,
    however, the UGV can only occupy landing ports exposed by
    SemanticNavAdapter.  Accepting a candidate merely because an arbitrary L0
    fine node is reachable can therefore create an assigned candidate that the
    sparse policy can never reach.  This Dijkstra uses the exact agent-visible
    maneuver transitions and succeeds only when a reachable landing port lies in
    the frozen service-node pool.

    Returns (reachable, min_distance_m, reachable_service_port_count).
    """
    import heapq
    pool = set(service_node_ids or ())
    if not current_port or not pool:
        return False, None, 0
    unbounded = (max_route_m is None or max_route_m <= 0 or
                 max_route_m == float("inf"))
    dist = {current_port: 0.0}
    pq = [(0.0, current_port)]
    hits = set()
    best = None
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u):
            continue
        if not unbounded and du > max_route_m:
            continue
        if u in pool:
            hits.add(u)
            if best is None:
                best = du
            # Do not stop immediately: count all reachable service landing ports
            # inside the same route bound for diagnostics/SFT provenance.
        try:
            menu = adapter.get_maneuver_options(
                u, intersection_max_distance_m=intersection_max_distance_m)
        except Exception:
            menu = {}
        for opt in menu.values():
            v = opt.target_fine_node_id
            nd = du + float(opt.path_distance_m)
            if (unbounded or nd <= max_route_m) and nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return bool(hits), best, len(hits)


def _polygon_area(polygon):
    """多边形面积（鞋带公式，返回非负值）"""
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _region_area(region):
    """任务区域面积"""
    return (region["x_max"] - region["x_min"]) * (region["y_max"] - region["y_min"])

class MessageBus:
    def __init__(self, dedup_enabled=True):
        self.messages = []
        self._msg_seq = 0
        self._dropped_count = 0  # P0-3: track suppressed duplicates
        self._dropped_log = []   # P0-3: log dropped messages for report
        # UGV 获得 target 信息的唯一来源（无候选/grid-to-ground），重复消息的
        self.dedup_enabled = dedup_enabled
    def send(self, sender, content, sim_time=0.0, receiver="", message_type="info",
             related_candidate_id="", step=0):
        # P0-3: Duplicate suppression - same sender+receiver+type+candidate+normalized text
        # within 3 steps is considered a duplicate and suppressed.
        # 2026-08-02 修复（bug review）：旧实现按"最近 5 条消息"开窗——若发送方
        # 持续输出相同内容且无其他消息入列，窗口永不滑动，后续合法重复被永久
        # 按 step 距离开窗：同键且 |Δstep| ≤ 3 才压制，与注释语义一致。
        if self.dedup_enabled:
            norm_content = content.strip().lower() if content else ""
            dedup_key = (sender, receiver or "ALL", message_type, related_candidate_id, norm_content)
            for m in self.messages[-5:]:
                m_key = (getattr(m, "sender", ""), getattr(m, "receiver", "") or "ALL",
                         getattr(m, "message_type", ""), getattr(m, "related_candidate_id", ""),
                         getattr(m, "content", "").strip().lower() if getattr(m, "content", "") else "")
                if m_key == dedup_key and abs(step - getattr(m, "step_index", step)) <= 3:
                    self._dropped_count += 1
                    self._dropped_log.append({
                        "sender": sender, "receiver": receiver or "ALL",
                        "message_type": message_type, "reason": "DUPLICATE",
                        "content_preview": content[:80] if content else "",
                        "step": step,
                    })
                    return  # suppress duplicate
        self._msg_seq += 1
        m = MessageInfo(sender, content, sim_time)
        # Spec v2 §4.9 fields (attached dynamically to keep MessageInfo dataclass unchanged)
        m.message_id = "M%03d" % self._msg_seq
        m.step_index = step
        m.receiver = receiver
        m.message_type = message_type
        m.related_candidate_id = related_candidate_id
        self.messages.append(m)
    def get_recent_for(self, role, sim_time, n=5):
        """获取对指定角色可见的消息（P0-3: 按 sender/receiver 过滤）"""
        visible = []
        for m in self.messages:
            if m.timestamp_step > sim_time:
                continue
            # P0-3: Don't show messages sent by the role itself
            if getattr(m, "sender", "") == role:
                continue
            # P0-3: Only show messages addressed to this role or to ALL
            m_receiver = getattr(m, "receiver", "") or "ALL"
            if m_receiver not in ("ALL", role):
                continue
            visible.append(m)
        return visible[-n:]
    def get_recent(self, n=5):
        return self.messages[-n:]

    def get_visible_for(self, role, sim_time, n=None):
        """All messages this role may legally see at ``sim_time``.

        The default five-message tail silently drops unread traffic between two
        decision boundaries; cooperative_search consumes the full public stream
        and de-duplicates by message id inside the method.
        """
        visible = []
        for m in self.messages:
            if m.timestamp_step > sim_time:
                continue
            if getattr(m, "sender", "") == role:
                continue
            receiver = getattr(m, "receiver", "") or "ALL"
            if receiver not in ("ALL", role):
                continue
            visible.append(m)
        return visible[-n:] if n else visible



class PatrolGuardState:
    """G1 UGV patrol guard（防死循环 watchdog）的纯判定状态机。

    H2（v3 外审 P1-01）：从 runner 主循环剥离，便于单测。调用方每步喂
    (action, active_candidate_id, trace_route_guidance)，返回 None（未触发）
    或触发信息 dict；ledger/msg_bus/memory 等副作用留在 runner 侧。

    mode=steps（旧行为，仅兼容保留）：同 active candidate 下连续非终局动作
        步数 > max_steps 即触发——与无界 approach route 矛盾（长 route 走
        一半被误杀）。
    mode=route_progress（新默认）：同 active candidate 下 route 剩余里程
        （trace_route_guidance.route_length_m，仅 activated 时有效）连续
        stall_max 步无下降（容差 tol_m）才触发。route 未激活/无里程
        不累计。candidate 切换或终局动作
        （STOP_OBSERVE/CONFIRM/REJECT）重置全部计数。
    """

    __slots__ = ("mode", "max_steps", "stall_max", "tol_m",
                 "count", "cand_id", "route_stall", "rem_prev",
                 "topology_history")

    TERMINAL_ACTIONS = ("STOP_OBSERVE", "CONFIRM", "REJECT")

    def __init__(self, mode="route_progress", max_steps=8, stall_max=8, tol_m=0.5):
        if mode not in ("steps", "route_progress"):
            raise ValueError(
                "patrol_guard_mode must be 'steps' or 'route_progress', got %r" % (mode,))
        self.mode = mode
        self.max_steps = int(max_steps)
        self.stall_max = int(stall_max)
        self.tol_m = float(tol_m)
        self.reset()

    def reset(self):
        self.count = 0
        self.cand_id = None
        self.route_stall = 0
        self.rem_prev = None
        # R29 external audit: deterministic sparse-graph liveness fallback.
        # Tokens are (logical_port, executed_action) at UGV decision time.
        # Repeating the exact same topological cycle three times cannot make
        # progress in a deterministic sparse action graph, so it is safe to
        # abort the current candidate even when no route-distance metric exists.
        self.topology_history = []

    @staticmethod
    def _topology_cycle_period(history, min_period=2, max_period=6, repeats=3):
        n = len(history)
        for period in range(int(min_period), int(max_period) + 1):
            need = period * int(repeats)
            if n < need:
                continue
            tail = history[-need:]
            block = tail[:period]
            if all(tail[i * period:(i + 1) * period] == block
                   for i in range(1, int(repeats))):
                return period
        return None

    def update(self, action, active_cand_id, route_guidance=None, topology_state=None):
        """喂入本步动作与上下文；返回 None 或触发 dict（触发后自动复位）。

        ``route_guidance`` accepts both the ``remaining_m`` schema and the
        legacy TRACE ``route_length_m`` schema.  When neither route metric is
        available, ``topology_state`` provides a method-neutral sparse-port
        liveness fallback.
        """
        if active_cand_id is None or action in self.TERMINAL_ACTIONS:
            keep_cand = active_cand_id if action in self.TERMINAL_ACTIONS else None
            self.reset()
            self.cand_id = keep_cand
            return None

        switched = self.cand_id != active_cand_id
        if switched:
            # candidate 切换：全新计数窗口
            self.reset()
            self.cand_id = active_cand_id
            self.count = 1
        else:
            self.count += 1

        # Record the sparse pre-action state even on the first candidate step.
        # Keep only what is needed for 3 repeats of the longest (6-state) cycle.
        if topology_state not in (None, ""):
            self.topology_history.append((str(topology_state), str(action)))
            if len(self.topology_history) > 18:
                self.topology_history = self.topology_history[-18:]

        if self.mode == "route_progress" and not switched:
            rg = route_guidance or {}
            rem = None
            if rg.get("activated"):

                # route_length_m.  R28 only read the latter, disconnecting G1

                for key in ("remaining_m", "route_length_m"):
                    value = rg.get(key)
                    if isinstance(value, (int, float)) and value >= 0.0:
                        rem = float(value)
                        break
            if rem is not None:
                if self.rem_prev is None:
                    pass  # 首个样本：建立基线，不计停滞
                elif rem <= self.rem_prev - self.tol_m:
                    self.route_stall = 0  # 剩余里程有效下降 = 行进进展
                else:
                    self.route_stall += 1
                self.rem_prev = rem
            else:
                # No distance metric: do not manufacture route progress.  The
                # sparse topological recurrence check below remains active.
                self.route_stall = 0
                self.rem_prev = None

        if self.mode == "route_progress":
            fired = self.route_stall > self.stall_max
            if fired:
                info = {
                    "mode": "route_progress",
                    "candidate_id": active_cand_id,
                    "stall_steps": self.route_stall,
                    "route_remaining_m": (round(self.rem_prev, 2)
                                          if self.rem_prev is not None else None),
                    "wander_steps": None,
                    "cycle_period": None,
                    "cycle_repeats": None,
                }
                self.reset()
                return info

            period = self._topology_cycle_period(self.topology_history)
            if period is not None:
                info = {
                    "mode": "topology_cycle",
                    "candidate_id": active_cand_id,
                    "stall_steps": None,
                    "route_remaining_m": None,
                    "wander_steps": period * 3,
                    "cycle_period": period,
                    "cycle_repeats": 3,
                }
                self.reset()
                return info
            return None

        # Legacy steps mode remains byte-compatible in threshold semantics.
        if self.count <= self.max_steps:
            return None
        info = {
            "mode": "steps",
            "candidate_id": active_cand_id,
            "stall_steps": None,
            "route_remaining_m": None,
            "wander_steps": self.count,
            "cycle_period": None,
            "cycle_repeats": None,
        }
        self.reset()
        return info


def fcs_should_intercept(shield, action, oracle_positive):
    """FCS 诊断拦截判定（N0 Go/No-Go，纯函数便于单测）。

    shield=fcs 且动作为 CONFIRM 且该步 oracle 判定阴性（不可确认：不可见/
    超距/LOS 挡/距离未知）时返回 True——runner 将其转 STOP_OBSERVE 不终止
    不耗预算（would_false_confirm 计数）；oracle 阳性 CONFIRM 照常放行
    （正确确认=成功终局）。GNG-3 用于解开盲 CONFIRM 对 UAV 搜索的 censoring。
    """
    return shield == "fcs" and action == "CONFIRM" and not bool(oracle_positive)


def _build_oracle_candidate_polygon(target_xy, half_m):
    """GNG-1 oracle 注入用的 candidate 多边形（以真实 target 为中心的方框）。"""
    tx, ty = float(target_xy[0]), float(target_xy[1])
    h = float(half_m)
    return [(tx - h, ty - h), (tx + h, ty - h), (tx + h, ty + h), (tx - h, ty + h)]


class JointLiteRunner:
    """联合评测主循环"""
    def __init__(self, config, agent_mode="cooperative_search", vlm_config=None, sibling_episodes=None):
        if agent_mode not in method_registry.PRIMARY_METHODS:
            raise ValueError(f'Unknown evaluation method: {agent_mode!r}')
        config = copy.deepcopy(config)
        config.setdefault('success', {}).setdefault('max_confirm_distance_m', confirm_distance_m(config))
        self.config = config
        self.success_protocol = success_protocol(config)
        self.agent_mode = agent_mode
        self.vlm_config = vlm_config
        self.sibling_episodes = sibling_episodes or []  # 其他 episode 元数据，用于 CECM
        self.method = None
        # N0（Go/No-Go 诊断基建）：diagnostic 组三开关，默认全关 = 零行为影响。
        #   shield: none|fcs —— fcs=oracle 阴性 CONFIRM 被框架拦截（转 STOP_OBSERVE
        #           不终止不耗预算，记 would_false_confirm；oracle 阳性 CONFIRM 放行）
        #   save_all_frames: 每 UGV 决策步全量保存六视图（GNG-2 verifier 数据集）
        #   oracle_candidate_injection: episode 启动即注入覆盖真实 target 的
        #           candidate（GNG-1 route/service 执行层测试，UAV 冻结）
        _diag = dict(config.get("diagnostic", {}) or {})
        _shield = str(_diag.get("shield", "none") or "none").strip().lower()
        if _shield not in ("none", "fcs"):
            raise ValueError("diagnostic.shield must be 'none' or 'fcs', got %r" % _shield)
        self._diag_shield = _shield
        self._diag_save_all_frames = bool(_diag.get("save_all_frames", False))
        # R18 dataset expansion: save every UAV decision RGB frame (raw + grid)
        # and optionally persist per-view oracle visibility boxes for offline
        # labels. Both are diagnostic-only and default off.
        self._diag_save_all_uav_frames = bool(_diag.get("save_all_uav_frames", False))
        self._diag_save_full_visibility_debug = bool(_diag.get("save_full_visibility_debug", False))
        # R28-V0 插桩：每个 UGV 决策步把六个原始视图（原生 480x270）平铺落盘。
        # 纯增量落盘、flag 门控默认关，不进入任何决策路径；V0 离线 TIP/r18
        # 评估用（与 save_all_uav_frames 的 UAV 侧对称）。
        self._diag_save_ugv_views = bool(_diag.get("save_ugv_views_per_step", False))
        # 演示素材采集（2026-09-13 对外演示任务）：每 VLM 决策步保存
        # UAV 航拍输入图 + 每仿真步全局俯视(含双机轨迹叠加)图。纯增量、
        # flag 门控默认关，不影响生产语义。UGV 六视图由 save_all_frames 覆盖。
        self._diag_demo_record = bool(_diag.get("demo_record", False))
        # 常设口径（2026-09-13 用户定）：常规实验只落轨迹/动作/VLM 输出等
        # 结构化数据，图片仅演示素材任务才收。此处防呆提醒，不拦截。
        if (self._diag_save_all_frames or self._diag_save_all_uav_frames
                or self._diag_save_ugv_views or self._diag_demo_record):
            logger.warning(
                "image collection ENABLED (save_all_frames=%s, save_all_uav_frames=%s, "
                "save_ugv_views_per_step=%s, demo_record=%s) — "
                "diagnostic/V0 instrumentation runs only; normal experiments should keep these off "
                "(structured JSON/trajectories only per 2026-09-13 standing rule)",
                self._diag_save_all_frames, self._diag_save_all_uav_frames,
                self._diag_save_ugv_views, self._diag_demo_record)
        self._diag_cfg = _diag

    def _demo_save(self, img, path):
        """demo_record 存图助手：兼容 bytes / PIL.Image 输入。"""
        import io as _io
        try:
            if isinstance(img, (bytes, bytearray)):
                with open(path, "wb") as f:
                    f.write(img)
            else:
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                with open(path, "wb") as f:
                    f.write(buf.getvalue())
        except Exception as e:
            logger.debug("demo_record save failed (%s): %s", path, e)

    def _demo_save_annotated_topdown(self, map_frame, store, ledger, region,
                                     global_rgb_bytes, img_dir, step, state,
                                     target_actor=None, fallback_topdown=None):
        """演示素材：增强版全局俯视（大图标/粗轨迹/真实目标红星）。仅存图，不进 VLM。"""
        try:
            tp = None
            if target_actor is not None and getattr(target_actor, "is_alive", False):
                _tl = target_actor.get_location()
                tp = (_tl.x, _tl.y)
            ann = render_topdown_annotated(map_frame,
                (state.uav_pos["x"], state.uav_pos["y"]),
                (state.ugv_pos["x"], state.ugv_pos["y"]), ledger.candidates,
                state.uav_trajectory, state.ugv_trajectory, road_graph_store=store,
                background_image_bytes=global_rgb_bytes, show_road_network=False,
                region=region, target_pos=tp)
            self._demo_save(ann, os.path.join(img_dir, "topdown_s%03d.png" % step))
        except Exception as e:
            logger.debug("annotated topdown failed: %s", e)
            if fallback_topdown is not None:
                self._demo_save(fallback_topdown, os.path.join(img_dir, "topdown_s%03d.png" % step))

    @staticmethod
    def _framework_activate_uav(state, msg_bus, kind, candidate_id="", extra=""):
        """框架层确定性激活 UAV（不依赖 UGV 手动发消息）。

        根源性杜绝 UAV 无限冻结：只要 UGV 抵达疑似目标区域（成功/失败/开始观察），
        就由框架替 UGV 给 UAV 发一条消息并解冻 UAV，引导后续协同。

        kind:
          - "arrived"   : UGV 抵达疑似目标区域 -> 引导 UAV 协同（精定位引导）
          - "unreachable": UGV 无法抵达 -> 请 UAV 重新标定疑似目标区域
          - "observing" : UGV 到达并开始 STOP 观察 -> 引导 UAV 协同
        """
        was_frozen = state.uav_frozen
        if kind == "arrived":
            text = "UGV arrived at candidate area %s. Please guide fine localization." % (candidate_id or "")
            mtype = "arrival_notification"
        elif kind == "unreachable":
            text = "UGV cannot reach candidate %s. Please re-annotate a new candidate area." % (candidate_id or "")
            mtype = "reannotate_request"
        else:  # observing
            text = "UGV is observing candidate %s area." % (candidate_id or "")
            mtype = "arrival_notification"
        if extra:
            text = text + " " + extra
        msg_bus.send("UGV", text, state.ugv_sim_time, receiver="UAV",
                     message_type=mtype, related_candidate_id=candidate_id or "", step=state.step)
        state.uav_frozen = False
        state.uav_frozen_since_step = None  # 解冻后清除冻结计时
        if was_frozen:
            logger.info("[%s] UAV framework-activated (%s) for candidate %s",
                        state.episode_id, kind, candidate_id or "-")

    def _build_episode_metadata(self, episode_plan):
        """为 CECM 构造当前 episode 元数据。"""
        from agos.eval.contamination import _extract_xy
        tgt = episode_plan.get('target_oracle', {})
        tgt_tf = tgt.get('transform', {})
        start = episode_plan.get('start', {})
        ugv_start = start.get('ugv', {})
        return {
            'episode_id': episode_plan.get('episode_id', 'unknown'),
            'difficulty': '',
            'target_class': tgt.get('blueprint_filter', '') or tgt.get('blueprint_id', ''),
            # B22（2026-09-17 审计）：运行时 CECM 同样优先用人类可读描述匹配泄露
            # （blueprint 串 VLM 输出不可能包含），与 contamination.py 同源逻辑
            'target_description': episode_plan.get('target_task', {}).get('target_description', ''),
            'target_loc': (float(tgt_tf.get('x', 0.0)), float(tgt_tf.get('y', 0.0))),
            'ugv_start': (float(ugv_start.get('x', 0.0)), float(ugv_start.get('y', 0.0))),
            'distractor_locs': [xy for d in episode_plan.get('distractors_oracle', []) if (xy := _extract_xy(d)) is not None],
        }

    def _build_vlm_failure_result(self, state, role):
        """2026-07-26 鲁棒评测：VLM 连续失败时构建终止结果"""
        return {
            "episode_id": state.episode_id,
            "agent_mode": self.agent_mode,
            "success_protocol": success_protocol(self.config),
            "max_confirm_distance_m": confirm_distance_m(self.config),
            "success": False,
            "failure_stage": "VLM_INFRA_FAILURE",
            "failure_reason": "VLM_CONSECUTIVE_FAILURE",
            "failure_reason_detail": f"{role} VLM 连续 {state.consecutive_vlm_failures} 次调用失败",
            "completed": False,
            "outcome_class": "infra_error",
            "steps": state.step,
            "vlm_calls": state.vlm_calls,
            **http_accounting_result(state),
            "uav_vlm_calls": state.uav_vlm_calls,
            "ugv_vlm_calls": state.ugv_vlm_calls,
            "infra_error_count": getattr(state, "infra_error_count", 0),
            "consecutive_vlm_failures": state.consecutive_vlm_failures,
            "uav_trajectory": [(round(x,1), round(y,1)) for x,y in state.uav_trajectory],
            "ugv_trajectory": [(round(x,1), round(y,1)) for x,y in state.ugv_trajectory],
            "candidates": [],
            "messages": [],
            "step_logs": [],
            "total_sim_time_s": round(getattr(state, 'uav_sim_time', 0) + getattr(state, 'ugv_sim_time', 0), 1),
        }

    def _los_fail_closed(self) -> bool:
        """B3（2026-09-17 审计）：LOS raycast 异常口径（默认 False=fail-open 旧行为）。"""
        return bool(((self.config or {}).get("perception") or {}).get("los_fail_closed", False))

    def _lookup_ground_z(self, x, y, carla_iface):
        """B9（2026-09-17 审计）：查询 (x, y) 处地图路面高程。

        地图高程不是目标 GT（允许使用）。失败/无 SDK/无路网覆盖时回退 0.0（=旧行为）。
        """
        try:
            if carla is None:
                return 0.0
            wp = carla_iface.get_world().get_map().get_waypoint(
                carla.Location(x=float(x), y=float(y)))
            if wp is not None:
                return float(wp.transform.location.z)
        except Exception as e:
            logger.debug("ground elevation lookup failed at (%.1f, %.1f): %s", x, y, e)
        return 0.0

    def _lookup_public_ground_z(self, x, y, carla_iface):
        """R33 nearest PUBLIC road z; off-road/failed lookups stay unknown."""
        try:
            if carla is None:
                return None
            wp = carla_iface.get_world().get_map().get_waypoint(
                carla.Location(x=float(x),y=float(y)))
            if wp is None:
                return None
            loc = wp.transform.location
            if math.hypot(float(loc.x)-float(x),float(loc.y)-float(y))>40.:
                return None
            return float(loc.z)
        except Exception:
            return None

    def run_episode(self, carla_iface, episode_plan, output_dir):
        """运行单个 episode - 离散事件仿真（带 try/finally actor 清理）"""
        self.last_model_call_accounting = None
        validate_plan_success_protocol(episode_plan, self.config)

        state = EpisodeState(self.config, episode_plan)
        state.paper_fault_events = []
        map_protocol = ObservationMapProtocol(self.config, episode_plan, self.agent_mode, output_dir)
        # Same PUBLIC road elevation source as UAV candidate grid projection.
        # The service caches it once on a navigation-height lattice, no RGB/GT.
        map_protocol.set_ground_elevation(lambda x,y:self._lookup_public_ground_z(x,y,carla_iface))
        msg_bus = MessageBus(dedup_enabled=True)

        # only how model-REJECT regions are suppressed; default OFF preserves the
        # historical permanent suppression behavior.
        _observation_cfg = ((self.config or {}).get("observation_map") or {}) if isinstance(self.config, dict) else {}
        _r19_reject_cd_on = bool(_observation_cfg.get("r19_reject_cooldown_enabled", False))
        # ---- B7/B9/B10/B11（2026-09-17 审计）新增 config 读取（全部默认=旧行为）----
        _cand_cfg = ((self.config or {}).get("candidate") or {})
        # B7：重复抑制模式（默认 "radius"=旧 30m 半径语义；"cell_aware" 见
        # perception.CandidateLedger.create 文档）
        _dup_suppression_mode = str(_cand_cfg.get("duplicate_suppression_mode", "radius") or "radius").strip().lower()
        # B9 is wired by ObservationMapProtocol into the public proposal projection
        # height snapshot; disabled/missing road height retains the zero plane.
        # B10：approach goal 必须可确认（最远角 <= 服务半径；默认 False=旧行为）
        _approach_require_confirmable = bool(
            ((self.config or {}).get("ugv") or {}).get("approach_require_confirmable", False))
        # R27 movement-contract experiment: broad serviceability is nearest-edge <= R;
        # "verification-ready" is conservative farthest-corner <= R. Default false
        # preserves R26 behavior byte-for-byte until explicitly enabled.
        _confirmable_service_arrival = bool(
            ((self.config or {}).get("ugv") or {}).get("confirmable_service_arrival_enabled", False))
        # B11：UAV 框架反馈回注（默认 False=零变化）
        _uav_framework_feedback = bool(
            ((self.config or {}).get("ux") or {}).get("uav_framework_feedback", False))
        ledger = CandidateLedger(
            self.config["candidate"]["duplicate_suppression_radius_m"],
            self.config["candidate"]["max_candidates_total"],
            rejected_cooldown_steps=(int(_observation_cfg.get("r19_reject_cooldown_steps", 12) or 12)
                                     if _r19_reject_cd_on else None),
            rejected_max_reopens=int(_observation_cfg.get("r19_reject_max_reopens", 1) or 0))
        # block once per episode; all R16 hooks below are no-ops when disabled
        # (P0 behavior byte-identical).
        _q_enabled = False
        _q_max_shadow = max(1, int(_observation_cfg.get("q_max_shadow", 2) or 2))
        # R17 modules: all default off. The opportunity probe is diagnostic-only;
        # visual ranking is shadow-only unless dispatch is explicitly unlocked.
        _r20_revision_min_shift = float(_cand_cfg.get("r20_revision_min_shift_m", 6.0))
        _policy_horizon_cfg = int((((self.config or {}).get("eval") or {}).get("policy_horizon_steps") or 0))
        _last_shadow_event = {"event": "", "candidate_id": None}
        checker = SuccessChecker(self.config)

        target_ref_img = b""
        ref_views = {}
        ref_path = episode_plan.get("target_task", {}).get("target_reference_image", "")
        if ref_path:
            # Spec v2 fix: resolve relative reference image against the plan file's
            # own directory (injected as _plan_dir), NOT the output dir parent.
            # r9.5: 统一走 load_reference_assets（collage + 独立 views 双资产）
            from agos.episode_io import load_reference_assets
            target_ref_img, ref_views = load_reference_assets(
                episode_plan, extra_search_dirs=(os.path.dirname(output_dir),))

        # Spec v2 §15: persist key reference images into episode images dir
        img_dir = os.path.join(output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        target_reference_image_path = ""
        if target_ref_img:
            target_reference_image_path = os.path.join(img_dir, "target_reference.png")
            try:
                with open(target_reference_image_path, "wb") as f:
                    f.write(target_ref_img)
            except Exception as e:
                logger.warning("Failed to save target reference image: %s", e)

        self.method = method_registry.create(self.agent_mode, self.config, services=MethodServices())
        uav_agent = ugv_agent = self.method

        uav_start = episode_plan["start"]["uav"]
        ugv_start = episode_plan["start"]["ugv"]
        # 2026-07-26 鲁棒评测：连续 VLM 失败阈值（从 config 读取，默认5）
        state.max_consecutive_vlm_failures = self.config.get("vlm", {}).get("max_consecutive_failures", 5)
        state.consecutive_vlm_failures = 0
        carla_iface.set_uav_pose(uav_start['x'], uav_start['y'], uav_start['z'], 0)
        _require_identity_ref = bool((self.config.get('benchmark') or {}).get('require_teammate_reference',False)
                                     or (self.config.get('observation_map') or {}).get('variant_id')=='R33_SHARED_MAP')
        from agos.episode_io import load_teammate_reference
        # Reject missing/invalid assets before spawn or model use; legacy is explicit.
        load_teammate_reference(episode_plan,require_new=_require_identity_ref)
        # Target spawning configures this too, but the UGV is placed first here.
        # Select the plan's deployment contract before its initial spawn so V3
        # reapplies/acknowledges the exact command after CARLA's spawn auto-lift.
        if hasattr(carla_iface,'configure_vehicle_deployment'):
            carla_iface.configure_vehicle_deployment(episode_plan.get('vehicle_deployment_protocol'))
        if hasattr(carla_iface,'configure_ugv_identity'):
            carla_iface.configure_ugv_identity(episode_plan.get('ugv_identity'))
        ugv_tf = carla_iface.set_ugv_transform(ugv_start["x"], ugv_start["y"], ugv_start["z"], ugv_start["yaw"],
                                                   strict_kinematic=_sparse_nav_enabled(self.config))
        # P0-fix: UGV spawn 失败（碰撞/偏移过大）时立即终止 episode，不进入主循环空转
        if ugv_tf is None:
            logger.error("[%s] UGV spawn failed at (%.1f, %.1f), episode aborted", state.episode_id, ugv_start["x"], ugv_start["y"])
            result = {"episode_id": state.episode_id, "agent_mode": self.agent_mode,
            "success_protocol": success_protocol(self.config),
            "max_confirm_distance_m": confirm_distance_m(self.config), "success": False,
                      "failure_stage": "UGV_SPAWN_FAILED", "steps": 0, "vlm_calls": 0,
                      "uav_vlm_calls": 0, "ugv_vlm_calls": 0, "uav_distance_m": 0.0, "ugv_distance_m": 0.0,
                      "failure_reason": "UGV_SPAWN_FAILED", "failure_reason_detail": "UGV 起点被遮挡或偏移过大，无法生成",
                      "completed": False, "outcome_class": "infra_error"}
            result["run_meta"] = _build_run_meta(self.config, self.agent_mode)
            from agos.runtime.paper_audit import finalize_fault_audit
            finalize_fault_audit(self.config, state, result)
            return result
        # 2026-07-25 新增：用 actual_tf 更新 state.ugv_pos，确保和 CARLA 实际位置一致
        state.ugv_pos = {"x": ugv_tf.location.x, "y": ugv_tf.location.y, "z": ugv_tf.location.z, "yaw": ugv_tf.rotation.yaw}
        state.uav_trajectory.append((uav_start["x"], uav_start["y"]))
        state.ugv_trajectory.append((state.ugv_pos["x"], state.ugv_pos["y"]))

        # P0-fix（2026-07-21）：捕获全局 RGB 俯视图作为 topdown_map 背景
        # 避免 topdown_map 是纯色背景，改为实际 RGB 俯视图 + 叠加轨迹/标注
        # 2026-07-24 修正：RGB 背景图必须用 map_region（整张地图），不是 episode region
        # 延迟到 map_region 计算后捕获（见下方 map_region 代码块）
        region = episode_plan["region"]
        global_rgb_bytes = None

        ep_meta = self._build_episode_metadata(episode_plan)
        cecm = None

        target_actor = None
        distractor_actors = []
        result = None
        try:
            if self.sibling_episodes:
                cecm = RuntimeContaminationMonitor(ep_meta, self.sibling_episodes)

            target_actor = carla_iface.spawn_target_from_plan(episode_plan)
            if target_actor is None:
                result = {"episode_id": state.episode_id, "agent_mode": self.agent_mode,
            "success_protocol": success_protocol(self.config),
            "max_confirm_distance_m": confirm_distance_m(self.config), "success": False,
                          "failure_stage": "TARGET_SPAWN_FAILED", "steps": 0, "vlm_calls": 0,
                          # R2-P2-03：early return 统一字段口径（与 UGV_SPAWN_FAILED 一致）
                          "failure_reason": "RUNTIME_EXCEPTION",
                          "failure_reason_detail": "目标 actor 生成失败（CARLA 仿真器问题），需重跑",
                          "schema_version": "lite_result_v3_spec_r2",
                          "completed": False, "outcome_class": "infra_error"}
                return result
            carla_iface._runtime_target = target_actor

            # P0-5: respawn distractors at runtime
            distractor_actors = carla_iface.spawn_distractors_from_plan(episode_plan)

            target_loc = target_actor.get_location()
            state.target["transform"]["x"] = target_loc.x
            state.target["transform"]["y"] = target_loc.y
            state.target["transform"]["z"] = target_loc.z
            tgt_xy = (target_loc.x, target_loc.y)

            region = episode_plan["region"]

            # Load frozen road graph for this episode (moved up - need full-map extent for RGB capture)
            store, locator, router, option_provider, approach_planner, anchor_catalog = _load_roadnet_for_episode(self.config, episode_plan)
            # sparse_nav_v1：锚点网络六动作底座（外审四轮 P0-BENCH；门控见 _sparse_nav_enabled）
            sparse_nav = sparse_nav_sha = None
            if _sparse_nav_enabled(self.config):
                sparse_nav, sparse_nav_sha = _load_sparse_nav_adapter(self.config, episode_plan)
                state.sparse_current_port = _validate_sparse_start_contract(
                    episode_plan, store, sparse_nav, actual_tf=ugv_tf)
                logger.info("[%s] UGV_ACTION_SPACE=sparse_nav_v1 release=%s sha=%s start_port=%s",
                            state.episode_id, sparse_nav.release_id, sparse_nav_sha[:12],
                            state.sparse_current_port)
            state.road_graph_id = store.manifest.graph_id
            state.road_graph_asset_sha256 = store.manifest.asset_sha256
            state.anchor_visit_history = []
            state.consecutive_backtracks = 0
            # r7: 把 road services 注入 MethodSession（三方法相同注入；
            if self.method is not None:
                self.method.services = MethodServices(
                    road_graph=store, router=router,
                    approach_planner=approach_planner,
                    semantic_nav=sparse_nav)
                # r7 review fix: honor the MethodSession episode lifecycle.

                # runtime integration must not rely on constructor side effects.
                _method_start_episode(self.agent_mode, self.method, {
                    "episode_plan": episode_plan,
                    "episode_id": state.episode_id,
                    "target_reference_image": target_ref_img,
                    "target_reference_views": ref_views,
                })

            teammate_ref_img = (load_teammate_reference(episode_plan,
                actor=carla_iface._ugv_actor,require_new=_require_identity_ref)
                )
            if teammate_ref_img is None:
                teammate_ref_img = b""
                try:
                    teammate_ref_img = carla_iface.capture_ugv_topdown_reference(
                        state.ugv_pos["x"], state.ugv_pos["y"], width=512, height=512, fov=90)
                except Exception as e:
                    logger.warning("Failed to capture legacy UGV top-down reference: %s", e)
            # Spec v2 §4.8.2: persist UGV teammate reference image for UAV
            teammate_reference_image_path = ""
            if teammate_ref_img:
                teammate_reference_image_path = os.path.join(img_dir, "teammate_reference_uav.png")
                try:
                    with open(teammate_reference_image_path, "wb") as f:
                        f.write(teammate_ref_img)
                except Exception as e:
                    logger.warning("Failed to save teammate reference image: %s", e)

            # Compute full-map world extent from road graph nodes.
            # v1 approach: road network extent + 50m margin = ~548m coverage.
            # This covers all UAV/UGV trajectory ranges (they move on roads).
            _all_nodes = list(store.nodes.values())
            if _all_nodes:
                _map_x_min = min(n.x for n in _all_nodes)
                _map_x_max = max(n.x for n in _all_nodes)
                _map_y_min = min(n.y for n in _all_nodes)
                _map_y_max = max(n.y for n in _all_nodes)
                _map_margin = 50
                map_region = {
                    "x_min": _map_x_min - _map_margin,
                    "x_max": _map_x_max + _map_margin,
                    "y_min": _map_y_min - _map_margin,
                    "y_max": _map_y_max + _map_margin,
                }
            else:
                map_region = region
                logger.warning("Road graph has no nodes, falling back to episode region for map extent")

            # Spec v2 §6.3: capture full-map RGB top-down base image once per run.
            # P0-09: run-level files go to the map directory (parent of episodes/),
            # NOT the episode's own dir. This prevents cross-run overwrites.
            # New layout: <run_id>/maps/<map>/global_rgb_topdown.png
            #             <run_id>/maps/<map>/episodes/<eid>/...
            # 2026-07-24 修正：单 episode 模式（如 Human Baseline）也正确捕获 RGB
            # 不依赖 output_dir 的目录结构，直接在当前 episode 的 map 目录下查找/保存
            _ep_dir = output_dir.rstrip("/\\")
            # 检查是否是 run-level 目录结构（有 episodes/ 子目录）
            _parent = os.path.dirname(_ep_dir)
            if os.path.basename(_parent) == "episodes":
                # 标准结构: <run>/maps/<map>/episodes/<eid> -> _map_root = <run>/maps/<map>
                _map_root = os.path.dirname(_parent)
            else:
                # 单 episode 模式: <output>/<eid> -> _map_root = <output>
                _map_root = _parent
            global_rgb_bytes = _load_observation_global_rgb(
                carla_iface, map_region, _map_root, state.episode_id, map_protocol)
            # MapFrame uses full-map extent (not single episode region) so map_metadata covers entire Town01
            map_frame = MapFrame(map_region)
            uav_wps = episode_plan["uav_grid_waypoints"]
            ugv_wps = episode_plan["ugv_road_waypoints"]
            uav_speed = state.uav_speed_mps
            ugv_speed = state.ugv_speed_mps
            wp_spacing = self.config["uav"]["waypoint_spacing_m"]

            step_logs = []
            uav_distance = 0.0
            ugv_distance = 0.0

            # Spec v2 §4: per-episode structured logs (jsonl + trajectory json)
            action_log_records = []       # §4.4 full action log
            candidate_log_records = []    # §4.7 successfully created candidates
            candidate_attempt_records = [] # R19 audit: every ANNOTATE attempt + ledger/serviceability disposition
            ag_os_trace_records = []      # P0-6: AG-OS 模块级证据 trace
            r17_opportunity_records = []  # R17 diagnostic sidecar; never exposed to policy
            uav_traj_records = []         # §4.5 UAV decision-level trajectory
            ugv_decision_records = []     # §4.6.1 UGV decision waypoints
            ugv_fine_segments = []        # §4.6.2 UGV executed fine path segments
            _last_annotate_img = {"path": ""}  # mutable holder for annotate input image path
            warnings_list = []            # spec: missing data must surface as warnings

            # ---- UGV loop guards (deterministic watchdogs against dead loops) ----
            # G1: UGV patrol guard — H2（v3 外审 P1-01）双模式 steps|route_progress，
            #     判定核心在模块级 PatrolGuardState（纯函数式，便于单测）。
            #     route_progress 为新默认：route 剩余里程连续停滞才 abort，
            #     与无界 approach route 兼容；steps 保留旧行为供复现。
            _ugv_cfg = self.config.get("ugv", {}) or {}
            _patrol_guard = PatrolGuardState(
                mode=_ugv_cfg.get("patrol_guard_mode", "route_progress"),
                max_steps=_ugv_cfg.get("patrol_guard_max_steps", 8),
                stall_max=_ugv_cfg.get("patrol_stall_steps", 8),
                tol_m=_ugv_cfg.get("patrol_route_progress_tol_m", 0.5),
            )
            # G2: nav-stall guard — candidate stays in assigned_to_ugv status for too
            #     many runner steps without ever becoming visited.
            _nav_stall_cand_id = None
            _nav_stall_start_step = None
            # B4（2026-09-17 审计）：阈值 config 化（nav 块优先、ugv 块回退，默认 20=旧行为）
            NAV_STALL_MAX = _resolve_nav_stall_max_steps(self.config)
            _nav_stall_guard_events = 0  # B4：guard 触发计数器（终局写入 result 增量字段）
            # R29: G1 liveness trigger count (route stall or deterministic topology cycle).
            _patrol_loop_guard_events = 0
            # R27: unambiguous sparse logical-port vs executed XY consistency telemetry.
            # The older locator-ID mismatch is representation-dependent at junctions.
            _sparse_pose_drift_max_m = 0.0
            _sparse_pose_drift_over_2m_count = 0
            _guard_abort_info = None  # (kind, candidate_id) when a guard fires terminal abort
            # N5a TRACE-Core：route advisory episode 级状态（白板模式专用开关）
            _route_advisory_on = False
            _advisory_goal_cache = {}
            _route_advisory_steps = 0
            _route_advisory_activated = 0

            # ---- ASE / ACov tracking (2026-07-24 新增) ----
            # ASE: 目标首次进入 Oracle 可识别空中视野的 UAV 本地有效决策步
            # ACov: UAV 已观察过的搜索网格数（用 mark_visited 的网格 ID 衡量）
            _ase_first_oracle_visible_uav_step = None  # int | None
            _ase_uav_local_step = 0                    # UAV 本地有效决策步（不含被动冻结）
            _ase_sampling_valid = True
            _acov_observed_cells = set()               # 已观察网格 ID 集合
            _acov_observed_events = []                 # P1-2: (uav_local_step, cell_id) 时间线
            # 2026-08-02 修复（bug review）：分母改为"区域激活"航点数。
            # 旧实现 len(uav_wps) 用的是全地图网格目录，而分子(observed)只可能来自
            # UAVGridCatalog 的 active 过滤——分母远大于分子，ACov 被系统性压低
            # （easy 150m 区域的上限仅 ~0.1 量级）。与 UAVGridCatalog 同用 active 默认值。
            _acov_total_cells = sum(1 for w in uav_wps if w.get("active", True) and inside_task_region((w["x"], w["y"]), region))

            ugv_start_xy = (ugv_start["x"], ugv_start["y"])
            uav_start_xy = (uav_start["x"], uav_start["y"])

            uav_traj_records.append({
                "step_index": 0, "sim_time_s": 0.0,
                "world_x": round(uav_start["x"], 2), "world_y": round(uav_start["y"], 2),
                "world_z": round(uav_start.get("z", 80.0), 2),
                "yaw": round(uav_start.get("yaw", -90.0), 2),
                "selected_waypoint_id": "",
                "action_type": "SPAWN"})
            ugv_decision_records.append({
                "step_index": 0, "sim_time_s": 0.0,
                # Sparse benchmark start is already an exact control port; persist it so
                # SFT/export tooling can prove the very first discrete state as well.
                "waypoint_id": (state.sparse_current_port or ""),
                "world_x": round(ugv_start["x"], 2), "world_y": round(ugv_start["y"], 2),
                "yaw": round(ugv_start.get("yaw", 0.0), 2),
                "action_type": "SPAWN",
                "candidate_id_if_any": ""})

            # Locate UGV and target on the frozen graph
            ugv_loc = locator.locate(carla_iface.get_world(), ugv_start["x"], ugv_start["y"], ugv_start.get("z", 0.5), ugv_start.get("yaw", 0))
            target_loc_f = target_actor.get_location() if target_actor else None
            target_loc_pos = (target_loc_f.x, target_loc_f.y, target_loc_f.z) if target_loc_f else tgt_xy + (0.5,)
            target_loc_res = locator.locate(carla_iface.get_world(), target_loc_pos[0], target_loc_pos[1], target_loc_pos[2])
            current_fine_node = ugv_loc.fine_node_id if ugv_loc else None
            target_fine_node = target_loc_res.fine_node_id if target_loc_res else None

            from agos.eval.navigation_oracle import (oracle_required, deployment_binding,
                validate_certificate, fine_path_distance, spl_measurement, ORACLE_VERSION,
                LAYERED_ORACLE_VERSIONS, verify_deployed_certificate)
            _strict_spl = oracle_required(episode_plan, self.config)
            _navigation_certificate = None
            if _strict_spl:
                _navigation_binding = deployment_binding(episode_plan, self.config, store,
                    sparse_nav, carla_iface, target_actor, distractor_actors, sparse_nav_sha)
                # Evaluation-private evidence on preflight failure; never an
                # observation/Prompt field. Preserve exact mismatch magnitudes.
                with open(os.path.join(output_dir, "navigation_binding_preflight.json"), "w", encoding="utf-8") as _bf:
                    json.dump(_navigation_binding, _bf, indent=2, ensure_ascii=False)
                if _navigation_binding.get("oracle_version") in LAYERED_ORACLE_VERSIONS:
                    _navigation_certificate = verify_deployed_certificate(
                        episode_plan, self.config, store, sparse_nav, carla_iface,
                        target_actor, distractor_actors, sparse_nav_sha,
                        os.path.join(output_dir, "navigation_deployment_qa"))
                else:
                    try:
                        _navigation_certificate = validate_certificate(
                            episode_plan.get("navigation_oracle"), _navigation_binding)
                    except Exception as _nav_err:
                    # GSAM 两实验（2026-10-10 设计 §9.2 处置路径）：actor 物理沉降
                    # 噪声（z 轴 1-2mm，xy 全同）会让全有或全无的绑定 hash 判定随机
                    # 拒绝合法重放。按设计"重新生成合法证书、不关闭校验"处置：
                    # 严格审计约束（差异仅限 deployment.location.z 且 |dz|<=1cm，
                    # SPL 度量字段逐字段原样保留）下重绑；越界差异仍按原错误抛出。
                        _rebound = _rebind_navigation_certificate(
                            episode_plan.get("navigation_oracle"), _navigation_binding)
                        if _rebound is None:
                            raise
                        _navigation_certificate = _rebound[0]
                        logger.warning("[%s] navigation_oracle certificate rebound "
                                       "(physical settle noise dz=%s): %s",
                                       state.episode_id, _rebound[1], _nav_err)
                with open(os.path.join(output_dir, "navigation_oracle.json"), "w", encoding="utf-8") as _nf:
                    json.dump(_navigation_certificate, _nf, indent=2, ensure_ascii=False)

            # Compute oracle UGV-SPL distance via road graph (explicit legacy only).
            # 2026-07-23 修正: oracle 最短路径终点 = 最近合法确认位置(certified approach node),
            # 而非"目标最近路网节点"。用 plan 的 static_solvability_precheck.certified_route_length_m。
            oracle_ugv_distance_m = 0.0
            precheck = episode_plan.get("static_solvability_precheck") or {}
            certified_route_len = precheck.get("certified_route_length_m")
            if _strict_spl:
                oracle_ugv_distance_m = float(_navigation_certificate["distance_m"])
            elif certified_route_len and certified_route_len > 0:
                oracle_ugv_distance_m = float(certified_route_len)
                logger.info("[%s] SPL oracle distance from certified approach node: %.1fm",
                            state.episode_id, oracle_ugv_distance_m)
            elif current_fine_node and target_fine_node:
                # fallback: 起点到目标最近节点
                route = router.plan(current_fine_node, target_fine_node, cost_mode="distance")
                if isinstance(route, RouteFailure):
                    logger.warning(f"Road graph routing failed for oracle UGV distance: {route.code}")
                    oracle_ugv_distance_m = math.hypot(tgt_xy[0] - ugv_start_xy[0], tgt_xy[1] - ugv_start_xy[1])
                else:
                    oracle_ugv_distance_m = route.path_length_m
            if not _strict_spl and oracle_ugv_distance_m <= 0:
                oracle_ugv_distance_m = math.hypot(tgt_xy[0] - ugv_start_xy[0], tgt_xy[1] - ugv_start_xy[1])
            try:
                oracle_view = episode_plan.get("solvability_oracle", {})
                oracle_uav_wp_id = oracle_view.get("oracle_uav_waypoint_id")
                oracle_uav_wp = next((w for w in uav_wps if w["id"] == oracle_uav_wp_id), None)
                if oracle_uav_wp:
                    oracle_uav_xy = (oracle_uav_wp["x"], oracle_uav_wp["y"])
                    oracle_uav_distance_m = math.hypot(oracle_uav_xy[0] - uav_start_xy[0], oracle_uav_xy[1] - uav_start_xy[1])
                else:
                    oracle_uav_distance_m = math.hypot(tgt_xy[0] - uav_start_xy[0], tgt_xy[1] - uav_start_xy[1])
            except Exception:
                oracle_uav_distance_m = math.hypot(tgt_xy[0] - uav_start_xy[0], tgt_xy[1] - uav_start_xy[1])

            max_confirm_distance_m = confirm_distance_m(self.config)

            # 冻结超时兜底：UAV 冻结后若 UGV 长时间未激活（迷路/卡死/未发消息），
            # 框架强制激活 UAV，从机制上杜绝无限冻结。单位：scheduler step。
            UAV_FROZEN_TIMEOUT_STEPS = int(self.config.get("uav", {}).get("frozen_timeout_steps", 25))
            if bool(self._diag_cfg.get("oracle_candidate_injection", False)):
                # GNG-1：UAV 全程冻结（只测 UGV 执行层），禁用冻结超时强制解冻，
                # 防止 UAV 决策挤占共享 VLM 预算污染归因。
                UAV_FROZEN_TIMEOUT_STEPS = 10 ** 9

            # ---- SIGTERM 保底：supervisor 墙钟超时时发 SIGTERM（先 SIGTERM 再 SIGKILL），
            # 收到信号后设标志位，runner 在下一步检查时提前退出，确保能走到 result 写入。
            # R2-P1-07：同时触发 VLM 取消（retry sleep 分片检查），阻塞在 backoff sleep
            # 中的调用会在 <=0.5s 内抛 VLMCancelledError，由主循环 except 走同款收尾。
            clear_cancel()  # 防同进程上一 episode 残留 cancel flag
            _sigterm_received = {"flag": False}
            def _sigterm_handler(signum, frame):
                _sigterm_received["flag"] = True
                request_cancel()  # R2-P1-07: 打断 VLM retry sleep
                logger.warning("[%s] SIGTERM received (wall-clock timeout), will write partial result and exit.",
                               state.episode_id)
            _prev_sigterm_handler = None
            try:
                import signal as _signal
                _prev_sigterm_handler = _signal.signal(_signal.SIGTERM, _sigterm_handler)
            except Exception:
                pass  # Windows or restricted env

            def _check_local_osr():
                _check_ugv_osr(state, target_actor, self.config, carla_iface)

            # I1 帧转储跨步状态（Phase-G Step 6；config 无 i1_frame_dump 键时零开销）
            _i1_fvs = None           # 目标首次 oracle 可见步（decide 后回填，供后续步触发判定）
            _i1_gt_vis_prev = False  # 上一步 oracle 六视图可见（decide 前只有上一步信息）
            _i1_assigned_prev = False
            _i1_arrival_step = None  # candidate 到达沿（assigned 上升沿）所在步
            _i1_last_trig = None     # P0-I01：本步 I1 触发原因（decide 后写 meta_post.json 用）

            _task_attempt = int(((self.config or {}).get("evaluation") or {}).get("task_attempt", 1) or 1)
            while not state.terminal and state.step < state.max_steps:

                uav_turn = state.uav_sim_time <= state.ugv_sim_time
                # Oracle mode does not consume VLM budget; skip the VLM budget check.
                if state.vlm_calls >= state.max_vlm_calls:
                    break
                if _sigterm_received["flag"]:
                    state.terminal = True
                    # Supervisor budget SIGTERM ends model evaluation through
                    # normal metric/fault finalization; user stop is fenced by parent.
                    state.failure_stage = "EPISODE_WALL_TIMEOUT"
                    logger.warning("[%s] Exiting loop due to SIGTERM at step %d/%d.",
                                   state.episode_id, state.step, state.max_steps)
                    break


                # ---- G2 nav-stall guard (checked every scheduler tick) ----
                _ac = ledger.get_active()
                if _ac is not None and _ac.status == "assigned_to_ugv":
                    if _nav_stall_cand_id != _ac.candidate_id:
                        _nav_stall_cand_id = _ac.candidate_id
                        _nav_stall_start_step = state.step
                    elif state.step - _nav_stall_start_step > NAV_STALL_MAX:
                        logger.warning("[%s] GUARD nav-stall: candidate %s stuck in assigned_to_ugv for %d steps; marking unserviceable",
                                       state.episode_id, _ac.candidate_id, state.step - _nav_stall_start_step)
                        _nav_stall_guard_events += 1  # B4：每次触发 +1
                        ledger.mark_unserviceable(_ac.candidate_id, reason="nav_stall_guard")
                        msg_bus.send("UGV", "Candidate %s navigation stalled; aborting verification." % _ac.candidate_id,
                                     state.ugv_sim_time, receiver="UAV", message_type="unreachable",
                                     related_candidate_id=_ac.candidate_id, step=state.step)
                        state.add_ugv_memory(action="GUARD_NAV_STALL",
                            params={"candidate_id": _ac.candidate_id},
                            result="candidate marked unserviceable", outcome="candidate aborted")
                        state.uav_frozen = False
                        _nav_stall_cand_id = None
                        _nav_stall_start_step = None
                        _patrol_guard.reset()
                        _guard_abort_info = ("UGV_NAV_STALLED", _ac.candidate_id)
                else:
                    _nav_stall_cand_id = None
                    _nav_stall_start_step = None

                # Frozen UAV: do not consume a step. Advance only the UAV clock so the
                # Public map publication does not advance either scheduling clock.
                # event scheduler picks the UGV next; the UAV stays suspended until the
                # framework (or UGV) activates it again.
                if _is_coop_mode(self.agent_mode):
                    state.uav_frozen = False  # dedicated methods continue aerial observation
                if uav_turn and state.uav_frozen:
                    # 冻结超时兜底：机制性强制激活，杜绝任何路径下的无限冻结
                    if (state.uav_frozen_since_step is not None
                            and state.step - state.uav_frozen_since_step > UAV_FROZEN_TIMEOUT_STEPS):
                        logger.warning("[%s] GUARD uav-frozen-timeout: UAV frozen for %d steps with no activation; force-activating",
                                       state.episode_id, state.step - state.uav_frozen_since_step)
                        self._framework_activate_uav(state, msg_bus, "unreachable",
                            candidate_id="", extra="(UAV frozen-timeout fallback; resume exploration)")
                        # fall through to normal UAV decision below (do NOT skip step)
                    else:
                        dt = compute_action_sim_time("WAIT", "UAV",
                            (state.uav_pos["x"], state.uav_pos["y"]),
                            (state.uav_pos["x"], state.uav_pos["y"]), uav_speed,
                            time_model=state.time_model)
                        state.uav_sim_time += dt
                        continue

                state.step += 1

                if uav_turn:
                    logger.info("[%s] UAV step %d sim_t=%.1fs active_cand=%s frozen=%s", state.episode_id, state.step, state.uav_sim_time, ledger.get_active(), state.uav_frozen)
                    uav_img_raw = _capture_observation_uav(carla_iface, map_protocol, state.uav_sim_time)
                    uav_img = uav_img_raw  # current methods consume clean RGB
                    if self._diag_save_all_uav_frames:
                        try:
                            with open(os.path.join(img_dir, "uav_raw_s%03d.png" % state.step), "wb") as _f:
                                _f.write(uav_img_raw)
                        except Exception as _se:
                            logger.warning("[%s] save_all_uav_frames failed at step %d: %s", state.episode_id, state.step, _se)
                    topdown = b"" if map_protocol.excludes_fullcity else render_topdown(map_frame, (state.uav_pos["x"], state.uav_pos["y"]),
                        (state.ugv_pos["x"], state.ugv_pos["y"]), ledger.candidates,
                        state.uav_trajectory, state.ugv_trajectory, road_graph_store=store,
                        background_image_bytes=global_rgb_bytes, show_road_network=False, region=region)
                    if self._diag_demo_record and not map_protocol.excludes_fullcity:
                        # 演示素材：本步 VLM 输入（UAV 航拍+网格）；增强版全局俯视仅存图
                        # （大图标/粗轨迹/Target 红星），不进 VLM 输入；干净版另存留档
                        self._demo_save(uav_img, os.path.join(img_dir, "uav_input_s%03d.png" % state.step))
                        self._demo_save(topdown, os.path.join(img_dir, "topdown_clean_s%03d.png" % state.step))
                        self._demo_save_annotated_topdown(map_frame, store, ledger, region,
                            global_rgb_bytes, img_dir, state.step, state,
                            target_actor=target_actor, fallback_topdown=topdown)
                    uav_catalog = UAVGridCatalog(uav_wps, (state.uav_pos["x"], state.uav_pos["y"]), spacing=wp_spacing, region=region)
                    avail_wps = uav_catalog.get_available()
                    avail_ids = {w["waypoint_id"] for w in avail_wps}
                    active_cand = ledger.get_active()
                    # GSAM T1（2026-10-10 设计 §4.1）：正常轮干净图，ANNOTATE 菜单项
                    # 换成 SUSPECT_CANDIDATE（仅触发信号，method 层 pipeline 必然转换）。
                    _gsam_on = bool(_observation_cfg.get("gsam_localizer_enabled", False))
                    uav_available_actions = ["MOVE_TO_UAV_WAYPOINT", "WAIT"]
                    if active_cand is None:
                        # r7: benchmark 给三方法同一动作全集；
                        uav_available_actions.append(
                            "SUSPECT_CANDIDATE" if _gsam_on else "ANNOTATE_CANDIDATE")
                        # 2026-07-23 改动2a：允许 UAV 弃权(无active候选时才可标注/弃权)
                        uav_available_actions.append("NO_RELIABLE_CANDIDATE")
                    elif _q_enabled and ledger.shadow_count() < _q_max_shadow:

                        # annotating into the shadow queue (queue not full).
                        uav_available_actions.append(
                            "SUSPECT_CANDIDATE" if _gsam_on else "ANNOTATE_CANDIDATE")
                    uav_obs = LiteObservation(
                        episode_id=state.episode_id, step=state.step, max_steps=state.max_steps,
                        role="UAV",
                        agent_image=uav_img_raw,
                        topdown_map=topdown,
                        agent_position=(state.uav_pos["x"], state.uav_pos["y"], 80),
                        observation_mode="UAV_AERIAL", target_description=state.target_description,
                        candidates=ledger.candidates, active_candidate=active_cand,
                        available_waypoints=avail_wps, available_actions=uav_available_actions,
                        recent_messages=(msg_bus.get_visible_for("UAV", state.uav_sim_time)
                                         if _is_coop_mode(self.agent_mode) else
                                         msg_bus.get_recent_for("UAV", state.uav_sim_time)),
                        annotation_count=state.annotation_attempt_count,
                        max_annotations=self.config["uav"]["max_candidate_annotations"],
                        region_boundary=(region["x_min"], region["x_max"], region["y_min"], region["y_max"]),
                        target_reference_image=target_ref_img, teammate_reference_image=teammate_ref_img,
                        target_reference_views=ref_views,
                        uav_frozen=state.uav_frozen)
                    uav_obs.candidate_publication_check = _public_point_publication_check(
                        ledger, q_enabled=_q_enabled, q_max_shadow=_q_max_shadow,
                        suppression_mode=_dup_suppression_mode,
                        min_shift_m=_r20_revision_min_shift)
                    map_protocol.attach(uav_obs, state.uav_sim_time, state.ugv_sim_time,
                                        (state.ugv_pos["x"], state.ugv_pos["y"]), state.ugv_trajectory)
                    if _is_coop_mode(self.agent_mode):
                        uav_obs.coop_event_time = state.uav_sim_time
                    uav_obs.uav_memory_text = state.get_uav_memory_lines()
                    # R18 visual tools need the canonical raw sensor image, not
                    # the grid overlay used by the legacy VLM prompt.
                    uav_obs.raw_agent_image = uav_img_raw
                    if _bind_uav_frame(self.agent_mode, _gsam_on):


                        # 绑定图像、位姿与 frame ID。
                        # agent_image。
                        _attach_uav_observation_frame(uav_obs,
                            getattr(map_protocol, 'last_frame', None), uav_img_raw)

                    # Budget-isolation support: a 100-step run can advertise 80
                    # for steps <=80, preserving the first-80 prompt horizon.
                    if _policy_horizon_cfg > 0:
                        uav_obs.policy_horizon_steps = (_policy_horizon_cfg if state.step <= _policy_horizon_cfg
                                                        else state.max_steps)

                    # Oracle mode does not consume VLM budget (OracleAgent has no VLM call).
                    uav_obs.internal_vlm_calls_remaining = max(
                        0, state.max_vlm_calls - state.vlm_calls -
                        int(_role_uses_vlm(self.agent_mode, uav_agent, "UAV")))


                    _uav_step_uses_vlm = _role_uses_vlm(self.agent_mode, uav_agent, "UAV")
                    if _uav_step_uses_vlm:
                        state.vlm_calls += 1
                        state.uav_vlm_calls += 1
                    _task_attempt = int(((self.config or {}).get("evaluation") or {})
                                       .get("task_attempt", 1) or 1)
                    try:
                        from agos.runtime.paper_audit import decide_and_account
                        uav_action = decide_and_account(
                            lambda: _method_decide(self.agent_mode, uav_agent,
                                uav_obs, task_attempt=_task_attempt), state, uav_agent, "UAV")
                    except VLMCancelledError:
                        # R2-P1-07: 墙钟超时打断阻塞中的 VLM 调用——回退预算计数，
                        # 终止 episode 并回循环顶部（terminal 已置位 → 正常退出走统一收尾）
                        if _uav_step_uses_vlm:
                            state.vlm_calls = max(0, state.vlm_calls - 1)
                            state.uav_vlm_calls = max(0, state.uav_vlm_calls - 1)
                        state.terminal = True
                        state.failure_stage = "EPISODE_WALL_TIMEOUT"
                        continue

                    # consume the same VLM budget as primary decisions.

                    # P0-6: 收集 AG-OS 模块证据 trace

                    # （build_trace 的 agent 字段区分 UAV/UGV；纯观测性改动，零行为影响）
                    _audit_role = "UAV"
                    if not (_collect_method_trace(self.agent_mode) and hasattr(uav_agent, "build_trace")):
                        try:
                            from agos.runtime.paper_audit import record_tool_faults
                            record_tool_faults(state, uav_obs, {})
                        except Exception as _audit_error:
                            from agos.runtime.paper_audit import record_audit_failure
                            record_audit_failure(state, _audit_role, state.step, _audit_error)
                    if _collect_method_trace(self.agent_mode) and hasattr(uav_agent, "build_trace"):
                        try:
                            _tr = uav_agent.build_trace(uav_obs); _tr["step"] = state.step
                            from agos.runtime.paper_audit import record_tool_faults
                            record_tool_faults(state, uav_obs, _tr)
                            if hasattr(uav_obs, "map_metadata"):
                                _tr["online_map"] = uav_obs.map_metadata
                            ag_os_trace_records.append(_tr)
                        except Exception as _e:
                            from agos.runtime.paper_audit import record_audit_failure
                            record_audit_failure(state, _audit_role, state.step, _e)
                            logger.warning("trace collect failed: %s", _e)
                    # P0-09: infra error - skip this step entirely, don't count as model decision
                    # 2026-09-04 用户指令：不再因 VLM 连续失败终止 episode（API 不稳定只等待+告警）。
                    # VLMClient 已对 429/5xx/超时/网络抖动无限重试；此处能计数的只剩
                    # 鉴权/参数类确定性错误——只周期性 ERROR 告警提示人工介入，永不放弃任务。
                    if uav_action.action == "INFRA_ERROR":
                        from agos.runtime.paper_audit import record_primary_infra
                        record_primary_infra(state, uav_obs, uav_action)
                        # R2-P1-06：infra 未产生有效模型决策，回退本步预算计数
                        # （vlm_calls 口径=有效决策数；否则偶发空回复会误耗预算）。
                        if _uav_step_uses_vlm:
                            state.vlm_calls = max(0, state.vlm_calls - 1)
                            state.uav_vlm_calls = max(0, state.uav_vlm_calls - 1)
                        state.consecutive_vlm_failures = getattr(state, "consecutive_vlm_failures", 0) + 1
                        state.infra_error_count = getattr(state, "infra_error_count", 0) + 1
                        _max_fail = getattr(state, "max_consecutive_vlm_failures", 5)
                        logger.warning("[%s] UAV VLM infra error (%d consecutive): %s",
                                       state.episode_id, state.consecutive_vlm_failures, uav_action.reason)
                        if state.consecutive_vlm_failures == _max_fail or state.consecutive_vlm_failures % 25 == 0:
                            logger.error("[%s] UAV VLM 已连续失败 %d 次（疑似鉴权/参数类确定性错误，"
                                         "请人工检查 API 配置）；按机制继续等待重试，不终止本任务",
                                         state.episode_id, state.consecutive_vlm_failures)
                        continue  # 不消耗步数，不推进仿真时间，直接下一步

                    # 成功调用：清零连续失败计数器
                    state.consecutive_vlm_failures = 0

                    # ---- ASE/ACov: UAV 本地有效决策步 + 首次 Oracle 空中可见检测 ----
                    # 2026-07-24 新增。UAV 每执行一次有效决策（已 capture 新图像、未触发 INFRA_ERROR），
                    # 计为一次本地有效步骤；同时检查目标是否在当前 UAV 相机视野中 Oracle 可见。
                    _ase_uav_local_step += 1
                    # ACov: 记录当前 UAV 所在网格为已观察（该网格对应一个 40m 搜索单元）
                    _cur_uav_cell_id = None
                    for _w in uav_wps:
                        if abs(_w["x"] - state.uav_pos["x"]) < wp_spacing / 2 and abs(_w["y"] - state.uav_pos["y"]) < wp_spacing / 2:
                            _cur_uav_cell_id = _w["id"]
                            break
                    if _cur_uav_cell_id:
                        if _cur_uav_cell_id not in _acov_observed_cells:
                            _acov_observed_events.append((_ase_uav_local_step, _cur_uav_cell_id))
                        _acov_observed_cells.add(_cur_uav_cell_id)
                    # ASE: actual captured predecision frame, fixed decision budget.
                    # Failed Oracle probes are metric invalidity, not ordinary misses.
                    if _ase_first_oracle_visible_uav_step is None:
                        _ase_probe = probe_uav_observation(target_actor,
                            getattr(map_protocol, 'last_frame', None), carla_iface.get_world(),
                            self._los_fail_closed())
                        if not _ase_probe['valid']:
                            _ase_sampling_valid = False
                            state.metric_sampling_errors.append({'metric': 'ASE',
                                'step': state.step, 'detail': _ase_probe})
                        elif _ase_probe['visible']:
                            _ase_first_oracle_visible_uav_step = _ase_uav_local_step

                    if cecm:
                        raw = getattr(uav_agent, "last_raw_response", "")
                        if raw:
                            cecm.check_vlm(raw)
                        cecm.check_trajectory(state.uav_trajectory, threshold_m=5.0)
                    # 2026-07-26 plain-baseline 可观测性：区分 模型真实WAIT / 解析降级 / 校验降级
                    _uav_policy_debug = {}
                    try:
                        if hasattr(uav_agent, "get_last_policy_debug"):
                            _uav_policy_debug = dict(uav_agent.get_last_policy_debug("UAV") or {})
                    except Exception:
                        _uav_policy_debug = {}
                    _uav_proposed = uav_action  # legacy field: method-resolved pre-validator action
                    _uav_parse_status = getattr(uav_agent, "last_parse_status", "")
                    _uav_validation_success = True
                    _uav_fallback_reason = ""
                    valid, reason = validate_action(uav_action.__dict__ if hasattr(uav_action, "__dict__") else {}, uav_obs, avail_ids)
                    if not valid:
                        state.invalid_action_count += 1
                        _uav_validation_success = False
                        _uav_fallback_reason = reason
                        _raw_before_downgrade = getattr(uav_action, "action", "?")
                        uav_action = LiteAction(role="UAV", action="WAIT", reason="invalid: %s" % reason)
                        if _uav_framework_feedback:
                            # B11（2026-09-17 审计）：框架反馈回注——让 UAV 在下轮
                            # 决策记忆中看到降级原因（默认关=零变化）
                            state.add_uav_memory(
                                "[framework] 上一次动作 %s 被拒绝：%s，本轮已置 WAIT"
                                % (_raw_before_downgrade, reason))

                    # 连续 WAIT >= 3 步且仍有未访问 waypoint 时强制 MOVE 打破循环。
                    uav_action = enforce_uav_move(uav_action, uav_catalog)
                    prev = (state.uav_pos["x"], state.uav_pos["y"])
                    msg = getattr(uav_action, "message", "")
                    if msg:
                        msg_bus.send("UAV", msg, state.uav_sim_time,
                            receiver="UGV", message_type="info", step=state.step)
                    if uav_action.action == "MOVE_TO_UAV_WAYPOINT" and uav_action.waypoint_id:
                        for wp in uav_wps:
                            if wp["id"] == uav_action.waypoint_id:
                                # 2026-07-24: UAV yaw 统一为 0（朝北），与 set_uav_pose 默认值一致
                                carla_iface.set_uav_pose(wp["x"], wp["y"], wp["z"], 0)
                                state.uav_pos = {"x": wp["x"], "y": wp["y"], "z": wp["z"], "yaw": 0}
                                state.uav_trajectory.append((wp["x"], wp["y"]))
                                uav_catalog.mark_visited(wp["id"])
                                uav_distance += math.sqrt((wp["x"]-prev[0])**2 + (wp["y"]-prev[1])**2)
                                break
                    elif uav_action.action == "NO_RELIABLE_CANDIDATE":
                        # 2026-07-23 改动2a：UAV 弃权（同色多车/车型不清/像素过小）。
                        # 不创建候选、不冻结 UAV、不消耗标注机会, 记录后让其继续探索。
                        # 实测(3.2)证明 abstain 能减少"凭颜色硬猜"的错误候选(它们会浪费 UGV 全部预算)。
                        state.add_uav_memory(
                            action="NO_RELIABLE_CANDIDATE",
                            params={"confidence": uav_action.confidence},
                            result="abstained: no reliable candidate (same-color ambiguity or low resolution)",
                            outcome="continue_exploration")
                        logger.info("[%s] UAV abstained (NO_RELIABLE_CANDIDATE): %s",
                                    state.episode_id, (uav_action.reason or "")[:120])
                    elif uav_action.action == "ANNOTATE_CANDIDATE":
                        state.annotation_action_outputs += 1
                        cand = None  # ensure defined for memory logging below even if grid_cell missing
                        _attempt_rec = {
                            "episode_id": state.episode_id, "step_index": state.step,
                            "grid_cell": getattr(uav_action, "grid_cell", None),
                            "confidence": getattr(uav_action, "confidence", None),
                            "active_candidate_id_before": getattr(ledger.get_active(), "candidate_id", None),
                            "active_candidate_status_before": getattr(ledger.get_active(), "status", None),
                            "shadow_requested": False, "ledger_status": "not_attempted",
                            "final_status": "not_attempted", "candidate_id": None,
                            "geometry_source": None,
                            "r20_revision_enabled": False,
                            "r20_revision_mode": "shadow",
                            "r20_revision_would_apply": False,
                            "r20_revision_parent_id": None,
                            "r20_revision_parent_status": None,
                            "r20_revision_shift_m": None,
                            "r20_revision_reason": None,
                        }
                        candidate_attempt_records.append(_attempt_rec)

                        # the shadow queue holds items), a new annotation becomes a
                        # FIFO shadow candidate instead of blocking the search.
                        _q_shadow_this = bool(
                            _q_enabled and (ledger.get_active() is not None or ledger.shadow_count() > 0))
                        _attempt_rec["shadow_requested"] = bool(_q_shadow_this)
                        _point_geom = str(getattr(uav_action, "geometry_type", "") or "")
                        _point_pos = getattr(uav_action, "position_xy", None)
                        if _point_geom == "point" and _point_pos:
                            # === GSAM 点候选分支（2026-10-10 设计 §5/§9.1）===
                            # position 来自工具投影（源帧位姿+几何真源）；polygon 空、
                            # grid_cell=None；服务池=点级三级可达观察port；导航/到达/终局
                            # 全部复用 CandidateServiceContract 与 SparseRouteTool 原路径。
                            _pos = (float(_point_pos[0]), float(_point_pos[1]))
                            _point_cfg = (_observation_cfg)
                            _gsam_unc = float(_point_cfg.get("gsam_point_uncertainty_m", 15.0) or 15.0)
                            cand, st = ledger.create("POINT", [], _pos, _gsam_unc,
                                uav_action.confidence, uav_action.evidence, state.step,
                                shadow=_q_shadow_this, max_shadow=_q_max_shadow,
                                suppression_mode=_dup_suppression_mode,
                                min_shift_m=_r20_revision_min_shift)
                            _attempt_rec["ledger_status"] = st
                            _attempt_rec["geometry_source"] = (getattr(uav_action, "localization_source", "") or "gsam_point")
                            _attempt_rec["candidate_id"] = getattr(cand, "candidate_id", None)
                            if cand is not None:
                                cand.geometry_type = "point"
                                cand.grid_cell = None
                                cand.source_frame_id = getattr(uav_action, "source_frame_id", None)
                                cand.source_image_sha256 = getattr(uav_action, "source_image_sha256", "")
                                # 服务性预检：点级三级池（5-20/5-35/5-50m，首非空档）
                                _pt_bands = _observation_cfg.get("gsam_point_approach_bands_m", (20, 35, 50))
                                _pt_min = float(_observation_cfg.get("gsam_point_approach_min_m", 5.0) or 5.0)
                                _pt_pool, _pt_band = compute_point_service_nodes(
                                    store, _pos, bands_m=(tuple(_pt_bands)
                                                          if isinstance(_pt_bands, (list, tuple)) else (20, 35, 50)),
                                    min_m=_pt_min,
                                    max_band_m=float(self.config["success"]["max_confirm_distance_m"]))
                                _attempt_rec["point_service_band_m"] = _pt_band
                                if current_fine_node is not None and sparse_nav is not None and _pt_pool:
                                    _serviceable, _svc_dist, _inside_cnt = sparse_service_pool_reachable(
                                        sparse_nav, current_fine_node, _pt_pool,
                                        intersection_max_distance_m=_sparse_intersection_max_distance(self.config),
                                        max_route_m=self.config["ugv"].get("approach_route_max_m", None))
                                    _reason = ("GSAM_POINT_SERVICEABLE band=%.0fm ports=%d best_m=%.1f"
                                               % (_pt_band or 0.0, _inside_cnt, _svc_dist)) if _serviceable \
                                              else ("GSAM_POINT_NO_SERVICE_PORT band=%.0fm" % (_pt_band or 0.0))
                                else:
                                    _serviceable, _reason, _inside_cnt = False, "GSAM_POINT_EMPTY_POOL", 0
                                if not _serviceable:
                                    ledger.rollback_reopen_charge(cand)
                                    ledger.candidates.remove(cand)
                                    state.candidates = ledger.candidates
                                    logger.info("[%s] Point candidate %s at (%.1f,%.1f) rejected: no reachable UGV node (%s)",
                                                state.episode_id, cand.candidate_id, _pos[0], _pos[1], _reason)
                                    state.add_uav_memory(
                                        action="ANNOTATE_CANDIDATE",
                                        params={"geometry_type": "point", "confidence": uav_action.confidence},
                                        result="rejected: no reachable UGV observation port near the point",
                                        outcome="candidate_rejected_unserviceable")
                                    _attempt_rec["final_status"] = "serviceability_rejected"
                                    _attempt_rec["serviceability_reason"] = _reason
                                    cand = None
                                else:
                                    _attempt_rec["serviceability_reason"] = _reason
                                    cand.service_node_ids = tuple(_pt_pool)
                                    cand.confirmable_service_node_ids = tuple(
                                        node_id for node_id, node in store.nodes.items()
                                        if math.hypot(node.x - _pos[0], node.y - _pos[1])
                                        <= float(self.config["success"]["max_confirm_distance_m"]))
                                    cand.prefer_confirmable_service = bool(_confirmable_service_arrival)
                            if cand:
                                _attempt_rec["final_status"] = ("created_shadow" if cand.status == "shadow" else "created_primary")
                                state.annotation_attempt_count += 1
                                state.candidate_created_count += 1
                                state.candidates = ledger.candidates
                                # 评估诊断方形（仅 grid_hit/ACG/地图显示，不入服务/路由合同）
                                poly = point_diagnostic_square(_pos, ACG_POINT_HALF_SIZE_M)
                                _cand_hit = False
                                if target_actor:
                                    tl = target_actor.get_location()
                                    if _point_in_polygon((tl.x, tl.y), poly):
                                        state.grid_hit = True
                                        state.first_hit_candidate = cand
                                        state.candidate_hit_count += 1
                                        _cand_hit = True
                                if target_actor:
                                    _tl = target_actor.get_location()
                                    _acg_score, _acg_dist, _acg_scale = compute_acg_score(
                                        (_tl.x, _tl.y), poly)
                                    state.candidate_acg_scores.append({
                                        "candidate_id": cand.candidate_id,
                                        "step_index": state.step,
                                        "grid_hit": _cand_hit,
                                        "distance_m": round(_acg_dist, 6),
                                        "distance_to_candidate_polygon_m": round(_acg_dist, 6),
                                        "candidate_scale_m": round(_acg_scale, 6),
                                        "acg_score": round(_acg_score, 6),
                                        "diagnostic_half_size_m": ACG_POINT_HALF_SIZE_M,
                                        "point_center_error_m": math.hypot(_tl.x-_pos[0], _tl.y-_pos[1]),
                                    })
                                msg_bus.send("UAV", "Candidate %s at point (%.1f, %.1f) (approximate location)."
                                    % (cand.candidate_id, _pos[0], _pos[1]),
                                    state.uav_sim_time, receiver="UGV", message_type="candidate_annotation",
                                    related_candidate_id=cand.candidate_id, step=state.step)
                                _is_shadow = cand.status == "shadow"
                                _last_shadow_event = {"event": ("enqueue" if _is_shadow else ""),
                                                      "candidate_id": cand.candidate_id}
                                _note_nadir_lifecycle(state, uav_action, shadow=_is_shadow)
                                img_path = os.path.join(img_dir, "uav_candidate_%s_input.png" % _fs_tag(cand.candidate_id))
                                with open(img_path, "wb") as f:
                                    f.write(uav_img_raw)
                                _last_annotate_img["path"] = img_path
                                candidate_log_records.append({
                                    "candidate_id": cand.candidate_id, "shadow": _is_shadow,
                                    "source_step_index": state.step, "source_agent_id": "UAV",
                                    "source_frame_id": cand.source_frame_id,
                                    "source_image_sha256": cand.source_image_sha256,
                                    "source_image_path": img_path,
                                    "uav_world_x": float(state.uav_pos["x"]),
                                    "uav_world_y": float(state.uav_pos["y"]),
                                    "uav_world_z": float(state.uav_pos.get("z", 80.0)),
                                    "uav_yaw": float(state.uav_pos.get("yaw", 0.0)),
                                    "geometry_type": "point", "grid_cell": None,
                                    "geometry_source": _attempt_rec["geometry_source"],
                                    "candidate_center_world_x": _pos[0],
                                    "candidate_center_world_y": _pos[1],
                                    "candidate_polygon_world": [],
                                    "diagnostic_polygon_world": poly,
                                    "uncertainty_radius_m": _gsam_unc,
                                    "vlm_confidence": uav_action.confidence,
                                    "vlm_reason": uav_action.reason,
                                    "raw_vlm_text": getattr(uav_agent, "last_raw_response", "")[:800],
                                    "grid_hit": _cand_hit,
                                    "acg_distance_m": _acg_dist if target_actor else None,
                                    "acg_scale_m": _acg_scale if target_actor else None,
                                    "acg_score": _acg_score if target_actor else None,
                                })
                    dt = compute_action_sim_time(uav_action.action, "UAV", prev,
                        (state.uav_pos["x"], state.uav_pos["y"]), uav_speed,
                        time_model=state.time_model)
                    state.uav_sim_time += dt
                    if uav_action.action == "ANNOTATE_CANDIDATE":
                        state.add_uav_memory(action=uav_action.action,
                            params={"geometry_type": getattr(uav_action, "geometry_type", None),
                                    "position_xy": getattr(uav_action, "position_xy", None),
                                    "confidence": getattr(uav_action, "confidence", 0.5)},
                            result=("annotated candidate %s" % cand.candidate_id) if cand else "annotation suppressed")
                    elif uav_action.action == "MOVE_TO_UAV_WAYPOINT":
                        state.add_uav_memory(action=uav_action.action,
                            params={"waypoint_id": uav_action.waypoint_id or "?"}, result="moved")
                    elif uav_action.action == "WAIT":
                        state.add_uav_memory(action="WAIT", result="waiting")
                    else:
                        state.add_uav_memory(action=uav_action.action, result="done")
                    step_logs.append({"step": state.step, "role": "UAV", "action": uav_action.action,
                        "sim_time": round(state.uav_sim_time, 1), "visual_description": uav_action.visual_description,
                        "pos_x": round(state.uav_pos["x"], 1), "pos_y": round(state.uav_pos["y"], 1),
                        "agent_yaw": round(state.uav_pos.get("yaw", -90.0), 1),
                        "waypoint_id": uav_action.waypoint_id, "grid_cell": getattr(uav_action, "grid_cell", ""),
                        "confidence": getattr(uav_action, "confidence", None),
                        "reason": getattr(uav_action, "reason", "") or "",
                        "input_image_path": _last_annotate_img["path"] if uav_action.action == "ANNOTATE_CANDIDATE" else "",
                        "raw_response": getattr(uav_agent, "last_raw_response", "")[:500]})
                    # Spec v2 §4.4/§4.5: action log + UAV decision trajectory
                    state.uav_action_count += 1
                    uav_traj_records.append({
                        "step_index": state.step, "sim_time_s": round(state.uav_sim_time, 1),
                        "world_x": round(state.uav_pos["x"], 2), "world_y": round(state.uav_pos["y"], 2),
                        "world_z": round(state.uav_pos.get("z", 80.0), 2),
                        "yaw": round(state.uav_pos.get("yaw", -90.0), 2),
                        "selected_waypoint_id": uav_action.waypoint_id or "",
                        "action_type": uav_action.action})
                    action_log_records.append({
                        "step_index": state.step, "sim_time_s": round(state.uav_sim_time, 1),
                        "agent_id": "UAV", "action_type": uav_action.action,
                        "raw_action_text": getattr(uav_agent, "last_raw_response", "")[:800],
                        "parse_success": (True if not _uav_parse_status else _uav_parse_status == "ok"),
                        "parse_status": _uav_parse_status or "unknown",
                        "validation_success": _uav_validation_success,
                        "fallback_applied": (not _uav_validation_success) or (_uav_parse_status not in ("", "ok")),
                        "fallback_reason": _uav_fallback_reason or (_uav_parse_status if _uav_parse_status not in ("", "ok") else ""),
                        # R18 canonical action provenance.  Legacy proposed_* below
                        # is the method-resolved pre-validator action, not raw VLM.
                        "model_raw_action_type": ((_uav_policy_debug.get("raw") or {}).get("action") or ""),
                        "model_raw_grid_cell": ((_uav_policy_debug.get("raw") or {}).get("grid_cell") or ""),
                        "model_raw_waypoint_id": ((_uav_policy_debug.get("raw") or {}).get("waypoint_id") or ""),
                        "method_resolved_action_type": ((_uav_policy_debug.get("resolved") or {}).get("action") or getattr(_uav_proposed,"action","")),
                        "method_resolved_grid_cell": ((_uav_policy_debug.get("resolved") or {}).get("grid_cell") or getattr(_uav_proposed,"grid_cell","") or ""),
                        "proposed_action_type": getattr(_uav_proposed, "action", ""),
                        "proposed_waypoint_id": getattr(_uav_proposed, "waypoint_id", "") or "",
                        "proposed_grid_cell": getattr(_uav_proposed, "grid_cell", "") or "",
                        "proposed_candidate_id": getattr(_uav_proposed, "candidate_id", "") or "",
                        "proposed_message": getattr(_uav_proposed, "message", "") or "",
                        "executed_action_type": uav_action.action,
                        "agent_world_x": round(state.uav_pos["x"], 2),
                        "agent_world_y": round(state.uav_pos["y"], 2),
                        "agent_world_z": round(state.uav_pos.get("z", 80.0), 2),
                        "agent_yaw": round(state.uav_pos.get("yaw", -90.0), 2),
                        "selected_waypoint_id": uav_action.waypoint_id or "",
                        "target_candidate_id": getattr(uav_action, "candidate_id", "") or "",
                        "grid_cell": getattr(uav_action, "grid_cell", "") or "",
                        "geometry_type": getattr(uav_action, "geometry_type", None),
                        "position_xy": getattr(uav_action, "position_xy", None),
                        "localization_source": getattr(uav_action, "localization_source", ""),
                        "input_image_paths": [_last_annotate_img["path"]] if uav_action.action == "ANNOTATE_CANDIDATE" and _last_annotate_img["path"] else [],
                        "confidence": getattr(uav_action, "confidence", None),
                        "reason": getattr(uav_action, "reason", "") or "",
                        "error_message": _uav_fallback_reason or (_uav_parse_status if _uav_parse_status not in ("", "ok") else "")})

                    # P0-7: 传 candidate 中心, 让 SPB belief 用候选区位置而非 UAV 高空位置
                    _cc = getattr(uav_action, "candidate_id", "") or ""
                    _cxy = None
                    if _cc:
                        for _cand in state.candidates:
                            if getattr(_cand, "candidate_id", "") == _cc:
                                _cxy = [round(_cand.region_center_xy[0], 2), round(_cand.region_center_xy[1], 2)]
                                break
                    _uav_receipt = {
                        "step": state.step, "action": uav_action.action,
                        "candidate_id": _cc,
                        "grid_cell": getattr(uav_action, "grid_cell", "") or "",
                        "geometry_type": getattr(uav_action, "geometry_type", None),
                        "position_xy": getattr(uav_action, "position_xy", None),
                        "localization_source": getattr(uav_action, "localization_source", ""),
                        "waypoint_id": uav_action.waypoint_id or "",
                        "candidate_center_xy": _cxy,
                        "pos": [round(state.uav_pos["x"], 2), round(state.uav_pos["y"], 2)],
                    }
                    if _is_coop_mode(self.agent_mode):
                        _uav_receipt = _coop_public_receipt(
                            uav_action, role="UAV",
                            pos=(state.uav_pos["x"], state.uav_pos["y"]),
                            effect_at=state.uav_sim_time - dt, ready_at=state.uav_sim_time,
                            extra={"candidate_center_xy": _cxy})
                    _safe_on_transition(uav_agent, uav_obs,
                                        _uav_proposed if _is_coop_mode(self.agent_mode) else uav_action,
                                        uav_action, _uav_receipt)
                else:
                    logger.info("[%s] UGV step %d sim_t=%.1fs", state.episode_id, state.step, state.ugv_sim_time)
                    if not hasattr(carla_iface, "_ugv_actor") or carla_iface._ugv_actor is None:
                        carla_iface.set_ugv_transform(state.ugv_pos["x"], state.ugv_pos["y"],
                            state.ugv_pos.get("z", 0.5), state.ugv_pos.get("yaw", 0),
                            strict_kinematic=(sparse_nav is not None))
                    if not hasattr(carla_iface, "_ugv_actor") or carla_iface._ugv_actor is None:
                        logger.warning("UGV actor None, skip")
                        continue
                    active_cand = ledger.get_active()
                    # 2026-09-04 视图 v2（用户裁定）：全程六视图（含方向标注），移动中亦可 360° 感知
                    # r7：benchmark 只提供原始六视图（canonical sensor channel）；
                    # prompt 用的 2x3 collage 由方法内 legacy adapter 渲染，
                    # trace 落盘的 collage 仍由 runner 用同一 adapter 生成（格式与历史可比）
                    ugv_views = _capture_metric_ugv_views(carla_iface, self.config)
                    # R2-P2-02：黑图 panel 计数（相机超时静默降级会污染核验证据链），累计进 warnings
                    _blk = getattr(carla_iface, "last_6view_black_count", 0)
                    if _blk:
                        state.sixview_black_panel_total = getattr(state, "sixview_black_panel_total", 0) + _blk
                        if _blk >= 3:
                            logger.warning("[%s] UGV sixview degraded: %d/6 black panels at step %d",
                                           state.episode_id, _blk, state.step)
                    state.ugv_observation_mode = "UGV_SIXVIEW"
                    if self._diag_save_ugv_views:
                        try:
                            _save_ugv_step_views(ugv_views, img_dir, state.step)
                        except Exception as _se:
                            logger.warning("[%s] save_ugv_views_per_step failed at step %d: %s",
                                           state.episode_id, state.step, _se)
                    topdown = b"" if map_protocol.excludes_fullcity else render_topdown(map_frame, (state.uav_pos["x"], state.uav_pos["y"]),
                        (state.ugv_pos["x"], state.ugv_pos["y"]), ledger.candidates,
                        state.uav_trajectory, state.ugv_trajectory, road_graph_store=store,
                        background_image_bytes=global_rgb_bytes, show_road_network=False, region=region)
                    if self._diag_demo_record and not map_protocol.excludes_fullcity:
                        # 演示素材：UGV 步全局俯视（增强版仅存图）；UGV 六视图输入由
                        # save_all_frames 在决策点保存（同名 ugv_<action>_*_sN.png）
                        self._demo_save(topdown, os.path.join(img_dir, "topdown_clean_s%03d.png" % state.step))
                        self._demo_save_annotated_topdown(map_frame, store, ledger, region,
                            global_rgb_bytes, img_dir, state.step, state,
                            target_actor=target_actor, fallback_topdown=topdown)
                    # sparse_nav_v1（外审五轮 E5a 试点修复）：逻辑状态以
                    # sparse_current_port 为权威（外审四轮 §6.2 禁止物理反推）；
                    # 物理定位降级为诊断（不拦截），仅 legacy 模式保留定位 gate。
                    if sparse_nav is not None and getattr(state, "sparse_current_port", None):
                        # sparse_current_port is the authoritative logical state.
                        # Physical localization is diagnostic only and MUST NOT rewrite
                        # service-arrival, route-history or method observations.
                        current_fine_node = state.sparse_current_port
                        _logical_node = store.nodes.get(state.sparse_current_port)
                        if _logical_node is not None:
                            _xy_drift = math.hypot(float(state.ugv_pos["x"]) - float(_logical_node.x),
                                                   float(state.ugv_pos["y"]) - float(_logical_node.y))
                            _sparse_pose_drift_max_m = max(_sparse_pose_drift_max_m, _xy_drift)
                            if _xy_drift > 2.0:
                                _sparse_pose_drift_over_2m_count += 1
                                logger.error("[%s] sparse logical/physical XY drift %.2fm: port=%s logical=(%.2f,%.2f) executed=(%.2f,%.2f)",
                                             state.episode_id, _xy_drift, state.sparse_current_port,
                                             _logical_node.x, _logical_node.y, state.ugv_pos["x"], state.ugv_pos["y"])
                        try:
                            ugv_loc = locator.locate(carla_iface.get_world(), state.ugv_pos["x"], state.ugv_pos["y"],
                                                      state.ugv_pos.get("z", 0.5), state.ugv_pos.get("yaw", 0),
                                                      check_heading=False)
                            if (ugv_loc and ugv_loc.fine_node_id and
                                    ugv_loc.fine_node_id != state.sparse_current_port):
                                logger.warning(
                                    "[%s] sparse locator mismatch: logical=%s physical=%s (diagnostic only)",
                                    state.episode_id, state.sparse_current_port, ugv_loc.fine_node_id)
                        except Exception as _le:
                            logger.warning("[%s] sparse locator diagnostic failed: %s", state.episode_id, _le)
                    else:
                        ugv_loc = locator.locate(carla_iface.get_world(), state.ugv_pos["x"], state.ugv_pos["y"],
                                                  state.ugv_pos.get("z", 0.5), state.ugv_pos.get("yaw", 0),
                                                  check_heading=True)
                        if ugv_loc is None or not ugv_loc.is_valid:
                            logger.warning("UGV position not valid on road graph; skipping action")
                            current_fine_node = None
                            continue
                        current_fine_node = ugv_loc.fine_node_id

                    # Maintain UGV route history for backward option
                    if not hasattr(state, "ugv_route_history"):
                        state.ugv_route_history = []
                    if not state.ugv_route_history or state.ugv_route_history[-1][0] != current_fine_node:
                        state.ugv_route_history.append((current_fine_node, ugv_distance))
                    # 锚点经过检测：当前节点命中锚则记录（R2-P1-03：供轨迹审计/复盘，
                    # 不再供回退动作——v2 回退由 REVERSE 承担）
                    if anchor_catalog is not None and current_fine_node in anchor_catalog.by_fine_node:
                        if not state.anchor_visit_history or state.anchor_visit_history[-1] != current_fine_node:
                            state.anchor_visit_history.append(current_fine_node)

                    # ===== 2026-09-04 动作空间 v2：操纵式动作菜单（前进1/2节点、路口左右转、倒车）=====
                    # sparse_nav_v1（2026-09-11 外审四轮 P0-BENCH）：菜单数据源换锚点网络
                    # SemanticNavAdapter（六动作同键，~30m 节拍，路口判定 50m）；落点/边序列/
                    # fine_path 均为 L2 图权威。MANEUVER_V2 分支保留（legacy 对照）。
                    if sparse_nav is not None:
                        _sparse_port = getattr(state, "sparse_current_port", None) or current_fine_node
                        try:
                            maneuver_menu = sparse_nav.get_maneuver_options(
                                _sparse_port,
                                intersection_max_distance_m=_sparse_intersection_max_distance(self.config))
                        except Exception as _e:
                            logger.warning("sparse_nav menu failed at %s: %s", _sparse_port, _e)
                            maneuver_menu = {}
                    else:
                        maneuver_menu = option_provider.get_maneuver_options(
                            current_fine_node,
                            state.ugv_route_history,
                            intersection_max_distance_m=self.config["ugv"]["intersection_max_distance_m"],
                            consecutive_backtracks=state.consecutive_backtracks)
                    if sparse_nav is not None:
                        _maneuver_zh = {
                            "FORWARD_1": "沿控制路网执行当前短前进选项（距离以菜单数值为准）",
                            "FORWARD_2": "沿控制路网执行当前长前进选项（可能跨多个控制段；距离以菜单数值为准）",
                            "TURN_LEFT": "执行当前可用左转路口动作（距离以菜单数值为准）",
                            "TURN_RIGHT": "执行当前可用右转路口动作（距离以菜单数值为准）",
                            "TURN_STRAIGHT": "执行当前可用路口直行动作（距离以菜单数值为准）",
                            "REVERSE": "执行当前可用物理倒车选项（距离以菜单数值为准）",
                        }
                    else:
                        _maneuver_zh = {
                            "FORWARD_1": "向前移动(约10m)",
                            "FORWARD_2": "前向大幅移动(约20m)",
                            "TURN_LEFT": "路口左转",
                            "TURN_RIGHT": "路口右转",
                            "TURN_STRAIGHT": "路口直行",
                            "REVERSE": "原地倒车(原路回退约10m)",
                        }
                    avail_wps = []  # 兼容字段：旧路点列表已由动作菜单取代
                    avail_action_menu = []
                    for act in UGV_MANEUVER_ACTIONS:
                        mopt = maneuver_menu.get(act)
                        if mopt is None:
                            continue
                        _tn = store.nodes.get(mopt.target_fine_node_id)
                        avail_action_menu.append({
                            "action": act,
                            "description": _maneuver_zh.get(act, mopt.display_text),
                            "distance_m": round(mopt.path_distance_m, 1),
                            "target_xy": (round(_tn.x, 1), round(_tn.y, 1)) if _tn else None,
                            # Internal road metadata: benchmark-legal frozen graph,

                            "target_fine_node_id": mopt.target_fine_node_id,
                            "option_type": mopt.option_type,
                        })
                    # GO_TO_CANDIDATE 已移除（用户裁定 2026-09-04）：UGV 用操纵动作自行抵近候选；
                    # 到达契约（CandidateServiceContract，外审 P0-C01 修复，Phase-G2 Stage 0
                    # Step G1，方法中立）：当前 fine node ∈ 候选服务节点池即框架标记"到达"
                    # （assigned_to_ugv）并唤醒冻结的 UAV（对应原 GO_TO 的到达语义）。
                    # 旧判定"距候选中心 <= approach_distance_min_m(30m)"被删除——它与三级
                    # 规划优先选 30-50m 外环的策略数学冲突（外环节点到中心 >= ~39m，永远
                    # 触发不了赋值），是 TRACE 到达合法 goal 仍停 unverified 的根因。
                    # 本判定与候选创建点、adapter 可服务性检查三处共用同一池（半径同源
                    # = max_confirm_distance_m）；fine node 缺失/不在图中时有 <=5m 欧氏兜底。
                    if active_cand is not None and active_cand.status == "unverified":
                        _arrival_pool = (active_cand.confirmable_service_node_ids
                                         if _confirmable_service_arrival else active_cand.service_node_ids)
                        if service_arrival_hit(current_fine_node, _arrival_pool,
                                               ugv_xy=(state.ugv_pos["x"], state.ugv_pos["y"]),
                                               node_xy=lambda nid: (
                                                   (store.nodes[nid].x, store.nodes[nid].y)
                                                   if nid in store.nodes else None)):
                            ledger.assign_to_ugv(active_cand.candidate_id)
                            active_cand = ledger.get_active()
                            self._framework_activate_uav(state, msg_bus, "arrived",
                                candidate_id=active_cand.candidate_id if active_cand else "",
                                extra=("(UGV reached conservative verification-ready node)"
                                       if _confirmable_service_arrival
                                       else "(UGV reached broad candidate service node)"))
                    # r7: TRACE route guidance 已迁入 TraceOSMethod._prepare_trace_ugv
                    # （方法内 policy；road services 经 MethodServices 注入）
                    avail_ids = {item["action"] for item in avail_action_menu}
                    # R2-P0-01 修复（外部二轮审查）：删除 avail_ids 被空列表覆盖的行——
                    # 旧代码 avail_ids = {w["waypoint_id"] for w in avail_wps} 把上面正确的
                    # 动作集合覆盖为空集，且 available_waypoints 传空列表导致 UGV prompt
                    # 的动作菜单渲染为空（模型看不到本步可选 FORWARD/TURN/REVERSE）。
                    ugv_obs = LiteObservation(
                        episode_id=state.episode_id, step=state.step, max_steps=state.max_steps,
                        role="UGV", agent_image=b"", ugv_views=ugv_views, topdown_map=topdown,
                        agent_position=(state.ugv_pos["x"], state.ugv_pos["y"], 0.5),
                        observation_mode=state.ugv_observation_mode,
                        target_description=state.target_description,
                        candidates=ledger.candidates, active_candidate=active_cand,
                        available_waypoints=avail_action_menu,  # R2-P0-01：v2 动作菜单（_render_maneuver_menu 从此渲染；UGV-only 分支同口径）
                        available_actions=compute_ugv_available_actions(state, active_cand, maneuver_menu),
                        recent_messages=(msg_bus.get_visible_for("UGV", state.ugv_sim_time)
                                         if _is_coop_mode(self.agent_mode) else
                                         msg_bus.get_recent_for("UGV", state.ugv_sim_time)),
                        latest_stop_context=state.latest_stop_context,
                        confirm_count=state.confirm_count, max_confirms=self.config["ugv"]["confirm_budget"],
                        confirm_budget=state.confirm_budget,
                        region_boundary=(region["x_min"], region["x_max"], region["y_min"], region["y_max"]),
                        target_reference_image=target_ref_img,
                        target_reference_views=ref_views)
                    map_protocol.attach(ugv_obs, state.ugv_sim_time, state.ugv_sim_time,
                                        (state.ugv_pos["x"], state.ugv_pos["y"]), state.ugv_trajectory)
                    if _is_coop_mode(self.agent_mode):
                        ugv_obs.coop_event_time = state.ugv_sim_time
                    # batch2 停发复盘：本步六图采集黑帧数暴露给异常分级
                    # （sixview 合同 fail-closed 拒绝黑帧时区分采集瞬态故障）
                    ugv_obs.sixview_black_panels = int(_blk)
                    ugv_obs.ugv_memory_text = state.get_ugv_memory_lines()


                    # R18 non-depth ranging uses sensor geometry only.  Prefer
                    # the actual CARLA camera anchor height for this vehicle;
                    # fall back to the configured nominal height.
                    ugv_obs.r18_ugv_yaw_deg = float(state.ugv_pos.get("yaw", 0.0))
                    try:
                        _r18_anchor = carla_iface._get_ugv_camera_anchor()
                        ugv_obs.r18_camera_height_m = max(0.2, float(_r18_anchor.z) - float(state.ugv_pos.get("z",0.5)))
                    except Exception:
                        ugv_obs.r18_camera_height_m = float(_observation_cfg.get("r18_ugv_camera_height_m",2.2) or 2.2)
                    if _policy_horizon_cfg > 0:
                        ugv_obs.policy_horizon_steps = (_policy_horizon_cfg if state.step <= _policy_horizon_cfg
                                                        else state.max_steps)

                    # r7: benchmark 事实字段（路网定位 + 原始动作菜单），方法内
                    # route guidance 由 TraceOSMethod 读取这些字段计算。
                    ugv_obs.current_fine_node_id = current_fine_node
                    ugv_obs.ugv_maneuver_menu = maneuver_menu
                    # 开关开启且存在 active candidate 时，用与 trace method 同源的
                    # 冻结路网计算挂 obs.trace_route_guidance（preferred_action 供
                    # I1 帧转储钩子（Phase-G Step 6）：obs 构造后、VLM decide 前；
                    # 默认关（config 无 i1_frame_dump 键零开销），任何失败不影响主循环
                    try:
                        _i1_cfg = self.config.get("i1_frame_dump")
                        if _i1_cfg is not None:
                            _assigned_now = (active_cand is not None
                                             and active_cand.status == "assigned_to_ugv")
                            if _assigned_now and not _i1_assigned_prev:
                                _i1_arrival_step = state.step
                            _i1_trig = i1_dump_trigger(
                                state.step, _i1_fvs, _i1_gt_vis_prev,
                                _assigned_now, _i1_assigned_prev, _i1_cfg,
                                arrival_step=_i1_arrival_step)
                            if _i1_trig:
                                _i1_dump_frames(output_dir, state, ugv_obs,
                                                _i1_trig, target_actor)
                                # P0-I01：持久化触发原因，decide 后写 meta_post.json
                                _i1_last_trig = _i1_trig
                            _i1_assigned_prev = _assigned_now
                    except Exception as e:
                        logger.warning("i1 frame dump failed: %s", e)
                    # HYBRID owns its 0/1-call receipts; do not reserve a phantom main call.
                    ugv_obs.internal_vlm_calls_remaining = max(
                        0, state.max_vlm_calls - state.vlm_calls -
                        int(_role_uses_vlm(self.agent_mode, ugv_agent, "UGV")))


                    _ugv_step_uses_vlm = _role_uses_vlm(self.agent_mode, ugv_agent, "UGV")
                    if _ugv_step_uses_vlm:
                        state.vlm_calls += 1
                        state.ugv_vlm_calls += 1
                    # Private Oracle snapshot belongs to the captured input pose;
                    # never attach this GT information to ugv_obs or its Prompt.
                    _ugv_metric_snapshot = probe_ugv_observation(
                        state, target_actor, self.config, carla_iface, checker, ugv_views)
                    try:
                        from agos.runtime.paper_audit import decide_and_account
                        ugv_action = decide_and_account(
                            lambda: _method_decide(self.agent_mode, ugv_agent,
                                ugv_obs, task_attempt=_task_attempt,
                                capture_black_count=getattr(ugv_obs, "sixview_black_panels", 0)),
                            state, ugv_agent, "UGV")
                    except VLMCancelledError:
                        # R2-P1-07: 墙钟超时打断阻塞中的 VLM 调用（UGV 段，与 UAV 段同款收尾）
                        if _ugv_step_uses_vlm:
                            state.vlm_calls = max(0, state.vlm_calls - 1)
                            state.ugv_vlm_calls = max(0, state.ugv_vlm_calls - 1)
                        state.terminal = True
                        state.failure_stage = "EPISODE_WALL_TIMEOUT"
                        continue


                    # P0-6: 收集 AG-OS 模块证据 trace
                    _audit_role = "UGV"
                    if not (_collect_method_trace(self.agent_mode) and hasattr(ugv_agent, "build_trace")):
                        try:
                            from agos.runtime.paper_audit import record_tool_faults
                            record_tool_faults(state, ugv_obs, {})
                        except Exception as _audit_error:
                            from agos.runtime.paper_audit import record_audit_failure
                            record_audit_failure(state, _audit_role, state.step, _audit_error)
                    if _collect_method_trace(self.agent_mode) and hasattr(ugv_agent, "build_trace"):
                        try:
                            _tr = ugv_agent.build_trace(ugv_obs); _tr["step"] = state.step
                            from agos.runtime.paper_audit import record_tool_faults
                            record_tool_faults(state, ugv_obs, _tr)
                            if hasattr(ugv_obs, "map_metadata"):
                                _tr["online_map"] = ugv_obs.map_metadata
                            ag_os_trace_records.append(_tr)
                        except Exception as _e:
                            from agos.runtime.paper_audit import record_audit_failure
                            record_audit_failure(state, _audit_role, state.step, _e)
                            logger.warning("trace collect failed: %s", _e)
                    # P0-I01（发车前外审 20260907）：I1 两阶段 artifact——pre-dump 发生在


                    # artifact_id 关联。try/except 全包不影响主循环。
                    if _i1_last_trig is not None:
                        try:
                            _step_dir = os.path.join(output_dir, "i1_frames",
                                                     "step_%04d" % state.step)
                            _post = {
                                "artifact_id": "%s:%04d" % (state.episode_id, state.step),
                                "step": state.step, "trigger": _i1_last_trig,
                                "dove_board_text": str(getattr(ugv_obs, "dove_board_text", "") or "")[:400] or None,
                                "prompt_audit": dict(getattr(ugv_agent, "_last_prompt_audit", None) or {}),
                                "raw_action": getattr(getattr(ugv_agent, "_last_raw_action", None), "action", None),
                                "executed_action": ugv_action.action,
                                "resolution_reason": getattr(ugv_agent, "_last_resolution_reason", None),
                                "changed_by_trace_policy": bool(getattr(ugv_agent, "_last_resolution_changed", False)),
                            }
                            with open(os.path.join(_step_dir, "meta_post.json"), "w",
                                      encoding="utf-8") as f:
                                json.dump(_post, f, ensure_ascii=False, indent=2)
                        except Exception as _e:
                            logger.warning("i1 post-decision dump failed: %s", _e)
                        _i1_last_trig = None
                    # P0-09: infra error - skip this step entirely, don't count as model decision
                    # 2026-09-04 用户指令：不再因 VLM 连续失败终止 episode（API 不稳定只等待+告警）。
                    if ugv_action.action == "INFRA_ERROR":
                        from agos.runtime.paper_audit import record_primary_infra
                        record_primary_infra(state, ugv_obs, ugv_action)
                        # R2-P1-06：infra 未产生有效模型决策，回退本步预算计数（口径=有效决策数）。
                        if _ugv_step_uses_vlm:
                            state.vlm_calls = max(0, state.vlm_calls - 1)
                            state.ugv_vlm_calls = max(0, state.ugv_vlm_calls - 1)
                        state.consecutive_vlm_failures = getattr(state, "consecutive_vlm_failures", 0) + 1
                        state.infra_error_count = getattr(state, "infra_error_count", 0) + 1
                        _max_fail = getattr(state, "max_consecutive_vlm_failures", 5)
                        logger.warning("[%s] UGV VLM infra error (%d consecutive): %s",
                                       state.episode_id, state.consecutive_vlm_failures, ugv_action.reason)
                        if state.consecutive_vlm_failures == _max_fail or state.consecutive_vlm_failures % 25 == 0:
                            logger.error("[%s] UGV VLM 已连续失败 %d 次（疑似鉴权/参数类确定性错误，"
                                         "请人工检查 API 配置）；按机制继续等待重试，不终止本任务",
                                         state.episode_id, state.consecutive_vlm_failures)
                        continue  # 不消耗步数，不推进仿真时间，直接下一步

                    # Commit only valid decision opportunities; INFRA retries above
                    # do not create extra metric opportunities.
                    record_ugv_observation(state, _ugv_metric_snapshot)
                    # 成功调用：清零连续失败计数器
                    state.consecutive_vlm_failures = 0
                    if cecm:
                        raw = getattr(ugv_agent, "last_raw_response", "")
                        if raw:
                            cecm.check_vlm(raw)
                        cecm.check_trajectory(state.ugv_trajectory, threshold_m=5.0)
                    # 2026-07-26 plain-baseline 可观测性：区分 模型真实WAIT / 解析降级 / 校验降级
                    _ugv_policy_debug = {}
                    try:
                        if hasattr(ugv_agent, "get_last_policy_debug"):
                            _ugv_policy_debug = dict(ugv_agent.get_last_policy_debug("UGV") or {})
                    except Exception:
                        _ugv_policy_debug = {}
                    _ugv_proposed = ugv_action  # legacy field: method-resolved pre-validator action
                    _ugv_parse_status = getattr(ugv_agent, "last_parse_status", "")
                    _ugv_validation_success = True
                    _ugv_fallback_reason = ""
                    valid, reason = validate_action(ugv_action.__dict__ if hasattr(ugv_action, "__dict__") else {}, ugv_obs, avail_ids)
                    if not valid:
                        state.invalid_action_count += 1
                        _ugv_validation_success = False
                        _ugv_fallback_reason = reason
                        # 2026-09-04 动作空间 v2：WAIT 已移除，非法输出降级为原地等候(STOP_OBSERVE)
                        ugv_action = LiteAction(role="UGV", action="STOP_OBSERVE", reason="invalid: %s" % reason)
                    prev = (state.ugv_pos["x"], state.ugv_pos["y"])
                    road_distance_m = 0.0
                    goal_wp = None
                    _ugv_last_img_path = ""
                    # Free-form UGV messages are communication only.  UAV lifecycle is
                    # framework-controlled by CandidateServiceContract (_framework_activate_uav).
                    # Treating every model-authored message as an arrival_notification used to
                    # wake a frozen UAV prematurely during ordinary navigation.
                    ugv_msg = getattr(ugv_action, "message", "")
                    if ugv_msg:
                        _msg_type = ("arrival_notification"
                                     if active_cand is not None and active_cand.status == "assigned_to_ugv"
                                     else "agent_message")
                        msg_bus.send("UGV", ugv_msg, state.ugv_sim_time, receiver="UAV",
                            message_type=_msg_type,
                            related_candidate_id=getattr(ugv_action, "candidate_id", "") or "",
                            step=state.step)
                    # ===== 2026-09-04 动作空间 v2：操纵式移动动作（菜单查表 -> goal_wp）=====
                    if ugv_action.action in UGV_MANEUVER_ACTIONS:
                        _mopt = maneuver_menu.get(ugv_action.action)
                        if _mopt is not None:
                            goal_wp = {"id": _mopt.target_fine_node_id,
                                       "fine_path": list(_mopt.fine_path) if _mopt.fine_path else None,
                                       "option_type": _mopt.option_type,
                                       "distance_m": _mopt.path_distance_m}
                            # sparse_nav REVERSE：车头朝向锁定行进方向（倒车物理语义）
                            if getattr(_mopt, "yaw_override_deg", None) is not None:
                                goal_wp["yaw_override_deg"] = _mopt.yaw_override_deg
                            if ugv_action.action in ("TURN_LEFT", "TURN_RIGHT"):
                                msg_bus.send("UGV", "Turning %s toward candidate road." % ugv_action.action, state.ugv_sim_time,
                                    receiver="UAV", message_type="candidate_navigation",
                                    related_candidate_id=getattr(ugv_action, "candidate_id", "") or "",
                                    step=state.step)
                    elif ugv_action.action == "MOVE_TO_ROAD_WAYPOINT" and ugv_action.waypoint_id:
                        # P0-fix: 先从当步 available_waypoints 查找（option provider 输出的 roadnet 节点），
                        # fallback 到旧全局 ugv_road_waypoints 列表
                        for wp in avail_wps:
                            if wp["waypoint_id"] == ugv_action.waypoint_id:
                                goal_wp = {"id": wp["waypoint_id"],
                                           "fine_path": wp.get("fine_path"),
                                           "option_type": wp.get("option_type"),
                                           "distance_m": wp.get("distance_m")}
                                break
                        if goal_wp is None:
                            for wp in ugv_wps:
                                if wp["id"] == ugv_action.waypoint_id:
                                    goal_wp = wp
                                    break

                    if goal_wp and current_fine_node is not None:
                        # 2026-09-04 动作空间 v2：操纵动作的 fine_path 是选项权威给出的
                        # （REVERSE 为反向节点序列，非合法有向路径），优先直走；其余走 router 规划
                        # （BACK_TO_PREVIOUS_ANCHOR 仅作旧 trace 兼容名保留在白名单）
                        _opt_fine_path = goal_wp.get("fine_path") if isinstance(goal_wp, dict) else None
                        _sparse_option_types = (
                            "REVERSE_NODE", "FORWARD_1", "FORWARD_2",
                            "INTERSECTION_LEFT", "INTERSECTION_RIGHT", "INTERSECTION_STRAIGHT")
                        if sparse_nav is not None and (
                                not _opt_fine_path or goal_wp.get("option_type") not in _sparse_option_types):
                            raise RuntimeError(
                                "SPARSE_ROUTE_CONTRACT_VIOLATION: sparse_nav_v1 movement must use "
                                "the SemanticNavAdapter-resolved fine_path; refusing RoadRouter fallback")
                        if _opt_fine_path and goal_wp.get("option_type") in (
                                "BACK_TO_PREVIOUS_ANCHOR",) + _sparse_option_types:
                            node_path = list(_opt_fine_path)
                            _yaw_ov = goal_wp.get("yaw_override_deg") if isinstance(goal_wp, dict) else None
                            road_path = [(store.nodes[nid].x, store.nodes[nid].y, store.nodes[nid].z,
                                          (_yaw_ov if _yaw_ov is not None else store.nodes[nid].yaw_deg))
                                         for nid in node_path[1:]]
                            road_length = (fine_path_distance(store.nodes, node_path) if _strict_spl
                                           else goal_wp.get("distance_m", 0.0) or 0.0)
                            normal_turns = 1 if goal_wp.get("option_type") in ("INTERSECTION_LEFT", "INTERSECTION_RIGHT") else 0  # STRAIGHT 直行不计转弯罚时
                            u_turns = 0
                        else:
                            route = router.plan(current_fine_node, goal_wp["id"], cost_mode="distance")
                            if isinstance(route, RouteFailure):
                                logger.warning(f"UGV routing failed: {route.code}; not moving")
                                road_path = []
                                road_length = 0.0
                                normal_turns, u_turns = 0, 0
                            else:
                                node_path = list(route.node_path)
                                road_path = [(store.nodes[nid].x, store.nodes[nid].y, store.nodes[nid].z, store.nodes[nid].yaw_deg) for nid in node_path[1:]]
                                road_length = route.path_length_m
                                normal_turns = route.turn_count
                                u_turns = 0  # currently counted in turn_count as needed
                        if not road_path:
                            logger.warning("UGV route plan empty; not moving")
                        else:
                            # Spec v2 §4.6.2: record executed road-graph fine path segment
                            ugv_fine_segments.append({
                                "segment_id": "seg_%03d" % (len(ugv_fine_segments) + 1),
                                "from_step": state.step, "to_step": state.step,
                                "action": ugv_action.action,
                                "from_fine_node_id": (node_path[0] if node_path else ""),
                                "to_fine_node_id": (node_path[-1] if node_path else ""),
                                "fine_node_ids": list(node_path),
                                "path_points": [[round(x, 2), round(y, 2)] for (x, y, z, yaw) in road_path],
                                "source": ("sparse_nav_v1" if sparse_nav is not None
                                           else "road_graph_shortest_path"),
                                "action_space_version": ("sparse_nav_v1" if sparse_nav is not None
                                                         else "MANEUVER_V2"),
                                "control_graph_sha256": (sparse_nav_sha if sparse_nav is not None else ""),
                                "length_m": round(road_length, 2),
                                "length_m_exact": road_length})
                            # 锚点经过检测：沿执行路径命中的锚按序记录
                            if anchor_catalog is not None:
                                for _nid in node_path:
                                    if _nid in anchor_catalog.by_fine_node:
                                        if not state.anchor_visit_history or state.anchor_visit_history[-1] != _nid:
                                            state.anchor_visit_history.append(_nid)
                        road_distance_m = road_length
                        for i, (x, y, z, yaw) in enumerate(road_path):
                            is_final = (i == len(road_path) - 1)
                            ugv_tf = carla_iface.set_ugv_transform(
                                x, y, z, yaw, fast_teleport=not is_final,
                                strict_kinematic=(sparse_nav is not None))
                            if ugv_tf is None:
                                # Do not advance sparse logical state after a partial/failed physical
                                # execution.  A transform failure is evaluation infrastructure failure,
                                # not a legal model transition; fail the episode so the job layer can retry.
                                raise RuntimeError(
                                    "UGV_TRANSFORM_FAILED episode=%s step=%s waypoint=(%.2f,%.2f)" %
                                    (state.episode_id, state.step, x, y))
                            state.ugv_pos = {"x": x, "y": y, "z": z, "yaw": yaw}
                            state.ugv_trajectory.append((x, y))
                        ugv_distance += road_distance_m
                        # sparse_nav_v1：逻辑状态以 resolved edge 落点 port 为权威
                        # （外审四轮 §6.2：禁止 CARLA get_waypoint 反推逻辑状态）
                        if sparse_nav is not None and goal_wp.get("id"):
                            state.sparse_current_port = goal_wp["id"]
                        # P0-2 修复：route_history 展开为节点级（步级历史隔 5-15 个节点，
                        # 会使 BACK 同链直退判定失效）；欧氏累加里程 ≈ 路径长
                        if len(node_path) > 1:
                            _cum = ugv_distance - road_distance_m
                            _prev = store.nodes[node_path[0]]
                            for _nid in node_path[1:]:
                                _n = store.nodes[_nid]
                                _cum += math.hypot(_n.x - _prev.x, _n.y - _prev.y)
                                if not state.ugv_route_history or state.ugv_route_history[-1][0] != _nid:
                                    state.ugv_route_history.append((_nid, _cum))
                                _prev = _n
                        # 连续回退计数：BACK 动作 +1（防震荡），任何其他实际移动清零
                        if goal_wp.get("option_type") in ("BACK_TO_PREVIOUS_ANCHOR", "REVERSE_NODE"):
                            state.consecutive_backtracks += 1
                        elif road_distance_m > 0:
                            state.consecutive_backtracks = 0
                        if road_path:
                            state.ugv_observation_mode = "UGV_SIXVIEW"
                            # Any actual movement invalidates previous STOP context
                            if state.latest_stop_context is not None:
                                state.latest_stop_context.is_fresh = False

                    # R2-P2-04（2026-09-04 外审）：candidate_id 提前归一化——
                    # STOP_OBSERVE/CONFIRM/REJECT 存在 active candidate 时，
                    # 省略 id 则绑定 active（此前 auto-bind 在 GVA event/checker 之后，
                    # 导致日志、gva_events、success_source 与 ledger 状态不一致）。
                    # 显式给出不一致 id 时不覆盖模型选择，仅告警（validate_action
                    # 已对 CONFIRM/REJECT 的 mismatch 做拦截）。
                    # R22 外审 P0 SEM-01：记录绑定来源（explicit/auto_active/none）——
                    # 归一化**之前**记录，两类保持可区分；绑定来源不是 UAV 因果引导证据。
                    _candidate_binding_origin = ("explicit" if ugv_action.candidate_id else "none")
                    if active_cand and ugv_action.action in ("STOP_OBSERVE", "CONFIRM", "REJECT"):
                        if not ugv_action.candidate_id:
                            ugv_action.candidate_id = active_cand.candidate_id
                            _candidate_binding_origin = "auto_active"
                            logger.info("[%s] %s auto-bound candidate_id=%s",
                                        state.episode_id, ugv_action.action, ugv_action.candidate_id)
                        elif ugv_action.candidate_id != active_cand.candidate_id:
                            logger.warning("[%s] %s candidate_id=%s != active %s (kept model choice)",
                                           state.episode_id, ugv_action.action,
                                           ugv_action.candidate_id, active_cand.candidate_id)

                    gt_vis = _ugv_metric_snapshot['visible']
                    vis_views = _ugv_metric_snapshot['visible_views']
                    vis_debug = _ugv_metric_snapshot['debug']
                    los_clear = _ugv_metric_snapshot['los_clear']
                    d2t = _ugv_metric_snapshot['distance_m']

                    if ugv_action.action == "STOP_OBSERVE":
                        # 2026-09-04 语义 v2（用户裁定）：STOP 仅表示原地等候 10s。
                        # 六视图已全程常态提供，无需停车采集，也不再建立/门控 stop context。
                        # P0-C02 修复（Phase-G2 Stage 0 Step G1）：STOP 是观察事件，不是
                        # 候选生命周期转换——不再 mark_visited（visited 仍算 active 但两家
                        # prompt 都不处理，状态机会进死区）；只计数+记录观察步，status 保持
                        # 不变（CONFIRM/REJECT 才做生命周期裁决）。
                        if ugv_action.candidate_id:
                            if not ledger.record_stop_observation(ugv_action.candidate_id, state.step):
                                logger.warning("[%s] STOP_OBSERVE ledger.record_stop_observation(%s) no-op",
                                               state.episode_id, ugv_action.candidate_id)
                        # 框架层激活：UGV 在候选附近驻留观察，激活 UAV 引导（死锁保护）
                        self._framework_activate_uav(state, msg_bus, "observing",
                            candidate_id=ugv_action.candidate_id or "")
                        msg_bus.send("UGV", "Waiting in place (10s).", state.ugv_sim_time,
                            receiver="UAV", message_type="stop_observe",
                            related_candidate_id=ugv_action.candidate_id or "", step=state.step)

                    # Reuse input-pose Oracle evidence, including for Confirm/Reject.
                    # MOVE has already happened here: do not score its unseen endpoint
                    # as an observation the agent had a chance to interpret.
                    if True:
                        _oracle_pos_now = bool(_ugv_metric_snapshot['confirmable'])
                        # I1 帧转储（Phase-G Step 6）：回填首次 oracle 可见步 + 上一步可见态
                        if gt_vis and _i1_fvs is None:
                            _i1_fvs = state.step
                        _i1_gt_vis_prev = bool(gt_vis)
                        # N0 GNG-2：diagnostic.save_all_frames 开启时每 UGV 决策步
                        # 全量保存六视图（与 vis_debug json 的 oracle 标签同 step 对齐，
                        # 离线 verifier 数据集用）；默认仅终局/观察动作存帧。
                        _save_frame_now = (ugv_action.action in ("CONFIRM", "REJECT", "STOP_OBSERVE")
                                           or self._diag_save_all_frames)
                        if _save_frame_now:
                            _cand_tag = _fs_tag(ugv_action.candidate_id)
                            img_path = os.path.join(img_dir, "ugv_%s_%s_s%d.png" % (ugv_action.action.lower(), _cand_tag, state.step))
                            with open(img_path, "wb") as f:
                                # r7：trace 存档沿用 legacy 2x3 collage 格式（与历史可比）
                                f.write(render_legacy_sixview_collage(ugv_views))
                            _ugv_last_img_path = img_path
                            # R19 dataset contract: when full-frame collection is
                            # enabled, persist the six canonical RGB panels as
                            # independent images.  A collage cannot be assigned a
                            # single panel-level visibility/bbox label.
                            if self._diag_save_all_frames:
                                for _vv in (ugv_views or []):
                                    if not getattr(_vv, "valid", False) or not getattr(_vv, "image_bytes", b""):
                                        continue
                                    _vid = str(getattr(_vv, "view_id", "unknown"))
                                    _vp = os.path.join(img_dir, "ugv_view_%s_s%d.png" % (_vid, state.step))
                                    with open(_vp, "wb") as _vf:
                                        _vf.write(_vv.image_bytes)
                        vis_debug_path = os.path.join(img_dir, "%s_vis_debug_s%d.json" % (state.episode_id, state.step))
                        with open(vis_debug_path, "w") as f:
                            _vis_payload = {"step": state.step, "gt_visible": gt_vis,
                                "visible_views": vis_views, "dist_to_target": round(d2t, 1) if d2t is not None else None,
                                "los_clear": los_clear,
                                "ugv_pos": _ugv_metric_snapshot['pose_xy'],
                                "ugv_yaw": _ugv_metric_snapshot['yaw'],
                                "observation_phase": "predecision",
                                "candidate_id": ugv_action.candidate_id,
                                "model_action": ugv_action.action,
                                "is_false_reject": gt_vis and ugv_action.action == "REJECT"}
                            if self._diag_save_full_visibility_debug:
                                _vis_payload["per_view_visibility_debug"] = vis_debug
                            json.dump(_vis_payload, f, indent=2, default=str)

                    if ugv_action.action in ("CONFIRM", "REJECT"):
                        # GVA：记录 verification event（2026-07-21 metrics v3）
                        # 2026-09-04: CONFIRM/REJECT 免停靠（六视图常态化），oracle 判定不再依赖 stop context
                        _has_fresh_stop = True
                        # Reuse the all-step opportunity predicate above.
                        _oracle_pos = bool(_oracle_pos_now)
                        _model_pos = (ugv_action.action == "CONFIRM")
                        _verdict_correct = (_model_pos == _oracle_pos)
                        # N0 GNG-3：FCS 诊断拦截判定（oracle 阴性 CONFIRM 不终止）
                        _fcs = fcs_should_intercept(self._diag_shield, ugv_action.action, _oracle_pos)
                        # P0-1 (2026-09-04 external review): episode 级 GVA 分母条件必须
                        # 同步置位到 state。此前只写入 gva_events，oracle_verifiable_reached
                        # 恒 False，导致成功 episode 全部不进 GVA 分母（eval.py 端只能
                        # fallback 到 osr 宽口径）。与 lite_core.py 注释声明的契约对齐。
                        if _oracle_pos:
                            state.oracle_verifiable_reached = True
                        state.gva_events.append({
                            "verification_id": f"V{len(state.gva_events) + 1:03d}",
                            "step_index": state.step,
                            "candidate_id": ugv_action.candidate_id or "",
                            # R22 外审 P0 SEM-01：绑定来源随核验事件落盘（与 result 级字段同源）
                            "candidate_binding_origin": _candidate_binding_origin,
                            "model_verdict": ugv_action.action,
                            "has_fresh_stop_context": _has_fresh_stop,
                            "gt_visible": bool(gt_vis),
                            "distance_to_target_m": round(d2t, 2) if d2t is not None else None,
                            "max_confirm_distance_m": float(max_confirm_distance_m),
                            "require_los": bool(checker.require_los),
                            "los_clear": bool(los_clear),
                            "oracle_positive": _oracle_pos,
                            "oracle_valid": bool(_ugv_metric_snapshot["valid"]),
                            "observation_phase": "predecision",
                            "verdict_correct": _verdict_correct,
                            "intercepted_by_fcs": bool(_fcs),
                            "confidence": float(getattr(ugv_action, "confidence", 0.5) or 0.5),
                        })
                        # verdict 写入后消费 fresh STOP context
                        if state.latest_stop_context is not None:
                            state.latest_stop_context.is_fresh = False

                        if _fcs:
                            ok, res = False, "fcs_intercepted"
                        else:
                            ok, res = checker.check(state, ugv_action, gt_vis, d2t, los_clear=los_clear)
                        if _fcs:
                            # N0 GNG-3（FCS 式拦截）：oracle 阴性 CONFIRM 不终止不耗预算
                            # （checker 不被调用 → confirm_count/budget/terminal 均不变），
                            # 转 STOP_OBSERVE 继续搜索。would_false_confirm 单列诊断口径，
                            # 不污染 false_confirm_count（正式口径）。
                            state.would_false_confirm_count = getattr(state, "would_false_confirm_count", 0) + 1
                            if d2t is not None:
                                state.diagnostic_would_confirm_distance_m = d2t
                            logger.info("[%s] FCS intercept: oracle-negative CONFIRM on %s withheld (d=%.1fm, gt_vis=%s) -> STOP_OBSERVE, episode continues",
                                        state.episode_id, ugv_action.candidate_id or "independent",
                                        (d2t if d2t is not None else -1.0), bool(gt_vis))
                            if ugv_action.candidate_id:
                                ledger.record_stop_observation(ugv_action.candidate_id, state.step)
                            msg_bus.send("UGV", "CONFIRM withheld (diagnostic shield); STOP_OBSERVE.",
                                state.ugv_sim_time, receiver="UAV", message_type="fcs_intercept",
                                related_candidate_id=ugv_action.candidate_id or "", step=state.step)
                            state.add_ugv_memory(action="FCS_INTERCEPT",
                                params={"candidate_id": ugv_action.candidate_id or None,
                                        "dist_to_target_m": round(d2t, 1) if d2t is not None else None,
                                        "gt_visible": bool(gt_vis)},
                                result="oracle-negative confirm withheld", outcome="stop_observe")
                            ugv_action = LiteAction(role="UGV", action="STOP_OBSERVE",
                                candidate_id=getattr(ugv_action, "candidate_id", None),
                                reason="fcs_intercept: oracle-negative confirm withheld")
                        elif ugv_action.action == "CONFIRM":
                            # R2-P2-04: auto-bind 已前移至验证事件记录前（normalize 块）
                            if d2t is not None:
                                state.ne_distance = d2t
                            if ok:
                                if ugv_action.candidate_id:
                                    # R2-P2-04: ledger 操作返回值检查（错误 cid 不再静默 no-op）
                                    if not ledger.confirm(ugv_action.candidate_id):
                                        logger.warning("[%s] CONFIRM ledger.confirm(%s) no-op: candidate not in ledger",
                                                       state.episode_id, ugv_action.candidate_id)
                                # 2026-07-24: 记录成功来源（UAV-guided / UGV-independent）
                                state.success_source = "uav_guided" if ugv_action.candidate_id else "ugv_independent"
                                # R22 外审 P0 SEM-01：旧 success_source 仅反映 candidate_id
                                # 是否非空（auto-bind 后几乎恒 guided），保留只为旧报表兼容，
                                # 禁止当作"UAV 因果引导成功"口径。
                                state.success_confirm_binding_origin = _candidate_binding_origin
                                state.success_confirm_candidate_id = ugv_action.candidate_id or ""
                                msg_bus.send("UGV", "Confirmed %s." % (ugv_action.candidate_id or "independent"),
                                    state.ugv_sim_time, receiver="UAV", message_type="confirm",
                                    related_candidate_id=ugv_action.candidate_id or "", step=state.step)
                            else:
                                state.terminal = True
                                state.failure_stage = "FALSE_CONFIRM"
                                msg_bus.send("UGV", "False confirm %s." % (ugv_action.candidate_id or "independent"), state.ugv_sim_time,
                                    receiver="UAV", message_type="false_confirm",
                                    related_candidate_id=ugv_action.candidate_id or "", step=state.step)
                        elif ugv_action.action == "REJECT":
                            # R2-P2-04: auto-bind 已前移至验证事件记录前（normalize 块）
                            # False reject: projected_visible AND los_clear AND distance <= effective confirm radius
                            projected_visible = bool(gt_vis)
                            is_false_reject = (projected_visible and los_clear
                                               and d2t is not None and d2t <= max_confirm_distance_m)
                            if is_false_reject:
                                state.false_reject_count += 1
                                # 契约（2026-07-21 用户拍板）：REJECT 不终止 episode。
                                # false reject 只计数，episode 继续，UGV/UAV 恢复自由搜索。
                                logger.info("[%s] FALSE_REJECT: target was visible but UGV rejected (d=%.1fm), episode continues",
                                            state.episode_id, d2t)
                            if ugv_action.candidate_id:
                                # R2-P2-04: 返回值检查——错误 cid 不再静默 no-op
                                if not ledger.reject(ugv_action.candidate_id, step=state.step):
                                    logger.warning("[%s] REJECT ledger.reject(%s) no-op: candidate not in ledger",
                                                   state.episode_id, ugv_action.candidate_id)
                            # REJECT 后恢复自由搜索（不 terminal）
                            state.ugv_observation_mode = "UGV_SIXVIEW"
                            msg_bus.send("UGV", "Rejected %s." % (ugv_action.candidate_id or "independent"),
                                state.ugv_sim_time, receiver="UAV", message_type="reject",
                                related_candidate_id=ugv_action.candidate_id or "", step=state.step)
                            # 框架层激活：candidate 被拒，请 UAV 重新标定疑似目标区域
                            self._framework_activate_uav(state, msg_bus, "unreachable",
                                candidate_id=ugv_action.candidate_id or "",
                                extra="(candidate rejected by UGV)")

                    # ---- G1 UGV patrol guard（H2：判定核心移至 PatrolGuardState）----
                    # route_progress 模式：route 剩余里程连续 patrol_stall_steps 步
                    # 无下降（容差 0.5m）才触发；steps 模式保持旧行为。
                    _ac_now = ledger.get_active()
                    _guard_guidance = (getattr(ugv_obs, "pace_route_guidance", None)
                                       or getattr(ugv_obs, "trace_route_guidance", None))
                    _patrol_fire = _patrol_guard.update(
                        ugv_action.action,
                        _ac_now.candidate_id if _ac_now is not None else None,
                        _guard_guidance,
                        getattr(ugv_obs, "current_fine_node_id", None),
                    )
                    if _patrol_fire is not None:
                        _patrol_loop_guard_events += 1
                        if _patrol_fire["mode"] == "route_progress":
                            logger.warning("[%s] GUARD patrol-loop(route_progress): route remaining stalled %d steps at %.1fm for candidate %s; marking unserviceable",
                                           state.episode_id, _patrol_fire["stall_steps"],
                                           _patrol_fire["route_remaining_m"] or 0.0,
                                           _patrol_fire["candidate_id"])
                            _g_msg = ("Cannot verify candidate %s; route stalled %d steps; aborting."
                                      % (_patrol_fire["candidate_id"], _patrol_fire["stall_steps"]))
                        elif _patrol_fire["mode"] == "topology_cycle":
                            logger.warning("[%s] GUARD patrol-loop(topology_cycle): exact sparse cycle period=%d repeated=%d for candidate %s; marking unserviceable",
                                           state.episode_id, _patrol_fire["cycle_period"],
                                           _patrol_fire["cycle_repeats"],
                                           _patrol_fire["candidate_id"])
                            _g_msg = ("Cannot verify candidate %s; deterministic sparse movement cycle "
                                      "period %d repeated %d times; aborting."
                                      % (_patrol_fire["candidate_id"], _patrol_fire["cycle_period"],
                                         _patrol_fire["cycle_repeats"]))
                        else:
                            logger.warning("[%s] GUARD patrol-loop: UGV wandered %d steps around candidate %s without verification; marking unserviceable",
                                           state.episode_id, _patrol_fire["wander_steps"],
                                           _patrol_fire["candidate_id"])
                            _g_msg = ("Cannot verify candidate %s after %d moves; aborting."
                                      % (_patrol_fire["candidate_id"], _patrol_fire["wander_steps"]))
                        ledger.mark_unserviceable(_patrol_fire["candidate_id"], reason="patrol_loop_guard")
                        msg_bus.send("UGV", _g_msg,
                                     state.ugv_sim_time, receiver="UAV", message_type="unreachable",
                                     related_candidate_id=_patrol_fire["candidate_id"], step=state.step)
                        state.add_ugv_memory(action="GUARD_PATROL_LOOP",
                            params=_patrol_fire,
                            result="candidate marked unserviceable", outcome="candidate aborted")
                        state.uav_frozen = False
                        # Record for post-loop failure classification; if nothing else
                        # resolves the episode, step budget will end it and we surface
                        # the guard reason instead of a generic timeout.
                        _guard_abort_info = ("UGV_PATROL_LOOP", _patrol_fire["candidate_id"])

                    dt = compute_action_sim_time(ugv_action.action, "UGV", prev,
                        (state.ugv_pos["x"], state.ugv_pos["y"]), ugv_speed,
                        road_distance_m=(road_distance_m if goal_wp else None),
                        normal_turns=(normal_turns if goal_wp else 0),
                        u_turns=(u_turns if goal_wp else 0),
                        time_model=state.time_model)
                    state.ugv_sim_time += dt
                    if ugv_action.action == "CONFIRM":
                        state.add_ugv_memory(action=ugv_action.action,
                            params={"candidate_id": getattr(ugv_action, "candidate_id", "?")},
                            result="confirmed candidate")
                    elif ugv_action.action == "REJECT":
                        state.add_ugv_memory(action=ugv_action.action,
                            params={"candidate_id": getattr(ugv_action, "candidate_id", "?"),
                                    "reason": getattr(ugv_action, "reason", ""),
                                    "visual_description": getattr(ugv_action, "visual_description", "")},
                            result="rejected candidate")
                    elif ugv_action.action == "STOP_OBSERVE":
                        state.add_ugv_memory(action=ugv_action.action,
                            params={"candidate_id": getattr(ugv_action, "candidate_id", "?")},
                            result="waited in place (10s)")
                    elif ugv_action.action in UGV_MANEUVER_ACTIONS:
                        state.add_ugv_memory(action=ugv_action.action,
                            params={"distance_m": (goal_wp.get("distance_m") if goal_wp else None)},
                            result=("moved %.0fm" % (goal_wp.get("distance_m") or 0)) if goal_wp else "no maneuver option")
                    elif ugv_action.action == "MOVE_TO_ROAD_WAYPOINT":
                        state.add_ugv_memory(action=ugv_action.action,
                            params={"waypoint_id": ugv_action.waypoint_id or "?"},
                            result="driving")
                    elif ugv_action.action == "WAIT":
                        state.add_ugv_memory(action="WAIT", result="waiting")
                    else:
                        state.add_ugv_memory(action=ugv_action.action, result="done")
                    step_logs.append({"step": state.step, "role": "UGV", "action": ugv_action.action,
                        "sim_time": round(state.ugv_sim_time, 1), "visual_description": ugv_action.visual_description,
                        "pos_x": round(state.ugv_pos["x"], 1), "pos_y": round(state.ugv_pos["y"], 1),
                        "agent_yaw": round(state.ugv_pos.get("yaw", 0.0), 1),
                        "candidate_id": getattr(ugv_action, "candidate_id", ""),
                        "confidence": getattr(ugv_action, "confidence", None),
                        "reason": getattr(ugv_action, "reason", "") or "",
                        "waypoint_id": (goal_wp["id"] if goal_wp else getattr(ugv_action, "waypoint_id", None)),
                        "input_image_path": _ugv_last_img_path,
                        "gt_visible": gt_vis if ugv_action.action in ("CONFIRM","REJECT") else None,
                        "dist_to_target": round(d2t, 1) if ugv_action.action in ("CONFIRM","REJECT") else None,
                        "raw_response": getattr(ugv_agent, "last_raw_response", "")[:500]})
                    # Spec v2 §4.4/§4.6: action log + UGV decision trajectory
                    state.ugv_action_count += 1
                    if ugv_action.action == "STOP_OBSERVE":
                        state.stop_observe_count += 1
                    elif ugv_action.action == "REJECT":
                        state.reject_count += 1
                    ugv_decision_records.append({
                        "step_index": state.step, "sim_time_s": round(state.ugv_sim_time, 1),
                        "waypoint_id": (goal_wp["id"] if goal_wp else ""),
                        "world_x": round(state.ugv_pos["x"], 2), "world_y": round(state.ugv_pos["y"], 2),
                        "yaw": round(state.ugv_pos.get("yaw", 0.0), 2),
                        "action_type": ugv_action.action,
                        "candidate_id_if_any": getattr(ugv_action, "candidate_id", "") or ""})
                    action_log_records.append({
                        "step_index": state.step, "sim_time_s": round(state.ugv_sim_time, 1),
                        "agent_id": "UGV", "action_type": ugv_action.action,
                        "raw_action_text": getattr(ugv_agent, "last_raw_response", "")[:800],
                        "parse_success": (True if not _ugv_parse_status else _ugv_parse_status == "ok"),
                        "parse_status": _ugv_parse_status or "unknown",
                        "validation_success": _ugv_validation_success,
                        "fallback_applied": (not _ugv_validation_success) or (_ugv_parse_status not in ("", "ok")),
                        "fallback_reason": _ugv_fallback_reason or (_ugv_parse_status if _ugv_parse_status not in ("", "ok") else ""),
                        "model_raw_action_type": ((_ugv_policy_debug.get("raw") or {}).get("action") or ""),
                        "model_raw_candidate_id": ((_ugv_policy_debug.get("raw") or {}).get("candidate_id") or ""),
                        "method_resolved_action_type": ((_ugv_policy_debug.get("resolved") or {}).get("action") or getattr(_ugv_proposed,"action","")),
                        "proposed_action_type": getattr(_ugv_proposed, "action", ""),
                        "proposed_waypoint_id": getattr(_ugv_proposed, "waypoint_id", "") or "",
                        "proposed_message": getattr(_ugv_proposed, "message", "") or "",
                        "executed_action_type": ugv_action.action,
                        "agent_world_x": round(state.ugv_pos["x"], 2),
                        "agent_world_y": round(state.ugv_pos["y"], 2),
                        "agent_world_z": round(state.ugv_pos.get("z", 0.5), 2),
                        "agent_yaw": round(state.ugv_pos.get("yaw", 0.0), 2),
                        "selected_waypoint_id": (goal_wp["id"] if goal_wp else getattr(ugv_action, "waypoint_id", "") or ""),
                        "target_candidate_id": getattr(ugv_action, "candidate_id", "") or "",
                        "input_image_paths": [_ugv_last_img_path] if _ugv_last_img_path else [],
                        "confidence": getattr(ugv_action, "confidence", None),
                        "reason": getattr(ugv_action, "reason", "") or "",
                        "error_message": _ugv_fallback_reason or (_ugv_parse_status if _ugv_parse_status not in ("", "ok") else "")})

                    _ugv_receipt = {
                        **(_hybrid_navigation_receipt(ugv_action.action, goal_wp)
                           if False or False else {}),
                        "step": state.step, "action": ugv_action.action,
                        "candidate_id": getattr(ugv_action, "candidate_id", "") or "",
                        "waypoint_id": (goal_wp["id"] if goal_wp else getattr(ugv_action, "waypoint_id", "") or ""),
                        "pos": [round(state.ugv_pos["x"], 2), round(state.ugv_pos["y"], 2)],
                        "gt_visible": gt_vis if ugv_action.action in ("CONFIRM", "REJECT") else None,
                        "dist_to_target": round(d2t, 1) if d2t is not None and ugv_action.action in ("CONFIRM", "REJECT") else None,
                    }
                    if _is_coop_mode(self.agent_mode):
                        _ugv_receipt = _coop_public_receipt(
                            ugv_action, role="UGV",
                            pos=(state.ugv_pos["x"], state.ugv_pos["y"]),
                            effect_at=state.ugv_sim_time - dt, ready_at=state.ugv_sim_time)
                    _safe_on_transition(ugv_agent, ugv_obs,
                                        _ugv_proposed if _is_coop_mode(self.agent_mode) else ugv_action,
                                        ugv_action, _ugv_receipt)

                if state.vlm_calls >= state.max_vlm_calls and not state.terminal:
                    state.terminal = True
                    state.failure_stage = "VLM_BUDGET_EXHAUSTED"
                if not state.terminal and max(state.uav_sim_time, state.ugv_sim_time) > state.max_sim_time:
                    state.terminal = True
                    state.failure_stage = "SIM_TIME_EXHAUSTED"

            if not state.terminal:
                state.terminal = True
                state.failure_stage = "STEP_BUDGET_EXHAUSTED"

            ccg = None
            if state.candidate_created_count > 0:
                ccg = state.candidate_hit_count / state.candidate_created_count

            # OSR: legal Confirm opportunity at consumed predecision six-view poses.
            osr = bool(state.osr)
            # UGV-SPL requires directed road graph; placeholder until graph refactor is done
            _spl_measure = (spl_measurement(state.success, oracle_ugv_distance_m, ugv_distance)
                            if _strict_spl else None)
            ugv_spl_contribution = (_spl_measure["value"] if _strict_spl else
                (state.success * (oracle_ugv_distance_m / max(ugv_distance, oracle_ugv_distance_m))
                 if state.success and oracle_ugv_distance_m > 0 else 0.0))
            spl_value = round(ugv_spl_contribution, 4) if ugv_spl_contribution is not None else None
            state.spl = spl_value
            total_sim = max(state.uav_sim_time, state.ugv_sim_time)

            # Spec v2 §4.10: normalized failure reason + natural-language detail
            failure_reason, failure_reason_detail = _terminal_failure_reason(state, total_sim, _guard_abort_info)
            _outcome_class, _completed = _terminal_outcome_fields(state, failure_reason)
            if not state.success and not candidate_log_records:
                warnings_list.append("UAV 未产生任何 candidate 标注")
            if not target_reference_image_path:
                warnings_list.append("缺少目标参考图 target_reference.png")
            if not teammate_reference_image_path:
                warnings_list.append("缺少 UGV 队友参考图 teammate_reference_uav.png")
            # R2-P2-02：六视图黑图 panel 累计（相机超时降级痕迹，供审计）
            _sixview_black_total = getattr(state, "sixview_black_panel_total", 0)
            if _sixview_black_total:
                warnings_list.append("六视图存在黑图 panel 累计 %d 个（相机超时降级）" % _sixview_black_total)

            result = {
                "schema_version": "lite_result_v3_spec",
                "metric_protocol": metric_protocol(),
                "region_policy_revision": REGION_POLICY_REVISION,
                "aerial_metrics_applicable": True,
                "ase_sampling_valid": bool(_ase_sampling_valid),
                "gva_sampling_valid": bool(state.gva_sampling_valid),
                "metric_sampling_errors": state.metric_sampling_errors,
                # I0 run_meta（Phase-G Step 4）：代码/配置指纹，事后审计用
                "run_meta": _build_run_meta(self.config, self.agent_mode),
                "success_protocol": success_protocol(self.config),
                'episode_id': state.episode_id, 'mode': 'joint_' + self.agent_mode,
                "success": state.success, "failure_stage": state.failure_stage,
                "success_source": getattr(state, "success_source", None),
                # R22 外审 P0 SEM-01：旧字段语义标注 + 绑定来源 + 后验 oracle 字段。
                # 后验字段只进最终 result JSON，绝不进 prompt/观测/控制决策。
                "success_source_semantics": "legacy_candidate_id_presence_not_causal_guidance",
                "success_confirm_binding_origin": getattr(state, "success_confirm_binding_origin", None),
                "success_confirm_candidate_id": getattr(state, "success_confirm_candidate_id", None),
                "success_confirm_candidate_exact_hit_posthoc": (
                    next((bool(s.get("grid_hit")) for s in reversed(state.candidate_acg_scores)
                          if s.get("candidate_id") == getattr(state, "success_confirm_candidate_id", None)), None)
                    if state.success else None),
                "success_confirm_candidate_distance_to_polygon_m_posthoc": (
                    next((s.get("distance_to_candidate_polygon_m", s.get("distance_m"))
                          for s in reversed(state.candidate_acg_scores)
                          if s.get("candidate_id") == getattr(state, "success_confirm_candidate_id", None)), None)
                    if state.success else None),
                "success_any_candidate_exact_hit_posthoc": (
                    bool(state.candidate_hit_count > 0) if state.success else None),
                "steps": state.step, "vlm_calls": state.vlm_calls,
                "uav_vlm_calls": state.uav_vlm_calls, "ugv_vlm_calls": state.ugv_vlm_calls,
                "annotation_action_outputs": state.annotation_action_outputs,
                "annotation_attempt_count": state.annotation_attempt_count,
                "candidate_created_count": state.candidate_created_count,
                "candidate_hit_count": state.candidate_hit_count,
                "ccg": round(ccg, 4) if ccg is not None else None,
                "invalid_action_count": state.invalid_action_count, "confirm_count": state.confirm_count,
                "false_confirm_count": state.false_confirm_count, "false_reject_count": state.false_reject_count,
                "infra_error_count": getattr(state, "infra_error_count", 0),
                # N0 Go/No-Go 诊断标记（diagnostic run 不计入正式 SR）
                "diagnostic_mode": (self._diag_shield if self._diag_shield != "none" else None),
                "diagnostic_save_all_frames": bool(self._diag_save_all_frames),
                "diagnostic_save_all_uav_frames": bool(self._diag_save_all_uav_frames),
                "diagnostic_save_ugv_views_per_step": bool(self._diag_save_ugv_views),
                "diagnostic_save_full_visibility_debug": bool(self._diag_save_full_visibility_debug),
                "oracle_candidate_injected": bool(getattr(state, "oracle_candidate_injected", False)),
                "would_false_confirm_count": getattr(state, "would_false_confirm_count", 0),
                # N5a TRACE-Core：route advisory 观测（非绑定，不改动作）
                "route_advisory_enabled": bool(_route_advisory_on),
                "route_advisory_steps": int(_route_advisory_steps),
                "route_advisory_activated": int(_route_advisory_activated),
                # 2026-07-26 plain baseline 可观测性计数器（其他模式恒为 0）
                "raw_model_wait_count": getattr(state, "raw_model_wait_count", 0),
                "json_parse_failure_count": getattr(state, "json_parse_failure_count", 0),
                "missing_required_action_count": getattr(state, "missing_required_action_count", 0),
                "fallback_wait_count": getattr(state, "fallback_wait_count", 0),
                "uav_wait_with_message_count": getattr(state, "uav_wait_with_message_count", 0),
                "uav_wait_without_message_count": getattr(state, "uav_wait_without_message_count", 0),
                "anti_stall_force_move_count": getattr(
                    state, "anti_stall_force_move_count",
                    getattr(self.method, "anti_stall_force_move_count", 0)
                    if self.method is not None else 0),
                "grid_hit": state.grid_hit, "gt_visible_in_stop6": state.gt_visible_in_stop6,
                **finalize_navigation_error(state, actor_xy(target_actor), actor_xy(getattr(carla_iface, '_ugv_actor', None))),
                "oracle_ugv_distance_m": round(oracle_ugv_distance_m, 1),
                "oracle_uav_distance_m": round(oracle_uav_distance_m, 1),
                "uav_distance_m": round(uav_distance, 1), "ugv_distance_m": round(ugv_distance, 1),
                "osr": osr,
                "UGV_SPL_contribution": (ugv_spl_contribution if _strict_spl else
                                          round(ugv_spl_contribution, 4)),
                "spl_oracle_version": _navigation_certificate["version"] if _strict_spl else "legacy_static_v0",
                "spl_valid": _spl_measure["valid"] if _strict_spl else True,
                "spl_invalid_reason": _spl_measure["reason"] if _strict_spl else None,
                "oracle_ugv_distance_m_exact": oracle_ugv_distance_m,
                "ugv_distance_m_exact": ugv_distance,
                "navigation_oracle_sha256": (_navigation_certificate["certificate_sha256"]
                                             if _navigation_certificate else None),
                "spl": spl_value,
                "oracle_success": osr,
                "failure_reason": failure_reason,
                "failure_reason_detail": failure_reason_detail,
                "completed": _completed,
                "outcome_class": _outcome_class,
                "budget_exhausted": (state.failure_stage in ("VLM_BUDGET_EXHAUSTED", "SIM_TIME_EXHAUSTED", "STEP_BUDGET_EXHAUSTED", "EPISODE_WALL_TIMEOUT")),
                "uav_action_count": state.uav_action_count,
                "ugv_action_count": state.ugv_action_count,
                "stop_observe_count": state.stop_observe_count,
                "reject_count": state.reject_count,
                # B4（2026-09-17 审计）：nav-stall guard 触发计数（增量字段，0=未触发）
                "nav_stall_guard_events": int(_nav_stall_guard_events),
                "patrol_loop_guard_events": int(_patrol_loop_guard_events),
                "candidate_arrival_contract": ("confirmable_farthest_corner" if _confirmable_service_arrival
                                               else "broad_nearest_edge"),
                "sparse_pose_drift_max_m": round(float(_sparse_pose_drift_max_m), 4),
                "sparse_pose_drift_over_2m_count": int(_sparse_pose_drift_over_2m_count),
                "candidate_count": state.candidate_created_count,
                "max_confirm_distance_m": max_confirm_distance_m,
                # B14（2026-09-17 审计）：目标运行时实际 xyz（spawn 后回填 state.target，
                # 修复 plan z 恒 0.5 的离线复算误差；增量字段）
                "target_actual_xyz": [round(float(state.target["transform"]["x"]), 3),
                                      round(float(state.target["transform"]["y"]), 3),
                                      round(float(state.target["transform"]["z"]), 3)],
                "target_reference_image_path": target_reference_image_path,
                "teammate_reference_image_path": teammate_reference_image_path,
                "warnings": warnings_list,
                "uav_sim_time_s": round(state.uav_sim_time, 1), "ugv_sim_time_s": round(state.ugv_sim_time, 1),
                "total_sim_time_s": round(total_sim, 1),
                "time_model_version": state.time_model.get("version", "lite_time_v2"),
                "uav_trajectory": [(round(x,1), round(y,1)) for x,y in state.uav_trajectory],
                "ugv_trajectory": [(round(x,1), round(y,1)) for x,y in state.ugv_trajectory],
                "candidates": [{"candidate_id": c.candidate_id, "grid_cell": c.grid_cell,
                    "status": c.status, "region_center_xy": [round(c.region_center_xy[0],1), round(c.region_center_xy[1],1)],
                    "confidence": c.confidence,
                    "geometry_type": getattr(c, "geometry_type", "region"),
                    "source_frame_id": getattr(c, "source_frame_id", None),
                    "source_image_sha256": getattr(c, "source_image_sha256", ""),
                    "uncertainty_radius_m": c.uncertainty_radius_m,
                    "created_step": c.created_step,
                    "service_node_count": len(getattr(c, "service_node_ids", ()) or ()),
                    "confirmable_service_node_count": len(getattr(c, "confirmable_service_node_ids", ()) or ())} for c in ledger.candidates],
                "messages": [{"message_id": getattr(m, "message_id", "M???"),
                    "sender": m.sender, "receiver": getattr(m, "receiver", ""),
                    "message_type": getattr(m, "message_type", "info"),
                    "related_candidate_id": getattr(m, "related_candidate_id", ""),
                    "step_index": getattr(m, "step_index", 0),
                    "content": m.content, "sim_time": round(float(m.timestamp_step), 1)} for m in msg_bus.messages],
                "step_logs": step_logs,
                "distractors_spawned": len(distractor_actors),
                "runtime_error": None,
                # ACG/GVA metrics (2026-07-21 metrics v3)
                # Metric generation v5; ACG/GVA formulas remain v3.
                "ACG_episode": compute_acg_episode_v3(state.candidate_acg_scores, state.max_steps)["score"],
                "ACG_episode_detail": compute_acg_episode_v3(state.candidate_acg_scores, state.max_steps),
                "oracle_verifiable_reached": bool(state.oracle_verifiable_reached),
                "first_oracle_verifiable_step": getattr(state, "first_oracle_verifiable_step", None),
                # P0-I02：predecision 口径（VLM 输入 pose 对齐），机制分析主指标
                "first_predecision_oracle_verifiable_step": getattr(
                    state, "first_predecision_oracle_verifiable_step", None),
                "min_ugv_target_distance_m": (round(float(state.min_ugv_target_distance_m), 1)
                    if getattr(state, "min_ugv_target_distance_m", None) is not None else None),
                "min_visible_target_distance_m": (round(float(state.min_visible_target_distance_m), 1)
                    if getattr(state, "min_visible_target_distance_m", None) is not None else None),
                "GVA_episode": aggregate_verification_actions([{
                    "gva_events": state.gva_events, "gva_sampling_valid": state.gva_sampling_valid}])["GVA"],
                "Verification_Opportunity_Conversion_episode": (compute_gva_episode_v3(
                    state.success, state.oracle_verifiable_reached, not state.success and state.confirm_count > 0)
                    if state.gva_sampling_valid else None),
                "candidate_acg_scores": state.candidate_acg_scores,
                "gva_events": state.gva_events,
                "gva_event_count": len(state.gva_events),
                "gva_positive_event_count": sum(1 for e in state.gva_events if e["oracle_positive"]),
                "gva_negative_event_count": sum(1 for e in state.gva_events if not e["oracle_positive"]),
                "gva_tp": sum(1 for e in state.gva_events if e["model_verdict"] == "CONFIRM" and e["oracle_positive"]),
                "gva_tn": sum(1 for e in state.gva_events if e["model_verdict"] == "REJECT" and not e["oracle_positive"]),
                "gva_fp": sum(1 for e in state.gva_events if e["model_verdict"] == "CONFIRM" and not e["oracle_positive"]),
                "gva_fn": sum(1 for e in state.gva_events if e["model_verdict"] == "REJECT" and e["oracle_positive"]),
                # ASE/ACov metrics (2026-07-24 metrics v4)
                "metrics_schema_version": 6,
                "ase_episode": (compute_ase_episode(_ase_first_oracle_visible_uav_step, state.max_steps)
                                if _ase_sampling_valid else None),
                # Legacy actual-horizon diagnostic, not the paper's fixed-T ASE.
                "ase_episode_uavnorm": compute_ase_episode(
                    _ase_first_oracle_visible_uav_step, max(int(_ase_uav_local_step), 1)),
                "ase_uav_local_steps_final": int(_ase_uav_local_step),
                "ase_first_oracle_visible_uav_step": _ase_first_oracle_visible_uav_step,
                "ase_max_uav_decision_steps": state.max_steps,
                "acov_at_25": compute_acov_at_k(_acov_observed_cells, _acov_total_cells, 25, _acov_observed_events),
                "acov_at_50": compute_acov_at_k(_acov_observed_cells, _acov_total_cells, 50, _acov_observed_events),
                "acov_at_80": compute_acov_at_k(_acov_observed_cells, _acov_total_cells, 80, _acov_observed_events),
                "acov_total_cells": _acov_total_cells,
                "acov_observed_cells": len(_acov_observed_cells),
                "acov_observed_events": _acov_observed_events,  # P1-2: 时间线落盘，供离线按任意 K 重算
                # 2026-07-25: deterministic decoding 参数记录（可复现性）
                "request_params": {
                    "temperature": self.config.get("vlm", {}).get("temperature", 0.0),
                    "top_p": self.config.get("vlm", {}).get("top_p", None),
                    "max_output_tokens": self.config.get("vlm", {}).get("max_output_tokens", 800),
                    "seed": self.config.get("vlm", {}).get("seed", None),
                    "max_retries_per_call": self.config.get("vlm", {}).get("max_retries_per_call", self.config.get("vlm", {}).get("max_retries", 3)),
                    "timeout_s": self.config.get("vlm", {}).get("timeout_s", 120),
                    "total_timeout_s": self.config.get("vlm", {}).get("total_timeout_s", 180.0),
                    "wait_forever": False,
                    "model": self.config.get("vlm", {}).get("model", "unknown"),
                    "api_url": self.config.get("vlm", {}).get("api_url", "unknown"),
                },
            }


            result['aerial_metrics_applicable'] = True
            from agos.runtime.paper_audit import finalize_fault_audit
            finalize_fault_audit(self.config, state, result)

            # Spec v2 §4: write per-episode structured data files
            def _write_jsonl(path, records):
                with open(path, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            try:
                _write_jsonl(os.path.join(output_dir, "action_log.jsonl"), action_log_records)
                _write_jsonl(os.path.join(output_dir, "candidate_log.jsonl"), candidate_log_records)
                _write_jsonl(os.path.join(output_dir, "candidate_attempt_log.jsonl"), candidate_attempt_records)
                message_records = [{
                    "message_id": getattr(m, "message_id", "M???"),
                    "step_index": getattr(m, "step_index", 0),
                    "sim_time_s": round(float(m.timestamp_step), 1),
                    "sender_agent_id": m.sender,
                    "receiver_agent_id": getattr(m, "receiver", ""),
                    "message_type": getattr(m, "message_type", "info"),
                    "message_text": m.content,
                    "related_candidate_id": getattr(m, "related_candidate_id", ""),
                } for m in msg_bus.messages]
                _write_jsonl(os.path.join(output_dir, "message_log.jsonl"), message_records)
                if ag_os_trace_records:
                    _write_jsonl(os.path.join(output_dir, "ag_os_trace.jsonl"), ag_os_trace_records)
                if r17_opportunity_records:
                    _write_jsonl(os.path.join(output_dir, "r17_opportunity_log.jsonl"), r17_opportunity_records)
                # P0-3: Write dropped message stats for report sanity check
                if msg_bus._dropped_count > 0 or msg_bus._dropped_log:
                    with open(os.path.join(output_dir, "message_dropped_stats.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "total_dropped": msg_bus._dropped_count,
                            "dropped_messages": msg_bus._dropped_log,
                        }, f, indent=2, ensure_ascii=False)
                with open(os.path.join(output_dir, "uav_trajectory.json"), "w", encoding="utf-8") as f:
                    json.dump(uav_traj_records, f, indent=2, ensure_ascii=False)
                with open(os.path.join(output_dir, "ugv_trajectory.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "navigation_provenance": {
                            "action_space_version": ("sparse_nav_v1" if sparse_nav is not None else "ugv_maneuver_v2"),
                            "road_graph_id": state.road_graph_id,
                            "road_graph_asset_sha256": state.road_graph_asset_sha256,
                            "control_graph_release_id": (sparse_nav.release_id if sparse_nav is not None else ""),
                            "control_graph_sha256": (sparse_nav_sha or ""),
                            "semantic_adapter_sha256": episode_plan.get("semantic_adapter_sha256", ""),
                            "start_port_fine_node": episode_plan.get("ugv_start_port_fine_node", ""),
                        },
                        "decision_waypoints": ugv_decision_records,
                        "fine_path_segments": ugv_fine_segments,
                    }, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning("Failed to write spec data files: %s", e)
        finally:
            self.last_model_call_accounting = model_call_accounting_result(state)
            if self.method is not None:
                try:
                    self.method.close_episode()
                except Exception as _e:
                    logger.warning("method.close_episode failed: %s", _e)
            # R2-P1-07: 恢复进程原有 SIGTERM handler，避免同进程后续 episode
            # 受到本 episode 残留 handler 影响（handler 闭包引用已失效的 state）
            try:
                import signal as _signal
                if _prev_sigterm_handler is not None:
                    _signal.signal(_signal.SIGTERM, _prev_sigterm_handler)
            except Exception:
                pass
            # P2-5: 统一清理 target、干扰物和临时相机
            try:
                if target_actor and target_actor.is_alive:
                    target_actor.destroy()
            except Exception as e:
                _record_terminal_finalization_fault(state, result, "target_alive", e, "cleanup")
                logger.warning("Failed to clean up target actor: %s", e)
            for d in distractor_actors:
                try:
                    if d and d.is_alive:
                        d.destroy()
                except Exception as e:
                    _record_terminal_finalization_fault(state, result, "distractor_alive", e, "cleanup")
                    logger.warning("Failed to clean up distractor: %s", e)
            try:
                if carla_iface._cam_actor:
                    carla_iface._cam_actor.destroy()
                    carla_iface._cam_actor = None
            except Exception as e:
                _record_terminal_finalization_fault(state, result, "camera_cleanup", e, "cleanup")
                logger.warning("Failed to destroy camera: %s", e)
            if hasattr(carla_iface, "_runtime_target"):
                carla_iface._runtime_target = None
            # CECM and result writing also happen in finally so they run even on exception
            os.makedirs(output_dir, exist_ok=True)
            if cecm and result is not None:
                try:
                    img_dir = os.path.join(output_dir, "images")
                    imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)] if os.path.isdir(img_dir) else []
                    cecm.check_images(imgs)
                except Exception as e:
                    if not _record_terminal_finalization_fault(state, result, "cecm_check", e, "quality"):
                        raise
                try:
                    cecm.save(output_dir)
                except Exception as e:
                    if not _record_terminal_finalization_fault(state, result, "cecm_save", e, "quality"):
                        raise
            if result is not None:
                result.setdefault("run_meta", _build_run_meta(self.config, self.agent_mode))
                from agos.runtime.paper_audit import finalize_fault_audit
                finalize_fault_audit(self.config, state, result)
                result.update(uav_annotation_revision=UAV_ANNOTATION_REVISION,
                              success_protocol=success_protocol(self.config),
                              max_confirm_distance_m=confirm_distance_m(self.config))
                rp = os.path.join(output_dir, "%s_result.json" % state.episode_id)
                with open(rp, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                # Spec v2 §4.1/§4.2: run-level config + map metadata + road graph summary
                # P0-09: run-level files go to the map directory (shared across
                # episodes of the same map in the same run), not the episode dir.
                try:
                    _ep_dir = output_dir.rstrip("/\\")
                    _map_root = os.path.dirname(os.path.dirname(_ep_dir))
                    if os.path.basename(_map_root) == "episodes":
                        _map_root = os.path.dirname(_map_root)
                    _write_run_level_files(self, _map_root, episode_plan, state, region, store,
                                          locals().get("map_frame"), map_protocol=map_protocol)
                except Exception as e:
                    logger.warning("Failed to write run-level files: %s", e)
        return result


def _write_run_level_files(runner, output_dir, episode_plan, state, region, store, map_frame=None,
                           map_protocol=None):
    """写 run_config.json / map_metadata.json / road_graph_summary.json（幂等，已存在则跳过）"""
    import datetime as _dt
    cfg = runner.config
    _new_input_contract = bool(
        (map_protocol is not None and map_protocol.excludes_fullcity)
        or episode_plan.get("observation_contract") == "AGOS_ONLINE_VISUAL_MAP_V1"
        or (cfg.get("benchmark") or {}).get("observation_contract") == "AGOS_ONLINE_VISUAL_MAP_V1")
    _online_map = getattr(map_protocol, "online_map", None)
    _map_input_source = "raw_uav_rgb" if _online_map is not None else "none"
    if _new_input_contract:
        # The run-level MapFrame below belongs to navigation. Actual R32 image
        # dimensions/origin are episode-local and come from the active service.
        frame = None
        if _online_map is not None:
            frame = {
                "image_width_px": _online_map.width, "image_height_px": _online_map.height,
                "resolution_m": _online_map.resolution_m,
                "origin": {"x_max": _online_map.region["x_max"], "y_min": _online_map.region["y_min"]},
                "world_bounds": dict(_online_map.region),
                "world_to_pixel_transform": "px=(y-y_min)/resolution_m; py=(x_max-x)/resolution_m",
                "pixel_center_to_world_transform": "x=x_max-(row+0.5)*resolution_m; y=y_min+(col+0.5)*resolution_m",
                "projection": ("north_up_public_road_elevation_approximation" if getattr(map_protocol,'shared',False)
                               else "north_up_ground_plane_approximation"),
                "legend_footer_px": 48 if getattr(map_protocol,'shared',False) else 0,
            }
        if map_protocol is not None:
            os.makedirs(map_protocol.output_dir, exist_ok=True)
            with open(os.path.join(map_protocol.output_dir, "online_map_frame_metadata.json"), "w", encoding="utf-8") as out:
                json.dump({
                    "episode_id": episode_plan.get("episode_id"),
                    "observation_contract": "AGOS_ONLINE_VISUAL_MAP_V1",
                    "map_input_source": _map_input_source, "fullcity_rgb_allowed": False,
                    "map_frame_role": "observation_input_frame", "frame": frame,
                }, out, indent=2, ensure_ascii=False)

    # run_config.json — written once per run (skip if exists)
    run_cfg_path = os.path.join(output_dir, "run_config.json")
    if not os.path.exists(run_cfg_path):
        cfg = runner.config
        _run_id = (os.environ.get("AGOS_RUN_ID") or
                   os.path.basename(os.path.dirname(os.path.dirname(output_dir.rstrip("/\\")))) or
                   "unknown_run")
        _map_raw = str(episode_plan.get("map_name") or episode_plan.get("map") or "unknown_map")
        _map_name = os.path.basename(_map_raw).replace("_Opt", "") if "/" in _map_raw or "\\" in _map_raw else _map_raw.replace("_Opt", "")
        _action_space = str(episode_plan.get("action_space_version") or
                            cfg.get("ugv", {}).get("action_space_version") or "unknown")
        _agent_name = _agent_name_for_mode(runner.agent_mode)
        _navigator = (str(episode_plan.get("control_graph_release_id") or "sparse_semantic_control_graph")
                      if _action_space == "sparse_nav_v1" else "frozen_road_graph_v2")
        run_cfg = {
            "run_id": _run_id,
            "run_start_time": _dt.datetime.now().isoformat(timespec="seconds"),
            "agent_name": _agent_name,
            "agent_version": "lite_v1_spec_v3_r2",
            "vlm_model_name": (runner.vlm_config or {}).get("model", "N/A") if runner.vlm_config else "N/A",
            "prompt_version": "agents.py_v20261013_clean_uav_point_region_v1",
            "uav_annotation_revision": UAV_ANNOTATION_REVISION,
            "metric_protocol": metric_protocol(),
            "region_policy_revision": REGION_POLICY_REVISION,
            # R20 metadata fix: write the actual frozen action-space contract
            # instead of the historical hard-coded pre-sparse label.
            "action_space_version": _action_space,
            "view_protocol_version": "ugv_sixview_always_v2",
            "benchmark_name": "AG-OS-Bench",
            "benchmark_version": success_protocol(cfg)["id"],
            "success_protocol": success_protocol(cfg),
            "map_name": _map_name,
            # This file is map/run-level while formal runs mix difficulties.
            # Avoid presenting the first episode's difficulty as a run invariant.
            "difficulty": "mixed",
            "first_episode_difficulty": episode_plan.get("difficulty", "N/A"),
            "difficulty_scope": "per_episode",
            "max_steps": cfg.get("eval", {}).get("max_steps"),
            "max_vlm_calls": cfg.get("eval", {}).get("max_vlm_calls"),
            "max_sim_time_s": cfg.get("eval", {}).get("max_sim_time_s"),
            "success_max_confirm_distance_m": confirm_distance_m(cfg),
            "uav_altitude_m": cfg.get("uav", {}).get("altitude_m", 80),
            "uav_fov_deg": 90,
            "uav_waypoint_spacing_m": cfg.get("uav", {}).get("waypoint_spacing_m"),
            "uav_speed_mps": cfg.get("uav", {}).get("speed_mps", 5.0),
            "ugv_speed_mps": cfg.get("ugv", {}).get("speed_mps", 3.0),
            "ugv_confirm_budget": cfg.get("ugv", {}).get("confirm_budget"),
            "navigator": _navigator,
            "simulator": "CARLA-AIR 0.1.7",
            "time_model_version": state.time_model.get("version", "lite_time_v2"),
        }
        if _new_input_contract:
            run_cfg.update({
                "observation_contract": "AGOS_ONLINE_VISUAL_MAP_V1",
                "map_input_source": _map_input_source, "map_input_frame_scope": "per_episode",
            })
        with open(run_cfg_path, "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, indent=2, ensure_ascii=False)

    # map_metadata.json — written once per run (skip if exists)
    map_meta_path = os.path.join(output_dir, "map_metadata.json")
    if not os.path.exists(map_meta_path):
        mf = map_frame
        if mf is None:
            mf = MapFrame(region)
        map_w_m = mf.x_max - mf.x_min
        map_h_m = mf.y_max - mf.y_min
        _map_raw = str(episode_plan.get("map_name") or episode_plan.get("map") or "unknown_map")
        _map_name = os.path.basename(_map_raw).replace("_Opt", "") if "/" in _map_raw or "\\" in _map_raw else _map_raw.replace("_Opt", "")
        map_meta = {
            "map_name": _map_name,
            "world_x_min": round(mf.x_min, 2), "world_x_max": round(mf.x_max, 2),
            "world_y_min": round(mf.y_min, 2), "world_y_max": round(mf.y_max, 2),
            "map_width_m": round(map_w_m, 2), "map_height_m": round(map_h_m, 2),
            "image_width_px": mf.img_w, "image_height_px": mf.img_h,
            "meters_per_pixel_x": round(map_w_m / mf.img_w, 4),
            "meters_per_pixel_y": round(map_h_m / mf.img_h, 4),
            "origin_mode": "north_up_world",
            "axis_flip_applied": "image_y_down_is_world_south",
            "world_frame": {
                "north": "+X",
                "south": "-X",
                "east": "+Y",
                "west": "-Y"
            },
            "image_frame": {
                "right": "east/+Y",
                "down": "south/-X",
                "up": "north/+X",
                "left": "west/-Y"
            },
            "world_to_pixel_transform": "px=(y-y_min)/(y_max-y_min)*img_w; py=(x_max-x)/(x_max-x_min)*img_h",
            "pixel_to_world_transform": "x=x_max-py/img_h*(x_max-x_min); y=y_min+px/img_w*(y_max-y_min)",
            "note": "North-up map: image up=+X/north, right=+Y/east. Do not use legacy V8_CAMERA_YAW=-90 convention.",
        }
        if _new_input_contract:
            map_meta.update({
                "observation_contract": "AGOS_ONLINE_VISUAL_MAP_V1",
                "map_input_source": _map_input_source, "fullcity_rgb_allowed": False,
                "map_frame_role": "navigation_frame_only", "applies_to_map_input": False,
                "map_input_frame_scope": "per_episode",
                "map_input_frame_metadata": "each episode's online_map_frame_metadata.json",
            })
        with open(map_meta_path, "w", encoding="utf-8") as f:
            json.dump(map_meta, f, indent=2, ensure_ascii=False)

    # road_graph_summary.json — written once per run (skip if exists)
    rg_path = os.path.join(output_dir, "road_graph_summary.json")
    if not os.path.exists(rg_path) and store is not None:
        nodes = [{"node_id": nid, "x": round(n.x, 2), "y": round(n.y, 2)}
                 for nid, n in store.nodes.items()]
        edges = [{"edge_id": eid, "src": e.src, "dst": e.dst,
                  "length_m": round(e.length_m, 2),
                  "geometry": [[round(g[0], 2), round(g[1], 2)] for g in e.geometry] if e.geometry else []}
                 for eid, e in store.edges.items()]
        with open(rg_path, "w", encoding="utf-8") as f:
            json.dump({
                "graph_id": state.road_graph_id,
                "asset_sha256": state.road_graph_asset_sha256,
                "node_count": len(nodes), "edge_count": len(edges),
                "nodes": nodes, "edges": edges,
            }, f, indent=2, ensure_ascii=False)




