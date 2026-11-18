# -*- coding: utf-8 -*-
"""menu_oracle method（E5a benchmark 机制验证用 scripted agent，无 VLM）。

设计文档：docs/designs/2026-09-12-benchmark-readiness-verification.md D1/D4。
- UGV：GT 引导菜单贪心（target_xy 最近目标）+ 覆盖牵引（未用动作类型优先）
  + 20% 随机扰动；候选流转复刻 oracle（unverified 接近 → assigned STOP_OBSERVE
  带 message → visited 视距内 CONFIRM）。
- UAV：复刻 runner.OracleAgent 的投影 annotate 逻辑（GT 目标投影到俯视网格
  按点 annotate），其余 WAIT。
- 目的：验证 sparse_nav_v1 全链（plan→adapter→菜单→分发→fine_path 执行→
  状态提交→候选契约→UAV 生命周期），非搜索能力对照。
"""
import math
import random

from agos.methods.base import MethodServices

from agos.lite_core import LiteAction


_MOVEMENT_ACTIONS = ("FORWARD_1", "FORWARD_2", "TURN_LEFT",
                     "TURN_RIGHT", "TURN_STRAIGHT", "REVERSE")


class MenuOracleMethod:
    method_id = "menu_oracle"

    def __init__(self, config=None, services: MethodServices = None, profile=None):
        self.config = config if config is not None else {}
        self.services = services
        self.profile = profile
        self._target_xy = (0.0, 0.0)
        self._episode_id = ""
        self._used_actions = set()
        self._step_count = 0
        self._annotated = False
        self._last_move = ""
        self.sys_template = "default"
        # runner 兼容字段（i1 artifact / parse 可观测性读取）
        self.last_raw_response = ""
        self.last_parse_status = ""

    def get_mode_name(self):
        return "menu_oracle"

    # --- MethodSession 协议 ---
    def start_episode(self, episode_context):
        ctx = episode_context if isinstance(episode_context, dict) else {}
        plan = ctx.get("episode_plan", {}) or {}
        target = plan.get('target_oracle') or {}
        tf = target.get('transform') or target
        self._target_xy = (float(tf['x']), float(tf['y']))
        self._episode_id = ctx.get("episode_id", "")
        self._used_actions = set()
        self._step_count = 0
        self._annotated = False
        self._last_move = ""
        # L2 可解性证书路线（start→approach 的 L0 fine node 顺序序列）：
        # 沿序号贪心 = 永不迷路的理想导航（E5a 全量教训：欧氏贪心在复杂
        # 路网陷入局部陷阱，Town10HD min_dist 67.8m 打转 60 步）。
        pre = plan.get("static_solvability_precheck") or {}
        route = pre.get("certified_route_node_ids") or []
        self._route = list(route)
        self._route_actions = list(pre.get("certified_route_actions") or [])
        self._route_pos = {n: i for i, n in enumerate(route)}
        self._approach_node = pre.get("certified_ugv_approach_node_id") or ""

    def on_transition(self, observation, proposed_action=None,
                      executed_action=None, execution_result=None):
        return None

    def build_trace(self, obs):
        return {"method": self.method_id, "episode_id": self._episode_id,
                "step": getattr(obs, "step", 0),
                "used_actions": sorted(self._used_actions)}

    def close_episode(self):
        return None

    # --- 决策 ---
    def decide(self, obs):
        role = getattr(obs, "role", "")
        if role == "UAV":
            return self._decide_uav(obs)
        return self._decide_ugv(obs)

    def _decide_uav(self, obs):
        if getattr(obs, "uav_frozen", False):
            return LiteAction(role="UAV", action="WAIT",
                              reason="Frozen: waiting for framework arrival contract.")
        active = getattr(obs, "active_candidate", None)
        if active is not None:
            return LiteAction(role="UAV", action="WAIT",
                              reason="Candidate under UGV verification.")
        # 多轮：上一候选终局后允许再 annotate（与 OracleAgent 同口径）
        if self._annotated and not getattr(obs, "candidates", None):
            self._annotated = False
        elif self._annotated and all(
                getattr(c, "status", "") in ("rejected", "confirmed")
                for c in obs.candidates):
            self._annotated = False
        if not self._annotated:
            try:
                from agos.perception.perception import compute_grid_cell_from_world
                pos = obs.agent_position
                plan_yaw = -90
                cell, _px, _py = compute_grid_cell_from_world(
                    pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0,
                    plan_yaw, 90, 1024, 576,
                    self._target_xy[0], self._target_xy[1])
                if cell:
                    self._annotated = True
                    return LiteAction(
                        role="UAV", action="ANNOTATE_CANDIDATE", geometry_type='point', position_xy=list(self._target_xy),
                        localization_source='menu_oracle',
                        confidence=0.95,
                        visual_description="Target lies in the current aerial footprint.",
                        evidence="menu_oracle GT projection.",
                        reason="Annotating for service-contract validation.")
            except Exception:
                pass
        return LiteAction(role="UAV", action="WAIT", reason="menu_oracle UAV hold.")

    def _decide_ugv(self, obs):
        self._step_count += 1
        menu = [m for m in (getattr(obs, "available_waypoints", None) or [])
                if isinstance(m, dict) and m.get("action") in _MOVEMENT_ACTIONS
                and m.get("target_xy") is not None]
        ux, uy = obs.agent_position[0], obs.agent_position[1]
        tx, ty = self._target_xy
        dist_gt = math.hypot(ux - tx, uy - ty)
        max_confirm = (self.config.get("success", {})
                       .get("max_confirm_distance_m", 30))
        confirm_d = min(25.0, float(max_confirm))
        active = getattr(obs, "active_candidate", None)
        cid = (getattr(active, "candidate_id", "") or "") if active is not None else ""
        cur_node = getattr(obs, "current_fine_node_id", "") or ""

        # 有静态可解性证书时，必须走到 certificate 绑定的 approach node 才确认。
        # 不能因为中途欧氏距离偶然 <=25m 就提前 CONFIRM：道路可能隔着车道/护栏，
        # 更关键的是该位置并未通过生成时的 LOS + STOP6 可见性门。R12 外审实证
        # Town02/audit_easy_00003 在距 GT 22m 处提前确认且 gt_visible=False。
        has_certified_route = bool(self._route_pos and self._approach_node)
        if self._approach_node and cur_node == self._approach_node:
            return LiteAction(role="UGV", action="CONFIRM",
                              candidate_id=cid or None, confidence=0.95,
                              visual_description="Target visible in 6 views.",
                              evidence="certified approach reached (%.1fm)." % dist_gt,
                              reason="Confirm at certified approach.")
        if not has_certified_route and dist_gt <= confirm_d:
            return LiteAction(role="UGV", action="CONFIRM",
                              candidate_id=cid or None, confidence=0.95,
                              visual_description="Target visible in 6 views.",
                              evidence="within %.1fm." % dist_gt,
                              reason="Confirm after heuristic approach.")

        if menu:
            act = self._pick_route_action(menu, cur_node)
            if act:
                return self._move(act, "certified-route guided navigation")
        if has_certified_route:
            # Oracle/audit harness must fail closed on certificate/menu mismatch.
            # Falling back to Euclidean local-optimum confirmation could turn an
            # infrastructure/adapter mismatch into a false benchmark verdict.
            return LiteAction(role="UGV", action="WAIT",
                              reason="menu_oracle: certified route cannot advance at current port.")
        # Legacy plan without certificate: retain heuristic behavior.
        return self._decide_ugv_fallback(obs, menu, dist_gt, max_confirm, cid)

    def _pick_route_action(self, menu, cur_node):
        """Follow a certified executable semantic route deterministically.

        Certificate v2 stores one action for every consecutive landing-port pair.
        We verify the expected target before executing it.  Legacy certificates
        without actions fall back to monotonic route-index progress, but never
        take an off-route action merely for action-coverage statistics: a
        benchmark oracle must validate the mechanism, not perturb it.
        """
        if not self._route_pos:
            return None
        cur_idx = self._route_pos.get(cur_node, -1)
        if cur_idx < 0:
            return None

        # v2 exact semantic certificate.
        if cur_idx < len(self._route_actions) and cur_idx + 1 < len(self._route):
            expected_action = self._route_actions[cur_idx]
            expected_target = self._route[cur_idx + 1]
            for item in menu:
                if (item.get("action") == expected_action and
                        item.get("target_fine_node_id") == expected_target):
                    self._used_actions.add(expected_action)
                    self._last_move = expected_action
                    return expected_action
            # Fail closed into legacy monotonic matching rather than choosing an
            # arbitrary movement.  The caller may fall back, and audit logs can
            # expose a certificate/adapter mismatch.

        # Legacy certificate: choose the legal landing furthest forward on route.
        best, best_idx = None, cur_idx
        for item in menu:
            idx = self._route_pos.get(item.get("target_fine_node_id"), -1)
            if idx > best_idx:
                best, best_idx = item, idx
        if best is None:
            return None
        action = best["action"]
        self._used_actions.add(action)
        self._last_move = action
        return action

    def _decide_ugv_fallback(self, obs, menu, dist_gt, max_confirm, cid):
        """欧氏贪心兜底（无证书路线的 legacy plan 或偏离后无路线落点时）。"""
        tx, ty = self._target_xy
        best, best_d = None, None
        for m in menu:
            d = math.hypot(m["target_xy"][0] - tx, m["target_xy"][1] - ty)
            if best_d is None or d < best_d:
                best, best_d = m, d
        if best is None:
            return LiteAction(role="UGV", action="WAIT",
                              reason="menu_oracle: empty menu at non-excluded port.")
        if best_d >= dist_gt - 3.0 and dist_gt <= max_confirm:
            return LiteAction(role="UGV", action="CONFIRM",
                              candidate_id=cid or None, confidence=0.95,
                              visual_description="Best reachable view of target.",
                              evidence="no closer menu landing (%.1fm vs %.1fm)." % (best_d, dist_gt),
                              reason="Confirm at local optimum within range.")
        act = self._pick_menu_action(menu, self._target_xy)
        if act:
            return self._move(act, "GT-guided menu navigation (fallback)")
        return LiteAction(role="UGV", action="WAIT",
                          reason="menu_oracle: no usable menu action.")

    # --- 菜单选择策略 ---
    def _pick_menu_action(self, menu, goal_xy):
        """GT 引导贪心 + 覆盖牵引 + 随机扰动 + REVERSE 防抖。menu 项: action/distance_m/target_xy。"""
        if not menu:
            return None
        unused = [m for m in menu if m["action"] not in self._used_actions]
        # 覆盖牵引：存在未用动作类型时以较高概率优先（保证六动作覆盖矩阵）
        pool = unused if unused and random.random() < 0.4 else menu
        # REVERSE 防抖（E5a 全量教训）：宏动作 overshoot 后贪心会反复选 REVERSE
        # 回退形成振荡耗尽步数；连续 REVERSE 禁用（除非唯一选项）。
        non_rev = [m for m in pool if m["action"] != "REVERSE"]
        if self._last_move == "REVERSE" and non_rev:
            pool = non_rev
        if non_rev and random.random() < 0.20:
            pick = random.choice(non_rev)
        else:
            gx, gy = goal_xy
            pick = min(pool, key=lambda m: math.hypot(
                m["target_xy"][0] - gx, m["target_xy"][1] - gy))
        self._used_actions.add(pick["action"])
        self._last_move = pick["action"]
        return pick["action"]

    def _move(self, act, why):
        return LiteAction(role="UGV", action=act,
                          visual_description="menu_oracle scripted move.",
                          evidence=why, reason="menu_oracle: %s" % why)
