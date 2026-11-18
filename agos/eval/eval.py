"""eval.py — SuccessChecker + Metrics + Report Writer"""
import json, math, os, logging
from typing import List, Dict
from agos.benchmark_protocol import confirm_distance_m, assert_uniform_result_protocol, success_protocol
from .metric_contract import (METRIC_REVISION, V5_PROTOCOL, aggregate_osr,
                             aggregate_verification_actions, assert_uniform_metric_protocol,
                             aggregate_navigation_error, finite_distance)

logger = logging.getLogger("LiteV1.Eval")


class SuccessChecker:
    def __init__(self, config):
        self.require_stop = config["success"]["require_stop_observe"]
        self.require_gt_visible = config["success"]["require_gt_visible_in_stop_6view"]
        self.require_los = config["success"].get("require_los", False)
        self.max_confirm_dist = confirm_distance_m(config)
        self.protocol = success_protocol(config)
        self.confirm_budget = config["ugv"].get("confirm_budget", 1)

    def observation_is_confirmable(self, gt_visible, distance_to_target,
                                    los_clear, observation_mode="UGV_SIXVIEW"):
        """Pure production CONFIRM predicate, shared with the SPL oracle."""
        return (distance_to_target is not None
                and distance_to_target <= self.max_confirm_dist
                and (not self.require_stop or observation_mode == "UGV_SIXVIEW")
                and (not self.require_gt_visible or bool(gt_visible))
                and (not self.require_los or los_clear is True))

    def check(self, state, action, gt_visible, distance_to_target, los_clear=None, projected_visible=None):
        """检查 CONFIRM 是否成功。只有 CONFIRM 改变 terminal success/failure。

        契约（2026-07-21 用户拍板）：
        - CONFIRM：对=success+terminal，错=false_confirm+terminal
        - REJECT：永远不终止任务；关闭 candidate，恢复搜索。false reject 计数由上层统一处理。
        """
        if action and action.action == "CONFIRM":
            state.confirm_count += 1
            state.confirm_budget -= 1

            if distance_to_target is None:
                # P1-11: distance unknown (no STOP_OBSERVE performed), treat as failure
                state.false_confirm_count += 1
                state.terminal = True
                state.failure_stage = "FALSE_CONFIRM"
                return False, "distance_unknown"

            if self.require_stop and state.ugv_observation_mode != "UGV_SIXVIEW":
                # R2 自查（2026-09-05）：v2 六视图常态化后 runner 只设置 UGV_SIXVIEW，
                # 旧串 UGV_STOP_6VIEW 永不相等——require_stop 开启时所有 CONFIRM
                # 会恒判 FALSE_CONFIRM。当前 require_stop_observe=false 不触发，
                # 此处修正比对串使该开关恢复正确语义（六视图模式即可 confirm）。
                state.false_confirm_count += 1
                state.terminal = True
                state.failure_stage = "FALSE_CONFIRM"
                state.ne_distance = distance_to_target
                return False, "invalid_confirm_no_stop"
            if self.require_gt_visible and not gt_visible:
                state.false_confirm_count += 1
                state.terminal = True
                state.failure_stage = "FALSE_CONFIRM"
                state.ne_distance = distance_to_target
                return False, "false_confirm"
            if distance_to_target > self.max_confirm_dist:
                state.false_confirm_count += 1
                state.terminal = True
                state.failure_stage = "FALSE_CONFIRM"
                state.ne_distance = distance_to_target
                return False, "too_far_%.0fm" % distance_to_target
            if self.require_los:
                if los_clear is None or los_clear is False:
                    state.false_confirm_count += 1
                    state.terminal = True
                    state.failure_stage = "FALSE_CONFIRM"
                    state.ne_distance = distance_to_target
                    return False, "los_blocked" if los_clear is False else "los_unclear"

            state.success = True
            state.terminal = True
            state.osr = True
            state.ne_distance = distance_to_target
            return True, "success"

        elif action and action.action == "REJECT":
            # REJECT never changes terminal state. Caller handles false_reject_count.
            return False, "rejected"

        return False, "no_confirm"


class Metrics:
    def __init__(self):
        self.episodes = []

    def add_episode(self, result):
        self.episodes.append(result)

    def compute_joint_metrics(self):
        # B2（2026-09-17 审计）：聚合前剔除 infra_error 与 diagnostic episode——
        # 前者是基础设施崩溃（非模型能力），后者是诊断 run（不计正式指标）。
        # 两类分别计数写入返回 dict；total 为有效 episode 数。infra 优先归类
        # （同时命中两者时只计入 infra），避免双重计数。
        valid_episodes = []
        n_infra_excluded = 0
        n_diagnostic_excluded = 0
        n_implementation_excluded = 0
        for e in self.episodes:
            if str(e.get('outcome_class', '') or '') == 'implementation_error':
                n_implementation_excluded += 1
            elif str(e.get("outcome_class", "") or "") == "infra_error":
                n_infra_excluded += 1
            elif e.get("diagnostic_mode", False):
                n_diagnostic_excluded += 1
            else:
                valid_episodes.append(e)
        total = len(valid_episodes)
        protocol_keys = assert_uniform_result_protocol(valid_episodes)
        metric_keys = assert_uniform_metric_protocol(valid_episodes)
        action_metrics = metric_keys == [METRIC_REVISION]
        current_metrics = action_metrics or metric_keys == [V5_PROTOCOL["id"]]
        if total == 0:
            if not self.episodes:
                return {}
            # 全部被剔除时仍返回剔除计数，便于排查（不再返回空 dict 掩盖事实）
            return {"total": 0, "n_infra_excluded": n_infra_excluded,
                    "n_implementation_excluded": n_implementation_excluded,
                    "n_diagnostic_excluded": n_diagnostic_excluded,
                    "infra_error_count": n_infra_excluded,
                    "infra_error_rate": (n_infra_excluded / len(self.episodes)
                                         if self.episodes else 0),
                    "infra_error_rate_denominator": len(self.episodes)}
        successes = sum(1 for e in valid_episodes if e["success"])
        success_steps = [e.get("steps", 0) for e in valid_episodes if e["success"]]
        cand_created = sum(1 for e in valid_episodes if e.get("candidate_created_count", 0) > 0)
        annot_attempts = sum(1 for e in valid_episodes if e.get("annotation_attempt_count", 0) > 0)
        grid_hits = sum(1 for e in valid_episodes if e.get("grid_hit", False))
        ugv_go = 0
        ugv_stop = 0
        gt_vis = 0
        ugv_confirm_vis = 0
        eps_with_cand = 0
        eps_with_stop = 0
        eps_with_vis = 0
        eps_with_candidate_and_stop = 0
        for e in valid_episodes:
            logs = e.get("step_logs", [])
            has_cand = e.get("candidate_created_count", 0) > 0
            if has_cand:
                eps_with_cand += 1
            did_stop = any(s.get("action") == "STOP_OBSERVE" for s in logs if s.get("role") == "UGV")
            did_go = any(s.get("action") == "GO_TO_CANDIDATE" for s in logs if s.get("role") == "UGV")
            if has_cand and did_go:
                ugv_go += 1
            if did_stop:
                ugv_stop += 1
                eps_with_stop += 1
                if has_cand:
                    eps_with_candidate_and_stop += 1
            if e.get("gt_visible_in_stop6", False):
                gt_vis += 1
                eps_with_vis += 1
                if any(s.get("action") == "CONFIRM" for s in logs if s.get("role") == "UGV"):
                    ugv_confirm_vis += 1

        # CCG = macro average: per-episode hit/created, then average across episodes
        ccg_per_ep = []
        total_hit = 0
        total_created = 0
        for e in valid_episodes:
            created = e.get("candidate_created_count", 0)
            hit = e.get("candidate_hit_count", 0)
            if created > 0:
                ccg_per_ep.append(hit / created)
                total_hit += hit
                total_created += created
        avg_ccg = sum(ccg_per_ep) / len(ccg_per_ep) if ccg_per_ep else None
        micro_ccg = total_hit / total_created if total_created > 0 else None

        # OSR and UGV-SPL require full directed road graph; placeholders updated when graph is ready
        osr_count = sum(1 for e in valid_episodes if e.get("osr", False))
        # A scientific SPL aggregate cannot mix metric generations or silently
        # exclude an oracle/executor contradiction. Other metrics remain usable.
        spl_versions = {e.get("spl_oracle_version", "legacy_static_v0") for e in valid_episodes}
        spl_invalid = sum(e.get("spl_valid", True) is not True for e in valid_episodes)
        spl_aggregate_valid = len(spl_versions) == 1 and not spl_invalid
        if action_metrics and any('UGV_SPL_contribution' not in e or e.get('spl_valid') is not True
                                  for e in valid_episodes):
            spl_aggregate_valid = False
        spls = [e.get("UGV_SPL_contribution", 0) for e in valid_episodes]
        if any(v is None or not isinstance(v, (int, float)) or isinstance(v, bool)
               or not math.isfinite(v) or not 0 <= v <= 1 for v in spls):
            spl_aggregate_valid = False
        avg_spl = sum(spls) / len(spls) if spl_aggregate_valid and spls else None

        # Current NE covers every terminal pose; incomplete current data invalidates NE.
        ne_metrics = aggregate_navigation_error(valid_episodes, current_metrics)

        # P0-09: infra error count (not model failure)
        infra_errors = sum(e.get("infra_error_count", 0) for e in valid_episodes)

        # GVA v3 (2026-07-24): episode 级条件成功率 = task_success / (oracle_verifiable_reached OR invalid_confirm)。
        # 2026-08-02 修复（bug review）：runner 现已在 CONFIRM/REJECT 时刻落盘
        # oracle_verifiable_reached 字段（gt可见+距离≤max_confirm+LOS 的 oracle 条件）；
        # 旧结果无此字段时 fallback 到 osr（口径偏宽，仅用于兼容历史数据，
        # 历史数据的正确口径以 dist/投影修复_离线重算报告 为准）。
        # B15（2026-09-17 审计）：gva_strict 标记本次聚合是否发生过 osr fallback
        # （False=历史宽口径，True=严格口径），两代数据可区分。
        gva_episodes = []
        gva_fallback_osr_used = False
        gva_complete = True
        for e in valid_episodes:
            success = bool(e.get("success", False))
            _has_strict_field = ("oracle_verifiable_reached" in e)
            oracle_reached = bool(e.get("oracle_verifiable_reached",
                                       False if current_metrics else e.get("osr", False)))
            if not _has_strict_field:
                gva_fallback_osr_used = True
            if current_metrics and (not _has_strict_field or e.get('gva_sampling_valid') is not True
                                    or (success and not oracle_reached)):
                gva_complete = False
            invalid_confirm = not success and e.get("confirm_count", 0) > 0
            gva_episodes.append({
                "task_success": success,
                "oracle_verifiable_reached": oracle_reached,
                "invalid_confirm": invalid_confirm,
            })
        from .metric_primitives import compute_gva_v3, compute_ase, compute_acov
        gva_result = compute_gva_v3(gva_episodes)
        gva = gva_result["gva"] if gva_complete else None
        gva_support = gva_result["gva_support"]
        action_gva = aggregate_verification_actions(valid_episodes) if action_metrics else {}
        strict_osr = aggregate_osr(valid_episodes) if action_metrics else {}

        # ACG: average per-episode ACG scores (0 for episodes with no candidates)
        aerial_episodes = [e for e in valid_episodes if e.get('aerial_metrics_applicable', True)]
        acg_values = [e.get("ACG_episode", 0.0) for e in aerial_episodes]
        acg_complete = all(finite_distance(v) and v <= 1 for v in acg_values)
        if current_metrics:
            acg_complete = acg_complete and all('ACG_episode' in e for e in aerial_episodes)
        acg = sum(acg_values) / len(acg_values) if acg_values and acg_complete else None

        # ASE/ACov metrics (2026-07-24 metrics v4)
        ase_complete = all(e.get('ase_sampling_valid') is True
                           and 'ase_first_oracle_visible_uav_step' in e
                           and 'ase_max_uav_decision_steps' in e for e in aerial_episodes) if current_metrics else True
        try:
            ase = compute_ase(aerial_episodes) if ase_complete else None
        except (ValueError, TypeError):
            ase, ase_complete = None, False
        # P1-2 (2026-09-04): 聚合层优先平均 episode 级 acov_at_K（runner 已按 UAV 本地步
        # 截断）；旧 result 无该字段时退回 compute_acov 的累积口径（K 被忽略）。
        def _acov_agg(field: str, k: int):
            vals = [e.get(field) for e in valid_episodes if e.get(field) is not None]
            if vals:
                return round(sum(vals) / len(vals), 6)
            return compute_acov(valid_episodes, k=k)
        acov_25 = _acov_agg("acov_at_25", 25)
        acov_50 = _acov_agg("acov_at_50", 50)
        acov_80 = _acov_agg("acov_at_80", 80)

        # Team DS includes all executed role decisions, including unsuccessful
        # episodes. Internal perception/model requests are accounted separately.
        ds_values = []
        for e in valid_episodes:
            counts = [e.get('uav_action_count'), e.get('ugv_action_count')]
            logs = e.get('step_logs', [])
            if (not all(type(v) is int and v >= 0 for v in counts) or
                    any(sum(s.get('role') == role for s in logs) != counts[i]
                        for i, role in enumerate(('UAV', 'UGV')))):
                continue
            ds_values.append(sum(counts))
        ds_valid = len(ds_values) == total

        return {
            "metrics_schema_version": 6 if action_metrics else 5 if current_metrics else 4,
            "success_protocol_keys": protocol_keys,
            "total": total,
            # B2（2026-09-17 审计）：剔除计数（total 为剔除后的有效数）
            "n_infra_excluded": n_infra_excluded,
            "n_diagnostic_excluded": n_diagnostic_excluded,
            "n_implementation_excluded": n_implementation_excluded,
            "DS": sum(ds_values) / total if ds_valid else None,
            "ds_aggregation_valid": ds_valid,
            "ds_valid_samples": len(ds_values),
            "SR": successes / total,
            "OSR": osr_count / total if total else 0,
            "UGV_SPL": avg_spl,
            "spl_aggregation_valid": spl_aggregate_valid,
            "spl_oracle_versions": sorted(spl_versions),
            "spl_invalid_count": spl_invalid,
            "spl_zero_optimum_count": sum(e.get("oracle_ugv_distance_m_exact") == 0
                                          for e in valid_episodes),
            **ne_metrics,
            "metric_protocol_keys": metric_keys,
            "ACG": acg,
            "acg_aggregation_valid": acg_complete,
            "GVA": gva,
            "gva_aggregation_valid": gva_complete,
            "gva_support": gva_support,
            "Verification_Opportunity_Conversion": (gva_result["gva"]
                if gva_complete and gva_support else None),
            "verification_opportunity_support": gva_support,
            **action_gva,
            **strict_osr,
            # B15（2026-09-17 审计）：True=严格口径（oracle_verifiable_reached 字段），
            # False=发生过 osr fallback（历史宽口径）
            "gva_strict": True if action_metrics else not gva_fallback_osr_used,
            "ASE": ase,
            "ase_aggregation_valid": ase_complete,
            "aerial_metric_support": len(aerial_episodes),
            "ACov@25": acov_25,
            "ACov@50": acov_50,
            "ACov@80": acov_80,
            "num_oracle_verifiable": gva_result["num_oracle_verifiable"],
            "num_invalid_confirm": gva_result["num_invalid_confirm"],
            "avg_CCG": avg_ccg,
            "micro_CCG": micro_ccg,
            "total_candidate_hit": total_hit,
            "total_candidate_created": total_created,
            "UAV_annotation_attempt_rate": annot_attempts / total,
            "UAV_candidate_create_rate": cand_created / total,
            "UAV_grid_hit_rate": grid_hits / total if total else 0,
            "UGV_go_to_candidate_rate": ugv_go / eps_with_cand if eps_with_cand else 0,
            "UGV_stop_observe_rate": ugv_stop / total if total else 0,
            "candidate_followup_stop_rate": eps_with_candidate_and_stop / eps_with_cand if eps_with_cand else 0,
            "independent_stop_rate": (eps_with_stop - eps_with_candidate_and_stop) / total if total else 0,
            "GT_visible_in_stop6_rate": gt_vis / eps_with_stop if eps_with_stop else 0,
            # B5（2026-09-17 审计）：任意步曾可见率——分子同 gt_visible_in_stop6 的
            # "任意步曾可见"口径，分母改为全部有效 episode（旧字段分母为停过车的
            # episode，两口径并存便于对照）
            "gt_visible_ever_rate": gt_vis / total if total else 0,
            "UGV_confirm_when_visible_rate": ugv_confirm_vis / eps_with_vis if eps_with_vis else 0,
            "false_confirm_rate": sum(e.get("false_confirm_count", 0) for e in valid_episodes) / total,
            "false_reject_rate": sum(e.get("false_reject_count", 0) for e in valid_episodes) / total,
            # R22 外审 P0 MET-01：episode 级 infra 发生率必须以**全部提交 episode**
            # 为分母（剔除之前计算）；剔除后再算恒为 0。事件计数单列，两口径不再混用。
            "infra_error_rate": n_infra_excluded / len(self.episodes) if self.episodes else 0,
            "infra_error_count": n_infra_excluded,
            "infra_error_rate_denominator": len(self.episodes),
            "infra_error_event_count_within_valid_episodes": infra_errors,
            "avg_annotation_attempts": sum(e.get("annotation_attempt_count", 0) for e in valid_episodes) / total,
            "avg_candidates_created": sum(e.get("candidate_created_count", 0) for e in valid_episodes) / total,
            "avg_confirms": sum(e.get("confirm_count", 0) for e in valid_episodes) / total,
            "avg_invalid_actions": sum(e.get("invalid_action_count", 0) for e in valid_episodes) / total,
            "avg_uav_vlm_calls": sum(e.get("uav_vlm_calls", 0) for e in valid_episodes) / total,
            "avg_ugv_vlm_calls": sum(e.get("ugv_vlm_calls", 0) for e in valid_episodes) / total,
            **{f"avg_{key}": (sum(e[key] for e in valid_episodes) / total
                if all(type(e.get(key)) is int and e[key] >= 0 for e in valid_episodes) else None)
               for key in ('vlm_http_attempts', 'uav_vlm_http_attempts', 'ugv_vlm_http_attempts')},
            "avg_steps_to_success": sum(success_steps) / len(success_steps) if success_steps else 0,
            "avg_steps": sum(e.get("steps", 0) for e in valid_episodes) / total,
            "avg_uav_sim_time": sum(e.get("uav_sim_time_s", 0) for e in valid_episodes) / total,
            "avg_ugv_sim_time": sum(e.get("ugv_sim_time_s", 0) for e in valid_episodes) / total,
            "avg_total_sim_time": sum(e.get("total_sim_time_s", 0) for e in valid_episodes) / total,
            "avg_uav_distance_m": sum(e.get("uav_distance_m", 0) for e in valid_episodes) / total,
            "avg_ugv_distance_m": sum(e.get("ugv_distance_m", 0) for e in valid_episodes) / total,
            "failure_stages": self._failure_breakdown(valid_episodes),
            "runtime_exception_count": sum(1 for e in valid_episodes if e.get("failure_stage") == "RUNTIME_EXCEPTION"),
        }

    def compute_uav_only_metrics(self):
        """Deprecated: UAV-only is no longer a formal benchmark mode."""
        total = len(self.episodes)
        if total == 0:
            return {}
        successes = sum(1 for e in self.episodes if e["success"])
        return {
            "total": total,
            "UAV_Annotation_SR_deprecated": successes / total,
            "note": "UAV-only is not a formal mode; results are for smoke test only.",
            "avg_annotation_attempts": sum(e.get("annotation_attempt_count", 0) for e in self.episodes) / total,
            "avg_candidates_created": sum(e.get("candidate_created_count", 0) for e in self.episodes) / total,
        }

    def compute_ugv_search_only_metrics(self):
        # Same scientific aggregation gates as the joint benchmark. Auxiliary
        # episodes without an Oracle SPL measurement report N/A, never zero.
        result = self.compute_joint_metrics()
        if not result:
            return result
        eligible = [e for e in self.episodes if e.get('outcome_class') != 'infra_error'
                    and not e.get('diagnostic_mode', False)]
        if any('UGV_SPL_contribution' not in e for e in eligible):
            result['UGV_SPL'] = None
            result['spl_aggregation_valid'] = False
        return result

    def _failure_breakdown(self, episodes=None):
        # B2：episodes 参数供 compute_joint_metrics 传入剔除后的有效集；
        # 不传时保持旧行为（全部 episode），兼容 deprecated 两个聚合入口。
        stages = {}
        for e in (self.episodes if episodes is None else episodes):
            if not e["success"]:
                stage = e.get("failure_stage", "UNKNOWN")
                stages[stage] = stages.get(stage, 0) + 1
        return stages


def write_report(results, metrics, output_path):
    assert_uniform_result_protocol([r for r in results
        if r.get('outcome_class') != 'infra_error' and not r.get('diagnostic_mode', False)])
    assert_uniform_metric_protocol([r for r in results
        if r.get('outcome_class') != 'infra_error' and not r.get('diagnostic_mode', False)])
    report = {"total_episodes": len(results), "metrics": metrics, "episodes": results}
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report written to {output_path}")
