# -*- coding: utf-8 -*-
"""r7 边界重构：MethodSession 协议与 MethodServices。

benchmark 与 method 的契约只有两个：
  Benchmark -> Method : LiteObservation（canonical raw observation）
  Method    -> Benchmark : LiteAction
每个方法管理自己的决策和观测表示。
"""
from typing import Any, Optional, Protocol, runtime_checkable

from agos.lite_core import LiteAction, LiteObservation


@runtime_checkable
class MethodSession(Protocol):
    """一级方法的统一会话接口（episode 级生命周期）。

    obs.role 决定 UAV/UGV 分支；episode 级共享资源（shared belief、
    reference embedding、episode trace）挂在 session 上。
    """

    method_id: str

    def start_episode(self, episode_context: Any) -> None:
        ...

    def decide(self, obs: LiteObservation) -> LiteAction:
        ...

    def on_transition(self, observation: LiteObservation, proposed_action: LiteAction = None,
                      executed_action: LiteAction = None, execution_result: Any = None) -> None:
        ...

    def build_trace(self, obs: LiteObservation) -> dict:
        ...

    def close_episode(self) -> None:
        ...


class MethodServices:
    """benchmark 提供给方法的只读服务能力（三方法相同注入，方法自行取舍）。

    road_graph/router/approach_planner 提供只读路网。
    Optional for methods without a road planner.
    """

    def __init__(self, road_graph: Optional[Any] = None,
                 router: Optional[Any] = None,
                 approach_planner: Optional[Any] = None,
                 semantic_nav: Optional[Any] = None):
        self.road_graph = road_graph
        self.router = router
        self.approach_planner = approach_planner
        # Read-only sparse semantic navigation adapter.  Methods may use this
        # to score/plan over the exact action graph exposed by the benchmark;
        # it contains no target GT and does not bypass the runner action menu.
        self.semantic_nav = semantic_nav
