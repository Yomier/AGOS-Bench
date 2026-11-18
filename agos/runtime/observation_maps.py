"""Publish causal RGB observation maps to cooperative-search agents."""
import hashlib
import json
import os
import copy

ONLINE_MAP_CONTRACT = "AGOS_ONLINE_VISUAL_MAP_V1"


class ObservationMapProtocol:
    def __init__(self, config, plan, agent_mode, output_dir):
        contract = plan.get("observation_contract")
        configured = (config.get("benchmark") or {}).get("observation_contract")
        if contract not in (None, "", ONLINE_MAP_CONTRACT):
            raise ValueError("unsupported episode observation_contract: %r" % contract)
        if configured not in (None, "", ONLINE_MAP_CONTRACT):
            raise ValueError("unsupported benchmark observation_contract: %r" % configured)
        map_config = config.get('observation_map') or {}
        capability = agent_mode == 'cooperative_search' and bool(map_config.get('online_map_enabled'))
        self.excludes_fullcity = bool(contract or configured or capability or agent_mode == 'minimal_agent')
        self.online_map = None
        self.output_dir = output_dir
        diag = {**(config.get("diagnostics") or {}), **(config.get("diagnostic") or {})}
        self.snapshot_interval = max(0, int(diag.get("online_map_snapshot_every_n_steps",
                                                   1 if diag.get("save_online_map_snapshots", False) else 0)))
        self.last_update = {"status": "not_updated"}
        self.shared = False
        self.teammate_reference_contract = plan.get('teammate_reference_contract', '')
        if isinstance(plan.get('teammate_reference'), dict):
            self.teammate_reference_contract = plan['teammate_reference'].get('protocol',
                plan['teammate_reference'].get('contract', self.teammate_reference_contract))
        self.last_frame = None
        self._projection_height_enabled = bool((config.get('candidate') or {}).get('use_ground_elevation', False))
        self._projection_height = None
        self._height_region = plan.get('region')
        self._height_resolution = map_config.get('online_map_resolution_m', .5)
        if capability:
            if map_config.get('variant_id') != 'R32_ONLINE_MAP':
                raise ValueError('Unsupported observation map variant')
            from agos.methods.common.online_map import OnlineVisualMap
            self.online_map = OnlineVisualMap(plan['region'], map_config.get('online_map_resolution_m', .5))
            start = (plan.get("start") or {}).get("ugv")
            if start:
                xy = (start["x"], start["y"])
                self.online_map.note_ugv(0.0, xy, [xy])

    def set_ground_elevation(self, provider):
        if self._projection_height_enabled and provider is not None:
            from agos.perception.public_road_elevation import PublicRoadElevation
            self._projection_height = PublicRoadElevation.from_provider(
                self._height_region, self._height_resolution, provider)


    def update_uav(self, frame, event_time):
        if self.online_map is not None:
            try:
                self.last_update = self.online_map.update(frame, available_at=event_time)
            except Exception as exc:
                self.last_update = {"status": "error", "error": str(exc)}
            self.last_update["available_at"] = float(event_time)
        return self.last_update

    def attach(self, obs, event_time, ugv_time, ugv_xy, ugv_trajectory):
        obs.candidate_use_ground_elevation = self._projection_height_enabled
        obs.public_ground_elevation = self._projection_height
        if self.teammate_reference_contract:
            obs.teammate_reference_contract = self.teammate_reference_contract
        if not self.excludes_fullcity:
            return
        obs.observation_contract = ONLINE_MAP_CONTRACT
        if self.online_map is None:
            obs.topdown_map = b""
            meta = {"status": "excluded", "source": "none", "update_count": 0,
                    "coverage_fraction": 0.0, "source_frames": [], "overlays": []}
        else:
            try:
                self.online_map.note_ugv(ugv_time, ugv_xy, ugv_trajectory)
                obs.topdown_map, meta = self.online_map.render(event_time, overlays=True)
                meta['last_update'] = (self.last_update if self.last_update.get('available_at', -1) <= event_time
                                       else {'status': 'pending_future_update'})
            except Exception as exc:
                obs.topdown_map = b""
                meta = {"status": "error", "source": "raw_uav_rgb", "error": str(exc),
                        "update_count": 0, "coverage_fraction": 0.0, "source_frames": []}
        meta["contract"] = ONLINE_MAP_CONTRACT
        meta["fullcity_rgb_allowed"] = False
        meta["input_sha256"] = hashlib.sha256(obs.topdown_map).hexdigest() if obs.topdown_map else None
        obs.map_metadata = meta
        os.makedirs(self.output_dir, exist_ok=True)
        row = {"step": obs.step, "role": obs.role, "event_time": float(event_time), **meta}
        with open(os.path.join(self.output_dir, "online_map_log.jsonl"), "a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False)+"\n")
        if self.snapshot_interval and (obs.step == 1 or obs.step % self.snapshot_interval == 0) and obs.topdown_map:
            folder = os.path.join(self.output_dir, "images")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "online_map_%s_s%03d.png" % (obs.role.lower(), obs.step)), "wb") as out:
                out.write(obs.topdown_map)
