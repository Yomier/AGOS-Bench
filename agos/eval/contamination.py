"""contamination.py — Cross-Episode Contamination Monitor (CECM)

用于检测多任务并行评测中的内容泄露：
1. VLM 输出中出现其他 episode 的 episode_id / 目标类别 / 目标位置。
2. 当前 episode 轨迹靠近其他 episode 的私有 actor（目标、UGV起点、干扰物）。
3. 图像缓存中出现其他 episode 文件。
4. 输出目录文件命名混用。

本模块同时支持：
- 独立后检（post_check）
- 运行时轻量检查（RuntimeContaminationMonitor）
"""
import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path


def _extract_xy(record):
    """兼容两种 distractor 结构：扁平 x/y 和嵌套 transform.x/y。缺失字段返回 None。"""
    if not isinstance(record, dict):
        return None
    transform = record.get("transform")
    source = transform if isinstance(transform, dict) else record
    if "x" not in source or "y" not in source:
        return None
    try:
        return float(source["x"]), float(source["y"])
    except (TypeError, ValueError):
        return None


class ContaminationMonitor:
    """多 episode 内容泄露监测器。"""

    def __init__(self, episodes: List[Dict[str, Any]]):
        """
        episodes: [
            {'episode_id': str, 'difficulty': str, 'target_class': str,
             'target_loc': (x, y), 'ugv_start': (x, y), 'distractor_locs': [(x,y), ...]},
            ...
        ]
        """
        self.episodes = {ep['episode_id']: ep for ep in episodes}
        self.alerts: List[Dict[str, Any]] = []

    # ---------- 单项检查 ----------
    def check_vlm_output(self, episode_id: str, output_text: str):
        """检查 VLM 输出是否包含其他 episode 的敏感信息。"""
        if not isinstance(output_text, str):
            return
        for other_id, other_ep in self.episodes.items():
            if other_id == episode_id:
                continue
            if other_id in output_text:
                self._alert('episode_id_leak', other_id, episode_id,
                            f'VLM output of {episode_id} contains episode_id {other_id}')
            # B22（2026-09-17 审计）：target_class 匹配优先用 episode 元数据的
            # target_description（颜色/车型人类可读描述，若存在）——blueprint 串
            # （如 vehicle.tesla.model3）是内部资产名，VLM 自然语言输出不可能
            # 原样包含，旧匹配恒空转；blueprint 串仅作 description 缺失时的 fallback。
            desc = other_ep.get('target_description', '')
            target_class = other_ep.get('target_class', '')
            probe = desc or target_class
            if probe and probe in output_text:
                self._alert('target_class_leak', other_id, episode_id,
                            f'VLM output of {episode_id} mentions target description/class of {other_id}: {probe}')
            target_loc = other_ep.get('target_loc')
            if target_loc and len(target_loc) >= 2:
                x_int, y_int = int(target_loc[0]), int(target_loc[1])
                # 匹配 "x=123, y=456" 或 "(123, 456)" 等常见坐标格式
                pattern = re.compile(rf'(?:{x_int}[.,;]?\s*{y_int}|{y_int}[.,;]?\s*{x_int})')
                if pattern.search(output_text):
                    self._alert('target_location_leak', other_id, episode_id,
                                f'VLM output of {episode_id} may contain target location of {other_id}: {target_loc}')

    def check_trajectory_contamination(self, episode_id: str, trajectory: List[Tuple[float, float]], threshold_m: float = 5.0):
        """检查当前 episode 轨迹是否靠近其他 episode 的私有 actor。"""
        if not trajectory:
            return
        for t, pos in enumerate(trajectory):
            if len(pos) < 2:
                continue
            px, py = float(pos[0]), float(pos[1])
            for other_id, other_ep in self.episodes.items():
                if other_id == episode_id:
                    continue
                for key in ('target_loc', 'ugv_start', 'distractor_locs'):
                    points = other_ep.get(key, [])
                    if key == 'distractor_locs':
                        pass
                    elif points and isinstance(points, (tuple,)):
                        points = [points]
                    elif points and isinstance(points, list) and len(points) > 0 and not isinstance(points[0], (list, tuple)):
                        points = [points]
                    for pt in points or []:
                        if not pt or len(pt) < 2:
                            continue
                        dx, dy = px - float(pt[0]), py - float(pt[1])
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < threshold_m:
                            self._alert('trajectory_contamination', other_id, episode_id,
                                        f'Episode {episode_id} trajectory step {t} at ({px:.1f},{py:.1f}) is {dist:.1f}m from {other_id} {key} ({float(pt[0]):.1f},{float(pt[1]):.1f})')

    def check_image_cache(self, episode_id: str, image_paths: List[str]):
        """检查图像缓存目录是否混入其他 episode。"""
        for p in image_paths:
            fname = os.path.basename(p)
            if episode_id not in fname:
                self._alert('image_cache_contamination', '-', episode_id,
                            f'Episode {episode_id} image cache contains file not belonging to it: {fname}')

    def check_file_isolation(self, result_dir: str, expected_episodes: List[str]):
        """检查输出目录下是否有命名不符合预期 episode_id 的文件。"""
        root = Path(result_dir)
        if not root.exists():
            return
        for diff_dir in root.iterdir():
            if not diff_dir.is_dir():
                continue
            if diff_dir.name == 'report_input':
                continue
            for f in diff_dir.iterdir():
                if not f.is_file():
                    continue
                fname = f.name
                if fname == 'metrics.json':
                    continue
                if not any(ep_id in fname for ep_id in expected_episodes):
                    self._alert('file_isolation_violation', '-', str(diff_dir),
                                f'File {fname} in {diff_dir} does not match any expected episode_id: {expected_episodes}')

    # ---------- 内部工具 ----------
    def _alert(self, typ: str, from_id: str, to_id: str, detail: str):
        self.alerts.append({'type': typ, 'from': from_id, 'to': to_id, 'detail': detail})

    def _count_types(self) -> Dict[str, int]:
        counts = {}
        for a in self.alerts:
            counts[a['type']] = counts.get(a['type'], 0) + 1
        return counts

    def summary(self) -> Dict[str, Any]:
        return {
            'total_episodes': len(self.episodes),
            'total_alerts': len(self.alerts),
            'alert_types': self._count_types(),
            'alerts': self.alerts,
            'clean': len(self.alerts) == 0,
        }


class RuntimeContaminationMonitor:
    """运行时 CECM：只关注当前 episode 与其他已知 episode 的交互。"""

    def __init__(self, current_episode: Dict[str, Any], sibling_episodes: List[Dict[str, Any]]):
        """
        current_episode: 当前 episode 元数据
        sibling_episodes: 其他 episode 元数据列表
        """
        self.episode_id = current_episode['episode_id']
        all_eps = [current_episode] + list(sibling_episodes)
        self.monitor = ContaminationMonitor(all_eps)
        self.alerts_seen = set()

    def check_vlm(self, raw_response: str):
        self.monitor.check_vlm_output(self.episode_id, raw_response)

    def check_trajectory(self, trajectory: List[Tuple[float, float]], threshold_m: float = 5.0):
        self.monitor.check_trajectory_contamination(self.episode_id, trajectory, threshold_m)

    def check_images(self, image_paths: List[str]):
        self.monitor.check_image_cache(self.episode_id, image_paths)

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'{self.episode_id}_contamination.json')
        summary = self.monitor.summary()
        summary['episode_id'] = self.episode_id
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def summary(self):
        return self.monitor.summary()


# ================= 独立后检入口 =================

def _extract_episode_metadata(plan_path: Path) -> Optional[Dict[str, Any]]:
    if not plan_path.exists():
        return None
    c = json.loads(plan_path.read_text(encoding='utf-8'))
    plan = c.get('episode_plan', c)  # 兼容直接就是 plan 的文件
    tgt = plan.get('target_oracle', {})
    tgt_tf = tgt.get('transform', {})
    start = plan.get('start', {})
    ugv_start = start.get('ugv', {})
    target_class = tgt.get('blueprint_filter', '') or tgt.get('blueprint_id', '')
    # B22（2026-09-17 审计）：抽取人类可读的 target_description（颜色/车型描述），
    # 供 check_vlm_output 优先匹配（blueprint 串 VLM 输出不可能包含，见该函数注释）
    target_description = plan.get('target_task', {}).get('target_description', '')
    return {
        'episode_id': plan.get('episode_id', plan_path.stem.replace('_combined', '')),
        'difficulty': plan_path.parent.name,
        'target_class': target_class,
        'target_description': target_description,
        'target_loc': (float(tgt_tf.get('x', 0.0)), float(tgt_tf.get('y', 0.0))),
        'ugv_start': (float(ugv_start.get('x', 0.0)), float(ugv_start.get('y', 0.0))),
        'distractor_locs': [xy for d in plan.get('distractors_oracle', []) if (xy := _extract_xy(d)) is not None],
    }


def post_check(result_dir: str, plan_dir: str = None):
    """读取结果目录，对所有 episode 进行内容泄露后检。"""
    root = Path(result_dir)
    episodes = []
    result_files = sorted(root.glob('*/*_result.json'))
    for rf in result_files:
        ep_id = rf.stem.replace('_result', '')
        difficulty = rf.parent.name
        # 优先从 combined 读取 plan；否则尝试 result 本身
        combined_path = rf.parent / f'{ep_id}_combined.json'
        ep = _extract_episode_metadata(combined_path) if combined_path.exists() else None
        if ep is None:
            # 从 result 文件提取 minimal 信息
            r = json.loads(rf.read_text(encoding='utf-8'))
            ep = {
                'episode_id': ep_id,
                'difficulty': difficulty,
                'target_class': '',
                'target_description': '',  # B22：与 _extract_episode_metadata 键集对齐
                'target_loc': (0.0, 0.0),
                'ugv_start': (0.0, 0.0),
                'distractor_locs': [],
            }
            # 尝试解析 trajectory 最后一个点作为目标位置兜底
            traj = r.get('uav_trajectory') or r.get('ugv_trajectory') or []
            if traj:
                ep['target_loc'] = (float(traj[-1][0]), float(traj[-1][1]))
        episodes.append(ep)

    monitor = ContaminationMonitor(episodes)
    for ep in episodes:
        ep_id = ep['episode_id']
        rf = root / ep['difficulty'] / f'{ep_id}_result.json'
        if rf.exists():
            text = rf.read_text(encoding='utf-8')
            monitor.check_vlm_output(ep_id, text)
            r = json.loads(text)
            if 'uav_trajectory' in r:
                monitor.check_trajectory_contamination(ep_id, r['uav_trajectory'], threshold_m=5.0)
            if 'ugv_trajectory' in r:
                monitor.check_trajectory_contamination(ep_id, r['ugv_trajectory'], threshold_m=5.0)
        img_dir = root / ep['difficulty'] / 'images'
        if img_dir.exists():
            imgs = [str(p) for p in img_dir.iterdir() if p.is_file()]
            monitor.check_image_cache(ep_id, imgs)

    expected_ids = [ep['episode_id'] for ep in episodes]
    monitor.check_file_isolation(result_dir, expected_ids)

    return monitor.summary()
