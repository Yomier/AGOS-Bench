"""Resolve released task addresses. Episode names are unique only within a split and town."""
import csv
import json
from pathlib import Path

from .scoring import sha256

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {'train1100': ('Train1100', 1100), 'validation275': ('Val275', 275),
          'test300': ('UnseenTest300', 300), 'seentest450': ('SeenTest450', 450)}


def select_keys(split, episodes):
    source = ROOT / 'dataset/agos-tasksets-splits-v1.json'
    split_index = json.loads(source.read_text(encoding='utf-8'))['splits']
    if split == 'all' and episodes:
        raise ValueError('select one split for an explicit episode address')
    wanted = set(episodes)
    selected = []
    for name in SPLITS if split == 'all' else [split]:
        label, count = SPLITS[name]
        keys = split_index[label]['episode_keys']
        if len(keys) != count or len(set(keys)) != count:
            raise ValueError(f'invalid membership for {name}')
        missing = wanted.difference(keys)
        if missing:
            raise ValueError(f'unknown episode in {name}: {sorted(missing)}')
        selected.extend((name, key) for key in keys if not wanted or key in wanted)
    return selected


def load_tasks(data_root, selected):
    from agos.episode_io import load_and_validate_episode_plan
    from agos.eval.contamination import _extract_xy
    with (ROOT / 'dataset/tasks.csv').open(encoding='utf-8-sig', newline='') as stream:
        index = {(row['split'], row['episode_key']): row for row in csv.DictReader(stream)}
    tasks, metadata = [], {}
    for split, key in selected:
        town, episode = key.split('/')
        entry = index[split, key]
        for asset in ('plan', 'target_reference', 'teammate_reference'):
            file = (data_root / entry[asset + '_path']).resolve()
            if not file.is_relative_to(data_root) or sha256(file) != entry[asset + '_sha256']:
                raise ValueError(f'{split}/{key}: {asset} differs from the released task index')
        path = data_root / entry['plan_path']
        plan = load_and_validate_episode_plan(str(path), formal=True, check_reference=True)
        actual = plan['map'].split('/')[-1].removesuffix('_Opt')
        if actual != town or plan['episode_id'] != episode:
            raise ValueError(f'manifest and embedded identity disagree at {path}')
        task = dict(split=split, town=town, episode=episode, path=str(path), key=f'{split}/{key}',
                    plan_sha256=sha256(path))
        tasks.append(task)
        tgt = plan['target_oracle']
        meta = dict(episode_id=episode, map_key=town, difficulty=plan.get('difficulty'),
                    target_class=tgt.get('blueprint_filter') or tgt.get('blueprint_id'),
                    target_loc=_extract_xy(tgt), ugv_start=_extract_xy(plan['start']['ugv']),
                    distractor_locs=[xy for d in plan.get('distractors_oracle', []) if (xy := _extract_xy(d))])
        metadata.setdefault(f'{split}/{town}', []).append(meta)
    return tasks, metadata
