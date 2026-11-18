"""Build mixed and role-specific training files from extracted expert trajectories."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .train import TrainError, validate_training_row


@contextmanager
def new_directory(destination):
    """Publish a complete dataset after every row and image has passed validation."""
    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        staged = Path(scratch) / 'corpus'
        (staged / 'images').mkdir(parents=True)
        yield staged
        staged.rename(output)


def row_role(row):
    content = row['messages'][1]['content']
    text = content if isinstance(content, str) else ''.join(
        part['text'] for part in content if part['type'] == 'text')
    payload = json.loads(text)
    role = payload.get('role') if isinstance(payload, dict) else None
    if role not in ('UAV', 'UGV') or role != row['meta']['role']:
        raise ValueError('Deployed prompt role and metadata must agree on UAV or UGV')
    return role


def build_corpus(source, destination):
    source = Path(source).resolve()
    counts = Counter()
    keys, episodes, images = set(), set(), set()
    hashes = {name: hashlib.sha256() for name in ('mixed', 'uav', 'ugv')}
    with new_directory(destination) as output:
        with ExitStack() as stack:
            files = {name: stack.enter_context((output / f'{name}.jsonl').open('xb')) for name in hashes}
            lines = stack.enter_context((source / 'rows.jsonl').open('rb'))
            for number, raw in enumerate(lines, start=1):
                row = json.loads(raw)
                validate_training_row(row, number)
                role = row_role(row)
                meta = row['meta']
                key = (meta['episode_key'], meta['call_id'])
                if not all(isinstance(v, str) and v for v in key) or key in keys:
                    raise ValueError(f'Line {number}: missing or duplicated episode/call identity {key}')
                if meta.get('heldout') or meta['capture_site'] not in (
                        'expert_decision', 'expert_verify', 'expert_confirm'):
                    raise ValueError(f'Line {number}: expected a training expert site')
                keys.add(key)
                episodes.add(key[0])
                for name in row['images']:
                    if name in images:
                        continue
                    path = (source / name).resolve()
                    if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != Path(name).stem:
                        raise ValueError(f'Line {number}: image SHA256 or path mismatch: {name}')
                    try:
                        os.link(path, output / name)
                    except OSError:
                        shutil.copyfile(path, output / name)
                    images.add(name)
                for split in ('mixed', role.lower()):
                    files[split].write(raw)
                    hashes[split].update(raw)
                    counts[split] += 1
        if not counts['mixed']:
            raise ValueError('No training rows found')
        report = {'episodes': len(episodes), 'unique_images': len(images),
                  'rows': {name: counts[name] for name in hashes},
                  'sha256': {name: value.hexdigest() for name, value in hashes.items()}}
        (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Directory containing rows.jsonl and images/')
    parser.add_argument('--output', type=Path, required=True, help='New corpus directory')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_corpus(args.input, args.output), indent=2))
    except (KeyError, OSError, ValueError, TrainError) as exc:
        parser.exit(1, f'corpus: {exc}\n')


if __name__ == '__main__':
    main()
