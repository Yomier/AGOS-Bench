"""Reassemble selected released plans and their reference PNGs from an extracted volume."""
import argparse
import csv
from pathlib import Path
import shutil

from benchmark.tasks import ROOT, SPLITS, load_tasks, select_keys


ASSETS = ('plan_path', 'target_reference_path', 'teammate_reference_path')


def assemble(data_root, split, episodes, output, dry_run):
    if not episodes:
        raise ValueError('select at least one episode')
    selected = select_keys(split, episodes)
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    load_tasks(data_root, selected)
    wanted = set(selected)
    with (ROOT / 'dataset/tasks.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [r for r in reader if (r['split'], r['episode_key']) in wanted]
    if len(rows) != len(selected):
        raise ValueError('task index and split selections disagree')
    if dry_run:
        return len(rows)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for row in rows:
            for column in ASSETS:
                relative = Path(row[column])
                dest = (output / relative).resolve()
                if not dest.is_relative_to(output):
                    raise ValueError('task index path escapes the output directory')
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(data_root / relative, dest)
        load_tasks(output, selected)
        with (output / 'tasks.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    except BaseException:
        shutil.rmtree(output)
        raise
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', choices=SPLITS, required=True)
    parser.add_argument('--episode', action='append', required=True, metavar='TOWN/EPISODE')
    parser.add_argument('--output', type=Path, required=True, help='new data root')
    parser.add_argument('--dry-run', action='store_true', help='verify the selection without writing')
    args = parser.parse_args()
    try:
        count = assemble(args.data_root, args.split, args.episode, args.output, args.dry_run)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'{exc}\n')
    print(f'{count} released task(s) verified' + ('' if args.dry_run else f' and copied to {args.output}'))


if __name__ == '__main__':
    main()
