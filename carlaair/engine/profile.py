"""Bind an existing Linux native guard, CARLA client library and engine by content hash."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys

from benchmark.native_runtime import load_profile, require_client_guard, sha256

PAPER_SHA = {
    'engine': '84ab7d701fe458ed8333210722a628a1204f3d6b1c9c661b7bde61f49e5f0d15',
    'guard': 'ef08bc2d39b203efad1fd485adaa7ec327f0eee20e358a454ea727a7b415a318',
    'libcarla': '4795659108e845fa1e8c5f2af0749477813ea2a895f00470402c431f0e66508d',
}


def identity(path):
    if not path:
        return 'upstream'
    profile = load_profile(path)
    return 'v13-matched' if all(profile[k]['sha256'] == v for k, v in PAPER_SHA.items()) else 'custom-guarded'


def admit(path, command):
    path = str(Path(path).resolve())
    profile = load_profile(path)
    digest = sha256(path)
    if (os.environ.get('AGOS_NATIVE_RUNTIME_PROFILE'), os.environ.get('AGOS_NATIVE_RUNTIME_PROFILE_SHA256')) != (path, digest):
        env = dict(os.environ, AGOS_NATIVE_RUNTIME_PROFILE=path, AGOS_NATIVE_RUNTIME_PROFILE_SHA256=digest,
                   LD_PRELOAD=profile['guard']['path'])
        os.execve(sys.executable, [sys.executable, '-m', 'benchmark.run_parallel_eval', *command], env)
    import carla  # Load the exact library whose mappings are inspected below.
    require_client_guard()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for role in PAPER_SHA:
        p.add_argument('--' + role, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    profile = dict(schema='agos_native_runtime_v1', system=platform.system(), machine=platform.machine(),
                   engine_flags=['-norhithread'])
    for role in PAPER_SHA:
        path = getattr(a, role).resolve(strict=True)
        profile[role] = dict(path=str(path), sha256=sha256(path))
    if profile['system'] != 'Linux' or profile['machine'] != 'x86_64':
        p.error('the native overlay targets Linux x86_64')
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(profile, stream, indent=2)
    print(identity(a.output))


if __name__ == '__main__':
    main()
