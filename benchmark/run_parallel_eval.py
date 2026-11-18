"""Run the released benchmark on local CarlaAir slots with a local or API model endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from carlaair.engine.profile import admit, identity
from carlaair.engine.slot import BINARY, Slot, run_owned
from .scoring import aggregate, classify, next_attempt, read_terminal, sha256, verify_binding, vision_counts, write_json
from .tasks import ROOT, SPLITS, load_tasks, select_keys


ARMS = ('agos_a', 'agos_b', 'agos_c', 'baseline_a', 'menu_oracle')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=os.getenv('AGOS_DATA_ROOT'), help='root of extracted task volumes')
    p.add_argument('--split', choices=[*SPLITS, 'all'], default='test300')
    p.add_argument('--episode', action='append', default=[], metavar='TOWN/EPISODE', help='select one address; repeatable')
    p.add_argument('--arm', choices=ARMS, default='agos_b')
    p.add_argument('--config', type=Path, help='agent YAML override')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--sim-root', type=Path, default=os.getenv('AGOS_SIM_ROOT'))
    p.add_argument('--profile', choices=['local', 'api'], default='local', help='local = 6 slots, API = 10 slots')
    p.add_argument('--native-profile', type=Path, help='optional build-specific CARLA native guard profile')
    p.add_argument('--workers', type=int, help='override within the selected profile')
    p.add_argument('--stagger-seconds', type=float, default=30, help='minimum interval between cold starts')
    p.add_argument('--episode-timeout', type=float, default=9000, help='episode child wall deadline (seconds)')
    p.add_argument('--max-attempts', type=int, default=2, help='pending attempts per task this invocation')
    p.add_argument('--resume', action='store_true', help='lock existing valid results; retry pending tasks')
    p.add_argument('--dry-run', action='store_true', help='verify task plans, reference PNGs, config and map registry offline')
    return p


def worker_count(profile, requested):
    maximum = {'local': 6, 'api': 10}[profile]
    if requested is not None and not 1 <= requested <= maximum:
        raise ValueError(f'{profile} supports 1..{maximum} workers')
    return requested or maximum


def verify_config(path, arm, towns):
    from agos.lite_core import load_config
    from agos.roadnet.registry import RoadGraphRegistry
    config = load_config(str(path))
    mode = 'menu_oracle' if arm == 'menu_oracle' else 'cooperative_search'
    if mode == 'cooperative_search' and config['cooperative_search']['arm'] != arm:
        raise ValueError('--arm must match cooperative_search.arm in the configuration')
    configured_mode = (config.get('paper_experiment') or {}).get('agent_mode')
    if configured_mode and configured_mode != mode:
        raise ValueError(f'{path}: expected agent_mode {mode}')
    registry = RoadGraphRegistry.load(config['road_graph']['registry'])
    for town in towns:
        registry.resolve(town)
    return mode, config


def check_services(mode, config, profile):
    if mode == 'menu_oracle':
        return
    for name in ('gsam_endpoint', 'ugv_endpoint'):
        parts = urlsplit(config['cooperative_search'][name])
        health = urlunsplit((parts.scheme, parts.netloc, '/health', '', ''))
        with urlopen(health, timeout=10) as response:
            if json.load(response).get('ok') is not True:
                raise ValueError(f'perception health failed for {name}')
    vlm = config['vlm']
    if not urlsplit(vlm['api_url']).netloc or not vlm.get('model'):
        raise ValueError('model identifier and absolute chat-completion URL are required')
    if profile == 'local':
        from urllib.request import Request
        parts = urlsplit(vlm['api_url'])
        base, _, _ = parts.path.rpartition('/chat/completions')
        request = Request(urlunsplit((parts.scheme, parts.netloc, base + '/models', '', '')))
        token = os.getenv(vlm.get('api_key_env') or 'VLLM_API_KEY')
        if token:
            request.add_header('Authorization', 'Bearer ' + token)
        with urlopen(request, timeout=10) as response:
            served = {row['id'] for row in json.load(response)['data']}
        expected = {vlm.get('uav_model') or vlm['model'], vlm.get('ugv_model') or vlm['model']}
        if expected - served:
            raise ValueError(f'local vLLM is missing model aliases: {sorted(expected - served)}')


def collect_attempt(folder, attempt, task, mode, manifest, error=None):
    result_file = attempt / (task['episode'] + '_result.json')
    row = dict(attempt=attempt.name, outcome='pending', result=None, vision_errors=0, vision_ok=0)
    if error:
        row['worker_error'] = str(error)
    if result_file.is_file():
        try:
            result = json.loads(result_file.read_text(encoding='utf-8'))
            verify_binding(result, task, manifest)
            errors, ok = vision_counts(attempt / 'ag_os_trace.jsonl')
            row.update(outcome=classify(result, errors, ok), vision_errors=errors, vision_ok=ok,
                       result=result_file.name, result_sha256=sha256(result_file))
            if mode == 'menu_oracle' and (result.get('vlm_calls') != 0 or result.get('vlm_http_attempts') != 0):
                raise ValueError('menu oracle reported nonzero or missing model call count')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row.update(outcome='pending', evidence_error=str(exc))
    write_json(attempt / 'score.json', row)
    write_json(folder / 'terminal.json', row)
    return row


def monitor(run_dir, done):
    with (run_dir / 'resources.jsonl').open('a', encoding='utf-8') as stream:
        while not done.is_set():
            sample = {'time': time.time(), 'load': os.getloadavg()}
            try:
                stat = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                       '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                sample['gpu'] = stat.stdout.splitlines() if stat.returncode == 0 else None
            except (OSError, subprocess.TimeoutExpired):
                sample['gpu'] = None
            stream.write(json.dumps(sample) + '\n')
            stream.flush()
            done.wait(15)


def execute(a, workers, tasks, metadata, mode, manifest):
    import fcntl
    root = a.run_dir
    if not a.resume:
        root.mkdir(parents=True, exist_ok=False)
    if not root.is_dir():
        raise ValueError('--resume requires an existing run')
    with (root / '.owner.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / 'manifest.json'
        if a.resume:
            if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
                raise ValueError('resume manifest differs in task, configuration, model or runtime identity')
        else:
            write_json(manifest_path, manifest)
        siblings = root / 'metadata'
        siblings.mkdir(exist_ok=True)
        for group, rows in metadata.items():
            write_json(siblings / (group.replace('/', '-') + '.json'), rows)
        work = queue.Queue()
        outcomes = {}
        for task in tasks:
            folder = root / 'episodes' / task['key']
            row = read_terminal(folder)
            previous = sorted(folder.glob('attempt-[0-9]*'))
            if previous and (not row or row['outcome'] == 'pending' and row['attempt'] != previous[-1].name):
                row = collect_attempt(folder, previous[-1], task, mode, manifest, 'recovering interrupted attempt')
            outcomes[task['key']] = row['outcome'] if row else 'pending'
            if outcomes[task['key']] == 'pending':
                work.put((task, 0))
        lock_state = threading.Lock()
        launch_lock = threading.Lock()
        next_launch = 0.0
        stop, done = threading.Event(), threading.Event()
        def stagger():
            nonlocal next_launch
            with launch_lock:
                if stop.wait(max(0., next_launch - time.monotonic())):
                    raise InterruptedError('shutdown during stagger')
                next_launch = time.monotonic() + a.stagger_seconds

        def worker(index):
            slot = Slot(index)
            while not stop.is_set():
                try:
                    task, count = work.get_nowait()
                except queue.Empty:
                    return
                folder = root / 'episodes' / task['key']
                attempt = next_attempt(folder)
                attempt.mkdir(parents=True)
                job = dict(task=task, output=str(attempt), config=str(a.config), config_sha256=manifest['config_sha256'],
                           arm=manifest['arm'], resolved_config_sha256=manifest['resolved_config_sha256'],
                           rpc=slot.rpc, airsim=slot.airsim, slot=index, mode=mode,
                           siblings=str(siblings / f"{task['split']}-{task['town']}.json"))
                job_path = attempt / 'job.json'
                write_json(job_path, job)
                cmd = [sys.executable, '-m', 'benchmark.episode', str(job_path)]
                error = None
                try:
                    run_owned(slot, root / 'slots' / f'slot-{index:02d}', a.sim_root, task['town'],
                              cmd, attempt, a.episode_timeout, stop, stagger)
                except (OSError, RuntimeError) as exc:
                    error = exc
                row = collect_attempt(folder, attempt, task, mode, manifest, error)
                with lock_state:
                    outcomes[task['key']] = row['outcome']
                    write_json(root / 'summary.json', aggregate(outcomes))
                if row['outcome'] == 'pending' and count + 1 < a.max_attempts:
                    work.put((task, count + 1))
        interrupted = []
        def cancel(signum, frame):
            interrupted.append(signum)
            stop.set()
        original = {s: signal.signal(s, cancel) for s in (signal.SIGINT, signal.SIGTERM)}
        mon = threading.Thread(target=monitor, args=(root, done), daemon=True)
        write_json(root / 'summary.json', aggregate(outcomes))
        mon.start()
        try:
            from carlaair.engine.slot import own_slot
            with ExitStack() as owners, ThreadPoolExecutor(max_workers=workers) as pool:
                count = min(workers, work.qsize())
                for i in range(count):
                    owners.enter_context(own_slot(Slot(i)))
                futures = [pool.submit(worker, i) for i in range(count)]
                for future in futures:
                    future.result()
        finally:
            done.set()
            mon.join()
            for s, previous in original.items():
                signal.signal(s, previous)
        summary = aggregate(outcomes)
        print(json.dumps(summary, ensure_ascii=False))
        if interrupted:
            return 130
        return 3 if summary['pending'] or stop.is_set() else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = parser()
    a = p.parse_args(argv)
    try:
        if a.data_root is None:
            raise ValueError('--data-root or AGOS_DATA_ROOT is required')
        if a.max_attempts < 1 or not 0 <= a.stagger_seconds < float('inf') or not 0 < a.episode_timeout < float('inf'):
            raise ValueError('invalid timeout, stagger interval or attempt count')
        a.run_dir = a.run_dir.resolve()
        a.config = (a.config or ROOT / 'agents/configs' / (a.arm + '.yaml')).resolve(strict=True)
        workers = worker_count(a.profile, a.workers)
        keys = select_keys(a.split, a.episode)
        tasks, metadata = load_tasks(a.data_root.resolve(), keys)
        mode, config = verify_config(a.config, a.arm, {task['town'] for task in tasks})
        digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        manifest = dict(schema=1, arm=a.arm, mode=mode, profile=a.profile, native=identity(a.native_profile),
                        native_profile_sha256=sha256(a.native_profile) if a.native_profile else None,
                        config_sha256=sha256(a.config), resolved_config_sha256=digest,
                        tasks=[{'key': t['key'], 'plan_sha256': t['plan_sha256']} for t in tasks])
        if a.sim_root:
            a.sim_root = a.sim_root.resolve()
            if not (a.sim_root / BINARY).is_file():
                raise ValueError(f'missing engine under {a.sim_root}: {BINARY}')
        if a.dry_run:
            print(json.dumps(dict(tasks=len(tasks), first=tasks[0]['key'], last=tasks[-1]['key'], arm=a.arm,
                                  mode=mode, slots=workers, stagger_seconds=a.stagger_seconds, native=manifest['native'],
                                  engine_path_checked=bool(a.sim_root), config=str(a.config))))
            return 0
        if sys.platform != 'linux' or os.geteuid() == 0 or not a.sim_root:
            raise ValueError('live evaluation requires Linux, a non-root engine owner and --sim-root or AGOS_SIM_ROOT')
        if a.native_profile:
            from benchmark.native_runtime import load_profile
            profile = load_profile(a.native_profile)
            if Path(profile['engine']['path']).resolve() != (a.sim_root / BINARY).resolve():
                raise ValueError('profile engine differs from --sim-root')
            admit(a.native_profile, argv)
        elif any(os.environ.get(k) for k in ('AGOS_NATIVE_RUNTIME_PROFILE', 'AGOS_NATIVE_RUNTIME_PROFILE_SHA256', 'LD_PRELOAD')):
            raise ValueError('use a clean client environment or pass --native-profile explicitly')
        os.environ['AGOS_DEPLOYMENT_IDENTITY'] = manifest['native']
        manifest['engine_sha256'] = sha256(a.sim_root / BINARY)
        import agos.control.carla_interface
        import agos.runtime.runner
        check_services(mode, config, a.profile)
        return execute(a, workers, tasks, metadata, mode, manifest)
    except (OSError, KeyError, ValueError, RuntimeError, ImportError) as exc:
        p.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
