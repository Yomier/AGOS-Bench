"""Own one engine and its episode child on a local Linux workstation."""
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from benchmark.native_runtime import engine_environment
from benchmark.scoring import write_json

BINARY = Path('CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping')


@dataclass(frozen=True)
class Slot:
    index: int

    @property
    def rpc(self):
        return 15500 + 10 * self.index

    @property
    def airsim(self):
        return 31050 + 10 * self.index

    @property
    def ports(self):
        return (self.rpc, self.rpc + 1, self.rpc + 2, self.airsim, 18000 + 10 * self.index)


def prepare_engine(slot, work, sim_root, town):
    work = work.resolve()
    home = work / 'home'
    config = work / 'xdg_config'
    runtime = work / 'xdg_runtime'
    airsim = home / 'Documents/AirSim'
    for directory in (airsim, config, runtime):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    settings = airsim / 'settings.json'
    write_json(settings, dict(ApiServerPort=slot.airsim, RpcEnabled=True, SettingsVersion=1.2,
                              LogMessagesVisible=False, RecordUIVisible=False))
    assets = {'Town01': 'Town01', 'Town02': 'Town02', 'Town03': 'Town03_Opt', 'Town04': 'Town04_Opt',
              'Town05': 'Town05_Opt', 'Town10HD': 'Town10HD'}
    cmd = [str(sim_root / BINARY), 'CarlaUE4', assets[town], f'-carla-rpc-port={slot.rpc}',
           f'-carla-streaming-port={slot.rpc+1}', '-quality-level=Epic', '-RenderOffScreen', '-nosound',
           '-unattended', '-NoVSync', '-norhithread', f'-settings={settings}']
    env = engine_environment(dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(config), XDG_RUNTIME_DIR=str(runtime)))
    return cmd, env


@contextmanager
def own_slot(slot):
    # Abstract sockets are machine-wide, nonpersistent locks on Linux. No PID-file recovery.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as lock:
        lock.bind('\0agos-carlaair-slot-rpc-' + str(slot.rpc))
        yield


def ports_free(slot):
    families = [(socket.AF_INET, '0.0.0.0')]
    if socket.has_ipv6:
        families.append((socket.AF_INET6, '::'))
    for port in slot.ports:
        for family, host in families:
            try:
                with socket.socket(family) as probe:
                    if os.name == 'nt':
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    if family == socket.AF_INET6:
                        probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                    probe.bind((host, port))
            except OSError as exc:
                if family == socket.AF_INET6 and exc.errno in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                    continue
                return False
    return True


def stop_process(proc, engine=False):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(45 if engine else 5)
    except subprocess.TimeoutExpired:
        if engine:
            raise RuntimeError(f'owned engine PID {proc.pid} did not exit after SIGTERM; inspect its log')
        proc.kill()
        proc.wait(10)


def wait_child(proc, engine, deadline, stop):
    while proc.poll() is None:
        if stop.wait(.5):
            raise InterruptedError('evaluation interrupted')
        if engine.poll() is not None:
            raise RuntimeError(f'engine exited with code {engine.returncode}')
        if time.monotonic() >= deadline:
            raise TimeoutError('wall-time limit reached')
    if proc.returncode:
        raise RuntimeError(f'child exited with code {proc.returncode}')


def run_owned(slot, work, sim_root, town, episode_cmd, attempt, timeout, stop, before_start):
    """One fresh engine per attempt. Only handles created by this invocation are signalled."""
    cmd, env = prepare_engine(slot, work, sim_root, town)
    state = dict(slot=slot.index, ports=slot.ports, town=town, attempt=str(attempt), monitor_pid=os.getpid())
    engine = child = None
    def record(stage, **extra):
        state.update(stage=stage, updated_at=time.time(), **extra)
        write_json(work / 'status.json', state)
    with (attempt / 'engine.log').open('wb') as engine_log, (attempt / 'worker.log').open('wb') as log:
        try:
            record('waiting_for_ports')
            deadline = time.monotonic() + 60
            while not ports_free(slot):
                if stop.wait(1) or time.monotonic() >= deadline:
                    raise RuntimeError(f'ports unavailable: {slot.ports}')
            record('waiting_for_stagger')
            before_start()
            if stop.is_set():
                raise InterruptedError('shutdown requested')
            engine = subprocess.Popen(cmd, cwd=sim_root, env=env, start_new_session=True,
                                      stdout=engine_log, stderr=subprocess.STDOUT)
            record('starting', engine_pid=engine.pid)
            probe_cmd = [sys.executable, '-m', 'carlaair.engine.probe', town, str(slot.rpc), str(slot.airsim)]
            child = subprocess.Popen(probe_cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record('checking_camera_and_rpc', child_pid=child.pid)
            wait_child(child, engine, time.monotonic() + 180, stop)
            child = subprocess.Popen(episode_cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record('evaluating', child_pid=child.pid)
            wait_child(child, engine, time.monotonic() + timeout, stop)
        finally:
            try:
                stop_process(child)
            finally:
                record('stopping')
                try:
                    stop_process(engine, engine=True)
                except Exception as exc:
                    stop.set()
                    record('stop_failed', error=str(exc))
                    raise
                record('stopped')
