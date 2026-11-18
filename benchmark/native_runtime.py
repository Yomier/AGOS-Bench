"""Build-specific admission for the tested Linux CARLA-Air native mitigation.

The preload must exist before Python starts. This is not a portable replacement
for rebuilding CARLA-Air/UE from matching sources.
"""
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path

STOP_SYMBOL = '_ZN5carla9streaming6detail3tcp6Client4StopEv'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_profile(path):
    profile = json.loads(Path(path).read_text(encoding='utf-8'))
    if profile.get('schema') != 'agos_native_runtime_v1':
        raise RuntimeError('unsupported native runtime profile schema')
    if (platform.system(), platform.machine()) != (profile.get('system'), profile.get('machine')):
        raise RuntimeError('native runtime platform mismatch; this overlay is build specific')
    if profile.get('engine_flags') != ['-norhithread']:
        raise RuntimeError('unverified native runtime engine flags')
    for role in ('guard', 'libcarla', 'engine'):
        item = profile[role]
        path = Path(item['path'])
        if not path.is_absolute() or not path.is_file() or sha256(path) != item['sha256']:
            raise RuntimeError(f'{role} binary SHA mismatch or unavailable file')
    return profile


def mapped_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            paths.add(str(Path(fields[5]).resolve()))
    return paths


def symbol_provider(name):
    class DlInfo(ctypes.Structure):
        _fields_ = [('name', ctypes.c_char_p), ('base', ctypes.c_void_p),
                    ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]
    process = ctypes.CDLL(None)
    info = DlInfo()
    dladdr = process.dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    if not dladdr(ctypes.cast(getattr(process, name), ctypes.c_void_p), ctypes.byref(info)):
        raise RuntimeError('cannot resolve active native Stop symbol provider')
    return str(Path(os.fsdecode(info.name)).resolve())


def require_client_guard():
    path = os.environ.get('AGOS_NATIVE_RUNTIME_PROFILE')
    if not path:
        return None  # Unconfigured historical installations retain their identity.
    if os.environ.get('AGOS_NATIVE_RUNTIME_PROFILE_SHA256') != sha256(path):
        raise RuntimeError('native runtime profile identity changed or not bound by launcher')
    profile = load_profile(path)
    mapped = mapped_libraries()
    for role in ('guard', 'libcarla'):
        if str(Path(profile[role]['path']).resolve()) not in mapped:
            raise RuntimeError(f'{role} not mapped; launch Python through the benchmark --native-profile option')
    expected_client = str(Path(profile['libcarla']['path']).resolve())
    if any(Path(p).name.startswith('libcarla') and p != expected_client for p in mapped):
        raise RuntimeError('a second libcarla binary is mapped; refusing ambiguous ABI')
    if symbol_provider(STOP_SYMBOL) != str(Path(profile['guard']['path']).resolve()):
        raise RuntimeError('native Stop symbol provider is not the admitted guard')
    native = ctypes.CDLL(None)
    for symbol in ('agos_stop_guard_calls', 'agos_stop_guard_completed'):
        getattr(native, symbol).restype = ctypes.c_ulonglong
    return profile


PAPER_PROFILE_SHA256 = '0141730ddce3ec7fcd4a52402b346b0b6ab9ba0cc230b6176eb81ef75a278c8c'


def paper_profile_relocation(expected_digest, observed_digest):
    """Approve a path-only relocation of the paper profile after in-process native admission."""
    from carlaair.engine.profile import PAPER_SHA
    if (expected_digest != PAPER_PROFILE_SHA256 or not observed_digest
            or observed_digest != os.environ.get('AGOS_NATIVE_RUNTIME_PROFILE_SHA256')):
        return False
    profile = require_client_guard()
    return bool(profile and all(profile[k]['sha256'] == v for k, v in PAPER_SHA.items()))


def engine_runtime_args(binary):
    profile = require_client_guard()
    if profile is None:
        return []
    if Path(binary).resolve() != Path(profile['engine']['path']).resolve():
        raise RuntimeError('engine binary differs from admitted native runtime')
    return list(profile['engine_flags'])


def engine_environment(environment):
    env = dict(environment)
    env.pop('LD_PRELOAD', None)
    return env
