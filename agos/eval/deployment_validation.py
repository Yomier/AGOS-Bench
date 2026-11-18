"""Evaluation-private, fixed numerical contract; immutable fields stay exact."""
import copy
import math
from decimal import Decimal, localcontext

LAYERED_PROTOCOL = 'AGOS_DEPLOYMENT_LAYERED_V2'
LAYERED_ORACLE_VERSION = 'semantic_meters_confirm_layered_v2'
DEPLOYMENT_POLICY = {'protocol': LAYERED_PROTOCOL, 'position_abs_m': 1e-4,
                     'angle_abs_deg': 1e-5}
TASK_SCALE_PROTOCOL = 'AGOS_DEPLOYMENT_LAYERED_V3'
TASK_SCALE_ORACLE_VERSION = 'semantic_meters_confirm_layered_v3'
TASK_SCALE_POLICY = {'protocol': TASK_SCALE_PROTOCOL, 'position_abs_m': 1e-4,
                     'angle_abs_deg': 1e-3}
LAYERED_PROTOCOLS = (LAYERED_PROTOCOL, TASK_SCALE_PROTOCOL)
LAYERED_ORACLE_VERSIONS = (LAYERED_ORACLE_VERSION, TASK_SCALE_ORACLE_VERSION)


def deployment_policy(protocol):
    # V2 remains immutable. V3 explicitly accounts for the shipping engine's
    # 1e-4 degree near-equal rotation gate; actual Oracle QA is still mandatory.
    if protocol == LAYERED_PROTOCOL:
        return copy.deepcopy(DEPLOYMENT_POLICY)
    if protocol == TASK_SCALE_PROTOCOL:
        return copy.deepcopy(TASK_SCALE_POLICY)
    from agos.eval.navigation_oracle import NavigationOracleError
    raise NavigationOracleError('unsupported deployment validation protocol')


def _decimal_delta(old, new, angular):
    # The certificate stores decimal readbacks. Binary subtraction, especially
    # adding/subtracting 180, can move an exact closed boundary outside policy.
    # Use their serialized values; never add an epsilon or round away excess.
    with localcontext() as ctx:
        ctx.prec = 400  # Also permits modulo on any finite float input.
        delta = Decimal(str(new)) - Decimal(str(old))
        if angular:
            delta = (delta + Decimal(180)) % Decimal(360)
            if delta < 0:
                delta += Decimal(360)
            delta -= Decimal(180)
        return delta


def compare_deployments(expected, actual):
    from agos.eval.navigation_oracle import NavigationOracleError, canonical_hash
    policy = deployment_policy((expected.get('deployment_validation') or {}).get('protocol'))
    if (expected.get('deployment_validation') != policy or
            actual.get('deployment_validation') != policy):
        raise NavigationOracleError('unsupported or changed deployment validation policy')
    left, right = copy.deepcopy(expected), copy.deepcopy(actual)
    changes = []
    actors = []
    for key in ('target_deployment', 'ugv_deployment'):
        if key not in left or key not in right:
            raise NavigationOracleError('missing authoritative deployment: '+key)
        actors.append((key, left[key], right[key]))
    a, b = left.get('distractor_deployment'), right.get('distractor_deployment')
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        raise NavigationOracleError('changed distractor deployment count')
    actors.extend((f'distractor_deployment[{i}]', x, y) for i,(x,y) in enumerate(zip(a,b)))
    for path, old, new in actors:
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise NavigationOracleError('invalid deployment actor: '+path)
        for field, limit in (('location', policy['position_abs_m']),
                             ('rotation', policy['angle_abs_deg'])):
            x, y = old.get(field), new.get(field)
            if not isinstance(x, list) or not isinstance(y, list) or len(x) != 3 or len(y) != 3:
                raise NavigationOracleError('invalid deployment vector: '+path+'.'+field)
            deltas = []
            for i,(v,w) in enumerate(zip(x,y)):
                if any(isinstance(n, bool) or not isinstance(n, (int,float)) or not math.isfinite(n) for n in (v,w)):
                    raise NavigationOracleError('nonfinite deployment vector: '+path+'.'+field)
                delta = _decimal_delta(v,w,field == 'rotation')
                deltas.append(delta)
                if abs(delta) > Decimal(str(limit)):
                    raise NavigationOracleError(f'deployment numerical limit exceeded: '
                        f'{path}.{field}[{i}] expected={v!r} actual={w!r} '
                        f'delta={delta} limit={limit}')
                if v != w:
                    changes.append({'path': f'{path}.{field}[{i}]', 'expected': v,
                                    'actual': w, 'delta': float(delta), 'limit': limit})
            if field == 'location' and sum(d*d for d in deltas) > Decimal(str(limit))**2:
                raise NavigationOracleError('deployment position norm limit exceeded: '+path)
            # Only these six numeric readbacks are removed from identity comparison.
            del old[field], new[field]
    # The public namespace relocation changes a fixed set of source byte hashes.
    # Record both identities and require the exact reviewed pair for each changed file.
    from agos.eval.source_compatibility import FROZEN_TO_PUBLIC
    relocated = {}
    for field, (frozen, public) in FROZEN_TO_PUBLIC.items():
        if left.get(field) == frozen and right.get(field) == public:
            relocated[field] = {'frozen_sha256': frozen, 'public_sha256': public}
            left[field] = public
    profile_field = 'native_runtime_profile_sha256'
    native_comparison = 'identical'
    if left.get(profile_field) != right.get(profile_field):
        import os
        from benchmark.native_runtime import PAPER_PROFILE_SHA256, paper_profile_relocation
        expected_profile, observed_profile = left.get(profile_field), right.get(profile_field)
        if paper_profile_relocation(expected_profile, observed_profile):
            native_comparison = 'verified_binary_relocation'
        elif (expected_profile == PAPER_PROFILE_SHA256 and observed_profile is None
              and os.environ.get('AGOS_DEPLOYMENT_IDENTITY') == 'upstream'
              and not any(os.environ.get(key) for key in ('AGOS_NATIVE_RUNTIME_PROFILE',
                          'AGOS_NATIVE_RUNTIME_PROFILE_SHA256', 'LD_PRELOAD'))):
            native_comparison = 'independent_upstream'
        else:
            raise NavigationOracleError('native profile identity mismatch')
        left[profile_field] = observed_profile
    try:
        same = canonical_hash(left) == canonical_hash(right)
    except (ValueError, TypeError) as error:
        raise NavigationOracleError('invalid immutable deployment identity') from error
    if not same:
        raise NavigationOracleError('immutable deployment/config/code identity mismatch')
    return {'policy': policy, 'changes': changes, 'source_relocations': relocated,
            'native_comparison': native_comparison, 'expected_profile_sha256': expected.get(profile_field),
            'actual_profile_sha256': actual.get(profile_field),
            'expected_sha256': canonical_hash(expected), 'actual_sha256': canonical_hash(actual)}
