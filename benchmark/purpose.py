"""Authority of the actual method instantiated for a released run."""
from agos.methods.cooperative_search.contracts import ARMS


def admission(config, arm, split, collect=False):
    coop = config.get('cooperative_search') or {}
    expert = coop.get('expert') or {}
    enabled = expert.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError('cooperative_search.expert.enabled must be a boolean')
    if collect and split != 'train1100':
        raise ValueError('expert collection is restricted to Train1100')
    if enabled:
        if not (collect and arm == coop.get('arm') == 'agos_b' and expert.get('mode', 'drive') == 'drive'):
            raise ValueError('privileged expert requires an explicit agos_b Train1100 drive collection')
        from training.expert import EventExpert
        EventExpert(expert)
        return 'expert_collect'
    if collect:
        raise ValueError('expert collection requires an enabled drive expert')
    if arm == 'menu_oracle':
        return 'oracle_diagnostic'
    if arm not in ARMS or coop.get('arm') != arm:
        raise ValueError('method arm differs from the resolved configuration')
    return 'evaluation'


def manifest_purpose(manifest):
    purpose, arm = manifest.get('purpose'), manifest.get('arm')
    if purpose == 'expert_collect':
        if arm != 'agos_b' or any(not task['key'].startswith('train1100/') for task in manifest['tasks']):
            raise ValueError('expert_collect requires agos_b and Train1100 tasks')
    elif purpose == 'oracle_diagnostic':
        if arm != 'menu_oracle':
            raise ValueError('oracle_diagnostic requires menu_oracle')
    elif purpose != 'evaluation' or arm not in ARMS:
        raise ValueError('missing or incompatible run purpose and arm')
    mode = 'menu_oracle' if arm == 'menu_oracle' else 'cooperative_search'
    if manifest.get('mode') != mode:
        raise ValueError('run purpose disagrees with method mode')
    return purpose


def verify_calls(row, purpose, *, include_http=True):
    if purpose == 'evaluation' and (row.get('oracle_candidate_injected') or row.get('diagnostic_mode') not in (None, '', 'none')):
        raise ValueError('privileged diagnostic result cannot be scored as evaluation')
    for suffix in (('vlm_calls', 'vlm_http_attempts') if include_http else ('vlm_calls',)):
        total = row.get(suffix)
        roles = [row.get(role + '_' + suffix) for role in ('uav', 'ugv')]
        if any(type(n) is not int or n < 0 for n in [total, *roles]) or sum(roles) != total:
            raise ValueError(f'inconsistent or missing {suffix} accounting')
        if purpose != 'evaluation' and total:
            raise ValueError(f'{purpose} requires zero {suffix}')
        if purpose == 'evaluation' and row.get('success') and not total:
            raise ValueError(f'successful model evaluation requires positive {suffix}')
