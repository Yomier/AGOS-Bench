"""Evaluation method registry. Cooperative-search arms are selected by configuration."""
from importlib import import_module

_METHODS = {
    'cooperative_search': ('agos.methods.cooperative_search.method', 'CooperativeSearchMethod'),
    'menu_oracle': ('agos.methods.menu_oracle.method', 'MenuOracleMethod'),
}
PRIMARY_METHODS = tuple(_METHODS)


def resolve_method_id(agent_mode):
    return (agent_mode, None) if agent_mode in _METHODS else (None, None)


def create(agent_mode, config=None, services=None):
    if agent_mode not in _METHODS:
        raise ValueError(f'Unknown evaluation method: {agent_mode!r}')
    module, cls = _METHODS[agent_mode]
    return getattr(import_module(module), cls)(config=config, services=services, profile=None)
