"""Count actual generation POST attempts during one synchronous decision."""
from contextlib import contextmanager
from contextvars import ContextVar

_active_attempts = ContextVar('vlm_http_attempts', default=None)


def record_http_attempt():
    counts = _active_attempts.get()
    if counts is not None:
        counts[0] += 1


@contextmanager
def capture_http_attempts():
    counts = [0]
    token = _active_attempts.set(counts)
    try:
        yield counts
    finally:
        _active_attempts.reset(token)
