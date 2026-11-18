"""Single-volume and complete checksum verification using small local fixtures."""
import hashlib

import pytest

from dataset.verify import VOLUMES, verify


def test_verification_supports_single_and_full_downloads(tmp_path):
    checksums = tmp_path / 'SHA256SUMS'
    lines = []
    for name in VOLUMES:
        path = tmp_path / f'{name}.tar.gz'
        payload = name.encode('ascii')
        path.write_bytes(payload)
        lines.append(f'{hashlib.sha256(payload).hexdigest()}  {path.name}\n')
    checksums.write_text(''.join(lines), encoding='ascii')
    verify(tmp_path, checksums, 'all')

    name = 'taskset-train1100'
    (tmp_path / f'{name}.tar.gz').write_bytes(b'truncated fixture')
    verify(tmp_path, checksums, 'taskset-test300')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        verify(tmp_path, checksums, 'all')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        verify(tmp_path, checksums, name)


def test_unreleased_volume_has_no_verifiable_checksum(tmp_path):
    checksums = tmp_path / 'SHA256SUMS'
    checksums.write_text('a' * 64 + '  taskset-test300.tar.gz\n', encoding='ascii')
    with pytest.raises(ValueError, match='no released checksum'):
        verify(tmp_path, checksums, 'certification')
