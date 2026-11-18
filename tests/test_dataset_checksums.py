"""Single-volume and complete checksum verification using small local fixtures."""
import hashlib
import sys
from types import SimpleNamespace

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


@pytest.mark.parametrize('source,package', [('hf', 'huggingface_hub'), ('modelscope', 'modelscope')])
def test_download_checks_selected_volume_against_local_manifest(tmp_path, monkeypatch, source, package):
    from dataset import download

    name = 'taskset-test300.tar.gz'
    content = b'local test archive'
    (tmp_path / 'SHA256SUMS').write_text(f'{hashlib.sha256(content).hexdigest()}  {name}\n', encoding='ascii')
    delivery = tmp_path / 'downloads'
    delivery.mkdir()

    def snapshot_download(**request):
        assert request['repo_type'] == 'dataset' and request['revision'] == 'release-1'
        assert request['allow_patterns'] == [name, 'SHA256SUMS']
        (delivery / name).write_bytes(content)
        (delivery / 'SHA256SUMS').write_text('untrusted remote metadata', encoding='ascii')

    monkeypatch.setattr(download, '__file__', str(tmp_path / 'download.py'))
    monkeypatch.setitem(sys.modules, package, SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr(sys, 'argv', ['download', '--volume', 'taskset-test300', '--source', source,
                                     '--revision', 'release-1', '--directory', str(delivery)])
    assert download.main() == 0
