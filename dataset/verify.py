"""Verify selected AGOS volume bytes against the release checksum file."""

import argparse
import hashlib
import re
import sys
from pathlib import Path

VOLUMES = (
    "taskset-train1100", "taskset-validation275", "taskset-test300",
    "taskset-seentest450", "certification", "trajectories-train1100",
)


def selected(volume, checksums):
    if volume != "all":
        if volume not in checksums:
            raise ValueError(f"{volume} has no released checksum")
        return (volume,)
    missing = set(VOLUMES) - checksums.keys()
    if missing == {"certification"}:
        print("certification is pending original replay evidence. Verifying the five available volumes.", file=sys.stderr)
    elif missing:
        raise ValueError("Incomplete release checksum file. Missing " + ", ".join(sorted(missing)))
    return tuple(name for name in VOLUMES if name in checksums)


def read_checksums(path):
    entries = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  ([a-z0-9-]+)\.tar\.gz", line)
        if not match or match[2] not in VOLUMES or match[2] in entries:
            raise ValueError(f"Invalid or duplicate checksum entry in {path.name}")
        entries[match[2]] = match[1]
    if not entries:
        raise ValueError("Empty checksum file")
    return entries


def verify(directory, checksums, volume):
    expected = read_checksums(checksums)
    for name in selected(volume, expected):
        path = directory / (name + ".tar.gz")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected[name]:
            raise ValueError(f"Checksum mismatch for {path.name}")
        print(f"OK  {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data"))
    parser.add_argument("--volume", choices=(*VOLUMES, "all"), default="all")
    parser.add_argument("--checksums", type=Path, default=Path(__file__).with_name("SHA256SUMS"))
    args = parser.parse_args()
    try:
        verify(args.directory, args.checksums, args.volume)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
