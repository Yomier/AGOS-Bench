"""Download selected AGOS archives from Hugging Face and verify their hashes."""

import argparse
import sys
from pathlib import Path

if __package__:
    from .verify import VOLUMES, read_checksums, selected, verify
else:
    from verify import VOLUMES, read_checksums, selected, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", choices=(*VOLUMES, "all"), default="all")
    parser.add_argument("--directory", type=Path, default=Path("data"))
    parser.add_argument("--repo-id", default="Yomier/AGOS-Dataset")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    checksums = Path(__file__).with_name("SHA256SUMS")
    try:
        released = read_checksums(checksums)
        names = selected(args.volume, released)
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo_id, repo_type="dataset", revision=args.revision,
                          local_dir=args.directory,
                          allow_patterns=[f"{name}.tar.gz" for name in names] + ["SHA256SUMS"])
        verify(args.directory, checksums, args.volume)
    except ImportError:
        parser.exit(1, "Install huggingface_hub to download archives.\n")
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
