#!/usr/bin/env python3
"""Hash the exact approved Linux extension on artifact and cache reuse paths."""
import argparse
import hashlib
import json
from pathlib import Path

APPROVED_EXTENSION_SHA256 = '09832798c20e663d7917a72d427308960925f34594e0345f69351fc2b08df6fb'


def verify_digest(path: Path, expected: str) -> str:
    with path.open('rb') as stream:
        observed = hashlib.file_digest(stream, 'sha256').hexdigest()
    if observed != expected:
        raise ValueError('Nautilus binary digest mismatch')
    return observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package_directory', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    extensions = list(args.package_directory.glob('_libnautilus*.so'))
    if len(extensions) != 1 or extensions[0].is_symlink() or not extensions[0].is_file():
        raise ValueError('exactly one regular approved Linux extension required')
    digest = verify_digest(extensions[0], APPROVED_EXTENSION_SHA256)
    report = {'extension_file': str(extensions[0]), 'extension_sha256': digest,
        'upstream_revision': '4f021bafc2e99c5490cee204b0fc2bd2c83baab4',
        'original_artifact_id': 11454954090, 'original_run_id': 37553124116}
    args.receipt.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
