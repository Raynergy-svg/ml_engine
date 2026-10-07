#!/usr/bin/env python3
"""Restore the single approved Linux artifact, with exact provenance checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
from urllib.parse import urlsplit
import zipfile

ARTIFACT_ID = 11454954090
RUN_ID = 37553124116
AXIOM_SHA = '9e996c8bd3e0a37fafed2793be1d1165d91209b3'
ARCHIVE_SHA256 = 'fad40bc654f5965a8bef2317036f1b9a20504138f30e2a516afcfa90414ec721'
MANIFEST_SHA256 = '6eaefb9f2c089663e5d59aecb7ae91cd4685b7459588129e04a75bc7f8134270'
API = 'https://api.github.com/repos/Raynergy-svg/ml_engine/actions/artifacts/'


class ArtifactRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != 'https' or not target.hostname.endswith('.blob.core.windows.net'):
            raise ValueError('Unapproved artifact redirect')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        redirected.remove_header('Authorization')
        return redirected


def main():
    root = Path('nautilus')
    manifest = root/'.axiom2-nautilus-build-manifest.json'
    expected_manifest = manifest.read_bytes()
    if hashlib.sha256(expected_manifest).hexdigest() != MANIFEST_SHA256:
        raise SystemExit('Current source/toolchain manifest differs from approved Linux artifact')
    request = urllib.request.Request(API+str(ARTIFACT_ID), headers={
        'Authorization': 'Bearer '+os.environ['GH_TOKEN'],
        'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    if (metadata['expired'] or metadata['digest'] != 'sha256:'+ARCHIVE_SHA256
            or metadata['workflow_run']['id'] != RUN_ID
            or metadata['workflow_run']['head_sha'] != AXIOM_SHA
            or metadata['name'] != 'nautilus-build-'+MANIFEST_SHA256):
        raise SystemExit('Artifact provenance mismatch')
    request = urllib.request.Request(API+str(ARTIFACT_ID)+'/zip', headers={
        'Authorization': 'Bearer '+os.environ['GH_TOKEN']})
    archive = Path(os.environ['RUNNER_TEMP'])/'approved-nautilus.zip'
    opener = urllib.request.build_opener(ArtifactRedirect())
    with opener.open(request, timeout=60) as response, archive.open('wb') as target:
        shutil.copyfileobj(response, target)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit('Artifact ZIP digest mismatch')
    if os.environ.get('EXPORT_APPROVED_ARTIFACT') == 'true':
        blob = archive.read_bytes()
        size = 24*1024*1024
        for number, start in enumerate(range(0, len(blob), size), 1):
            (archive.parent/f'approved-nautilus.part{number}').write_bytes(blob[start:start+size])
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if any(Path(name).is_absolute() or '..' in Path(name).parts for name in names):
            raise SystemExit('Unsafe archive member')
        archived_manifest_present = '.axiom2-nautilus-build-manifest.json' in names
        if archived_manifest_present and bundle.read('.axiom2-nautilus-build-manifest.json') != expected_manifest:
            raise SystemExit('Archived manifest differs from source/toolchain manifest')
        bundle.extractall(root)
    interpreter = root/'python/.venv/bin/python'
    archived_environment_present = interpreter.is_file()
    if archived_environment_present:
        interpreter.chmod(0o755)
    receipt = {'artifact_id': ARTIFACT_ID, 'run_id': RUN_ID, 'axiom_sha': AXIOM_SHA,
        'archive_sha256': ARCHIVE_SHA256, 'manifest_sha256': MANIFEST_SHA256,
        'manifest': json.loads(expected_manifest), 'member_count': len(names),
        'archived_manifest_present': archived_manifest_present,
        'archived_environment_present': archived_environment_present,
        'manifest_verification': 'reconstructed source/toolchain bytes equal approved artifact manifest key',
        'extension_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in names if name.endswith('.so') and '_libnautilus' in name}}
    (Path(os.environ['RUNNER_TEMP'])/'nautilus-reuse-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
