import hashlib
from pathlib import Path

import pytest
from scripts.axiom2_verify_nautilus_binary import APPROVED_EXTENSION_SHA256, verify_digest


def test_binary_bytes_must_match_expected_digest(tmp_path):
    extension = tmp_path/'extension.so'
    extension.write_bytes(b'changed native binary')
    with pytest.raises(ValueError, match='binary digest mismatch'):
        verify_digest(extension, APPROVED_EXTENSION_SHA256)


def test_binary_digest_verification_returns_exact_observed_hash(tmp_path):
    extension = tmp_path/'extension.so'
    extension.write_bytes(b'fixture native bytes')
    expected = hashlib.sha256(extension.read_bytes()).hexdigest()
    assert verify_digest(extension, expected) == expected
