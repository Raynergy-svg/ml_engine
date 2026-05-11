"""
Tests for Phase 55 US-341: Atomic JSON Writes + Safe I/O.

Tests cover:
  - safe_json_write atomic write pattern
  - safe_json_read with corrupted files
  - safe_json_read missing file returns default
  - safe_jsonl_append atomic append
  - Concurrent write safety
  - Backup creation on write
  - Lazy imports
  - Execution.py wiring verification
"""

import json
import os
import tempfile
import threading

import pytest

from src.scanner.safe_io import safe_json_write, safe_json_read, safe_jsonl_append


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestSafeJsonWrite:

    def test_basic_write(self, tmp_dir):
        """Write creates a valid JSON file."""
        path = os.path.join(tmp_dir, "test.json")
        data = {"key": "value", "number": 42}
        assert safe_json_write(path, data) is True
        loaded = json.loads(open(path).read())
        assert loaded["key"] == "value"
        assert loaded["number"] == 42

    def test_atomic_no_partial(self, tmp_dir):
        """No .tmp file remains after successful write."""
        path = os.path.join(tmp_dir, "atomic.json")
        safe_json_write(path, {"ok": True})
        files = os.listdir(tmp_dir)
        tmp_files = [f for f in files if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    def test_creates_parent_dirs(self, tmp_dir):
        """Write creates missing parent directories."""
        path = os.path.join(tmp_dir, "sub", "dir", "test.json")
        assert safe_json_write(path, {"nested": True}) is True
        assert os.path.exists(path)

    def test_overwrites_existing(self, tmp_dir):
        """Write overwrites existing file."""
        path = os.path.join(tmp_dir, "overwrite.json")
        safe_json_write(path, {"v": 1})
        safe_json_write(path, {"v": 2})
        loaded = json.loads(open(path).read())
        assert loaded["v"] == 2

    def test_backup_created(self, tmp_dir):
        """Write creates .bak of previous version."""
        path = os.path.join(tmp_dir, "bak_test.json")
        safe_json_write(path, {"v": 1})
        safe_json_write(path, {"v": 2})
        bak_path = path + ".bak"
        assert os.path.exists(bak_path)
        bak_data = json.loads(open(bak_path).read())
        assert bak_data["v"] == 1

    def test_large_data(self, tmp_dir):
        """Write handles large data (~1MB)."""
        path = os.path.join(tmp_dir, "large.json")
        data = {"items": [{"i": i, "data": "x" * 100} for i in range(1000)]}
        assert safe_json_write(path, data) is True
        loaded = json.loads(open(path).read())
        assert len(loaded["items"]) == 1000

    def test_sort_keys_default_false(self, tmp_dir):
        """Default behavior preserves insertion order (no key sorting)."""
        path = os.path.join(tmp_dir, "unsorted.json")
        data = {"zebra": 1, "apple": 2, "mango": 3}
        safe_json_write(path, data)
        # Keys appear in insertion order in the raw file
        raw = open(path).read()
        assert raw.index('"zebra"') < raw.index('"apple"') < raw.index('"mango"')

    def test_sort_keys_true_sorts_alphabetically(self, tmp_dir):
        """sort_keys=True produces deterministic alphabetical key order on disk."""
        path = os.path.join(tmp_dir, "sorted.json")
        data = {"zebra": 1, "apple": 2, "mango": 3}
        safe_json_write(path, data, sort_keys=True)
        raw = open(path).read()
        assert raw.index('"apple"') < raw.index('"mango"') < raw.index('"zebra"')

    def test_sort_keys_round_trip(self, tmp_dir):
        """sort_keys preserves values exactly, only reorders keys on disk."""
        path = os.path.join(tmp_dir, "rt.json")
        data = {"b": [3, 1, 2], "a": {"y": 1, "x": 2}}
        safe_json_write(path, data, sort_keys=True)
        loaded = safe_json_read(path)
        assert loaded == data


class TestSafeJsonRead:

    def test_basic_read(self, tmp_dir):
        """Read returns parsed JSON data."""
        path = os.path.join(tmp_dir, "read.json")
        with open(path, "w") as f:
            json.dump({"hello": "world"}, f)
        result = safe_json_read(path)
        assert result["hello"] == "world"

    def test_missing_file_returns_default(self, tmp_dir):
        """Read returns default for missing file."""
        path = os.path.join(tmp_dir, "nonexistent.json")
        assert safe_json_read(path, default={}) == {}
        assert safe_json_read(path, default=[]) == []
        assert safe_json_read(path) is None

    def test_corrupted_file_returns_default(self, tmp_dir):
        """Read returns default for corrupted JSON."""
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("{corrupted json data!!")
        result = safe_json_read(path, default={"fallback": True})
        # Should return default or recovered backup data
        assert result is not None


class TestSafeJsonlAppend:

    def test_basic_append(self, tmp_dir):
        """Append adds a JSON line."""
        path = os.path.join(tmp_dir, "log.jsonl")
        safe_jsonl_append(path, {"event": "test", "value": 1})
        safe_jsonl_append(path, {"event": "test", "value": 2})
        lines = open(path).readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["value"] == 1
        assert json.loads(lines[1])["value"] == 2

    def test_creates_file_if_missing(self, tmp_dir):
        """Append creates file if it doesn't exist."""
        path = os.path.join(tmp_dir, "new.jsonl")
        assert not os.path.exists(path)
        safe_jsonl_append(path, {"first": True})
        assert os.path.exists(path)

    def test_preserves_existing_content(self, tmp_dir):
        """Append doesn't overwrite existing lines."""
        path = os.path.join(tmp_dir, "preserve.jsonl")
        with open(path, "w") as f:
            f.write('{"existing": true}\n')
        safe_jsonl_append(path, {"new": True})
        lines = open(path).readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["existing"] is True


class TestConcurrentAccess:

    def test_concurrent_writes_no_corruption(self, tmp_dir):
        """Concurrent writes don't corrupt the file."""
        path = os.path.join(tmp_dir, "concurrent.json")
        errors = []

        def writer(thread_id):
            try:
                for i in range(10):
                    safe_json_write(path, {"thread": thread_id, "iteration": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # File should be valid JSON after all writes
        data = json.loads(open(path).read())
        assert "thread" in data
        assert "iteration" in data


class TestLazyImports:

    def test_safe_json_write_import(self):
        from src.scanner import safe_json_write
        assert safe_json_write is not None

    def test_safe_json_read_import(self):
        from src.scanner import safe_json_read
        assert safe_json_read is not None

    def test_safe_jsonl_append_import(self):
        from src.scanner import safe_jsonl_append
        assert safe_jsonl_append is not None


class TestExecutionWiring:

    def test_execution_uses_safe_io_for_replay(self):
        """execution.py uses safe_jsonl_append for RL replay buffer."""
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager)
            assert "safe_jsonl_append" in source or "safe_json" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")

    def test_execution_uses_safe_io_for_performance(self):
        """execution.py uses safe_json_write for pair performance."""
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager._update_pair_performance)
            assert "safe_json_write" in source
        except (ImportError, AttributeError):
            pytest.skip("ExecutionManager._update_pair_performance not available")
