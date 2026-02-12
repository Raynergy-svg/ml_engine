import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import pytest

pytest.importorskip("unified_talk")


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from unified_talk import select_latest_checkpoint  # noqa: E402


class TestCheckpointSync(unittest.TestCase):
    def test_select_latest_checkpoint_prefers_unified(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models"
            model_dir.mkdir(parents=True)

            other = model_dir / "best_model.pth"
            other.write_bytes(b"x")

            unified_old = model_dir / "unified_market_net.pth"
            unified_old.write_bytes(b"x")

            unified_new = model_dir / "unified_market_net_checkpoint.pth"
            unified_new.write_bytes(b"x")

            now = time.time()
            os.utime(other, (now - 10, now - 10))
            os.utime(unified_old, (now - 20, now - 20))
            os.utime(unified_new, (now - 5, now - 5))

            picked = select_latest_checkpoint(model_dir)
            self.assertIsNotNone(picked)
            self.assertEqual(picked.name, unified_new.name)
# — Raynergy-svg —
