import tempfile
import unittest
from pathlib import Path

from src.scanner.automation.background_activity import BackgroundActivityTracker


class BackgroundActivityTrackerTest(unittest.TestCase):
    def test_background_activity_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "background_activity.json"
            tracker = BackgroundActivityTracker(path=path)

            activity_id = tracker.start_activity(
                kind="training",
                title="Train all models",
                source="test",
                metadata={"model": "all"},
            )
            tracker.append_event(activity_id, "started")
            tracker.complete_activity(activity_id, summary="done")

            snapshot = tracker.get_snapshot()

            self.assertEqual(snapshot["active_count"], 0)
            self.assertEqual(snapshot["history_count"], 1)
            self.assertEqual(snapshot["history"][0]["id"], activity_id)
            self.assertEqual(snapshot["history"][0]["status"], "completed")
            self.assertEqual(snapshot["history"][0]["summary"], "done")
            self.assertEqual(snapshot["history"][0]["events"][0]["message"], "started")


if __name__ == "__main__":
    unittest.main()
