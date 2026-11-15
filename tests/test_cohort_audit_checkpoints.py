import json
import tempfile
import unittest
from pathlib import Path

from scripts.cohort_audit_checkpoints import (
    archive_incomplete_condition,
    load_condition_checkpoint,
    save_condition_checkpoint,
)


class CohortAuditCheckpointTests(unittest.TestCase):
    def test_incomplete_condition_is_preserved_before_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            folder = output / "local" / "seq2"
            folder.mkdir(parents=True)
            (folder / "partial.txt").write_text("unfinished")
            archived = archive_incomplete_condition(output, "seq2", "local")
            self.assertFalse(folder.exists())
            self.assertEqual((archived / "partial.txt").read_text(), "unfinished")
            folder.mkdir()
            (folder / "condition_checkpoint.json").write_text("{}")
            with self.assertRaises(ValueError):
                archive_incomplete_condition(output, "seq2", "local")
            self.assertTrue(folder.exists())

    def test_complete_checkpoint_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = [{"frame": 7}, {"frame": 8}]
            calls = [{"frame": frame, "layer": layer} for frame in (7, 8) for layer in range(4)]
            (folder / "frames.json").write_text(json.dumps(rows))
            save_condition_checkpoint(folder, "seq2", "local", rows, calls, {"targets": 2})
            self.assertEqual(load_condition_checkpoint(folder, "seq2", "local"), (rows, calls, {"targets": 2}))
            with self.assertRaises(ValueError):
                load_condition_checkpoint(folder, "seq3", "local")
            (folder / "attention_audit.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_condition_checkpoint(folder, "seq2", "local")

    def test_interrupted_or_incomplete_checkpoint_is_not_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = [{"frame": 7}]
            (folder / "frames.json").write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                save_condition_checkpoint(folder, "seq2", "local", rows, [{"frame": 7, "layer": 0}], {})
            self.assertFalse((folder / "condition_checkpoint.json").exists())
            with self.assertRaises(FileNotFoundError):
                load_condition_checkpoint(folder, "seq2", "local")


if __name__ == "__main__":
    unittest.main()
