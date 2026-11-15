"""Exercise resume orchestration with a small deterministic stand-in runner."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.cohort_audit_checkpoints import save_condition_checkpoint
from scripts import run_spatiotemporal_tome_value_exchange_durable as runner


class DurableCohortResumeTests(unittest.TestCase):
    def test_resume_reuses_complete_arm_and_preserves_incomplete_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "attempt"
            output.mkdir()
            protocol_file = root / "protocol.json"
            protocol_file.write_text("{}")
            native_file = root / "native.json"
            native_file.write_text("{}")
            native_protocol = root / "native_protocol.json"
            native_protocol.write_text("{}")
            protocol = {
                "output_root": str(root), "unique_post_prompt_targets": 2,
                "runtime": {"threads": 1, "seed": 1}, "statistics": {}, "source_hashes": {},
            }
            (output / "setup.json").write_text(json.dumps({
                "durable_schema": 1, "protocol": protocol,
                "protocol_sha256": runner.digest(protocol_file),
            }))
            rows = [{"frame": i, "present": True, "scores": {"dice": 0.5},
                     "prediction_sha256": "prediction", "stored_sha256": "stored",
                     "absent_fp": None} for i in (1, 2)]
            calls = [{"frame": i, "layer": layer} for i in (1, 2) for layer in range(4)]
            receipt = {"targets": 2, "parameter_hash_unchanged": True,
                       "temporal_slots_restored": True, "native_exact_frames_checked": 2}
            complete = output / "noop" / "seq2"
            complete.mkdir(parents=True)
            (complete / "frames.json").write_text(json.dumps(rows))
            save_condition_checkpoint(complete, "seq2", "noop", rows, calls, receipt)
            incomplete = output / "local" / "seq2"
            incomplete.mkdir(parents=True)
            (incomplete / "partial.txt").write_text("preserved")
            invoked = []

            def fake_run(*args, **kwargs):
                condition = kwargs["condition_name"]
                invoked.append(condition)
                folder = output / condition / "seq2"
                folder.mkdir(parents=True)
                (folder / "frames.json").write_text(json.dumps(rows))
                kwargs["memory_reader_factory"](None).extend(calls)
                return rows, receipt

            def fake_audit(audits, frames):
                self.assertEqual(list(frames), [1, 2])
                self.assertEqual({key: len(value) for key, value in audits.items()},
                                 {"noop": 8, "local": 8, "cross_frame": 8})
                return {"passed": True}

            with patch.object(runner, "load_protocol", return_value=protocol), \
                 patch.object(runner, "preflight", return_value={"passed": True}), \
                 patch.object(runner, "validate_output", side_effect=lambda path: path.resolve()), \
                 patch.object(runner, "SEQUENCES", ("seq2",)), \
                 patch.object(runner, "MANIFEST", root / "manifest"), \
                 patch.object(runner, "NATIVE_RESULTS", native_file), \
                 patch.object(runner, "NATIVE_PROTOCOL", native_protocol), \
                 patch.object(runner, "load_manifest", side_effect=lambda *a: {"seq2": [0, 1, 2]}), \
                 patch.object(runner, "native_reference", return_value=(0, {r["frame"]: r for r in rows}, None)), \
                 patch.object(runner, "RecordingValueExchange", side_effect=lambda model, mode, audit_sink: audit_sink), \
                 patch.object(runner, "run_condition", side_effect=fake_run), \
                 patch.object(runner, "summarize", return_value={"present_dice": 0.5}), \
                 patch.object(runner, "verify_fullprop_audits", side_effect=fake_audit), \
                 patch.object(runner, "cohort_comparison", return_value={"screen_passed": False}), \
                 patch.object(sys, "argv", ["runner", "--protocol", str(protocol_file), "--output-dir", str(output), "--resume"]):
                runner.main()

            self.assertEqual(invoked, ["local", "cross_frame"])
            archived = list((output / "interrupted_conditions").glob("seq2_local_*/partial.txt"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_text(), "preserved")
            self.assertTrue(json.loads((output / "completion.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
