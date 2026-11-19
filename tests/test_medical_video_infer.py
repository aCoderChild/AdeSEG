"""Hermetic orchestration test; this is NOT native checkpoint GPU validation."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from datasets.video import ManifestFrame
from scripts.infer_med_video import oracle_first_present, infer_sequence

class FakePredictor:
    def init_state(self, *, video_path, offload_video_to_cpu, offload_state_to_cpu):
        assert Path(video_path, "000000.jpg").is_file()
        return {"video_path": video_path}
    def add_new_mask(self, state, frame_idx, obj_id, mask):
        assert frame_idx == 0 and obj_id == 1 and bool(mask.any())
    def propagate_in_video(self, state, start_frame_idx, reverse):
        assert start_frame_idx == 0 and reverse is False
        yield 0, [1], torch.ones((1,1,4,4))
        yield 1, [1], torch.ones((1,1,4,4))

def test_oracle_video_inference_marks_prompt_and_exports(tmp_path):
    frames = []
    for i in (0,1):
        image = tmp_path / f"{i}.jpg"
        mask = tmp_path / f"{i}_mask.png"
        Image.fromarray(np.ones((4,4,3),dtype=np.uint8)*30).save(image)
        Image.fromarray(np.ones((4,4),dtype=np.uint8)).save(mask)
        frames.append(ManifestFrame("seq2", i, image, mask))
    out = tmp_path / "out"
    out.mkdir()
    result = infer_sequence(FakePredictor(), frames, out, record_logits=True)
    assert result["status"] == "completed"
    assert result["prompt_source"] == "oracle_gt_mask"
    assert result["predicted"] == 1
    assert (out/"predictions"/"seq2"/"1.png").exists()
    assert (out/"logits"/"seq2"/"1.npy").exists()

def test_no_foreground_is_explicitly_unpromptable(tmp_path):
    mask = tmp_path/"0.png"
    Image.fromarray(np.zeros((4,4),dtype=np.uint8)).save(mask)
    frame = ManifestFrame("seq1", 0, tmp_path/"missing.jpg", mask)
    index, initial = oracle_first_present([frame],1)
    assert index is None and initial is None
