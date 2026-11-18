"""Oracle frame is excluded; missing post-prompt prediction is an error."""
import json
import numpy as np
from PIL import Image
import pytest
from scripts.infer_med_video import infer_sequence, sha256
from scripts.evaluate_med_video import evaluate_run
from datasets.video import ManifestFrame

class FakePredictor:
    def init_state(self, **kwargs):
        return {}
    def add_new_mask(self, state, frame_idx, obj_id, mask):
        assert frame_idx == 0
    def propagate_in_video(self, state, start_frame_idx, reverse):
        yield 0, [1], __import__("torch").ones((1,1,4,4))
        yield 1, [1], __import__("torch").ones((1,1,4,4))

def test_post_prompt_only_evaluation(tmp_path):
    rows, frames = [], []
    for i in (0,1):
        image, mask = tmp_path/f"{i}.jpg", tmp_path/f"{i}.png"
        Image.fromarray(np.ones((4,4,3),dtype=np.uint8)*20).save(image)
        Image.fromarray(np.ones((4,4),dtype=np.uint8)).save(mask)
        rows.append({"split":"test", "video_id":"seq2","frame_index":i,
                     "image":image.name,"mask":mask.name})
        frames.append(ManifestFrame("seq2",i,image,mask))
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(x) for x in rows))
    root=tmp_path/"run"
    root.mkdir()
    outcome=infer_sequence(FakePredictor(),frames,root)
    record={"protocol":"oracle_first_present_mask_post_prompt_v1",
            "split":"test","manifest_sha256":sha256(manifest),
            "label_id":1,"results":{"seq2":outcome}}
    (root/"run.json").write_text(json.dumps(record))
    result, per_frame, per_sequence=evaluate_run(manifest,root)
    assert result["all_post_prompt_frames"] == 1
    assert result["present_frame_weighted_dice"] == 1
    assert per_frame[0]["frame"] == 1
    assert per_sequence[0]["dice"] == 1
    (root/"predictions"/"seq2"/"1.png").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_run(manifest,root)
