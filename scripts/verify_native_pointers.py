#!/usr/bin/env python3
"""Check that modeling/native_pointers.py builds exactly the object-pointer tokens native MedSAM2
feeds to memory attention (captured from a native run on seq20 and seq5, 25 frames each)."""
import sys, json
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MedSAM2")]
import torch
from pathlib import Path
from modeling.medsam2 import build_video_predictor
from modeling.native_pointers import append_native_object_pointers
from scripts.infer import first_ground_truth_box
from datasets.polypgen import get_video_frame_dir, get_frame_names

cfg = json.load(open(ROOT / "configs/polypgen.yaml")); root = ROOT / cfg["data_root"]
predictor = build_video_predictor(cfg["sam2_cfg"], ROOT / cfg["sam2_checkpoint"], "auto")
captured = []
attention = predictor.memory_attention.forward
def capture(curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens=0):
    captured.append((memory[-num_obj_ptr_tokens:].clone(), memory_pos[-num_obj_ptr_tokens:].clone(), num_obj_ptr_tokens))
    return attention(curr=curr, curr_pos=curr_pos, memory=memory, memory_pos=memory_pos, num_obj_ptr_tokens=num_obj_ptr_tokens)
predictor.memory_attention.forward = capture
worst = 0.0
for seq in ["seq20", "seq5"]:
    names = get_frame_names(get_video_frame_dir(root, seq))
    prompt_idx, box = first_ground_truth_box(root, seq, names)
    state = predictor.init_state(video_path=str(get_video_frame_dir(root, seq)), offload_video_to_cpu=True)
    predictor.add_new_points_or_box(state, frame_idx=prompt_idx, obj_id=1, box=box)
    with torch.inference_mode():
        for frame_idx, _, _ in predictor.propagate_in_video(state, start_frame_idx=prompt_idx):
            if frame_idx == prompt_idx:
                continue
            native_tokens, native_pos, count = captured[-1]
            memory, positions = [], []
            ours = append_native_object_pointers(
                predictor, frame_idx, state["output_dict"], state["num_frames"], False,
                native_tokens.device, 1, memory, positions,
            )
            assert ours == count, (seq, frame_idx, ours, count)
            worst = max(worst, (memory[0] - native_tokens).abs().max().item(), (positions[0] - native_pos).abs().max().item())
            if frame_idx >= prompt_idx + 25:
                break
    predictor.reset_state(state)
print(f"pointer tokens identical in count; max abs difference {worst:.2e}")
if worst > 0:
    raise SystemExit("native_pointers.py differs from MedSAM2")
