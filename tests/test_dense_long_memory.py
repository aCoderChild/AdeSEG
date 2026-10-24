import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from modeling.dense_long_memory import DenseLongMemory
from modeling.sparse_memory import temporal_slot
from scripts.run_long_history_screen import require_e2


def fixture(current):
    model = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=1,
                            maskmem_tpos_enc=torch.arange(7).float()[:, None, None, None].expand(7, 1, 1, 64))
    controller = DenseLongMemory(model, 0)
    controller.frame = current
    controller.bank = {i: {"features": torch.full((1, 64, 64, 64), float(i)),
                           "pos": torch.zeros(1, 64, 64, 64), "keep": torch.zeros(4096, dtype=torch.bool)}
                       for i in range(current)}
    frames = [0]+list(range(max(1, current-6), current))
    memory = torch.cat([controller.bank[f]["features"].flatten(2).permute(2, 0, 1) for f in frames]+[torch.full((4, 1, 64), -99.)])
    pos = torch.cat([torch.full((4096, 1, 64), float(temporal_slot(f, current, 0))) for f in frames]+[torch.full((4, 1, 64), -88.)])
    return controller, {"memory": memory, "memory_pos": pos, "num_obj_ptr_tokens": 4}


def test_dense_warmup_inputs_match_native_and_keep_every_cell():
    controller, kwargs = fixture(4)
    _, rebuilt = controller._assemble(None, (), kwargs)
    assert torch.equal(rebuilt["memory"], kwargs["memory"])
    assert torch.equal(rebuilt["memory_pos"], kwargs["memory_pos"])
    attention = SimpleNamespace(training=False)
    controller._before(0, attention, (), {"v": rebuilt["memory"].transpose(0, 1), "num_k_exclude_rope": 4})
    assert attention._sparse_mask.all()
    assert controller.layers[0]["selected_spatial_keys"] == 4*4096


def test_dense_long_control_is_capped_and_pointer_tails_unchanged():
    controller, kwargs = fixture(40)
    _, rebuilt = controller._assemble(None, (), kwargs)
    assert controller.assembled_frames == [0]+list(range(9, 40))
    assert rebuilt["memory"].shape[0] == 32*4096+4
    assert torch.equal(rebuilt["memory"][-4:], kwargs["memory"][-4:])
    assert torch.equal(rebuilt["memory_pos"][-4:], kwargs["memory_pos"][-4:])
    assert all(temporal_slot(f, 40, 0) == 5 for f in range(9, 35))


def test_failed_e2_rejected_before_any_e3_execution(tmp_path):
    (tmp_path/"validation.json").write_text(json.dumps({"passed": False}))
    (tmp_path/"delivery.json").write_text(json.dumps({"completed": True}))
    with pytest.raises(ValueError, match="E2"):
        require_e2(tmp_path)


def test_queue_failure_stops_before_checkpoint_or_output(tmp_path):
    gate = tmp_path/"E2"
    gate.mkdir()
    (gate/"failure.json").write_text("{}")
    output = tmp_path/"never_created"
    runner = Path(__file__).resolve().parents[1]/"scripts/run_long_history_screen.py"
    result = subprocess.run([sys.executable, str(runner), "--checkpoint", str(tmp_path/"missing.pt"),
                             "--e2-dir", str(gate), "--output-dir", str(output), "--wait-for-e2"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "do not launch E3" in result.stderr
    assert not output.exists()
