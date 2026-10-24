"""E3 C1 full-history control; native writes and author sparse core unchanged.

Only project frame assembly and the allowed dense control budget differ from
E2. Every token is actually gathered, projected and read, including older BG.
"""

import torch

from modeling.sparse_memory import SparseMemoryRead, temporal_slot
from scripts.bank_integrity import fingerprint


class DenseLongMemory(SparseMemoryRead):
    def __init__(self, model, prompt):
        super().__init__(model, prompt, "all_rebuilt", max_frames=32)

    def _assemble(self, module, args, kwargs):
        if args:
            raise RuntimeError("Unexpected positional memory-attention inputs")
        self.read_bank_hash = fingerprint(self.bank)["sha256"]
        memory, pos = kwargs["memory"], kwargs["memory_pos"]
        pointers = int(kwargs.get("num_obj_ptr_tokens", 0))
        frames = [self.prompt]+list(range(max(self.prompt+1, self.frame-31), self.frame))
        values, positions = [], []
        for frame in frames:
            item = self.bank[frame]
            values.append(item["features"].to(memory.device).flatten(2).permute(2, 0, 1))
            spatial = item["pos"].to(pos.device).flatten(2).permute(2, 0, 1)
            positions.append(spatial+self.model.maskmem_tpos_enc[temporal_slot(frame, self.frame, self.prompt)])
        if pointers:
            values.append(memory[-pointers:])
            positions.append(pos[-pointers:])
        assembled, assembled_pos = torch.cat(values), torch.cat(positions)
        if self.frame-self.prompt <= 7 and not (
            torch.equal(assembled, memory) and torch.equal(assembled_pos, pos)
        ):
            raise RuntimeError("C1 warmup differs from native inputs")
        self.assembled_frames = frames
        return args, {**kwargs, "memory": assembled, "memory_pos": assembled_pos}

    def _before(self, layer, module, args, kwargs):
        if args or module.training:
            raise RuntimeError("C1 requires eval-mode keyword attention")
        memory = kwargs["v"]
        pointers = int(kwargs.get("num_k_exclude_rope", 0))
        spatial = memory.shape[1]-pointers
        frames = self.assembled_frames
        if not 1 <= len(frames) <= 32 or spatial != len(frames)*4096 or memory.shape[0] != 1:
            raise RuntimeError("C1 frame/key budget mismatch")
        for index, frame in enumerate(frames):
            expected = self.bank[frame]["features"].flatten(2).transpose(1, 2).float().cpu()
            if not torch.equal(memory[:, index*4096:(index+1)*4096].float().cpu(), expected):
                raise RuntimeError("C1 actual memory values differ from captured frame")
        module._sparse_mask = torch.ones(max(0, spatial-8192), dtype=torch.bool, device=memory.device)
        self.layers.append({"layer": layer, "frames": frames, "raw_spatial_keys": spatial,
                            "selected_spatial_keys": spatial, "pointer_keys": pointers,
                            "temporal_slots": [temporal_slot(f, self.frame, self.prompt) for f in frames]})
