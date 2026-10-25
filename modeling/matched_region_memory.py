"""C2 and count/frame-matched C3 on the unchanged validated E2 reader.

C3 replays C2's allocation using the author's exact-count random-mask helper.
Allocation is experimental control glue, not an author-published method.
"""

import ast
from functools import lru_cache
import hashlib
import time

import numpy as np
import torch

from modeling.region_tokens import EFFICIENT_SAM2_REVISION, _verified_source
from modeling.sparse_memory import SparseMemoryRead


@lru_cache(maxsize=1)
def author_random_mask():
    path, text = _verified_source(
        "Efficient-SAM2", EFFICIENT_SAM2_REVISION, "build_model/window_bypass.py"
    )
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == "create_mask_randperm")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def random_cells(count, sequence, frame, seed):
    if not isinstance(count, int) or not 0 <= count <= 4096:
        raise ValueError("Invalid reference token count")
    derived = int.from_bytes(hashlib.sha256(f"{seed}|{sequence}|{frame}".encode()).digest()[:8], "little") % (2**63-1)
    # Author helper uses the default generator. Isolate it from model RNG.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(derived)
        mask = author_random_mask()(4096, (4096-count)/4096, device="cpu")
    if int(mask.sum()) != count:
        raise RuntimeError("Author random selector changed reference count")
    return mask


class MatchedRegionMemory(SparseMemoryRead):
    def __init__(self, model, prompt, sequence, *, reference=None, seed=None):
        super().__init__(model, prompt, "region_long", max_frames=32)
        if (reference is None) != (seed is None):
            raise ValueError("C3 requires both C2 allocation reference and fixed seed")
        self.reference, self.seed, self.sequence = reference, seed, sequence
        self.attention_seconds = 0.0

    def _before(self, layer, module, args, kwargs):
        super()._before(layer, module, args, kwargs)
        trace = self.layers[-1]
        frames = trace["frames"]
        counts = [4096 if i == 0 or i == len(frames)-1 else int(self.bank[f]["keep"].sum())
                  for i, f in enumerate(frames)]
        trace["per_frame_spatial_keys"] = counts
        trace["query_tokens"] = int(kwargs["q"].shape[1])
        if sum(counts) != trace["selected_spatial_keys"]:
            raise RuntimeError("Per-frame and actual gathered budget differ")
        if self.reference is not None:
            expected = self.reference[self.frame]["allocation"]
            actual = {"frames": frames, "per_frame_spatial_keys": counts,
                      "pointer_keys": trace["pointer_keys"],
                      "selected_spatial_keys": trace["selected_spatial_keys"]}
            if actual != expected:
                raise RuntimeError("C3 frame IDs/counts/pointers differ from C2 allocation")

    def capture(self, state, index):
        record = super().capture(state, index)
        if self.reference is not None:
            expected = self.reference[index]
            if expected["frame"] != index:
                raise RuntimeError("C2 reference frame is misaligned")
            self.bank[index]["keep"] = random_cells(expected["retained_tokens"], self.sequence, index, self.seed)
        keep = self.bank[index]["keep"].cpu().numpy()
        record.update(condition="C2" if self.reference is None else "C3", seed=self.seed,
                      retained_tokens=int(keep.sum()),
                      keep_bits_hex=np.packbits(keep, bitorder="big").tobytes().hex(),
                      allocation=None, memory_attention_seconds=self.attention_seconds)
        self.attention_seconds = 0.0
        if record["layers"]:
            first = record["layers"][0]
            record["allocation"] = {k: first[k] for k in ("frames", "per_frame_spatial_keys", "pointer_keys", "selected_spatial_keys")}
            for layer in record["layers"]:
                if {k: layer[k] for k in record["allocation"]} != record["allocation"]:
                    raise RuntimeError("Allocation differs across four attention layers")
        return record

    def _start_clock(self, module, args, kwargs):
        self.started = time.perf_counter()

    def _stop_clock(self, module, args, kwargs, output):
        self.attention_seconds += time.perf_counter()-self.started

    def __enter__(self):
        super().__enter__()
        # Start after E2 assembly; stop before E2 read-only bank hashing.
        self.handles.append(self.model.memory_attention.register_forward_pre_hook(self._start_clock, with_kwargs=True))
        self.handles.append(self.model.memory_attention.register_forward_hook(self._stop_clock, with_kwargs=True, prepend=True))
        return self
