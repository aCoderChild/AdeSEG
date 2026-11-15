"""Bounded on-demand CPU frame loading for the existing SAM2 eager pipeline.

This is an opt-in engineering component for one-shot forward video inference,
not part of the frozen capture runtime. It reuses the author's decoder and
matches the eager path's float32-before-normalization order. It requires a
source-hash list, rejects changing files, and does not claim a total RSS limit.
"""
from collections import OrderedDict
from contextlib import contextmanager
import ast
import inspect
import os
from pathlib import Path
import threading

from PIL import Image
import torch

from scripts.memory_token_io import digest

_LOCK = threading.Lock()


def require(value, message):
    if not value:
        raise ValueError(message)


def signature(path):
    path = Path(path).resolve(strict=True); s = path.stat()
    require(path.is_file(), 'Frame is not a regular file')
    return (str(path), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def verify_predictor_image_access(path, expected_sha):
    """Fail closed unless the pinned predictor has the inspected image accesses."""
    path = Path(path).resolve(strict=True)
    require(digest(path) == expected_sha, 'Predictor source changed')
    tree = ast.parse(path.read_text(), filename=str(path))
    require(digest(path) == expected_sha, 'Predictor source changed while parsing')
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    literals = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == 'images']
    accesses = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript) and
                isinstance(node.value, ast.Name) and node.value.id == 'inference_state' and
                isinstance(node.slice, ast.Constant) and node.slice.value == 'images']
    require(len(literals) == len(accesses) == 2, 'Predictor image-access sites differ')
    stores = [node for node in accesses if isinstance(node.ctx, ast.Store)]
    loads = [node for node in accesses if isinstance(node.ctx, ast.Load)]
    require(len(stores) == len(loads) == 1, 'Predictor image-access modes differ')
    indexed = parents.get(loads[0]); move = parents.get(indexed); call = parents.get(move)
    require(isinstance(indexed, ast.Subscript) and indexed.value is loads[0] and
            isinstance(indexed.slice, ast.Name) and indexed.slice.id == 'frame_idx' and
            isinstance(indexed.ctx, ast.Load) and isinstance(move, ast.Attribute) and move.attr == 'to' and
            isinstance(call, ast.Call) and len(call.args) == 1 and isinstance(call.args[0], ast.Name) and
            call.args[0].id == 'device', 'Predictor does not copy indexed frames to its device')
    def enclosing(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return None
    require(enclosing(stores[0]) == 'init_state' and enclosing(loads[0]) == '_get_image_feature',
            'Predictor image-access functions differ')
    return {'source_sha256': expected_sha, 'writes': 1, 'integer_frame_index_reads': 1,
            'reader': '_get_image_feature', 'cpu_frame_mutation_by_reader': False,
            'scope': 'Inspected pinned predictor source; model must use an MPS device.'}


class BoundedEagerFrameLoaderV1:
    """Decode a frame at first use and retain a bounded float32 CPU LRU."""
    def __init__(self, paths, hashes, image_size, decoder, *, limit_bytes,
                 mean=(.485, .456, .406), std=(.229, .224, .225)):
        self.paths = tuple(Path(x).resolve(strict=True) for x in paths)
        self.hashes = tuple(hashes)
        require(self.paths and len(self.paths) == len(self.hashes) and type(image_size) is int and image_size > 0,
                'Invalid frame plan')
        require(type(limit_bytes) is int and limit_bytes >= 3 * image_size * image_size * 4,
                'Frame cache must fit at least one float32 frame')
        require(all(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                    for h in self.hashes), 'Invalid frame hashes')
        require(len(mean) == 3 and len(std) == 3 and all(float(x) > 0 for x in std), 'Invalid normalization')
        self.size = image_size; self.decoder = decoder; self.limit_bytes = limit_bytes
        self.owner_thread = threading.get_ident()
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.initial_signatures = tuple(signature(path) for path in self.paths)
        self.cache = OrderedDict(); self.bytes = 0; self.peak_bytes = 0; self.hits = 0; self.misses = 0; self.evictions = 0
        self.read_indices = set(); self.final_sha256_verified = False
        with Image.open(self.paths[0]) as image:
            self.video_width, self.video_height = image.size
        require(signature(self.paths[0]) == self.initial_signatures[0], 'First frame changed during inspection')
        for i, path in enumerate(self.paths):
            with Image.open(path) as image:
                require(image.size == (self.video_width, self.video_height), 'Variable frame geometry is unsupported')
            require(signature(path) == self.initial_signatures[i], 'Frame changed during geometry inspection')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        require(threading.get_ident() == self.owner_thread, 'Frame access from a different thread is unsupported')
        require(type(index) is int and -len(self.paths) <= index < len(self.paths), 'Invalid frame index')
        index %= len(self.paths); self.final_sha256_verified = False
        path = self.paths[index]; expected = self.initial_signatures[index]
        require(signature(path) == expected, 'Frame metadata changed')
        if index in self.cache:
            self.hits += 1; result = self.cache.pop(index); self.cache[index] = result
            # Do not expose the retained tensor to accidental mutation. The
            # pinned predictor immediately copies this value to MPS.
            return result.clone()
        self.misses += 1
        require(digest(path) == self.hashes[index], 'Frame hash differs before decode')
        raw, height, width = self.decoder(str(path), self.size)
        require(digest(path) == self.hashes[index] and signature(path) == expected and
                (height, width) == (self.video_height, self.video_width), 'Frame changed while decoding')
        require(isinstance(raw, torch.Tensor) and raw.device.type == 'cpu' and raw.shape == (3, self.size, self.size)
                and torch.isfinite(raw).all(), 'Author decoder output differs')
        # Native eager stores the decoder output in a float32 batch first.
        image = raw.to(dtype=torch.float32).contiguous()
        image -= self.mean; image /= self.std
        require(torch.isfinite(image).all(), 'Normalized frame is nonfinite')
        size = image.numel() * image.element_size()
        while self.cache and self.bytes + size > self.limit_bytes:
            _, removed = self.cache.popitem(last=False)
            self.bytes -= removed.numel() * removed.element_size()
            self.evictions += 1
        self.cache[index] = image; self.bytes += size; self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.read_indices.add(index)
        return image.clone()

    def verify_sources(self):
        require(threading.get_ident() == self.owner_thread, 'Source verification from a different thread is unsupported')
        for index, path in enumerate(self.paths):
            require(signature(path) == self.initial_signatures[index] and digest(path) == self.hashes[index] and
                    signature(path) == self.initial_signatures[index], 'Frame changed during video')
        self.final_sha256_verified = True

    def stats(self):
        return {'limit_bytes': self.limit_bytes, 'peak_cached_tensor_bytes': self.peak_bytes,
            'cache_entries': len(self.cache), 'hits': self.hits, 'misses': self.misses,
            'evictions': self.evictions,
            'unique_decoded_frames': len(self.read_indices), 'final_sha256_verified': self.final_sha256_verified,
            'scope': 'Cached float32 image tensors only; excludes returned copies, process RSS, model state, decode buffers and device memory.'}


@contextmanager
def install_bounded_eager_loader(model, *, staged_dir, expected_paths, expected_sha256,
                                 author_module_file, author_module_sha256, author_misc_sha256,
                                 author_load_video_frames, author_decoder, limit_bytes):
    """Temporarily replace only the predictor's init-time frame-loader binding.

    The surrounding pipeline must ensure no concurrent inference in this
    process and must call ``verify_sources`` on the returned loader before
    accepting outputs. The original global binding is always restored.
    """
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Concurrent bounded-loader installation')
    bindings = None; original = None; created = []
    try:
        require(threading.current_thread() is threading.main_thread() and threading.active_count() == 1,
                'Bounded loader installation requires a single-threaded Python process')
        method = inspect.unwrap(model.init_state)
        bindings = method.__globals__
        module = Path(author_module_file).resolve(strict=True)
        misc_file = Path(inspect.getsourcefile(author_load_video_frames)).resolve(strict=True)
        decoder_file = Path(inspect.getsourcefile(author_decoder)).resolve(strict=True)
        require(Path(bindings['__file__']).resolve() == Path(author_module_file).resolve() and
                bindings.get('load_video_frames') is author_load_video_frames and
                getattr(author_decoder, '__name__', None) == '_load_img_as_tensor' and
                author_load_video_frames.__globals__.get('_load_img_as_tensor') is author_decoder and
                misc_file == decoder_file and digest(module) == author_module_sha256 and
                digest(misc_file) == author_misc_sha256,
                'Unexpected predictor or author loader binding')
        require(getattr(getattr(model, 'device', None), 'type', None) == 'mps',
                'Bounded loader requires an MPS predictor so device transfer copies CPU frames')
        verify_predictor_image_access(module, author_module_sha256)
        original = bindings['load_video_frames']; staged = Path(staged_dir).resolve(strict=True)
        expected = tuple(Path(path).resolve(strict=True) for path in expected_paths)
        def load(video_path, image_size, offload_video_to_cpu, img_mean=(.485,.456,.406),
                 img_std=(.229,.224,.225), async_loading_frames=False, compute_device=None):
            require(threading.current_thread() is threading.main_thread() and threading.active_count() == 1,
                    'Frame loader requires a single-threaded Python process')
            require(not created and Path(video_path).resolve(strict=True) == staged and
                    offload_video_to_cpu is True and async_loading_frames is False,
                    'Unsupported or repeated frame-loader call')
            names = [p for p in os.listdir(staged) if Path(p).suffix in ('.jpg', '.jpeg', '.JPG', '.JPEG')]
            names.sort(key=lambda name: int(Path(name).stem))
            paths = tuple((staged / name).resolve(strict=True) for name in names)
            require(paths == expected, 'Staged frame order or sources differ')
            loader = BoundedEagerFrameLoaderV1(paths, expected_sha256, image_size, author_decoder,
                                               limit_bytes=limit_bytes, mean=img_mean, std=img_std)
            created.append(loader)
            return loader, loader.video_height, loader.video_width
        bindings['load_video_frames'] = load
        yield created
        require(len(created) == 1, 'Predictor did not invoke the expected frame loader once')
        require(threading.current_thread() is threading.main_thread() and threading.active_count() == 1,
                'Concurrent Python thread appeared during frame-loader installation')
        require(digest(module) == author_module_sha256 and digest(misc_file) == author_misc_sha256,
                'Predictor or author decoder source changed during installation')
    finally:
        if bindings is not None and original is not None:
            bindings['load_video_frames'] = original
        _LOCK.release()
