import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

import modeling.bounded_sam2_video_loader_v2 as bounded
from modeling.bounded_sam2_video_loader_v2 import BoundedEagerFrameLoaderV2, install_bounded_eager_loader
from modeling.seq23_viewport import stage_cropped_frames
from scripts.memory_token_io import digest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('author_misc_for_bounded_test',
    ROOT/'external/Lean-SAM2-token-window/sam2/utils/misc.py')
AUTHOR = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AUTHOR)


def make_frames(folder, count=5, shape=(36, 53)):
    rng = np.random.default_rng(101)
    paths = []
    for i in range(count):
        path = folder/f'{i:06d}.jpg'
        Image.fromarray(rng.integers(0, 256, (*shape, 3), dtype=np.uint8)).save(path, quality=91)
        paths.append(path)
    return paths


def test_on_demand_matches_author_eager_random_access_and_eviction(tmp_path):
    paths = make_frames(tmp_path); size = 32
    eager, height, width = AUTHOR.load_video_frames_from_jpg_images(str(tmp_path), size, True)
    loader = BoundedEagerFrameLoaderV2(paths, [digest(p) for p in paths], size, AUTHOR._load_img_as_tensor,
                                      limit_bytes=2*3*size*size*4)
    assert (loader.video_height, loader.video_width) == (height, width)
    assert len(loader) == len(paths)
    for i in (0, 2, 1, 2, 4, 0, 2, -1):
        actual = loader[i]
        assert actual.dtype == torch.float32 and torch.equal(actual, eager[i])
    assert loader.hits > 0 and loader.misses > len(paths)
    assert loader.evictions > 0
    assert loader.peak_bytes <= loader.limit_bytes and len(loader.cache) <= 2
    borrowed = loader[2]; borrowed.zero_()
    assert torch.equal(loader[2], eager[2])  # Caller mutation cannot corrupt the retained frame.
    loader.verify_sources(); assert loader.stats()['final_sha256_verified'] is True


def test_seq23_lossless_cropped_frame_matches_author_eager(tmp_path):
    src = tmp_path/'source'; staged = tmp_path/'staged'; src.mkdir(); staged.mkdir()
    originals = make_frames(src, count=2, shape=(720, 1280))
    stage_cropped_frames([SimpleNamespace(image_path=p) for p in originals], staged)
    paths = [staged/f'{i:06d}.jpg' for i in range(2)]; size = 32
    eager, height, width = AUTHOR.load_video_frames_from_jpg_images(str(staged), size, True)
    loader = BoundedEagerFrameLoaderV2(paths, [digest(p) for p in paths], size, AUTHOR._load_img_as_tensor,
                                      limit_bytes=3*size*size*4)
    assert (height, width) == (720, 834) == (loader.video_height, loader.video_width)
    assert all(torch.equal(loader[i], eager[i]) for i in (1, 0, 1))
    loader.verify_sources()


def test_mutated_frame_and_nonuniform_geometry_fail(tmp_path):
    paths = make_frames(tmp_path, count=2); hashes = [digest(p) for p in paths]
    loader = BoundedEagerFrameLoaderV2(paths, hashes, 32, AUTHOR._load_img_as_tensor, limit_bytes=3*32*32*4)
    _ = loader[0]
    paths[0].write_bytes(paths[1].read_bytes())
    with pytest.raises(ValueError, match='metadata changed'): _ = loader[0]
    with pytest.raises(ValueError, match='changed during video'): loader.verify_sources()
    Image.fromarray(np.zeros((10, 12, 3), np.uint8)).save(paths[0])
    with pytest.raises(ValueError, match='Variable frame geometry'):
        BoundedEagerFrameLoaderV2(paths, [digest(p) for p in paths], 32, AUTHOR._load_img_as_tensor,
                                  limit_bytes=3*32*32*4)


# Simulate the predictor's method-global loader binding without requiring MPS.
load_video_frames = AUTHOR.load_video_frames


class FakePredictor:
    device = torch.device('mps')
    @torch.inference_mode()
    def init_state(self, video_path, offload_video_to_cpu=True, offload_state_to_cpu=False,
                   async_loading_frames=False):
        images, h, w = load_video_frames(video_path=video_path, image_size=32,
            offload_video_to_cpu=offload_video_to_cpu, async_loading_frames=async_loading_frames,
            compute_device=torch.device('cpu'))
        return {'images':images, 'num_frames':len(images), 'video_height':h, 'video_width':w,
                'cached_features':{0:images[0].clone()}}


class FakeSWR(FakePredictor):
    def _reset_swr_runtime(self):
        self.reset_count = getattr(self, 'reset_count', 0) + 1

    @torch.inference_mode()
    def init_state(self, *args, **kwargs):
        self._reset_swr_runtime()
        return super().init_state(*args, **kwargs)


def test_scoped_predictor_hook_state_and_restore(tmp_path, monkeypatch):
    paths = make_frames(tmp_path, count=3); model = FakeSWR(); previous = load_video_frames
    monkeypatch.setattr(bounded, 'verify_predictor_image_access', lambda *args: {})
    monkeypatch.setattr(bounded.threading, 'active_count', lambda: 1)
    with install_bounded_eager_loader(model, staged_dir=tmp_path, expected_paths=paths,
            expected_sha256=[digest(p) for p in paths], author_module_file=__file__,
            author_module_sha256=digest(Path(__file__)),
            wrapper_module_file=__file__, wrapper_module_sha256=digest(Path(__file__)),
            author_misc_sha256=digest(ROOT/'external/Lean-SAM2-token-window/sam2/utils/misc.py'),
            author_load_video_frames=previous, author_decoder=AUTHOR._load_img_as_tensor,
            limit_bytes=2*3*32*32*4) as created:
        state = model.init_state(str(tmp_path))
        assert state['num_frames'] == 3 and len(created) == 1 and state['images'] is created[0]
        assert model.reset_count == 1
        assert torch.equal(state['cached_features'][0], state['images'][0])
    assert load_video_frames is previous


    assert torch.equal(state['images'][2], AUTHOR.load_video_frames_from_jpg_images(str(tmp_path),32,True)[0][2])
    state['images'].verify_sources()
    with pytest.raises(ValueError, match='did not invoke'):
        with install_bounded_eager_loader(model, staged_dir=tmp_path, expected_paths=paths,
                expected_sha256=[digest(p) for p in paths], author_module_file=__file__,
                author_module_sha256=digest(Path(__file__)),
            wrapper_module_file=__file__, wrapper_module_sha256=digest(Path(__file__)),
                author_misc_sha256=digest(ROOT/'external/Lean-SAM2-token-window/sam2/utils/misc.py'),
                author_load_video_frames=previous, author_decoder=AUTHOR._load_img_as_tensor,
                limit_bytes=2*3*32*32*4):
            pass
    assert load_video_frames is previous
    with pytest.raises(RuntimeError, match='synthetic failure'):
        with install_bounded_eager_loader(model, staged_dir=tmp_path, expected_paths=paths,
                expected_sha256=[digest(p) for p in paths], author_module_file=__file__,
                author_module_sha256=digest(Path(__file__)),
            wrapper_module_file=__file__, wrapper_module_sha256=digest(Path(__file__)),
                author_misc_sha256=digest(ROOT/'external/Lean-SAM2-token-window/sam2/utils/misc.py'),
                author_load_video_frames=previous, author_decoder=AUTHOR._load_img_as_tensor,
                limit_bytes=2*3*32*32*4):
            raise RuntimeError('synthetic failure')
    assert load_video_frames is previous



def test_loader_rejects_concurrent_python_threads(tmp_path, monkeypatch):
    paths = make_frames(tmp_path, count=2); model = FakeSWR()
    monkeypatch.setattr(bounded.threading, 'active_count', lambda: 2)
    with pytest.raises(ValueError, match='single-threaded Python process'):
        with install_bounded_eager_loader(model, staged_dir=tmp_path, expected_paths=paths,
                expected_sha256=[digest(p) for p in paths], author_module_file=__file__,
                author_module_sha256=digest(Path(__file__)),
            wrapper_module_file=__file__, wrapper_module_sha256=digest(Path(__file__)),
                author_misc_sha256=digest(ROOT/'external/Lean-SAM2-token-window/sam2/utils/misc.py'),
                author_load_video_frames=load_video_frames, author_decoder=AUTHOR._load_img_as_tensor,
                limit_bytes=2*3*32*32*4):
            pass

def test_pinned_predictor_access_contract_rejects_changed_behavior(tmp_path):
    source = ROOT/'external/Lean-SAM2-token-window/sam2/sam2_video_predictor_topk_insurance.py'
    proof = bounded.verify_predictor_image_access(source, digest(source))
    assert proof['integer_frame_index_reads'] == 1 and proof['cpu_frame_mutation_by_reader'] is False
    altered = tmp_path/'predictor.py'
    altered.write_text(source.read_text().replace('inference_state["images"][frame_idx].to(device)',
        'inference_state["images"][frame_idx]', 1))
    with pytest.raises(ValueError, match='does not copy'):
        bounded.verify_predictor_image_access(altered, digest(altered))


def test_loader_rejects_access_from_non_owner_thread(tmp_path):
    import threading
    paths = make_frames(tmp_path, count=2)
    loader = BoundedEagerFrameLoaderV2(paths, [digest(p) for p in paths], 32,
        AUTHOR._load_img_as_tensor, limit_bytes=3*32*32*4)
    errors = []
    def access():
        try: loader[0]
        except ValueError as error: errors.append(str(error))
    thread = threading.Thread(target=access); thread.start(); thread.join()
    assert len(errors) == 1 and 'different thread' in errors[0]


def test_scoped_hook_rejects_same_file_wrong_decoder(tmp_path, monkeypatch):
    paths = make_frames(tmp_path, count=2); model = FakeSWR()
    monkeypatch.setattr(bounded.threading, 'active_count', lambda: 1)
    monkeypatch.setattr(bounded, 'verify_predictor_image_access', lambda *args: {})
    with pytest.raises(ValueError, match='author loader binding'):
        with install_bounded_eager_loader(model, staged_dir=tmp_path, expected_paths=paths,
                expected_sha256=[digest(p) for p in paths], author_module_file=__file__,
                author_module_sha256=digest(Path(__file__)),
            wrapper_module_file=__file__, wrapper_module_sha256=digest(Path(__file__)),
                author_misc_sha256=digest(ROOT/'external/Lean-SAM2-token-window/sam2/utils/misc.py'),
                author_load_video_frames=load_video_frames,
                author_decoder=AUTHOR.load_video_frames_from_jpg_images,
                limit_bytes=2*3*32*32*4):
            pass


def test_actual_swr_wrapper_resolves_base_without_bypassing_reset():
    import subprocess
    import sys
    code = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path('external/Lean-SAM2-token-window').resolve()))
from sam2.sam2_video_predictor_swr_wrapper import SAM2VideoPredictorWithSWR
from sam2.utils import misc
from modeling.bounded_sam2_video_loader_v2 import resolve_swr_init_chain
from scripts.memory_token_io import digest
root = Path('external/Lean-SAM2-token-window/sam2')
wrapper = root/'sam2_video_predictor_swr_wrapper.py'
base = root/'sam2_video_predictor_topk_insurance.py'
globals_ = resolve_swr_init_chain(SAM2VideoPredictorWithSWR, wrapper_file=wrapper,
    wrapper_sha256=digest(wrapper), predictor_file=base, predictor_sha256=digest(base))
assert globals_['load_video_frames'] is misc.load_video_frames
print('verified actual SWR reset-then-super chain')
'''
    output = subprocess.check_output([sys.executable, '-c', code], cwd=ROOT, text=True)
    assert 'verified actual SWR reset-then-super chain' in output


def test_wrapper_that_skips_swr_reset_is_rejected():
    class BadSWR(FakePredictor):
        def init_state(self, *args, **kwargs):
            return super().init_state(*args, **kwargs)
    source = Path(__file__); sha = digest(source)
    with pytest.raises(ValueError, match='reset runtime'):
        bounded.resolve_swr_init_chain(BadSWR, wrapper_file=source, wrapper_sha256=sha,
                                       predictor_file=source, predictor_sha256=sha)
