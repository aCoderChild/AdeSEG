"""Exercise SASVi's end-of-video confirmation in its released control loop."""
import inspect

import numpy as np
from PIL import Image
import torch

from scripts.run_sasvi_cmsa_polypgen import load_author


class Predictor:
    def __init__(self, count):
        self.count = count
        self.resets = 0
        self.start = 0

    def init_state(self, **kwargs):
        return {'video_height': 8, 'video_width': 8, 'point_inputs_per_obj': {}, 'mask_inputs_per_obj': {}}

    def reset_state(self, inference_state):
        self.resets += 1
        inference_state['point_inputs_per_obj'].clear()
        inference_state['mask_inputs_per_obj'].clear()

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        self.start = frame_idx
        inference_state['mask_inputs_per_obj'][obj_id] = mask

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, **kwargs):
        self.start = frame_idx
        inference_state['point_inputs_per_obj'][obj_id] = True

    def propagate_in_video(self, inference_state):
        for idx in range(self.start, self.count):
            yield idx, [0], torch.zeros((1, 1, 8, 8))


class Overseer:
    def __init__(self, present):
        self.present = set(present)

    def get_prediction(self, paths, reshape_size):
        return {1: np.ones((8, 8), dtype=bool)} if int(paths[0].split('/')[-1].split('.')[0]) in self.present else {}


def run_case(tmp_path, monkeypatch, count, present):
    video = tmp_path / 'input' / 'video'
    video.mkdir(parents=True)
    for i in range(count):
        (video / f'{i:06d}.jpg').touch()
    author = load_author()
    monkeypatch.setattr(author, 'save_masks_to_dir', lambda *args, **kwargs: None)
    monkeypatch.setattr(author, 'get_points_from_mask', lambda mask, label_ids: {label_ids[0]: [[2, 2]]})
    predictor = Predictor(count)
    with torch.inference_mode():
        inspect.unwrap(author.sasvi_inference)(predictor, str(tmp_path / 'input'), str(tmp_path / 'out'), 'video',
            'test', None, Overseer(present), None, 2, [0], False, [], 'POLYPGEN')
    return predictor.resets


def test_final_frame_detection_has_no_future_confirmation(tmp_path, monkeypatch):
    assert run_case(tmp_path, monkeypatch, 5, [4]) == 1


def test_detection_with_three_confirming_frames_reprompts(tmp_path, monkeypatch):
    assert run_case(tmp_path, monkeypatch, 5, [1, 2, 3, 4]) > 1


def test_sasvi_loader_uses_full_numeric_frame_index(tmp_path):
    load_author()
    from sam2.utils.misc import load_video_frames_from_jpg_images
    for i in range(21):
        Image.fromarray(np.full((8, 8, 3), i * 10, dtype=np.uint8)).save(tmp_path / f'{i:06d}.jpg')
    images, _, _ = load_video_frames_from_jpg_images(str(tmp_path), 8, True,
        img_mean=(0, 0, 0), img_std=(1, 1, 1), async_loading_frames=True, compute_device=torch.device('cpu'))
    assert [int(path.split('/')[-1].split('.')[0]) for path in images.img_paths] == list(range(21))
    assert all(abs(float(images[i].mean()) - i * 10 / 255) < 0.005 for i in range(21))


def test_sasvi_point_sampler_handles_small_and_empty_masks():
    author = load_author()
    mask = np.zeros((8, 8), dtype=bool)
    mask[3, 5] = True
    assert author.get_points_from_mask({1: mask}, label_ids=[1], num_points=20) == {1: [[5, 3]]}
    mask[3, 5] = False
    assert author.get_points_from_mask({1: mask}, label_ids=[1], num_points=20) == {1: []}
