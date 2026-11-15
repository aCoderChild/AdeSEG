import numpy as np
import pytest
import torch
from PIL import Image

from datasets.video import ManifestFrame
from modeling.prompt_feature_refinement import reference_prototypes, select_points
from scripts.run_prompt_feature_refinement import automatic_reference, box_iou, choose_cases, gt_box
from scripts.audit_prompt_feature_refinement import audit_mask
from scripts.memory_token_io import digest


def test_target_and_background_prototypes_and_coordinate_mapping():
    f = torch.zeros(2, 8, 8)
    f[0] = 1
    f[:, 2:4, 2:4] = torch.tensor([0., 1.])[:, None, None]
    mask = np.zeros((80, 80), np.uint8)
    mask[20:40, 20:40] = 1
    pos, neg = reference_prototypes(f, mask)
    assert torch.allclose(pos, torch.tensor([0., 1.]))
    assert torch.allclose(neg, torch.tensor([1., 0.]))
    out = select_points(f, pos, neg, [10, 10, 70, 70], (80, 80), 17)
    assert out['feature_margins'][0] > out['feature_margins'][1]
    assert all(10 <= x < 70 and 10 <= y < 70 for x, y in out['feature_points'])
    assert out == select_points(f, pos, neg, [10, 10, 70, 70], (80, 80), 17)


def test_unusable_reference_and_tiny_box_are_explicit():
    f = torch.ones(2, 8, 8)
    with pytest.raises(ValueError, match='foreground'):
        reference_prototypes(f, np.zeros((10, 10)))
    assert select_points(f, torch.ones(2), torch.ones(2), [0, 0, 1, 1], (80, 80), 1) is None


def test_ground_truth_boxes_and_frame_selection(tmp_path):
    frames = []
    for i in range(7):
        mask = np.zeros((5, 5), np.uint8)
        if i not in (0, 3):
            mask[1:3, 2:5] = 1
        image_path, mask_path = tmp_path / f'{i}.jpg', tmp_path / f'{i}.png'
        Image.new('RGB', (5, 5)).save(image_path)
        Image.fromarray(mask).save(mask_path)
        frames.append(ManifestFrame('s', i, image_path, mask_path))
    prompt, indices = choose_cases(frames)
    assert (prompt, indices) == (1, [2, 3, 5, 6])
    gt = np.zeros((5, 5), bool)
    gt[1:3, 2:5] = True
    assert gt_box(gt) == [2., 1., 5., 3.]
    assert box_iou([2, 1, 5, 3], gt_box(gt)) == 1
    assert box_iou(None, gt_box(gt)) == 0


def test_mask_audit_detects_wrong_dice_and_false_alarm(tmp_path):
    path = tmp_path / 'prediction.png'
    Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(path)
    truth = np.array([[True, True], [False, False]])
    expected = {'mask_sha256': digest(path), 'dice': 2/3, 'false_alarm': None}
    assert audit_mask(path, truth, expected) == pytest.approx(2/3)
    expected['dice'] = .1
    with pytest.raises(ValueError, match='Dice mismatch'):
        audit_mask(path, truth, expected)
    expected.update(dice=0., false_alarm=False)
    with pytest.raises(ValueError, match='False-alarm mismatch'):
        audit_mask(path, np.zeros_like(truth), expected)


def test_automatic_reference_no_detection_does_not_invent_prompt():
    class Predictor:
        def set_image(self, image):
            self.image_shape = image.shape
    class Detector:
        def predict(self, **kwargs):
            return [type('Result', (), {'boxes': None})()]
    image = np.zeros((7, 9, 3), dtype=np.uint8)
    box, conf, mask, positive, negative, status = automatic_reference(Predictor(), Detector(), image)
    assert box is None and conf is None and not mask.any()
    assert positive is None and negative is None and status == 'no_detection'
