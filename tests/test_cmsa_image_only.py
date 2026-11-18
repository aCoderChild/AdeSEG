"""Independent CMSA video loading must never read target mask contents."""
from pathlib import Path
import torch
from PIL import Image

from scripts.run_cmsa_image_only import ImageOnlyCMSADataset


class TrivialTransforms:
    def __call__(self, images, labels):
        # In the real author pipeline images and dummy labels are transformed
        # separately. This test proves no mask file is needed for inference.
        image_tensors = []
        label_tensors = []
        for img, label in zip(images, labels):
            image_tensors.append(torch.tensor(list(img.getdata())).reshape(
                img.height, img.width, 3).permute(2, 0, 1).float())
            label_tensors.append(torch.zeros(1, label.height, label.width))
        return image_tensors, label_tensors


def test_inference_dataset_does_not_open_missing_mask_files(tmp_path):
    img = tmp_path / "rgb.png"
    Image.new("RGB", (8, 6), (12, 25, 50)).save(img)
    missing_gt = tmp_path / "gt_does_not_exist.png"
    source = type("SourceDataset", (), {})()
    source.video_train_list = [[(str(img), str(missing_gt))] * 4]
    source.img_label_transform = TrivialTransforms()
    actual = ImageOnlyCMSADataset(source)
    images, targets, locations = actual[0]
    assert images.shape == (4, 3, 6, 8)
    assert targets.shape == (4, 1, 6, 8)
    assert targets.count_nonzero() == 0
    assert locations[0][1] == str(missing_gt)
