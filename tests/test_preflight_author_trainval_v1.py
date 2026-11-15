"""Real archived-input regression: no outcome metadata or labels at preflight."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import preflight_author_trainval_v1 as preflight


def test_real_trainval_prompt_does_not_parse_outcomes_or_open_labels():
    checkpoint = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt')
    detector = preflight.ROOT/'checkpoints/polypgen_yolov8n.pt'
    if not checkpoint.exists() or not detector.exists() or not preflight.TRAIN_PREFLIGHT.exists():
        pytest.skip('Original frozen local input receipts are unavailable')
    # Audit hooks persist; perform this real-source check in a child process.
    code = '''
from pathlib import Path
from scripts import preflight_author_trainval_v1 as p
from scripts.source_group_label_guard_v1 import install_mask_open_guard
old=p.read
def guarded(path):
    if Path(path).name=='records.json': raise AssertionError('outcome records were deserialized')
    return old(path)
p.read=guarded
guard=install_mask_open_guard(p.DATA)
frames,start,mask,provenance=p.trainval_inputs('seq2',Path(__import__('sys').argv[1]),Path(__import__('sys').argv[2]))
assert len(frames)>start and mask.any() and provenance['prompt_mask_sha256']
assert guard['open_attempts']==0
'''
    subprocess.run([sys.executable, '-c', code, str(checkpoint), str(detector)], check=True)
