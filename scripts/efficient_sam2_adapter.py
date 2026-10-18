"""Configuration and read-only measurements for the device-ported author release.

Sources: Efficient-SAM2 tools/vos_inference_main.py:set_predictor and
build_model/window_bypass.py:build_WB_model. No selection/attention algorithm
is implemented here. Run in a fresh process; the repository owns `sam2`.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

UPSTREAM_REVISION = "8e1ad06089b9c958ff5ebfc72b9b7d0b904f5ce5"


def configure(predictor, sparse):
    """Use author classes with SWR disabled and no encoder blocks replaced."""
    from build_model.window_bypass import build_WB_model

    settings = dict(
        apply_WB=True, disable_WB=True, apply_bypass=False,
        sam2_model="base+", selected_layers=[], win_sel_layer=[],
        fpn_feat_layer=[], WB_all_layer=False, small_bypass=False,
        Mem_filter=False, prune_memory=True, pool_memory=False,
        random_mask=False, uniform_mask=False, topk_mask=True,
        set_drop_ratio=0.95 if sparse else 0.0, MTP_theta=0.5, print_WS=False,
        scale_layer=[], Mem_Frame_Prune=False, num_frame_to_prune=0,
        WB_theta=0.7, obj_theta=5, sel_max_iou=False,
    )
    for name, value in settings.items():
        setattr(predictor, name, value)
    predictor.memory_temporal_stride_for_eval = 1
    predictor.dilate_mask = False
    predictor.init_memory_info(enable_MeP_info=True)
    build_WB_model(SimpleNamespace(**settings), predictor, selected_layers=[])
    predictor.time_log = {}
    predictor.FW_time_log = {}
    predictor.WS_log = {"Sel": [], "Mask": [], "PF": []}
    for module in predictor.modules():
        module.FW_time_log = []
    # vos_inference_wj initializes this before native sparse reads.
    for layer in predictor.memory_attention.layers:
        layer.cross_attn_image.drop_ratio_log = []
    predictor.image_encoder.trunk.FW_time_log = {}
    if predictor.image_encoder.trunk.selected_layers:
        raise RuntimeError("SMR-only must not replace encoder blocks")
    if not predictor.disable_WB or predictor.apply_bypass:
        raise RuntimeError("Encoder routing/bypass unexpectedly enabled")
    return settings


def selection_rows(layer_masks, categories):
    """Describe native retained indices; never select or modify tokens."""
    from modeling.region_tokens import CATEGORIES

    categories = np.asarray(categories)
    if categories.shape != (64, 64) or not np.isin(categories, range(4)).all():
        raise ValueError("Expected the declared 64x64 four-region taxonomy")
    rows = []
    for layer, mask in enumerate(layer_masks):
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != (4096,):
            raise ValueError("Native selection must contain exactly 4096 positions")
        selected = selected.reshape(64, 64)
        counts = {}
        for index, name in enumerate(CATEGORIES):
            support = categories == index
            total, retained = int(support.sum()), int((support & selected).sum())
            counts[name] = dict(available=total, retained=retained,
                                retention_fraction=retained / total if total else None)
        rows.append(dict(layer=layer, retained=int(selected.sum()),
                         retained_indices=np.flatnonzero(selected).tolist(), regions=counts))
    return rows


def save_overlay(image_path, categories, selected_indices, destination):
    """Image | mask-defined regions | native retained locations. Display only."""
    palette = np.array([[255, 180, 0], [0, 200, 100], [30, 170, 255], [145, 95, 190]], np.uint8)
    with Image.open(image_path) as source:
        original = source.convert("RGB")
        rgb = np.asarray(original)
        color = Image.fromarray(palette[np.asarray(categories)]).resize(original.size, Image.Resampling.NEAREST)
        region = Image.blend(original, color, 0.45)
        selected = np.zeros(4096, np.uint8)
        selected[np.asarray(selected_indices, dtype=int)] = 255
        support = np.asarray(Image.fromarray(selected.reshape(64, 64)).resize(original.size, Image.Resampling.NEAREST)) > 0
        retained = (rgb * 0.2).astype(np.uint8)
        retained[support] = rgb[support]
        canvas = Image.new("RGB", (original.width * 3, original.height))
        for index, panel in enumerate((original, region, Image.fromarray(retained))):
            canvas.paste(panel, (index * original.width, 0))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
