"""Read-only native replay and attribution helpers; no segmentation modification."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


def diagnostic_target_plan(prompt, length, offsets, allow_short_sequences=False):
    """Resolve coverage before inference; unavailable targets are never silent."""
    if not 0 <= prompt < length or not offsets:
        raise ValueError('Invalid prompt, video length or empty target schedule')
    if any(type(n) is not int or n <= 0 for n in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError('Target offsets must be distinct positive integers')
    available = [prompt + n for n in offsets if prompt + n < length]
    omitted = [n for n in offsets if prompt + n >= length]
    if omitted and not allow_short_sequences:
        raise ValueError(f'Target offsets exceed available video frames: {omitted}')
    if not available:
        raise ValueError('No post-prompt targets are available')
    return {'frames': sorted(available), 'unavailable_offsets': omitted,
            'prompt': prompt, 'available_video_frames': length}


def detached_copy(value):
    # Called from inference-mode hooks: clone outside inference mode so replay
    # inputs can subsequently participate in autograd without changing SAM2.
    if isinstance(value, torch.Tensor):
        with torch.inference_mode(False):
            return value.detach().clone()
    if isinstance(value, dict):
        return {k: detached_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detached_copy(v) for v in value)
    return value


def orient_partition(partition, features, foreground_prototype, background_prototype):
    """Orient a binary partition from initial-prompt prototypes, never target GT."""
    labels = torch.as_tensor(partition, dtype=torch.bool).flatten()
    if labels.numel() != features.shape[0] or labels.all() or not labels.any():
        raise ValueError("Partition must contain two nonempty spatial groups")
    prototypes = torch.stack([foreground_prototype, background_prototype])
    if (prototypes.norm(dim=-1) <= 1e-12).any():
        raise ValueError("Prompt prototype is undefined")
    similarities = F.normalize(features.float(), dim=-1) @ F.normalize(prototypes.float(), dim=-1).T
    evidence = similarities[:, 0] - similarities[:, 1]
    margin = evidence[labels].mean() - evidence[~labels].mean()
    if not torch.isfinite(margin) or margin.abs() <= 1e-8:
        raise ValueError("Prompt cannot orient this partition")
    return (labels if margin > 0 else ~labels).reshape(partition.shape).numpy(), float(margin)


def ranking_metrics(scores, target):
    """Threshold-free membership diagnostic; constant/one-class cases explicit."""
    scores = np.asarray(scores).ravel()
    target = np.asarray(target, dtype=bool).ravel()
    if scores.shape != target.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid score/target arrays")
    if not target.any() or target.all():
        return {"status": "one_class_target", "auroc": None, "average_precision": None}
    return {"status": "constant_score" if np.ptp(scores) == 0 else "ok",
            "auroc": float(roc_auc_score(target, scores)),
            "average_precision": float(average_precision_score(target, scores))}


class NativeReadCapture:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.memory = self.decoder = self.outputs = None
        self.layers = []

    def __enter__(self):
        def memory(module, args, kwargs):
            self.memory = detached_copy(kwargs)
        def decoder(module, args, kwargs):
            self.decoder = detached_copy(kwargs)
        def outputs(module, args, output):
            self.outputs = detached_copy(output)
        self.handles = [
            self.model.memory_attention.register_forward_pre_hook(memory, with_kwargs=True),
            self.model.sam_mask_decoder.register_forward_pre_hook(decoder, with_kwargs=True),
            self.model.sam_mask_decoder.register_forward_hook(outputs)]
        for index, layer in enumerate(self.model.memory_attention.layers):
            def record(module, args, output, i=index):
                self.layers.append(i)
            self.handles.append(layer.register_forward_hook(record))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def replay(model, capture):
    """Return raw decoder outputs and differentiable memory leaf after parity."""
    if capture.layers != [0, 1, 2, 3]:
        raise RuntimeError("Native capture did not observe four memory layers")
    kwargs = detached_copy(capture.memory)
    leaf = kwargs['memory'].requires_grad_(True)
    output = model.memory_attention(**kwargs)
    decoder = detached_copy(capture.decoder)
    image = output.permute(1, 2, 0).reshape_as(decoder['image_embeddings'])
    torch.testing.assert_close(image, decoder['image_embeddings'], rtol=1e-6, atol=1e-6)
    decoder['image_embeddings'] = image
    actual = model.sam_mask_decoder(**decoder)
    for expected, observed in zip(capture.outputs, actual):
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
    return actual, leaf
