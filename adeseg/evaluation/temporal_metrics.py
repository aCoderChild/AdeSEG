"""Temporal and sequence evaluation entrypoints.

Lazy imports isolate the optional OpenCV dependency from backbone/adapter
imports and from CPU-only model tests.
"""

def evaluate_sequence(*args, **kwargs):
    from evaluation.temporal import evaluate_sequence as original
    return original(*args, **kwargs)


def evaluate_masks(*args, **kwargs):
    from evaluation.temporal import evaluate_masks as original
    return original(*args, **kwargs)


__all__ = ["evaluate_sequence", "evaluate_masks"]
