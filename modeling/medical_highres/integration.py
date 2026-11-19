"""Scoped high-resolution skip integration into the native MedSAM2 mask decoder.

No vendored source is edited. Only high_res_features[0] is altered, leaving
the second skip, native prompts and recurrent update procedure unchanged.
"""
from contextlib import contextmanager


def _decoder_inputs(args, kwargs):
    # MaskDecoder.forward: image_embeddings is arg 0, high_res_features arg 6.
    if "image_embeddings" in kwargs:
        deep = kwargs["image_embeddings"]
    elif len(args) > 0:
        deep = args[0]
    else:
        raise RuntimeError("Mask decoder lacks image_embeddings")
    if "high_res_features" in kwargs:
        feats = kwargs["high_res_features"]
        location = "keyword"
    elif len(args) >= 7:
        feats = args[6]
        location = "positional"
    else:
        raise RuntimeError("Mask decoder lacks high_res_features")
    if not isinstance(feats, (list, tuple)) or len(feats) != 2:
        raise RuntimeError("Expected two projected high-resolution skip features")
    return deep, feats, location


@contextmanager
def attach_medical_highres_adapter(predictor, adapter, *, allow_untrained=False,
                                   telemetry=None):
    """Install adapter for one decoder execution scope; restore original afterward.

    A loaded historical checkpoint authenticates *weights*, not the missing
    original forward implementation. See README before efficacy comparisons.
    """
    if telemetry is not None and (not isinstance(telemetry, dict) or telemetry):
        raise ValueError("Adapter telemetry must be an empty dict")
    decoder = predictor.sam_mask_decoder
    if not getattr(decoder, "use_high_res_features", False):
        raise ValueError("Medical adapter requires SAM2 high-resolution decoder features")
    if not allow_untrained and adapter.checkpoint_sha256 is None:
        raise RuntimeError("A strictly verified trained medical adapter is required")
    if getattr(decoder, "_medical_highres_adapter_active", False):
        raise RuntimeError("Medical high-resolution adapter already installed")
    decoder._medical_highres_adapter_active = True

    def prehook(_module, args, kwargs):
        deep, highres, location = _decoder_inputs(args, kwargs)
        early = highres[0]
        refined = adapter(early, deep, early)
        if telemetry is not None:
            telemetry["calls"] = telemetry.get("calls", 0) + 1
            changed = bool((refined != early).any().item())
            telemetry["nonzero_calls"] = telemetry.get("nonzero_calls", 0) + int(changed)
        updated = [refined, highres[1]]
        kwargs = dict(kwargs)
        args = tuple(args)
        if location == "keyword":
            kwargs["high_res_features"] = updated
        else:
            args = args[:6] + (updated,) + args[7:]
        return args, kwargs

    try:
        hook = decoder.register_forward_pre_hook(prehook, with_kwargs=True)
    except BaseException:
        del decoder._medical_highres_adapter_active
        raise
    try:
        yield
    finally:
        hook.remove()
        del decoder._medical_highres_adapter_active
