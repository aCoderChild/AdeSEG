"""Mirror SAM2's native spatial-memory frame selection and key spans."""


def selected_memory_frames(model, frame_idx, output_dict, num_frames,
                           track_in_reverse=False):
    """Return nonempty native memory-frame spans in SAM2 concatenation order.

    The temporal selection below follows SAM2's
    ``_prepare_memory_conditioned_features``. Object-pointer tokens are
    appended separately by SAM2 and are intentionally not in these spans.
    """
    from sam2.modeling.sam2_utils import select_closest_cond_frames

    cond_outputs = output_dict["cond_frame_outputs"]
    selected, unselected = select_closest_cond_frames(
        frame_idx, cond_outputs, model.max_cond_frames_in_attn
    )
    t_pos_and_prevs = [(0, int(index), out) for index, out in selected.items()]
    stride = 1 if model.training else model.memory_temporal_stride_for_eval

    for t_pos in range(1, model.num_maskmem):
        t_rel = model.num_maskmem - t_pos
        if t_rel == 1:
            prev_frame_idx = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
        elif track_in_reverse:
            prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
            prev_frame_idx += (t_rel - 2) * stride
        else:
            prev_frame_idx = ((frame_idx - 2) // stride) * stride
            prev_frame_idx -= (t_rel - 2) * stride
        output = output_dict["non_cond_frame_outputs"].get(prev_frame_idx)
        if output is None:
            output = unselected.get(prev_frame_idx)
        t_pos_and_prevs.append((t_pos, int(prev_frame_idx), output))

    spans = []
    cursor = 0
    for t_pos, frame, output in t_pos_and_prevs:
        if output is None:
            continue
        features = output["maskmem_features"]
        tokens = features.shape[-2] * features.shape[-1]
        spans.append({
            "frame": frame,
            "t_pos": int(t_pos),
            "start": cursor,
            "stop": cursor + tokens,
        })
        cursor += tokens
    if not spans:
        raise RuntimeError("SAM2 selected no spatial memory frames")
    return spans
