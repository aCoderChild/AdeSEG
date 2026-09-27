"""Shared native object-pointer handling for single-state memory."""

import torch

from MedSAM2.sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames


def append_native_object_pointers(
    model,
    frame_idx,
    output_dict,
    num_frames,
    track_in_reverse,
    device,
    batch_size,
    memory_chunks,
    position_chunks,
):
    """Append SAM2's normal object-pointer tokens to recurrent state memory."""
    if not model.use_obj_ptrs_in_encoder:
        return 0

    selected_cond, unselected_cond = select_closest_cond_frames(
        frame_idx, output_dict["cond_frame_outputs"], model.max_cond_frames_in_attn
    )
    if not model.training and model.only_obj_ptrs_in_the_past_for_eval:
        pointer_conditions = {
            t: output
            for t, output in selected_cond.items()
            if (t >= frame_idx if track_in_reverse else t <= frame_idx)
        }
    else:
        pointer_conditions = selected_cond

    sign = -1 if track_in_reverse else 1
    positions_and_pointers = [
        (
            (frame_idx - t) * sign if model.use_signed_tpos_enc_to_obj_ptrs else abs(frame_idx - t),
            output["obj_ptr"],
        )
        for t, output in pointer_conditions.items()
    ]
    max_pointers = min(num_frames, model.max_obj_ptrs_in_encoder)
    for frame_distance in range(1, max_pointers):
        t = frame_idx + frame_distance if track_in_reverse else frame_idx - frame_distance
        if t < 0 or (num_frames is not None and t >= num_frames):
            break
        output = output_dict["non_cond_frame_outputs"].get(
            t, unselected_cond.get(t)
        )
        if output is not None:
            positions_and_pointers.append((frame_distance, output["obj_ptr"]))
    if not positions_and_pointers:
        return 0

    positions, pointers = zip(*positions_and_pointers)
    pointers = torch.stack(pointers, dim=0).to(device)
    if model.add_tpos_enc_to_obj_ptrs:
        position_dim = model.hidden_dim if model.proj_tpos_enc_in_obj_ptrs else model.mem_dim
        temporal_positions = get_1d_sine_pe(
            torch.tensor(positions, device=device)
            / (max_pointers - 1),
            dim=position_dim,
        )
        pointer_positions = model.obj_ptr_tpos_proj(temporal_positions)
        pointer_positions = pointer_positions.unsqueeze(1).expand(-1, batch_size, model.mem_dim)
    else:
        pointer_positions = pointers.new_zeros(len(positions), batch_size, model.mem_dim)
    if model.mem_dim < model.hidden_dim:
        pointers = pointers.reshape(-1, batch_size, model.hidden_dim // model.mem_dim, model.mem_dim)
        pointers = pointers.permute(0, 2, 1, 3).flatten(0, 1)
        pointer_positions = pointer_positions.repeat_interleave(
            model.hidden_dim // model.mem_dim, dim=0
        )
    memory_chunks.append(pointers)
    position_chunks.append(pointer_positions)
    return pointers.shape[0]
