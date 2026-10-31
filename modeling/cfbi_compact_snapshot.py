"""Compact recorded reads while preserving native spatial and pointer inputs."""
def compact_training_read(sample, spatial_frames):
    """Keep all pointers and conditioning outputs, plus actually read spatial frames."""
    spatial = set(spatial_frames)
    if not spatial or len(spatial) != len(spatial_frames):
        raise ValueError('Expected distinct native spatial frame IDs')
    outputs = sample['output_dict']
    available = set(outputs['cond_frame_outputs']) | set(outputs['non_cond_frame_outputs'])
    if not spatial <= available or any(i >= sample['frame'] for i in spatial):
        raise ValueError('Spatial frames must be present and strictly past')
    for bucket in ('cond_frame_outputs', 'non_cond_frame_outputs'):
        for frame, value in outputs[bucket].items():
            if frame in spatial and (value.get('maskmem_features') is None or value.get('pred_masks') is None
                                     or not value.get('maskmem_pos_enc')):
                raise ValueError('Selected spatial memory is incomplete')
    result = dict(sample)
    result['positions'] = [p.new_empty(0) for p in sample['positions'][:-1]] + [sample['positions'][-1]]
    result['output_dict'] = {'cond_frame_outputs': dict(outputs['cond_frame_outputs']),
                             'non_cond_frame_outputs': {}}
    for frame, value in outputs['non_cond_frame_outputs'].items():
        if frame in spatial:
            result['output_dict']['non_cond_frame_outputs'][frame] = dict(value)
        else:
            pointer = dict(value)
            pointer['maskmem_features'] = None
            pointer['maskmem_pos_enc'] = None
            pointer['pred_masks'] = None
            result['output_dict']['non_cond_frame_outputs'][frame] = pointer
    return result
