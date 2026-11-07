"""Persistent-instance routing for the frozen Run171 diagnostic.

Only ownership/provenance changes. The author selector, budget, group rule and
sparse attention implementation are reused unchanged.
"""
from contextlib import contextmanager

from modeling.token_contribution import TokenContributionRead
from modeling.complementary_memory.integration import _tensor_sha256
from scripts.bank_integrity import fingerprint


def instance_context(state, bank, replay_owner=None):
    """Use native ownership validation, including for private cloned banks."""
    if replay_owner is None:
        owners = [index for index, value in state['output_dict_per_obj'].items() if value is bank]
        if len(owners) != 1:
            raise RuntimeError('Read bank has no unique persistent instance')
        return state
    if replay_owner not in state['obj_id_to_idx']:
        raise ValueError('Replay owner is not a prompted persistent ID')
    index = state['obj_id_to_idx'][replay_owner]
    # The bank here is the private clone actually passed to track_step, not the
    # original sample captured before replay makes its second private copy.
    return {'output_dict_per_obj': {index: bank}, 'obj_idx_to_id': {index: replay_owner}}


class InstanceTokenContributionRead(TokenContributionRead):
    def __init__(self, model, state, video):
        super().__init__(model)
        self.state = state
        self.sequence = video
        self.replay_owner = None
        self.only_owner = None
        self.inference_state = state

    def _prepare(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        bank = bound.arguments['output_dict']
        previous = self.inference_state
        self.inference_state = instance_context(self.state, bank, self.replay_owner)
        try:
            result = super()._prepare(*args, **kwargs)
            if not bound.arguments['is_init_cond_frame']:
                call = self.calls[-1]
                owner = call['object_id']
                if owner not in self.state['obj_id_to_idx']:
                    raise RuntimeError('Read receipt lost its persistent object ID')
                sources = []
                for span in call['memory_frames']:
                    frame = span['frame']
                    output = bank['cond_frame_outputs'].get(frame, bank['non_cond_frame_outputs'].get(frame))
                    sources.append({'object_id': owner, 'frame': frame,
                        'stored_mask_sha256': _tensor_sha256(output['pred_masks'])})
                call['mask_sources'] = sources
            return result
        finally:
            self.inference_state = previous

    def _before_selection(self, layer, module, args, kwargs):
        owner = self.context['object_id']
        if owner not in self.state['obj_id_to_idx']:
            raise RuntimeError('Selection cache lacks a valid persistent owner')
        arm = self.arm
        if self.only_owner is not None and owner != self.only_owner and arm != 'collect':
            self.arm = 'native'
        try:
            super()._before_selection(layer, module, args, kwargs)
        finally:
            self.arm = arm
        self.selection_layers[-1]['object_id'] = owner

    @contextmanager
    def replaying(self, object_id):
        if self.replay_owner is not None or object_id not in self.state['obj_id_to_idx']:
            raise ValueError('Unknown or nested instance replay')
        self.replay_owner = object_id
        try:
            yield self
        finally:
            self.replay_owner = None

    def cache_signature(self, object_id=None):
        caches = {name: {key: value for key, value in getattr(self, name).items()
                         if object_id is None or key[0] == object_id}
                  for name in ('masks', 'anchors', 'usage')}
        return fingerprint(caches)['sha256']
