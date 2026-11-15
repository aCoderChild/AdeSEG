"""Scoped identity binding for a verifier-owned Lean/refiner runtime.

This is a correctness guard against accidentally mixing model objects,
weights, policies or hooks; it is not a security boundary against arbitrary
Python code in the same process. Only the runtime factory should issue leases.
"""
from modeling.lean_live_state_audit import module_tensor_fingerprints

_FACTORY_ISSUER = object()


def _method_identity(owner, name):
    method = getattr(owner, name)
    return getattr(method, '__func__', method)


class _RuntimeLease:
    """Identity for one live model, one refiner and currently active hooks."""

    def __init__(self, model, refiner, runtime, expected_model_state, expected_refiner_state, *, _issuer,
                 positions=None, selections=None):
        if _issuer is not _FACTORY_ISSUER:
            raise RuntimeError('Only the verifier-owned runtime factory can issue a lease')
        self._model = model
        self._refiner = refiner
        self._runtime = runtime
        self._model_state = expected_model_state
        self._refiner_state = expected_refiner_state
        self._positions = positions
        self._selections = selections
        self._hooks = ((model.memory_attention, 'forward', _method_identity(model.memory_attention, 'forward')),
                       (model, '_prepare_memory_conditioned_features', _method_identity(model, '_prepare_memory_conditioned_features')),
                       (model, '_cache_tarr_affinity_from_new_memory', _method_identity(model, '_cache_tarr_affinity_from_new_memory')),
                       (model, '_dump_retained_tokens_record', _method_identity(model, '_dump_retained_tokens_record')))
        self._active = True

    def validate_live(self, model, refiner, *, tensors):
        if not self._active or model is not self._model or refiner is not self._refiner:
            raise RuntimeError('Runtime lease is inactive or belongs to another predictor/refiner')
        if model.training or refiner.training or model.device.type != 'mps':
            raise RuntimeError('Predictor/refiner evaluation device changed')
        if (getattr(model, 'token_window_runtime', None) is not self._runtime or
                self._runtime.budget != 16384 or self._runtime.insurance_reserve != 3072 or
                model.mem_keep_ratio != .25 or model.num_maskmem != 7 or
                model.ema_history_replace_old_tail_in_attn or model.ema_history_resident_prune):
            raise RuntimeError('Verified memory policy changed')
        if any(_method_identity(owner, name) is not expected for owner, name, expected in self._hooks):
            raise RuntimeError('Verified runtime hook changed')
        if tensors and (module_tensor_fingerprints(model) != self._model_state or
                        module_tensor_fingerprints(refiner) != self._refiner_state):
            raise RuntimeError('Verified model or refiner tensor state changed')

    def close(self):
        self._active = False

    def observations(self):
        if not self._active or self._positions is None or self._selections is None:
            raise RuntimeError('Runtime observations are unavailable')
        return list(self._positions), list(self._selections)
