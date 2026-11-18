"""Fingerprint a live module's parameters *and* buffers for engineering audits.

These hashes are evidence of state consistency, not authorization to run a
checkpoint: they do not describe image inputs, mutable non-tensor state, hooks,
or runtime behavior.  The caller must separately verify those components.
"""
import hashlib

import torch


def module_tensor_fingerprints(module):
    """Return deterministic content hashes for every named parameter/buffer."""
    if not isinstance(module, torch.nn.Module):
        raise TypeError('Expected a PyTorch module')
    result = {}
    for kind, values in (('parameter', module.named_parameters()),
                         ('buffer', module.named_buffers())):
        for name, value in values:
            if value.layout != torch.strided:
                raise ValueError(f'Unsupported {kind} layout: {name}')
            # Viewing as bytes supports bfloat16 and complex tensors without
            # lossy dtype conversion.  A CPU copy also synchronizes MPS.
            raw = value.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8)
            result[f'{kind}:{name}'] = {
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'sha256': hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
            }
    return result


def assert_module_tensor_fingerprints(module, expected):
    actual = module_tensor_fingerprints(module)
    if actual != expected:
        keys = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError(f'Model parameters or buffers changed: {keys[:8]}')
