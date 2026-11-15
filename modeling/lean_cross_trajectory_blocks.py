"""Index-matched block swaps for frozen memory read diagnostics."""
import torch
from modeling.lean_cross_trajectory_swap import swap_nonconditioning_spatial

BLOCKS = {'first': (4096, 7168), 'middle': (7168, 8192), 'last': (8192, 12288)}

def swap_block(recipient, donor, block):
    # Reuse all query, geometry, conditioning and pointer checks.
    full = swap_nonconditioning_spatial(recipient, donor)
    start, end = BLOCKS[block]
    ptr = recipient['num_obj_ptr_tokens']
    if len(recipient['memory']) - ptr != 12288:
        raise ValueError('Unexpected spatial key count')
    result = dict(recipient)
    for key in ('memory', 'memory_pos'):
        original = recipient[key]
        result[key] = torch.cat((original[:start], full[key][start:end], original[end:]), dim=0)
        if not torch.equal(result[key][:start], original[:start]) or not torch.equal(result[key][end:], original[end:]):
            raise ValueError('Outside-block values changed')
    return result
