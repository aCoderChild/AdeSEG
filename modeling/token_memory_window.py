"""Choose a contiguous recent spatial-memory window under a token budget.

This policy does not modify or hook a model. Token counts must describe the
physically stored feature/position pairs supplied by the caller.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpatialMemory:
    frame_index: int
    token_count: int


def select_token_window(condition_tokens, history, budget, *, reserved_tokens=0):
    """Keep the protected conditioning memory and newest fitting history.

    History is in increasing time order. If an older whole entry does not fit,
    stop rather than skip it. ``reserved_tokens`` lets a caller include other
    protected spatial feature entries such as an insurance bank in the same
    budget. Pointer and auxiliary tensors are not counted by this function.
    """
    if type(condition_tokens) is not int or condition_tokens <= 0:
        raise ValueError('Condition token count must be a positive integer')
    if type(reserved_tokens) is not int or reserved_tokens < 0:
        raise ValueError('Reserved token count must be a nonnegative integer')
    if type(budget) is not int or budget < condition_tokens + reserved_tokens:
        raise ValueError('Budget must accommodate protected condition memory')
    if any(not isinstance(x, SpatialMemory) or type(x.token_count) is not int or x.token_count <= 0 or type(x.frame_index) is not int for x in history):
        raise ValueError('Invalid history entry')
    if any(a.frame_index >= b.frame_index for a, b in zip(history, history[1:])):
        raise ValueError('History frame indices must strictly increase')
    retained = []
    used = condition_tokens + reserved_tokens
    for entry in reversed(history):
        if used + entry.token_count > budget:
            if not retained:
                raise ValueError('Budget cannot hold conditioning and newest memory')
            break
        retained.append(entry)
        used += entry.token_count
    retained.reverse()
    return retained, used
