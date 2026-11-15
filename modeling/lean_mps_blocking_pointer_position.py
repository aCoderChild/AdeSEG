"""Experimental scoped MPS pointer-position transfer workaround.

Rebuilds only the pinned Lean method with its scalar pointer-position transfer
changed to blocking. Other non-blocking transfers are intentionally unchanged;
this is a diagnostic adaptation, not a verified global transfer fix.
"""
from contextlib import contextmanager
import inspect
import textwrap
from types import MethodType


@contextmanager
def blocking_mps_pointer_position(model):
    if model.device.type != 'mps':
        raise ValueError('Blocking pointer-position adaptation requires MPS')
    name='_prepare_memory_conditioned_features'
    if name in vars(model):
        raise RuntimeError('Cannot compose with existing instance-level memory hook')
    old=getattr(model,name)
    original=textwrap.dedent(inspect.getsource(old.__func__))
    needle='obj_pos = torch.tensor(pos_list).to(\n                        device=device, non_blocking=True\n                    )'
    replacement='obj_pos = torch.tensor(pos_list).to(\n                        device=device, non_blocking=False\n                    )'
    if original.count(needle)!=1:
        raise RuntimeError('Lean pointer-position source is not the expected version')
    namespace={}
    exec(compile(original.replace(needle,replacement),old.__func__.__code__.co_filename,'exec'),old.__func__.__globals__,namespace)
    modified=namespace[name]
    try:
        setattr(model,name,MethodType(modified,model))
        yield
    finally:
        if name in vars(model):delattr(model,name)
