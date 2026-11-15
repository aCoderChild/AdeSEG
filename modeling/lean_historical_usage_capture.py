"""Read-only capture of the author's existing pre-pruning usage statistic."""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch

@contextmanager
def capture_historical_usage(model, output, observations):
    original=model._dump_retained_tokens_record;seen=set()
    def capture(self, **kwargs):
        source=kwargs['latest_prev'].get('frame_idx');read=kwargs['frame_idx'];usage=kwargs['usage_scores'];keep=kwargs['keep_idx']
        if type(source) is not int or type(read) is not int or source>=read or not torch.is_tensor(usage) or usage.ndim!=1 or usage.numel()!=4096 or not torch.isfinite(usage).all() or not torch.is_tensor(keep) or keep.ndim!=1:
            raise ValueError('Unexpected author usage or source geometry')
        ids=keep.detach().cpu().long();values=usage.detach().cpu().float()
        if len(ids)!=1024 or len(torch.unique(ids))!=len(ids) or (ids<0).any() or (ids>=len(values)).any() or source in seen:
            raise ValueError('Unexpected retained usage identities')
        folder=Path(output)/'historical_usage';folder.mkdir(exist_ok=True)
        path=folder/f'{source:06d}.pt';torch.save({'source_frame':source,'read_frame':read,'usage':values.clone(),'keep_idx':ids.clone()},path)
        observations.append({'source_frame':source,'read_frame':read,'file':path.name});seen.add(source)
        return original(**kwargs)
    model._dump_retained_tokens_record=MethodType(capture,model)
    try:
        yield
    finally:
        model._dump_retained_tokens_record=original
