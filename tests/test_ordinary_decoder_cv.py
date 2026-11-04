import json
from pathlib import Path
import pytest
import torch
from torch import nn

from scripts.run_ordinary_decoder_cv import restore_core, schedule


def parent():
    return json.loads((Path(__file__).resolve().parents[1]/
        'protocols/encoder_decoder_bottleneck_v1.json').read_text())


def test_training_schedule_excludes_each_validation_fold():
    protocol=parent()
    seeds=set()
    for fold,validation in enumerate(protocol['ordinary_baseline']['folds'],1):
        updates=schedule(protocol,fold)
        assert len(updates)==560 and updates==schedule(protocol,fold)
        assert {name for epoch,name,seed in updates}==set(protocol['sequences'])-set(validation)
        for epoch in range(40):
            assert len({name for e,name,s in updates if e==epoch})==14
        seeds.update(seed for epoch,name,seed in updates)
    assert len(seeds)==1680


def test_schedule_rejects_duplicate_validation_sequence():
    protocol=parent()
    protocol['ordinary_baseline']['folds'][0][0]='seq3'
    with pytest.raises(RuntimeError,match='partition'):
        schedule(protocol,1)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=nn.Linear(1,1,bias=False)
        self.conv_s0=nn.Linear(1,1,bias=False)
        self.conv_s1=nn.Linear(1,1,bias=False)


def model():
    m=nn.Module()
    m.image_encoder=nn.Linear(1,1,bias=False)
    m.sam_mask_decoder=Decoder()
    return m


def test_restore_updates_only_exact_decoder_core():
    m=model()
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    restore_core(m,{'sam_mask_decoder.core.weight':torch.full((1,1),3.)})
    for name,p in m.named_parameters():
        if name=='sam_mask_decoder.core.weight':
            assert p.item()==3 and p.requires_grad
        else:
            assert torch.equal(p,before[name]) and not p.requires_grad
    assert not m.training and not m.image_encoder.training


def test_restore_rejects_extra_scope_and_nonfinite_weights():
    m=model()
    with pytest.raises(RuntimeError,match='scope'):
        restore_core(m,{'image_encoder.weight':torch.ones(1,1)})
    with pytest.raises(RuntimeError,match='Invalid'):
        restore_core(m,{'sam_mask_decoder.core.weight':torch.full((1,1),float('nan'))})
