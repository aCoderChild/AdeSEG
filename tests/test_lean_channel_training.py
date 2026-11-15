import torch
from modeling.lean_channel_training import build_trainable_gate,train_gate,parameter_hash


def test_training_controls_share_initialization_and_update_deterministically():
    torch.manual_seed(14);samples=[];foreign={}
    for n in range(3):
        seq=f'seq{n}';fg=torch.nn.functional.normalize(torch.randn(32),dim=0);bg=torch.nn.functional.normalize(torch.randn(32),dim=0)
        samples.append({'sequence':seq,'feature':torch.randn(32,8,8),'foreground':fg,'background':bg,'target':torch.linspace(0,1,4096)})
    for i,s in enumerate(samples):
        other=samples[(i+1)%len(samples)];foreign[s['sequence']]=(other['foreground'],other['background'])
    models=[build_trainable_gate(45) for _ in range(2)]
    assert parameter_hash(models[0])==parameter_hash(models[1])
    results=[train_gate(m,samples,'prompt_conditioned',foreign,2,.0003,.0001) for m in models]
    assert results[0]==results[1] and results[0]['steps']==6 and results[0]['initial_parameter_sha256']!=results[0]['final_parameter_sha256']
    constant=build_trainable_gate(45);summary=train_gate(constant,samples,'descriptor_independent',foreign,1,.0003,.0001)
    assert summary['steps']==3 and summary['initial_parameter_sha256']==results[0]['initial_parameter_sha256']
