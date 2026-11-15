"""Reusable deterministic author-channel-gate optimization primitives."""
import hashlib
import torch
from modeling.lean_channel_attention import PromptContrastChannelAttention
from modeling.lean_channel_training_loss import balanced_soft_bce


def parameter_hash(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.gate.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_trainable_gate(seed):
    torch.manual_seed(seed);model=PromptContrastChannelAttention(initialize_for_training=True).train()
    # Preserve the author first-layer random initialization. Zeroing only the
    # final affine layer starts at uniform weights without blocking its gradient.
    torch.nn.init.zeros_(model.gate.mlp[-1].weight);torch.nn.init.zeros_(model.gate.mlp[-1].bias)
    return model


def train_gate(model,samples,arm,foreign_prototypes,epochs,learning_rate,weight_decay):
    """Train one arm; caller verifies provenance and uses fixed protocol values."""
    if arm not in ('prompt_conditioned','descriptor_independent','mismatched_prompt') or not samples or epochs<=0:
        raise ValueError('Invalid training arm or schedule')
    if not model.training or model.checkpoint_metadata is not None:raise ValueError('Expected freshly initialized training gate')
    opt=torch.optim.Adam(model.gate.parameters(),lr=learning_rate,weight_decay=weight_decay)
    initial=parameter_hash(model);history=[];steps=0
    for _ in range(epochs):
        losses=[]
        for sample in samples:
            seq=sample['sequence'];feature=sample['feature'];fg=sample['foreground'];bg=sample['background'];target=sample['target']
            if arm=='prompt_conditioned':gfg,gbg=fg,bg
            elif arm=='descriptor_independent':gfg,gbg=torch.zeros_like(fg),torch.zeros_like(bg)
            else:
                if seq not in foreign_prototypes:raise ValueError('Missing mismatched-sequence descriptor')
                gfg,gbg=foreign_prototypes[seq]
            opt.zero_grad(set_to_none=True)
            score,_=model(feature,fg,bg,gate_foreground=gfg,gate_background=gbg)
            loss=balanced_soft_bce(score,target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if not any(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0 for p in model.gate.parameters()):raise ValueError('No finite training gradient')
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.gate.parameters()):raise ValueError('Nonfinite training gradient')
            opt.step();steps+=1;losses.append(float(loss.detach()))
        history.append(sum(losses)/len(losses))
    final=parameter_hash(model)
    if initial==final or steps!=epochs*len(samples):raise ValueError('Training did not update gate as expected')
    return {'initial_parameter_sha256':initial,'final_parameter_sha256':final,'steps':steps,'epoch_losses':history}
