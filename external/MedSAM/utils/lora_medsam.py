import torch
import torch.nn as nn
import math

class _LoRALayer(nn.Module):
    def __init__(self, linear, r, alpha):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / r

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling

def apply_lora_to_vit_encoder(vit_encoder, r=8, alpha=16, target_modules=("qkv", "proj")):
    """
    Recursively wraps all nn.Linear layers named in target_modules in the ViT encoder with LoRA layers.
    """
    for name, module in vit_encoder.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            setattr(vit_encoder, name, _LoRALayer(module, r, alpha))
        else:
            apply_lora_to_vit_encoder(module, r, alpha, target_modules)
    return vit_encoder
