import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.linear = linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros((r, linear.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((linear.out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = lora_alpha / r

    def forward(self, x):
        result = self.linear(x)
        lora_out = (self.lora_B @ (self.lora_A @ x.transpose(-1, -2))).transpose(-1, -2)
        lora_out = self.lora_dropout(lora_out) * self.scaling
        return result + lora_out

def inject_lora(model, r=8, lora_alpha=16, lora_dropout=0.1):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r, lora_alpha, lora_dropout))
        else:
            inject_lora(module, r, lora_alpha, lora_dropout)
    return model
