from torch import nn

from modeling.cfbi_region_qk_training import enable_memory_qk_training


class _CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn_image = _CrossAttention()
        self.feed_forward = nn.Linear(8, 8)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_attention = nn.Module()
        self.memory_attention.layers = nn.ModuleList(_Layer() for _ in range(4))


def test_only_memory_cross_attention_query_and_key_projections_are_trainable():
    model = _Model()
    names, parameters = enable_memory_qk_training(model)

    assert len(names) == len(parameters) == 16
    assert all(
        parameter.requires_grad
        for layer in model.memory_attention.layers
        for projection in (layer.cross_attn_image.q_proj, layer.cross_attn_image.k_proj)
        for parameter in projection.parameters()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".q_proj." not in name and ".k_proj." not in name
    )


def test_qk_parameter_names_are_explicit_and_layer_specific():
    model = _Model()
    names, _ = enable_memory_qk_training(model)
    assert len(set(names)) == 16
    assert {name.split(".layers.", 1)[1].split(".", 1)[0] for name in names} == {
        "0", "1", "2", "3"
    }
    assert all("cross_attn_image." in name for name in names)
