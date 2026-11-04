"""Read-only snapshots at the native encoder, memory and decoder interfaces."""
import torch


class NativeStageCapture:
    def __init__(self, model):
        self.model = model
        self.active = False
        self.tensors = {}
        self.layers = []

    def begin(self, active):
        self.active = active
        self.tensors = {}
        self.layers = []

    def memory_input(self, module, args, kwargs):
        if self.active:
            curr = kwargs['curr']
            self.tensors['current_image'] = (curr[0] if isinstance(curr, list) else curr).detach().clone()

    def memory_output(self, module, args, output):
        if self.active:
            self.tensors['memory_conditioned'] = output.detach().clone()

    def decoder_input(self, module, args, kwargs):
        if not self.active:
            return
        image = kwargs['image_embeddings']
        expected = self.tensors['memory_conditioned'].permute(1, 2, 0).reshape_as(image)
        if not torch.equal(image, expected):
            raise RuntimeError('Decoder input differs from native memory output')
        self.tensors['high_res_features'] = [t.detach().clone() for t in kwargs['high_res_features']]

    def decoder_output(self, module, args, output):
        if self.active:
            masks, ious, tokens, score = output
            self.tensors.update(raw_masks=masks.detach().clone(), predicted_ious=ious.detach().clone(),
                mask_tokens=tokens.detach().clone(), object_score=score.detach().clone())

    def __enter__(self):
        self.handles = [
            self.model.memory_attention.register_forward_pre_hook(self.memory_input, with_kwargs=True),
            self.model.memory_attention.register_forward_hook(self.memory_output),
            self.model.sam_mask_decoder.register_forward_pre_hook(self.decoder_input, with_kwargs=True),
            self.model.sam_mask_decoder.register_forward_hook(self.decoder_output)]
        for i, layer in enumerate(self.model.memory_attention.layers):
            def hook(module, args, kwargs, output, index=i):
                self.layers.append({'layer': index, 'queries': kwargs['q'].shape[1],
                    'keys': kwargs['k'].shape[1], 'pointer_keys': kwargs.get('num_k_exclude_rope', 0)})
            self.handles.append(layer.cross_attn_image.register_forward_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
