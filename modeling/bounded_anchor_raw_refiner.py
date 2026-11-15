"""Exploratory bounded-logit raw-candidate refiner, not live-video enabled.

A smooth transform reduces extreme native-mask magnitudes without a hard gate.
It changes probabilities and can change masks after interpolation, so it is not
native-equivalent; object-presence score and pointer are not touched. This
module requires separately trained weights and an explicitly verified router.
"""
from pathlib import Path
import torch
from modeling.foreground_conditioned_refiner import _source_sha256
from modeling.lean_channel_attention import author_sha256
from modeling.raw_multimask_refiner import RawMultimaskRefiner


class BoundedAnchorRawRefiner(RawMultimaskRefiner):
    """Use bounded logits in the head and skip; inputs and output are float32."""
    SCALE=16.0

    def forward(self,early,deep,logits,raw_candidates=None,predicted_ious=None):
        if raw_candidates is None or predicted_ious is None:
            raise RuntimeError('Raw multimask inputs are required; the existing live router does not support this model')
        if any(not isinstance(x,torch.Tensor) or x.dtype!=torch.float32 for x in (early,deep,logits,raw_candidates,predicted_ious)):
            raise ValueError('Bounded-anchor inputs must be float32 tensors')
        bounded=self.SCALE*torch.tanh(logits/self.SCALE)
        return super().forward(early,deep,bounded,raw_candidates,predicted_ious)

    def load_trained_state(self,payload,*,backbone_sha256,feature_pipeline_sha256,train_manifest_sha256,raw_manifest_sha256):
        self._trained_checkpoint_loaded=False;self._mask_only=False
        if not isinstance(payload,dict) or payload.get('format')!='bounded_anchor_raw_refiner_v1' or payload.get('research_only') is not True or type(payload.get('training_steps')) is not int or payload['training_steps']<=0:
            raise ValueError('Matching bounded-anchor checkpoint required')
        here=Path(__file__).parent
        files=(('bounded_source_sha256',__file__),('raw_model_sha256',here/'raw_multimask_refiner.py'),
               ('conditioned_source_sha256',here/'foreground_conditioned_refiner.py'),('base_source_sha256',here/'multilevel_contrast_refiner.py'))
        if payload.get('author_sha256')!=author_sha256() or any(payload.get(k)!=_source_sha256(path) for k,path in files):
            raise ValueError('Bounded refiner or parent source changed')
        if payload.get('anchor_scale')!=self.SCALE or payload.get('raw_mode')!=self.raw_mode or payload.get('gate_mode')!=self.gate_mode or payload.get('contrast_head') is not self.contrast_head:
            raise ValueError('Checkpoint settings differ')
        for key,expected in (('backbone_sha256',backbone_sha256),('feature_pipeline_sha256',feature_pipeline_sha256),
                             ('train_manifest_sha256',train_manifest_sha256),('raw_manifest_sha256',raw_manifest_sha256)):
            if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected) or payload.get(key)!=expected:
                raise ValueError(f'Checkpoint provenance mismatch: {key}')
        self.load_state_dict(payload['state_dict'],strict=True)
        if any(not torch.isfinite(p).all() for p in self.parameters()):raise ValueError('Nonfinite checkpoint')
        self._trained_checkpoint_loaded=True
