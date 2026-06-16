from __future__ import annotations

from torch import nn


def set_reliability_score_finetune_mode(model: nn.Module) -> list[nn.Parameter]:
    """
    Fine-tune only the MedSAM-feature-to-QDMN projection before QDMN Score.

    Everything else stays frozen, including:
    - MedSAM2
    - PromptMemoryAttention / memory adapter
    - QDMN score.*
    """

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model.reliability_estimator, "project"):
        trainable_module = model.reliability_estimator.project
    else:
        raise TypeError(
            "Reliability-score fine-tuning requires QDMNScoreReliabilityEstimator "
            "with a trainable `.project` layer."
        )

    trainable_params = []
    trainable_module.train()
    for param in trainable_module.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    model.image_encoder.eval()
    model.prompt_encoder.eval()
    model.mask_decoder.eval()
    model.prompt_memory_attention.eval()
    model.reliability_estimator.score.eval()
    return trainable_params
