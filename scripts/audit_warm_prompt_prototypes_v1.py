"""Recompute saved warm prompt prototypes from archived mask and early features."""
from pathlib import Path
import torch
from modeling.prompt_feature_refinement import reference_prototypes
from scripts.crossfit_observer_capture_shared_v1 import require
from scripts.memory_token_io import digest


def validate_prompt_prototypes(folder,rec,mask):
    folder=Path(folder).resolve(strict=True);early_file=folder/'prompt_early.pt';prototypes_file=folder/'prototypes.pt'
    require(digest(early_file)==rec['prompt_early_sha256'] and digest(prototypes_file)==rec['prototypes_sha256'],
            'Warm prompt feature or prototypes changed')
    early_record=torch.load(early_file,map_location='cpu',weights_only=True)
    prototypes=torch.load(prototypes_file,map_location='cpu',weights_only=True)
    require(digest(early_file)==rec['prompt_early_sha256'] and digest(prototypes_file)==rec['prototypes_sha256'] and
            early_record['frame']==rec['prompt_frame'] and early_record['early'].shape==(32,256,256) and
            torch.isfinite(early_record['early']).all() and prototypes['foreground'].shape==prototypes['background'].shape==(32,) and
            torch.isfinite(prototypes['foreground']).all() and torch.isfinite(prototypes['background']).all(),
            'Warm prompt feature or prototype geometry differs')
    foreground,background=reference_prototypes(early_record['early'],mask)
    require(torch.equal(foreground,prototypes['foreground']) and torch.equal(background,prototypes['background']),
            'Warm prototypes do not derive from archived conditioning mask')
    return prototypes
