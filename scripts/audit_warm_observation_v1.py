"""Validate a saved warm decoder observation and its native video geometry."""
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts.audit_crossfit_observer_capture_v1 import tensor_tree_finite
from scripts.memory_token_io import digest
from scripts.crossfit_observer_capture_shared_v1 import require


def validate_observation(item,folder,roi):
    values=[]
    for kind in ('feature','snapshot'):
        path=Path(item[kind+'_file']).resolve(strict=True)
        require(path.is_relative_to(Path(folder).resolve()) and digest(path)==item[kind+'_sha256'],
                'Warm tensor path or digest differs')
        value=torch.load(path,map_location='cpu',weights_only=True)
        require(digest(path)==item[kind+'_sha256'] and tensor_tree_finite(value),'Warm tensor changed or is nonfinite')
        values.append(value)
    feature,snapshot=values;frame=item['frame']
    early=feature['feature'];deep=snapshot['decoder_inputs']['backbone_features']
    low=snapshot['decoder_outputs'][3];high=snapshot['decoder_outputs'][4]
    require(feature['frame']==snapshot['frame']==frame and early.shape==(32,256,256) and
            feature['foreground'].shape==feature['background'].shape==(32,) and
            deep.shape==(1,256,64,64) and low.shape==(1,1,256,256) and high.shape==(1,1,1024,1024),
            'Warm observation frame or tensor geometry differs')
    high_features=snapshot['decoder_inputs'].get('high_res_features')
    require(isinstance(high_features,(list,tuple)) and len(high_features)>0 and
            torch.equal(early.unsqueeze(0),high_features[0]),'Warm early feature and decoder input differ')
    require((snapshot['video_height'],snapshot['video_width'])==(roi[3]-roi[1],roi[2]-roi[0]),
            'Warm observation viewport geometry differs')
    require(torch.allclose(F.interpolate(low.float(),size=(1024,1024),mode='bilinear',align_corners=False),
                           high.float(),atol=1e-5,rtol=1e-6),'Warm decoder resize differs')
    return feature,snapshot
