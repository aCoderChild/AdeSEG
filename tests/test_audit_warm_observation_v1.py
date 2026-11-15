import pytest
import torch
from scripts.audit_warm_observation_v1 import validate_observation
from scripts.memory_token_io import digest


def save_case(folder,frame=3,height=720,width=834,bad_resize=False):
    feature=folder/'feature.pt';snapshot=folder/'snapshot.pt'
    torch.save({'frame':frame,'feature':torch.zeros(32,256,256),
                'foreground':torch.zeros(32),'background':torch.zeros(32)},feature)
    high=torch.ones(1,1,1024,1024) if bad_resize else torch.zeros(1,1,1024,1024)
    torch.save({'frame':frame,'video_height':height,'video_width':width,
        'decoder_inputs':{'backbone_features':torch.zeros(1,256,64,64),
                          'high_res_features':[torch.zeros(1,32,256,256)]},
        'decoder_outputs':[None,None,None,torch.zeros(1,1,256,256),high]},snapshot)
    return {'frame':frame,'feature_file':str(feature),'feature_sha256':digest(feature),
            'snapshot_file':str(snapshot),'snapshot_sha256':digest(snapshot)}


def test_seq23_roi_frame_and_decoder_resize_are_verified(tmp_path):
    item=save_case(tmp_path);roi=[446,0,1280,720]
    validate_observation(item,tmp_path,roi)
    with pytest.raises(ValueError,match='viewport geometry'):
        validate_observation(item,tmp_path,[0,0,1280,720])
    item['frame']=4
    with pytest.raises(ValueError,match='frame or tensor geometry'):
        validate_observation(item,tmp_path,roi)
    item=save_case(tmp_path,bad_resize=True)
    with pytest.raises(ValueError,match='decoder resize'):
        validate_observation(item,tmp_path,roi)


def test_mismatched_early_decoder_input_is_rejected(tmp_path):
    item=save_case(tmp_path);path=tmp_path/'snapshot.pt'
    snapshot=torch.load(path,map_location='cpu',weights_only=True)
    snapshot['decoder_inputs']['high_res_features'][0][0,0,0,0]=1
    torch.save(snapshot,path);item['snapshot_sha256']=digest(path)
    with pytest.raises(ValueError,match='early feature and decoder input'):
        validate_observation(item,tmp_path,[446,0,1280,720])
